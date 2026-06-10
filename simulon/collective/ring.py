from functools import lru_cache

from simulon.collective.common import P2PFlow


@lru_cache(maxsize=128)
def _get_ring_topology(
    N: int, num_nodes: int, gpus_per_node: int, num_channels: int
) -> dict[int, dict[int, int]]:
    """Query MockNcclGroup for ring channel topology (cached)."""
    try:
        import simulon._mocknccl as _m

        nvswitches = [N + i for i in range(num_nodes)]
        g = _m.MockNcclGroup(N, gpus_per_node, N, 1, 1, 1, 1, nvswitches, _m.GPUType.H100)
        raw = g.genringchannels(0, _m.GroupType.TP)
    except Exception:
        ch_map: dict[int, dict[int, int]] = {}
        for ch in range(num_channels):
            ch_map[ch] = {i: (i + 1) % N for i in range(N)}
        return ch_map

    ch_ids = sorted(raw.keys())[:num_channels]
    topology: dict[int, dict[int, int]] = {}
    for ch_id in ch_ids:
        topology[ch_id] = {rank: info[1] for rank, info in raw[ch_id].items()}

    if not topology:
        for ch in range(num_channels):
            topology[ch] = {i: (i + 1) % N for i in range(N)}
    return topology


@lru_cache(maxsize=256)
def _infer_ring_params_cached(group_ranks: tuple[int, ...]) -> tuple[int, int]:
    """Return (num_nodes, gpus_per_node) inferred from group_ranks spacing (cached)."""
    N = len(group_ranks)
    if N <= 1:
        return 1, max(1, N)
    strides = [group_ranks[i + 1] - group_ranks[i] for i in range(N - 1)]
    min_stride = min(strides)
    if min_stride == 1:
        gpus_per_node = min(N, 8)
    elif min_stride < N:
        gpus_per_node = min_stride
    else:
        gpus_per_node = 1
    num_nodes = max(1, N // gpus_per_node)
    return num_nodes, gpus_per_node


def _infer_ring_params(group_ranks: list[int]) -> tuple[int, int]:
    return _infer_ring_params_cached(tuple(group_ranks))


def ring_reduce_scatter(
    group_ranks: list[int], data_size: int, num_channels: int = 1, flow_id_start: int = 0
) -> tuple[list[P2PFlow], int]:
    """Ring ReduceScatter across group_ranks."""
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    num_nodes, gpus_per_node = _infer_ring_params(group_ranks)
    topology = _get_ring_topology(N, num_nodes, gpus_per_node, num_channels)

    nsteps = N - 1
    chunk_size = data_size // N // num_channels
    chunk_count = N * num_channels

    fid = flow_id_start
    flows: list[P2PFlow] = []
    flow_table: dict[tuple[int, int, int], int] = {}

    for s in range(nsteps):
        for c in range(num_channels):
            next_of = topology.get(c, topology[min(topology)])
            prev_of = {v: k for k, v in next_of.items()}

            for i in range(N):
                src_rank = group_ranks[i]
                dst_rank = group_ranks[next_of[i]]
                chunk_id = (i - s) % N

                parent_flow_ids: list[int] = []
                if s > 0:
                    parent_fid = flow_table.get((s - 1, c, prev_of[i]))
                    if parent_fid is not None:
                        parent_flow_ids = [parent_fid]

                flow = P2PFlow(
                    flow_id=fid,
                    src=src_rank,
                    dst=dst_rank,
                    flow_size=chunk_size,
                    parent_flow_ids=parent_flow_ids,
                    child_flow_ids=[],
                    channel_id=c,
                    chunk_id=chunk_id,
                    chunk_count=chunk_count,
                    conn_type="RING",
                )
                flow_table[(s, c, i)] = fid
                flows.append(flow)
                fid += 1

    fid_to_flow = {f.flow_id: f for f in flows}
    for flow in flows:
        for pid in flow.parent_flow_ids:
            parent_flow = fid_to_flow.get(pid)
            if parent_flow and flow.flow_id not in parent_flow.child_flow_ids:
                parent_flow.child_flow_ids.append(flow.flow_id)

    return flows, fid


def ring_all_gather(
    group_ranks: list[int], data_size: int, num_channels: int = 1, flow_id_start: int = 0
) -> tuple[list[P2PFlow], int]:
    """Ring AllGather across group_ranks."""
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    num_nodes, gpus_per_node = _infer_ring_params(group_ranks)
    topology = _get_ring_topology(N, num_nodes, gpus_per_node, num_channels)

    nsteps = N - 1
    chunk_size = data_size // N // num_channels
    chunk_count = N * num_channels

    fid = flow_id_start
    flows: list[P2PFlow] = []
    flow_table: dict[tuple[int, int, int], int] = {}

    for s in range(nsteps):
        for c in range(num_channels):
            next_of = topology.get(c, topology[min(topology)])
            prev_of = {v: k for k, v in next_of.items()}

            for i in range(N):
                src_rank = group_ranks[i]
                dst_rank = group_ranks[next_of[i]]
                chunk_id = (i - s) % N

                parent_flow_ids: list[int] = []
                if s > 0:
                    parent_fid = flow_table.get((s - 1, c, prev_of[i]))
                    if parent_fid is not None:
                        parent_flow_ids = [parent_fid]

                flow = P2PFlow(
                    flow_id=fid,
                    src=src_rank,
                    dst=dst_rank,
                    flow_size=chunk_size,
                    parent_flow_ids=parent_flow_ids,
                    child_flow_ids=[],
                    channel_id=c,
                    chunk_id=chunk_id,
                    chunk_count=chunk_count,
                    conn_type="RING",
                )
                flow_table[(s, c, i)] = fid
                flows.append(flow)
                fid += 1

    fid_to_flow = {f.flow_id: f for f in flows}
    for flow in flows:
        for pid in flow.parent_flow_ids:
            parent_flow = fid_to_flow.get(pid)
            if parent_flow and flow.flow_id not in parent_flow.child_flow_ids:
                parent_flow.child_flow_ids.append(flow.flow_id)

    return flows, fid


def ring_all_reduce(
    group_ranks: list[int], data_size: int, num_channels: int = 1, flow_id_start: int = 0
) -> tuple[list[P2PFlow], int]:
    """Ring AllReduce = ReduceScatter + AllGather."""
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    rs_flows, fid = ring_reduce_scatter(group_ranks, data_size, num_channels, flow_id_start)
    ag_flows, fid = ring_all_gather(group_ranks, data_size, num_channels, fid)

    nsteps_rs = N - 1
    rs_final_step_flows = rs_flows[(nsteps_rs - 1) * N * num_channels :]

    rs_final_by_rank_channel: dict[tuple[int, int], int] = {}
    rs_final_by_fid: dict[int, P2PFlow] = {}
    idx = 0
    for c in range(num_channels):
        for i in range(N):
            flow = rs_final_step_flows[idx]
            rs_final_by_rank_channel[(c, i)] = flow.flow_id
            rs_final_by_fid[flow.flow_id] = flow
            idx += 1

    num_nodes, gpus_per_node = _infer_ring_params(group_ranks)
    topology = _get_ring_topology(N, num_nodes, gpus_per_node, num_channels)

    ag_step0_flows = ag_flows[: N * num_channels]
    idx = 0
    for c in range(num_channels):
        next_of = topology.get(c, topology[min(topology)])
        prev_of = {v: k for k, v in next_of.items()}
        for i in range(N):
            ag_flow = ag_step0_flows[idx]
            rs_parent_fid = rs_final_by_rank_channel.get((c, prev_of[i]))
            if rs_parent_fid is not None:
                ag_flow.parent_flow_ids = [rs_parent_fid]
                rs_flow_obj = rs_final_by_fid[rs_parent_fid]
                if ag_flow.flow_id not in rs_flow_obj.child_flow_ids:
                    rs_flow_obj.child_flow_ids.append(ag_flow.flow_id)
            idx += 1

    return rs_flows + ag_flows, fid


def ring_all_to_all(
    group_ranks: list[int], data_size: int, flow_id_start: int = 0
) -> tuple[list[P2PFlow], int]:
    """AllToAll using a conflict-free round-robin schedule."""
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    chunk_size = data_size // N
    chunk_count = N - 1

    fid = flow_id_start
    flows: list[P2PFlow] = []
    prev_send_fid: dict[int, int] = {}

    for r in range(N - 1):
        for i in range(N):
            dst_idx = (i + r + 1) % N
            src = group_ranks[i]
            dst = group_ranks[dst_idx]

            parent_flow_ids = [prev_send_fid[i]] if i in prev_send_fid else []

            flow = P2PFlow(
                flow_id=fid,
                src=src,
                dst=dst,
                flow_size=chunk_size,
                parent_flow_ids=parent_flow_ids,
                child_flow_ids=[],
                channel_id=0,
                chunk_id=dst_idx,
                chunk_count=chunk_count,
                conn_type="RING",
            )
            prev_send_fid[i] = fid
            flows.append(flow)
            fid += 1

    fid_to_flow = {f.flow_id: f for f in flows}
    for flow in flows:
        for pid in flow.parent_flow_ids:
            parent_flow = fid_to_flow.get(pid)
            if parent_flow and flow.flow_id not in parent_flow.child_flow_ids:
                parent_flow.child_flow_ids.append(flow.flow_id)

    return flows, fid
