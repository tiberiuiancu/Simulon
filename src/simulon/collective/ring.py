from simulon.collective.common import P2PFlow


def _get_ring_topology(
    N: int,
    num_nodes: int,
    gpus_per_node: int,
    num_channels: int,
) -> dict[int, dict[int, int]]:
    """Query MockNcclGroup for ring channel topology.

    Returns {channel_id: {internal_rank: next_internal_rank}}.
    Falls back to sequential ring if the C++ binding is unavailable.
    Uses only the first num_channels channels.
    """
    try:
        import simulon._mocknccl as _m

        nvswitches = [N + i for i in range(num_nodes)]
        g = _m.MockNcclGroup(
            N, gpus_per_node,
            N, 1, 1, 1, 1,
            nvswitches,
            _m.GPUType.H100,
        )
        raw = g.genringchannels(0, _m.GroupType.TP)
    except Exception:
        # Binding unavailable or topology query failed — fall back to sequential ring.
        ch_map: dict[int, dict[int, int]] = {}
        for ch in range(num_channels):
            ch_map[ch] = {i: (i + 1) % N for i in range(N)}
        return ch_map

    ch_ids = sorted(raw.keys())[:num_channels]
    topology: dict[int, dict[int, int]] = {}
    for ch_id in ch_ids:
        # raw[ch_id][rank] = [prev, next, ring_start, ring_end]
        topology[ch_id] = {rank: info[1] for rank, info in raw[ch_id].items()}

    # Safety net: if MockNcclGroup returned no channels, fall back to sequential.
    if not topology:
        for ch in range(num_channels):
            topology[ch] = {i: (i + 1) % N for i in range(N)}
    return topology


def _infer_ring_params(group_ranks: list[int]) -> tuple[int, int]:
    """Return (num_nodes, gpus_per_node) inferred from group_ranks spacing.

    Three cases:
    - Consecutive ranks (stride=1): intra-node TP-style group, gpus_per_node=min(N,8).
    - Stride < N: inter-node stride pattern, gpus_per_node=min_stride.
    - Stride >= N: each rank is on its own node (e.g. DP pair [0,64] in 8-GPU nodes),
      gpus_per_node=1, num_nodes=N.
    """
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
        # stride >= N: sparse group where each GPU is on a different node
        gpus_per_node = 1
    num_nodes = max(1, N // gpus_per_node)
    return num_nodes, gpus_per_node


def ring_reduce_scatter(
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    """Ring ReduceScatter across group_ranks.

    N = len(group_ranks), nsteps = N - 1
    chunk_size = data_size // N // num_channels
    Step s, channel c, rank i sends chunk (i - s) % N to its ring-next neighbor.
    Ring neighbor order comes from MockNcclGroup.genringchannels (falls back to
    sequential if the C++ binding is unavailable).
    parent_flow_ids: flow from step s-1, same channel, from the ring-previous sender.
    """
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

    # Track flows by (step, channel, rank_index) -> flow_id for parent linkage
    flow_table: dict[tuple[int, int, int], int] = {}

    for s in range(nsteps):
        for c in range(num_channels):
            # next_of[i] = internal rank index that rank i sends to in this channel
            next_of = topology.get(c, topology[min(topology)])
            # prev_of[i] = who sends to i (inverse of next_of)
            prev_of = {v: k for k, v in next_of.items()}

            for i in range(N):
                src_rank = group_ranks[i]
                dst_rank = group_ranks[next_of[i]]
                chunk_id = (i - s) % N

                parent_flow_ids: list[int] = []
                if s > 0:
                    # rank i receives from prev_of[i] at step s-1
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

    # Back-fill child_flow_ids
    for flow in flows:
        for child_fid in _get_children(flow, flows):
            if child_fid not in flow.child_flow_ids:
                flow.child_flow_ids.append(child_fid)

    return flows, fid


def ring_all_gather(
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    """Ring AllGather across group_ranks.

    Step s, channel c, rank i sends chunk (i - s) % N to its ring-next neighbor.
    Ring neighbor order comes from MockNcclGroup.genringchannels (falls back to
    sequential if the C++ binding is unavailable).
    parent_flow_ids: at step s, rank i receives from its ring-previous neighbor.
    """
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

    # Back-fill child_flow_ids
    for flow in flows:
        for child_fid in _get_children(flow, flows):
            if child_fid not in flow.child_flow_ids:
                flow.child_flow_ids.append(child_fid)

    return flows, fid


def ring_all_reduce(
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    """Ring AllReduce = ReduceScatter + AllGather.

    AllGather step-0 parent_flow_ids = ReduceScatter final-step flows
    on the same rank and channel. Ring topology from MockNcclGroup.
    """
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    num_nodes, gpus_per_node = _infer_ring_params(group_ranks)
    topology = _get_ring_topology(N, num_nodes, gpus_per_node, num_channels)

    rs_flows, fid = ring_reduce_scatter(group_ranks, data_size, num_channels, flow_id_start)
    ag_flows, fid = ring_all_gather(group_ranks, data_size, num_channels, fid)

    # Connect RS final-step flows to AG step-0 flows.
    # At AG step 0, rank i receives from its ring-previous neighbor (prev_of[i]).
    # So AG step-0 for rank i depends on the RS final-step flow *sent by* prev_of[i].
    nsteps_rs = N - 1
    rs_final_step_flows = rs_flows[(nsteps_rs - 1) * N * num_channels:]

    rs_final_by_rank_channel: dict[tuple[int, int], int] = {}
    idx = 0
    for c in range(num_channels):
        for i in range(N):
            rs_final_by_rank_channel[(c, i)] = rs_final_step_flows[idx].flow_id
            idx += 1

    ag_step0_flows = ag_flows[:N * num_channels]
    idx = 0
    for c in range(num_channels):
        next_of = topology.get(c, topology[min(topology)])
        prev_of = {v: k for k, v in next_of.items()}
        for i in range(N):
            ag_flow = ag_step0_flows[idx]
            rs_parent_fid = rs_final_by_rank_channel.get((c, prev_of[i]))
            if rs_parent_fid is not None:
                ag_flow.parent_flow_ids = [rs_parent_fid]
                rs_flow = next(f for f in rs_final_step_flows if f.flow_id == rs_parent_fid)
                if ag_flow.flow_id not in rs_flow.child_flow_ids:
                    rs_flow.child_flow_ids.append(ag_flow.flow_id)
            idx += 1

    return rs_flows + ag_flows, fid


def ring_all_to_all(
    group_ranks: list[int],
    data_size: int,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    """AllToAll using a conflict-free round-robin schedule.

    Naive all-parallel AllToAll (one direct flow per ordered pair, all independent)
    is physically wrong: each GPU would appear to send on N-1 links simultaneously
    at full bandwidth, implying N-1× the actual NVLink budget.

    The correct model uses N-1 rounds:
      Round r: GPU_i sends to GPU_{(i+r+1)%N} for all i (simultaneously).
    Within each round, all N flows are on distinct src-dst pairs — no link contention.
    Between rounds, each GPU's outgoing sends are serialized via parent_flow_ids.

    This gives:  time ≈ (N-1) × (chunk_size/link_bw + per_step_latency)
                 bus_bw → link_bw at saturation  (alg_bw = N/(N-1) × link_bw).
    """
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    chunk_size = data_size // N
    chunk_count = N - 1  # N-1 rounds

    fid = flow_id_start
    flows: list[P2PFlow] = []

    # prev_send_fid[i] = flow_id of the last outgoing flow from internal rank i.
    # Used to chain each GPU's sends serially across rounds.
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

    # Back-fill child_flow_ids from parent_flow_ids
    fid_to_flow = {f.flow_id: f for f in flows}
    for flow in flows:
        for pid in flow.parent_flow_ids:
            parent_flow = fid_to_flow.get(pid)
            if parent_flow and flow.flow_id not in parent_flow.child_flow_ids:
                parent_flow.child_flow_ids.append(flow.flow_id)

    return flows, fid


def _get_children(flow: P2PFlow, all_flows: list[P2PFlow]) -> list[int]:
    """Return flow_ids of flows whose parent_flow_ids include this flow's flow_id."""
    return [f.flow_id for f in all_flows if flow.flow_id in f.parent_flow_ids]
