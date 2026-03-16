from simulon.collective.common import P2PFlow


def ring_reduce_scatter(
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    """Ring ReduceScatter across group_ranks.

    N = len(group_ranks), nsteps = N - 1
    chunk_size = data_size // N // num_channels
    Step s, channel c, rank i sends chunk (i - s) % N to rank (i+1) % N.
    parent_flow_ids: flow from step s-1 on this rank for same channel (empty for s=0).
    """
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    nsteps = N - 1
    chunk_size = data_size // N // num_channels
    chunk_count = N * num_channels

    fid = flow_id_start
    flows: list[P2PFlow] = []

    # Track flows by (step, channel, rank_index) -> flow_id for parent linkage
    # rank_index i is the position in group_ranks
    flow_table: dict[tuple[int, int, int], int] = {}

    for s in range(nsteps):
        for c in range(num_channels):
            for i in range(N):
                src_rank = group_ranks[i]
                dst_rank = group_ranks[(i + 1) % N]
                chunk_id = (i - s) % N

                parent_flow_ids: list[int] = []
                if s > 0:
                    parent_fid = flow_table.get((s - 1, c, i))
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

    Step s, channel c, rank i sends chunk (i - s) % N to rank (i+1) % N.
    parent_flow_ids: at step s, rank i receives from rank (i-1)%N
      => parent is flow at step s-1 where src=ranks[(i-1)%N], dst=ranks[i]
      => that is flow at (step=s-1, channel=c, rank_index=(i-1)%N).
    """
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    nsteps = N - 1
    chunk_size = data_size // N // num_channels
    chunk_count = N * num_channels

    fid = flow_id_start
    flows: list[P2PFlow] = []

    flow_table: dict[tuple[int, int, int], int] = {}

    for s in range(nsteps):
        for c in range(num_channels):
            for i in range(N):
                src_rank = group_ranks[i]
                dst_rank = group_ranks[(i + 1) % N]
                chunk_id = (i - s) % N

                parent_flow_ids: list[int] = []
                if s > 0:
                    # rank i receives from (i-1)%N at step s-1
                    sender_idx = (i - 1) % N
                    parent_fid = flow_table.get((s - 1, c, sender_idx))
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
    on same rank and same channel.
    """
    N = len(group_ranks)
    if N == 1:
        return [], flow_id_start

    rs_flows, fid = ring_reduce_scatter(group_ranks, data_size, num_channels, flow_id_start)
    ag_flows, fid = ring_all_gather(group_ranks, data_size, num_channels, fid)

    # Connect: AllGather step-0 flows' parent_flow_ids = ReduceScatter final-step flows
    # RS final step = N-2; AG step 0 flows have no parents yet (parent_flow_ids=[])
    # RS final step flows: last N*num_channels flows in rs_flows
    nsteps_rs = N - 1
    rs_final_step_flows = rs_flows[(nsteps_rs - 1) * N * num_channels:]

    # Build lookup: (channel, rank_index) -> RS final step flow_id
    rs_final_by_rank_channel: dict[tuple[int, int], int] = {}
    step_idx = nsteps_rs - 1
    idx = 0
    for c in range(num_channels):
        for i in range(N):
            flo = rs_final_step_flows[idx]
            rs_final_by_rank_channel[(c, i)] = flo.flow_id
            idx += 1

    # AG step-0 flows are the first N*num_channels flows in ag_flows
    ag_step0_flows = ag_flows[:N * num_channels]
    idx = 0
    for c in range(num_channels):
        for i in range(N):
            ag_flow = ag_step0_flows[idx]
            rs_parent_fid = rs_final_by_rank_channel.get((c, i))
            if rs_parent_fid is not None:
                ag_flow.parent_flow_ids = [rs_parent_fid]
                # Update RS final flow's child_flow_ids
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
    """AllToAll: one direct flow for each ordered pair (i, j) where i != j."""
    N = len(group_ranks)
    chunk_size = data_size // N
    chunk_count = 1

    fid = flow_id_start
    flows: list[P2PFlow] = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            flow = P2PFlow(
                flow_id=fid,
                src=group_ranks[i],
                dst=group_ranks[j],
                flow_size=chunk_size,
                parent_flow_ids=[],
                child_flow_ids=[],
                channel_id=0,
                chunk_id=j,
                chunk_count=chunk_count,
                conn_type="RING",
            )
            flows.append(flow)
            fid += 1

    return flows, fid


def _get_children(flow: P2PFlow, all_flows: list[P2PFlow]) -> list[int]:
    """Return flow_ids of flows whose parent_flow_ids include this flow's flow_id."""
    return [f.flow_id for f in all_flows if flow.flow_id in f.parent_flow_ids]
