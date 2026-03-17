from dataclasses import dataclass, field

from simulon.collective.common import P2PFlow

NVLS_CHUNK_COUNT = 4


@dataclass
class NVSwitchReduceNode:
    node_id: int
    nvswitch_rank: int
    chunk_id: int
    chunk_count: int


@dataclass
class NVLSResult:
    gather_flows: list[P2PFlow]
    scatter_flows: list[P2PFlow]
    switch_nodes: list[NVSwitchReduceNode]


def nvls_all_reduce(
    group_ranks: list[int],
    data_size: int,
    nvswitch_id: int | None = None,
    flow_id_start: int = 0,
    node_id_start: int = 0,
) -> tuple[list[P2PFlow], list[NVSwitchReduceNode], int, int]:
    """NVLS AllReduce (intra-node only).

    Pattern per chunk:
      Phase 1 (gather): each GPU sends to NVSwitch
      NVSwitch reduce node
      Phase 2 (scatter): NVSwitch sends to each GPU

    Returns (all_flows, switch_nodes, next_flow_id, next_node_id).
    """
    N = len(group_ranks)
    chunk_size = data_size // NVLS_CHUNK_COUNT

    if nvswitch_id is None:
        nvswitch_id = max(group_ranks) + 1

    fid = flow_id_start
    nid = node_id_start

    gather_flows: list[P2PFlow] = []
    scatter_flows: list[P2PFlow] = []
    switch_nodes: list[NVSwitchReduceNode] = []

    for ck in range(NVLS_CHUNK_COUNT):
        # Phase 1: gather (each GPU -> NVSwitch)
        chunk_gather_fids: list[int] = []
        for gpu_rank in group_ranks:
            flow = P2PFlow(
                flow_id=fid,
                src=gpu_rank,
                dst=nvswitch_id,
                flow_size=chunk_size,
                parent_flow_ids=[],
                child_flow_ids=[],
                channel_id=0,
                chunk_id=ck,
                chunk_count=NVLS_CHUNK_COUNT,
                conn_type="NVLS",
            )
            gather_flows.append(flow)
            chunk_gather_fids.append(fid)
            fid += 1

        # NVSwitch reduce node
        sw_node = NVSwitchReduceNode(
            node_id=nid,
            nvswitch_rank=nvswitch_id,
            chunk_id=ck,
            chunk_count=NVLS_CHUNK_COUNT,
        )
        switch_nodes.append(sw_node)
        nid += 1

        # Phase 2: scatter (NVSwitch -> each GPU)
        # parent_flow_ids = all gather flows for this chunk: the NVSwitch must have
        # received data from every GPU before it can send the reduced result to any GPU.
        # The replayer only uses parent_flow_ids to build the dependency graph;
        # child_flow_ids is ignored. Without this, scatter flows start at t=0,
        # bypassing the gather+reduce phases entirely.
        chunk_scatter_fids: list[int] = []
        for gpu_rank in group_ranks:
            flow = P2PFlow(
                flow_id=fid,
                src=nvswitch_id,
                dst=gpu_rank,
                flow_size=chunk_size,
                parent_flow_ids=list(chunk_gather_fids),
                child_flow_ids=[],
                channel_id=0,
                chunk_id=ck,
                chunk_count=NVLS_CHUNK_COUNT,
                conn_type="NVLS",
            )
            scatter_flows.append(flow)
            chunk_scatter_fids.append(fid)
            fid += 1

        # Link: each gather flow's child = corresponding scatter flow (same GPU)
        for i, gpu_rank in enumerate(group_ranks):
            g_flow = gather_flows[-(N - i)]  # gather flow for this gpu in this chunk
            s_flow = scatter_flows[-(N - i)]  # scatter flow for this gpu in this chunk
            g_flow.child_flow_ids = [s_flow.flow_id]

    all_flows = gather_flows + scatter_flows
    return all_flows, switch_nodes, fid, nid
