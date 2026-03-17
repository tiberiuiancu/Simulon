from dataclasses import dataclass, field

from simulon.collective.common import P2PFlow

NVLS_CHUNK_COUNT = 4


@dataclass
class NVSwitchReduceNode:
    node_id: int
    nvswitch_rank: int
    chunk_id: int
    chunk_count: int
    flow_id: int                                        # virtual flow_id; scatter flows reference this
    parent_flow_ids: list[int] = field(default_factory=list)  # gather flow_ids for this chunk


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

    Dependency chain per chunk ck:
      gather flows (GPU→NVSwitch, parent=[])
        → NVSwitchReduceNode (parent=all gather flow_ids for ck, has its own flow_id)
          → scatter flows (NVSwitch→GPU, parent=[nvswitch_flow_id])

    The NVSwitchReduceNode is assigned a flow_id so that scatter CommNodes can
    list it in parent_flow_ids; the replayer resolves these via flow_to_node,
    enforcing gather→reduce→scatter ordering without a separate edge type.

    Returns (all_flows, switch_nodes, next_flow_id, next_node_id).
    """
    N = len(group_ranks)
    chunk_size = data_size // NVLS_CHUNK_COUNT

    if nvswitch_id is None:
        # Use a large offset to avoid colliding with real GPU ranks.
        # Real clusters are unlikely to exceed 100 000 GPUs; this places
        # NVSwitch IDs in the 1 000 000+ range so they never alias a GPU rank.
        nvswitch_id = 1_000_000 + max(group_ranks)

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

        # NVSwitch reduce node: depends on all gather flows for this chunk.
        # Assigned its own flow_id so scatter flows can reference it as a parent.
        sw_fid = fid
        fid += 1
        sw_node = NVSwitchReduceNode(
            node_id=nid,
            nvswitch_rank=nvswitch_id,
            chunk_id=ck,
            chunk_count=NVLS_CHUNK_COUNT,
            flow_id=sw_fid,
            parent_flow_ids=list(chunk_gather_fids),
        )
        switch_nodes.append(sw_node)
        nid += 1

        # Phase 2: scatter (NVSwitch -> each GPU)
        # parent_flow_ids = [sw_fid]: each scatter flow waits for the NVSwitch
        # to finish reducing before it sends the result back to its GPU.
        for gpu_rank in group_ranks:
            flow = P2PFlow(
                flow_id=fid,
                src=nvswitch_id,
                dst=gpu_rank,
                flow_size=chunk_size,
                parent_flow_ids=[sw_fid],
                child_flow_ids=[],
                channel_id=0,
                chunk_id=ck,
                chunk_count=NVLS_CHUNK_COUNT,
                conn_type="NVLS",
            )
            scatter_flows.append(flow)
            fid += 1

    all_flows = gather_flows + scatter_flows
    return all_flows, switch_nodes, fid, nid
