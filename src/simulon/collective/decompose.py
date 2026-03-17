from dataclasses import dataclass, field

from simulon.collective.common import P2PFlow
from simulon.collective.ring import (
    ring_all_gather,
    ring_all_reduce,
    ring_all_to_all,
    ring_reduce_scatter,
)
from simulon.collective.nvls import NVSwitchReduceNode, nvls_all_reduce


@dataclass
class CollectiveResult:
    flows: list[P2PFlow]
    switch_nodes: list[NVSwitchReduceNode] = field(default_factory=list)


def decompose_collective(
    collective_type: str,
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    algorithm: str = "ring",
    nvswitch_id: int | None = None,
    flow_id_start: int = 0,
    node_id_start: int = 0,
) -> tuple[CollectiveResult, int, int]:
    """Decompose a collective into P2PFlows (and optional NVSwitch nodes).

    Returns (result, next_flow_id, next_node_id).
    """
    flows: list[P2PFlow] = []
    switch_nodes: list[NVSwitchReduceNode] = []
    next_flow_id = flow_id_start
    next_node_id = node_id_start

    if algorithm == "nvls":
        if collective_type != "AllReduce":
            # NVLS only supports AllReduce (intra-node reduction via NVSwitch).
            # All other collective types fall back to ring.
            algorithm = "ring"

    if algorithm == "nvls":
        flows, switch_nodes, next_flow_id, next_node_id = nvls_all_reduce(
            group_ranks=group_ranks,
            data_size=data_size,
            nvswitch_id=nvswitch_id,
            flow_id_start=flow_id_start,
            node_id_start=node_id_start,
        )
    elif algorithm == "ring":
        if collective_type == "AllReduce":
            flows, next_flow_id = ring_all_reduce(
                group_ranks=group_ranks,
                data_size=data_size,
                num_channels=num_channels,
                flow_id_start=flow_id_start,
            )
        elif collective_type == "ReduceScatter":
            flows, next_flow_id = ring_reduce_scatter(
                group_ranks=group_ranks,
                data_size=data_size,
                num_channels=num_channels,
                flow_id_start=flow_id_start,
            )
        elif collective_type == "AllGather":
            flows, next_flow_id = ring_all_gather(
                group_ranks=group_ranks,
                data_size=data_size,
                num_channels=num_channels,
                flow_id_start=flow_id_start,
            )
        elif collective_type == "AllToAll":
            flows, next_flow_id = ring_all_to_all(
                group_ranks=group_ranks,
                data_size=data_size,
                flow_id_start=flow_id_start,
            )
        else:
            raise ValueError(f"Unknown collective_type: {collective_type}")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    result = CollectiveResult(flows=flows, switch_nodes=switch_nodes)
    return result, next_flow_id, next_node_id
