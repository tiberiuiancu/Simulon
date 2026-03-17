from dataclasses import dataclass, field

from simulon.collective.common import P2PFlow
from simulon.collective.ring import (
    ring_all_gather,
    ring_all_reduce,
    ring_all_to_all,
    ring_reduce_scatter,
)


@dataclass
class CollectiveResult:
    flows: list[P2PFlow]


def decompose_collective(
    collective_type: str,
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
    node_id_start: int = 0,
    **_kwargs,
) -> tuple[CollectiveResult, int, int]:
    """Decompose a collective into P2PFlows.

    Returns (result, next_flow_id, next_node_id).
    """
    flows: list[P2PFlow] = []

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

    return CollectiveResult(flows=flows), next_flow_id, node_id_start
