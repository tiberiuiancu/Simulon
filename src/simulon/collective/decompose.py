from dataclasses import dataclass, field

from simulon.collective.common import P2PFlow
from simulon.collective.collnet import collnet_chain_all_reduce, collnet_direct_all_reduce
from simulon.collective.nvls import nvls_all_reduce, nvls_tree_all_reduce
from simulon.collective.ring import (
    ring_all_gather,
    ring_all_reduce,
    ring_all_to_all,
    ring_reduce_scatter,
)
from simulon.collective.tree import tree_all_reduce


@dataclass
class CollectiveResult:
    flows: list[P2PFlow]


def decompose_collective(
    collective_type: str,
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    algorithm: str = "ring",
    flow_id_start: int = 0,
    node_id_start: int = 0,
) -> tuple[CollectiveResult, int, int]:
    """Decompose a collective into P2PFlows.

    Returns (result, next_flow_id, next_node_id).
    """
    flows: list[P2PFlow] = []

    if algorithm == "tree" and collective_type == "AllReduce":
        flows, next_flow_id = tree_all_reduce(
            group_ranks=group_ranks,
            data_size=data_size,
            num_channels=num_channels,
            flow_id_start=flow_id_start,
        )
    elif algorithm == "collnet_direct" and collective_type == "AllReduce":
        flows, next_flow_id = collnet_direct_all_reduce(
            group_ranks=group_ranks,
            data_size=data_size,
            num_channels=num_channels,
            flow_id_start=flow_id_start,
        )
    elif algorithm == "collnet_chain" and collective_type == "AllReduce":
        flows, next_flow_id = collnet_chain_all_reduce(
            group_ranks=group_ranks,
            data_size=data_size,
            num_channels=num_channels,
            flow_id_start=flow_id_start,
        )
    elif algorithm == "nvls" and collective_type == "AllReduce":
        flows, next_flow_id = nvls_all_reduce(
            group_ranks=group_ranks,
            data_size=data_size,
            num_channels=num_channels,
            flow_id_start=flow_id_start,
        )
    elif algorithm == "nvls_tree" and collective_type == "AllReduce":
        flows, next_flow_id = nvls_tree_all_reduce(
            group_ranks=group_ranks,
            data_size=data_size,
            num_channels=num_channels,
            flow_id_start=flow_id_start,
        )
    elif algorithm not in ("ring", "tree", "collnet_direct", "collnet_chain", "nvls", "nvls_tree"):
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose from: ring, tree, collnet_direct, collnet_chain, nvls, nvls_tree")
    elif collective_type == "AllReduce":
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
