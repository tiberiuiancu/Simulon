from dataclasses import dataclass
from typing import Callable

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


_REGISTRY: dict[tuple[str, str], Callable] = {
    ("ring", "AllReduce"):           ring_all_reduce,
    ("ring", "AllGather"):           ring_all_gather,
    ("ring", "ReduceScatter"):       ring_reduce_scatter,
    ("ring", "AllToAll"):            ring_all_to_all,
    ("tree", "AllReduce"):           tree_all_reduce,
    ("collnet_direct", "AllReduce"): collnet_direct_all_reduce,
    ("collnet_chain",  "AllReduce"): collnet_chain_all_reduce,
    ("nvls",      "AllReduce"):      nvls_all_reduce,
    ("nvls_tree", "AllReduce"):      nvls_tree_all_reduce,
}


def decompose_collective(
    collective_type: str,
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    algorithm: str = "ring",
    flow_id_start: int = 0,
) -> tuple[CollectiveResult, int]:
    """Decompose a collective into P2PFlows.

    Returns (result, next_flow_id).
    """
    key = (algorithm, collective_type)
    fn = _REGISTRY.get(key)
    if fn is None:
        supported = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unsupported combination: algorithm={algorithm!r}, collective_type={collective_type!r}. "
            f"Supported (algorithm, collective_type) pairs: {supported}"
        )

    # AllToAll does not use num_channels
    if collective_type == "AllToAll":
        flows, next_flow_id = fn(
            group_ranks=group_ranks,
            data_size=data_size,
            flow_id_start=flow_id_start,
        )
    else:
        flows, next_flow_id = fn(
            group_ranks=group_ranks,
            data_size=data_size,
            num_channels=num_channels,
            flow_id_start=flow_id_start,
        )

    return CollectiveResult(flows=flows), next_flow_id
