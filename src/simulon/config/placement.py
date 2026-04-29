from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, TYPE_CHECKING

from .dc import DatacenterConfig

if TYPE_CHECKING:
    from .scenario import WorkloadInstance
else:
    WorkloadInstance = Any


@dataclass(frozen=True, slots=True)
class NodeSlice:
    start_node: int
    end_node: int
    start_gpu_rank: int
    end_gpu_rank: int
    num_gpus: int


def _unwrap_workload(instance: WorkloadInstance) -> tuple[str, Any]:
    workload = getattr(instance, "workload", instance)
    name = getattr(instance, "name", None) or getattr(workload, "name", None)
    if not name:
        raise ValueError("workload instance is missing a name")
    return name, workload


def _raw_num_gpus(workload: Any, datacenter: DatacenterConfig) -> int:
    framework = getattr(workload, "framework", None)
    if framework == "megatron":
        return int(workload.training.num_gpus)
    if framework == "inference":
        return int(workload.inference.num_gpus)
    if framework == "collective":
        num_gpus = getattr(workload, "num_gpus", None)
        if num_gpus is not None:
            return int(num_gpus)
        return int(datacenter.cluster.num_nodes * datacenter.node.gpus_per_node)
    raise ValueError(f"unsupported workload framework: {framework!r}")


def place_workloads(workloads: list[WorkloadInstance], datacenter: DatacenterConfig) -> dict[str, NodeSlice]:
    gpus_per_node = datacenter.node.gpus_per_node
    if gpus_per_node is None:
        raise ValueError("datacenter.node.gpus_per_node is required for placement")

    placements: dict[str, NodeSlice] = {}
    current_node = 0

    for instance in workloads:
        name, workload = _unwrap_workload(instance)
        raw_num_gpus = _raw_num_gpus(workload, datacenter)
        aligned_num_gpus = max(gpus_per_node, ceil(raw_num_gpus / gpus_per_node) * gpus_per_node)
        num_nodes = aligned_num_gpus // gpus_per_node

        start_node = current_node
        end_node = start_node + num_nodes - 1
        start_gpu_rank = start_node * gpus_per_node
        end_gpu_rank = start_gpu_rank + aligned_num_gpus - 1

        placements[name] = NodeSlice(
            start_node=start_node,
            end_node=end_node,
            start_gpu_rank=start_gpu_rank,
            end_gpu_rank=end_gpu_rank,
            num_gpus=aligned_num_gpus,
        )
        current_node = end_node + 1

    return placements
