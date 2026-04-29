from pathlib import Path
from math import ceil
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

from .dc import DatacenterConfig
from .workload import WorkloadConfig
from .placement import _raw_num_gpus


# ---------------------------------------------------------------------------
# Collective communication library config
# ---------------------------------------------------------------------------


class NcclConfig(BaseModel):
    library: Literal["nccl"] = "nccl"
    algorithm: str = "auto"   # auto | ring | tree | collnet_direct | collnet_chain | nvls | nvls_tree
    num_channels: int = 1


class RcclConfig(BaseModel):
    library: Literal["rccl"] = "rccl"
    algorithm: str = "ring"
    num_channels: int = 1


CollectiveConfig = Annotated[
    Union[NcclConfig, RcclConfig],
    Field(discriminator="library"),
]


# ---------------------------------------------------------------------------
# Workload start / instances
# ---------------------------------------------------------------------------


class StartConfig(BaseModel):
    offset_ms: float = 0.0
    after_finish: list[str] = Field(default_factory=list)


class WorkloadInstance(BaseModel):
    name: str
    workload: Union[Path, WorkloadConfig]
    start: StartConfig = Field(default_factory=StartConfig)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class ScenarioConfig(BaseModel):
    datacenter: Union[Path, DatacenterConfig]
    workloads: list[WorkloadInstance] = Field(default_factory=list)
    collective: CollectiveConfig = Field(default_factory=NcclConfig)

    @property
    def workload(self) -> Union[Path, WorkloadConfig]:
        if len(self.workloads) != 1:
            raise AttributeError("ScenarioConfig.workload is only available when exactly one workload exists")
        return self.workloads[0].workload

    @model_validator(mode="before")
    @classmethod
    def _coerce_workload_alias(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "workloads" not in data and "workload" in data:
            data = dict(data)
            data["workloads"] = [
                {
                    "name": "default",
                    "workload": data["workload"],
                    "start": {},
                }
            ]
        return data

    @model_validator(mode="after")
    def _validate_workloads(self) -> "ScenarioConfig":
        workloads = self.workloads
        names = [instance.name for instance in workloads]

        if len(names) != len(set(names)):
            seen: set[str] = set()
            dupes: set[str] = set()
            for name in names:
                if name in seen:
                    dupes.add(name)
                else:
                    seen.add(name)
            raise ValueError(f"duplicate workload names: {', '.join(dupes)}")

        graph = {instance.name: list(instance.start.after_finish) for instance in workloads}
        all_names = set(graph)

        missing = sorted({dep for deps in graph.values() for dep in deps if dep not in all_names})
        if missing:
            raise ValueError(f"unknown after_finish dependency names: {', '.join(missing)}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(name: str, path: list[str]) -> None:
            if name in visiting:
                cycle_start = path.index(name)
                cycle = path[cycle_start:] + [name]
                raise ValueError(f"cycle detected in after_finish graph: {' -> '.join(cycle)}")
            if name in visited:
                return
            visiting.add(name)
            path.append(name)
            for dep in graph[name]:
                dfs(dep, path)
            path.pop()
            visiting.remove(name)
            visited.add(name)

        for name in graph:
            if name not in visited:
                dfs(name, [])

        if isinstance(self.datacenter, DatacenterConfig):
            gpus_per_node = self.datacenter.node.gpus_per_node
            if gpus_per_node is None:
                raise ValueError("datacenter.node.gpus_per_node is required for workload GPU budget validation")

            total_gpus = 0
            for instance in workloads:
                if isinstance(instance.workload, Path):
                    return self
                raw_num_gpus = _raw_num_gpus(instance.workload, self.datacenter)
                total_gpus += max(gpus_per_node, ceil(raw_num_gpus / gpus_per_node) * gpus_per_node)

            capacity = self.datacenter.cluster.num_nodes * gpus_per_node
            if total_gpus > capacity:
                raise ValueError(
                    f"workload GPU demand {total_gpus} exceeds cluster capacity {capacity}"
                )

        return self
