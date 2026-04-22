from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from .dc import DatacenterConfig
from .workload import WorkloadConfig


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
    workload: Union[Path, WorkloadConfig]
    collective: CollectiveConfig = Field(default_factory=NcclConfig)
