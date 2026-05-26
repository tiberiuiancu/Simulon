from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .dc import DatacenterConfig
from .workload import WorkloadConfig

# ---------------------------------------------------------------------------
# Collective communication library config
# ---------------------------------------------------------------------------


class NcclConfig(BaseModel):
    library: Literal["nccl"] = "nccl"
    algorithm: str = (
        "auto"  # auto | ring | tree | collnet_direct | collnet_chain | nvls | nvls_tree
    )
    num_channels: int = 1


class RcclConfig(BaseModel):
    library: Literal["rccl"] = "rccl"
    algorithm: str = "ring"
    num_channels: int = 1


CollectiveConfig = Annotated[NcclConfig | RcclConfig, Field(discriminator="library")]


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class ScenarioConfig(BaseModel):
    datacenter: Path | DatacenterConfig
    workload: Path | WorkloadConfig
    collective: CollectiveConfig = Field(default_factory=NcclConfig)
