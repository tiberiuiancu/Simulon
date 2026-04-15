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
    # Per-step collective overhead in µs (kernel launch + sync), additive on top of fabric
    # latency. Keyed by CollectiveType value ("AllReduce", "AllGather", etc.).
    # Calibrated from single-node measurements where fabric latency is negligible.
    per_step_latency_us: dict[str, float] = {}
    # Per-collective effective bandwidth override in GB/s. When set for a collective,
    # replaces the fabric bandwidth from the datacenter spec. Accounts for the fact that
    # different collectives achieve different effective NVLink utilization.
    per_collective_bw_GBps: dict[str, float] = {}


class RcclConfig(BaseModel):
    library: Literal["rccl"] = "rccl"
    algorithm: str = "ring"
    num_channels: int = 1
    per_step_latency_us: dict[str, float] = {}
    per_collective_bw_GBps: dict[str, float] = {}


CollectiveConfig = Annotated[
    Union[NcclConfig, RcclConfig],
    Field(discriminator="library"),
]


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class ScenarioConfig(BaseModel):
    datacenter: Union[Path, DatacenterConfig]
    workload: Union[Path, WorkloadConfig]
    collective: CollectiveConfig = Field(default_factory=NcclConfig)
