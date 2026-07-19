from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

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
    workload: Path | dict | WorkloadConfig
    collective: CollectiveConfig = Field(default_factory=NcclConfig)

    @model_validator(mode="after")
    def _resolve_refs(self) -> ScenarioConfig:
        if isinstance(self.datacenter, Path):
            from simulon.config.resolve import resolve_datacenter, resolve_node_spec

            self.datacenter = resolve_datacenter(self.datacenter)
            if self.datacenter.node and self.datacenter.node.from_:
                self.datacenter.node = resolve_node_spec(self.datacenter)
        if isinstance(self.workload, Path | dict):
            from simulon.config.resolve import resolve_workload

            self.workload = resolve_workload(self.workload)
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> ScenarioConfig:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
