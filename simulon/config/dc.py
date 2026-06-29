from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from .common import CostField, PowerModel
from .nccl_profile import NcclProfile

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QueueDiscipline(StrEnum):
    drop_tail = "drop_tail"
    red = "red"
    codel = "codel"
    fq_codel = "fq_codel"


class TopologyType(StrEnum):
    spectrum_x = "spectrum_x"
    alibaba_hpn = "alibaba_hpn"
    dcn_plus = "dcn_plus"
    fat_tree = "fat_tree"
    rail_optimized = "rail_optimized"
    dragonfly = "dragonfly"
    custom = "custom"


# ---------------------------------------------------------------------------
# Datacenter block
# ---------------------------------------------------------------------------


class RackCoolingSpec(BaseModel):
    capacity_kw: float | None = None
    tdp_w: float | None = None
    cost: CostField | None = None


class RackSpec(BaseModel):
    nodes_per_rack: int | None = None
    rack_units: int | None = None
    max_power_kw: float | None = None
    cost: CostField | None = None
    cooling: RackCoolingSpec | None = None


class DatacenterCoolingSpec(BaseModel):
    """Optional facility-level cooling unit (e.g. chillers, CRAC units)."""

    tdp_w: float | None = None
    cost: CostField | None = None


class DatacenterMeta(BaseModel):
    name: str | None = None
    profiles_dir: str | None = None
    traces_dir: str | None = None
    pue: float = 1.0
    electricity_cost_per_kwh: float | None = None
    datacenter_lifetime_years: float | None = None
    idle_fraction: float = 0.0
    cooling: DatacenterCoolingSpec | None = None
    rack: RackSpec | None = None


# ---------------------------------------------------------------------------
# Node block
# ---------------------------------------------------------------------------


class GPUSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str | None = Field(None, alias="from")
    name: str | None = None
    vendor: str | None = None
    flops_multiplier: float = 1.0
    peak_tflops_bf16: float | None = None
    memory_capacity_gb: float | None = None
    power_model: PowerModel | None = None
    cost: CostField | None = None


class CPUSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str | None = Field(None, alias="from")
    name: str | None = None
    vendor: str | None = None
    sockets: int = 2
    cores_per_socket: int | None = None
    memory_gb: float | None = None
    power_model: PowerModel | None = None
    cost: CostField | None = None
    memory_cost_per_gb: float | None = None


class NodeCoolingSpec(BaseModel):
    tdp_w: float | None = None
    cost: CostField | None = None


class NodeSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str | None = None
    from_: str | None = Field(None, alias="from")
    gpus_per_node: int | None = None
    nics_per_node: int | None = None
    gpu: str | GPUSpec | None = None
    cpu: str | CPUSpec | None = None
    cooling: NodeCoolingSpec | None = None
    scale_up: ScaleUpSpec | None = None
    scale_out: ScaleOutSpec | None = None
    nccl: NcclProfile | None = None
    cost: CostField | None = None

    @model_validator(mode="after")
    def set_display_name_default(self) -> NodeSpec:
        if self.nics_per_node is None:
            self.nics_per_node = self.gpus_per_node
        return self


# ---------------------------------------------------------------------------
# Scale-up block
# ---------------------------------------------------------------------------


class SwitchSpec(BaseModel):
    """Unified switch spec used by scale_up.switch and scale_out leaf/spine switches."""

    model_config = ConfigDict(populate_by_name=True)

    from_: str | None = Field(None, alias="from")
    name: str | None = None
    vendor: str | None = None
    port_count: int | None = None
    port_speed: str | None = None
    latency: str | None = None  # propagation latency, e.g. "0.000025ms"
    buffer_per_port: str | None = None
    queue_discipline: QueueDiscipline | None = None
    queue_params: dict[str, Any] | None = None  # discipline-specific; typed later
    power_model: PowerModel | None = None
    cost: CostField | None = None


# ---------------------------------------------------------------------------
# Network block (scale-up + scale-out)
# ---------------------------------------------------------------------------


class NICSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str | None = Field(None, alias="from")
    name: str | None = None
    vendor: str | None = None
    speed: str | None = None
    latency: str | None = None
    power_model: PowerModel | None = None
    cost: CostField | None = None
    bandwidth_efficiency: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Bandwidth efficiency factor (0.0-1.0) for network collectives. "
        "Accounts for protocol overhead, serialization, etc. Defaults to 0.85 (85% efficient) "
        "which is realistic for multi-node collective operations.",
    )


class LinkSpec(BaseModel):
    latency: str | None = None
    error_rate: float = 0.0
    cost: CostField | None = None
    cost_per_meter: float | None = None


# --- Topology params (one model per template type) ---


class SpectrumXParams(BaseModel):
    nics_per_leaf: int | None = None
    num_leaf_switches: int | None = None
    num_spine_switches: int | None = None
    leaf_to_spine_bandwidth: str | None = None
    switches_per_spine: int | None = None
    nvlink_switches_per_node: int | None = None
    dual_tor: bool = False
    dual_plane: bool = False


class AlibabaHPNParams(BaseModel):
    nics_per_leaf: int | None = None
    num_leaf_switches: int | None = None
    num_spine_switches: int | None = None
    leaf_to_spine_bandwidth: str | None = None
    dual_tor: bool = False
    dual_plane: bool = False


class DCNPlusParams(BaseModel):
    nics_per_leaf: int | None = None
    num_leaf_switches: int | None = None
    num_spine_switches: int | None = None
    uplink_bandwidth: str | None = None
    dual_tor: bool = False


class FatTreeParams(BaseModel):
    k: int | None = None
    num_tiers: int = 3
    oversubscription: float = 1.0


class RailOptimizedParams(BaseModel):
    num_rails: int | None = None
    nodes_per_rail: int | None = None
    num_spine_switches: int = 1
    rail_to_spine_links: int = 1


class DragonflyParams(BaseModel):
    group_size: int
    nodes_per_router: int | None = None
    intra_group_links: int | None = None
    inter_group_links: int = 1


class CustomTopologyParams(BaseModel):
    topology_file: str


class TopologySpec(BaseModel):
    """Scale-out topology type and parameters."""

    type: TopologyType
    params: dict[str, Any] | None = None


class ScaleUpSpec(BaseModel):
    """Intra-node (NVLink) network — one NVSwitch per node assumed."""

    switch: str | SwitchSpec | None = None  # NVSwitch spec


class ScaleOutSpec(BaseModel):
    """Inter-node network."""

    nic: str | NICSpec | None = None
    leaf_switch: str | SwitchSpec | None = None
    spine_switch: str | SwitchSpec | None = None
    topology: TopologySpec | None = None


class NetworkSpec(BaseModel):
    """Top-level network block containing scale-up and scale-out sub-configs."""

    scale_up: ScaleUpSpec | None = None
    scale_out: ScaleOutSpec | None = None


# ---------------------------------------------------------------------------
# Top-level datacenter config
# ---------------------------------------------------------------------------


def _coerce_node(v):
    if isinstance(v, str):
        return {"from": v}
    return v


class DatacenterConfig(BaseModel):
    datacenter: DatacenterMeta | None = None
    num_nodes: int
    node: Annotated[NodeSpec, BeforeValidator(_coerce_node)]

    @model_validator(mode="after")
    def _resolve_node(self) -> DatacenterConfig:
        if self.node and self.node.from_:
            from simulon.config.resolve import resolve_node_spec

            self.node = resolve_node_spec(self)
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> DatacenterConfig:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
