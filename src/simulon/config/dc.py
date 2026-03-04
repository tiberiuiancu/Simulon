from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .common import CostField


# ---------------------------------------------------------------------------
# GPU profiling results
# ---------------------------------------------------------------------------


class KernelRun(BaseModel):
    """A single kernel benchmark: name, parameters, and measured runtimes."""

    kernel: str
    params: dict[str, Any]
    times_ms: list[float]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScaleUpTopology(str, Enum):
    switched = "switched"
    p2p = "p2p"


class ScaleUpTechnology(str, Enum):
    nvlink = "nvlink"
    ualink = "ualink"
    infinity_fabric = "infinity_fabric"


class QueueDiscipline(str, Enum):
    drop_tail = "drop_tail"
    red = "red"
    codel = "codel"
    fq_codel = "fq_codel"


class TopologyType(str, Enum):
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
    capacity_kw: Optional[float] = None
    tdp_w: Optional[float] = None
    cost: Optional[CostField] = None


class RackSpec(BaseModel):
    nodes_per_rack: Optional[int] = None
    rack_units: Optional[int] = None
    max_power_kw: Optional[float] = None
    cost: Optional[CostField] = None
    cooling: Optional[RackCoolingSpec] = None


class DatacenterMeta(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    profiles_dir: Optional[str] = None
    pue: float = 1.0
    electricity_cost_per_kwh: Optional[float] = None
    rack: Optional[RackSpec] = None


# ---------------------------------------------------------------------------
# Cluster block
# ---------------------------------------------------------------------------


class ClusterSpec(BaseModel):
    num_nodes: int


# ---------------------------------------------------------------------------
# Node block
# ---------------------------------------------------------------------------


class GPUSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[str] = Field(None, alias="from")
    name: Optional[str] = None
    vendor: Optional[str] = None
    flops_multiplier: float = 1.0
    memory_capacity_gb: Optional[float] = None
    tdp_w: Optional[float] = None
    cost: Optional[CostField] = None
    # Populated by `simulon profile gpu`; empty when declared inline in a DC config.
    kernel_runs: list[KernelRun] = []


class CPUSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[str] = Field(None, alias="from")
    name: Optional[str] = None
    vendor: Optional[str] = None
    sockets: int = 2
    cores_per_socket: Optional[int] = None
    memory_gb: Optional[float] = None
    tdp_w: Optional[float] = None
    cost: Optional[CostField] = None
    memory_cost_per_gb: Optional[float] = None


class NodeCoolingSpec(BaseModel):
    tdp_w: Optional[float] = None
    cost: Optional[CostField] = None


class NodeSpec(BaseModel):
    gpus_per_node: int
    num_switches_per_node: Optional[int] = None
    gpus_per_nic: int = 1
    gpu: Union[str, GPUSpec]
    cpu: Optional[Union[str, CPUSpec]] = None
    cooling: Optional[NodeCoolingSpec] = None


# ---------------------------------------------------------------------------
# Scale-up block
# ---------------------------------------------------------------------------


class SwitchSpec(BaseModel):
    """Unified switch spec used by both scale_up.switch and scale_out.topology.switch."""

    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[str] = Field(None, alias="from")
    name: Optional[str] = None
    vendor: Optional[str] = None
    port_count: Optional[int] = None
    port_speed: Optional[str] = None
    buffer_per_port: Optional[str] = None
    queue_discipline: Optional[QueueDiscipline] = None
    queue_params: Optional[dict[str, Any]] = None  # discipline-specific; typed later
    tdp_w: Optional[float] = None
    cost: Optional[CostField] = None


class ScaleUpSpec(BaseModel):
    topology: ScaleUpTopology = ScaleUpTopology.switched
    technology: ScaleUpTechnology = ScaleUpTechnology.nvlink
    link_bandwidth: Optional[str] = None
    link_latency: Optional[str] = None
    link_cost: Optional[CostField] = None
    switch: Optional[Union[str, SwitchSpec]] = None


# ---------------------------------------------------------------------------
# Scale-out block
# ---------------------------------------------------------------------------


class NICSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[str] = Field(None, alias="from")
    name: Optional[str] = None
    vendor: Optional[str] = None
    speed: Optional[str] = None
    latency: Optional[str] = None
    tdp_w: Optional[float] = None
    cost: Optional[CostField] = None


class LinkSpec(BaseModel):
    latency: Optional[str] = None
    error_rate: float = 0.0
    cost: Optional[CostField] = None
    cost_per_meter: Optional[float] = None


# --- Topology params (one model per template type) ---


class SpectrumXParams(BaseModel):
    nics_per_leaf: Optional[int] = None
    num_leaf_switches: Optional[int] = None
    num_spine_switches: Optional[int] = None
    leaf_to_spine_bandwidth: Optional[str] = None
    switches_per_spine: Optional[int] = None
    nvlink_switches_per_node: Optional[int] = None
    dual_tor: bool = False
    dual_plane: bool = False


class AlibabaHPNParams(BaseModel):
    nics_per_leaf: Optional[int] = None
    num_leaf_switches: Optional[int] = None
    num_spine_switches: Optional[int] = None
    leaf_to_spine_bandwidth: Optional[str] = None
    dual_tor: bool = False
    dual_plane: bool = False


class DCNPlusParams(BaseModel):
    nics_per_leaf: Optional[int] = None
    num_leaf_switches: Optional[int] = None
    num_spine_switches: Optional[int] = None
    uplink_bandwidth: Optional[str] = None
    dual_tor: bool = False


class FatTreeParams(BaseModel):
    k: Optional[int] = None
    num_tiers: int = 3
    oversubscription: float = 1.0


class RailOptimizedParams(BaseModel):
    num_rails: Optional[int] = None
    nodes_per_rail: Optional[int] = None
    num_spine_switches: int = 1
    rail_to_spine_links: int = 1


class DragonflyParams(BaseModel):
    group_size: int
    nodes_per_router: Optional[int] = None
    intra_group_links: Optional[int] = None
    inter_group_links: int = 1


class CustomTopologyParams(BaseModel):
    topology_file: str


class ScaleOutTopologySpec(BaseModel):
    type: TopologyType
    # params are validated against the correct model for `type` at load time
    params: Optional[dict[str, Any]] = None
    switch: Optional[Union[str, SwitchSpec]] = None
    link: Optional[LinkSpec] = None


class ScaleOutSpec(BaseModel):
    nic: Union[str, NICSpec]
    topology: Optional[ScaleOutTopologySpec] = None


# ---------------------------------------------------------------------------
# Top-level datacenter config
# ---------------------------------------------------------------------------


class DatacenterConfig(BaseModel):
    datacenter: DatacenterMeta
    cluster: ClusterSpec
    node: NodeSpec
    scale_up: Optional[ScaleUpSpec] = None
    scale_out: Optional[ScaleOutSpec] = None
