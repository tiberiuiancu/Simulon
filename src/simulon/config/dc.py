from enum import Enum
from typing import Annotated, Any, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from .common import CostField, PowerModel
from .nccl_profile import NcclProfile


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


class DatacenterCoolingSpec(BaseModel):
    """Optional facility-level cooling unit (e.g. chillers, CRAC units)."""
    tdp_w: Optional[float] = None
    cost: Optional[CostField] = None


class DatacenterMeta(BaseModel):
    name: Optional[str] = None
    profiles_dir: Optional[str] = None
    pue: float = 1.0
    electricity_cost_per_kwh: Optional[float] = None
    datacenter_lifetime_years: Optional[float] = None
    idle_fraction: float = 0.0
    cooling: Optional[DatacenterCoolingSpec] = None
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
    power_model: Optional[PowerModel] = None
    cost: Optional[CostField] = None
    # Populated by `simulon profile gpu`; empty when declared inline in a DC config.
    kernel_runs: list[KernelRun] = []
    # Configs that hit OOM during profiling. Used at runtime to warn when
    # interpolating for a config known to exceed GPU memory.
    oom_configs: list[dict] = []


class CPUSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[str] = Field(None, alias="from")
    name: Optional[str] = None
    vendor: Optional[str] = None
    sockets: int = 2
    cores_per_socket: Optional[int] = None
    memory_gb: Optional[float] = None
    power_model: Optional[PowerModel] = None
    cost: Optional[CostField] = None
    memory_cost_per_gb: Optional[float] = None


class NodeCoolingSpec(BaseModel):
    tdp_w: Optional[float] = None
    cost: Optional[CostField] = None


class NodeSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = None
    from_: Optional[str] = Field(None, alias="from")
    gpus_per_node: Optional[int] = None
    gpus_per_nic: int = 1
    gpu: Optional[Union[str, GPUSpec]] = None
    cpu: Optional[Union[str, CPUSpec]] = None
    cooling: Optional[NodeCoolingSpec] = None
    # Intra-node fabric (NVLink/NVSwitch). Moved here from network.scale_up.
    scale_up: Optional["ScaleUpSpec"] = None
    # Embedded NCCL measurement profile for this node's GPU + topology combination.
    nccl: Optional[NcclProfile] = None


# ---------------------------------------------------------------------------
# Scale-up block
# ---------------------------------------------------------------------------


class SwitchSpec(BaseModel):
    """Unified switch spec used by scale_up.switch and scale_out leaf/spine switches."""

    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[str] = Field(None, alias="from")
    name: Optional[str] = None
    vendor: Optional[str] = None
    port_count: Optional[int] = None
    port_speed: Optional[str] = None
    latency: Optional[str] = None  # propagation latency, e.g. "0.000025ms"
    buffer_per_port: Optional[str] = None
    queue_discipline: Optional[QueueDiscipline] = None
    queue_params: Optional[dict[str, Any]] = None  # discipline-specific; typed later
    power_model: Optional[PowerModel] = None
    cost: Optional[CostField] = None


# ---------------------------------------------------------------------------
# Network block (scale-up + scale-out)
# ---------------------------------------------------------------------------


class NICSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[str] = Field(None, alias="from")
    name: Optional[str] = None
    vendor: Optional[str] = None
    speed: Optional[str] = None
    latency: Optional[str] = None
    nics_per_node: int = Field(
        default=1,
        ge=1,
        description="Number of NICs per node. Used to compute total inter-node bandwidth "
        "(nic_bw × nics_per_node) for collective operations.",
    )
    power_model: Optional[PowerModel] = None
    cost: Optional[CostField] = None
    bandwidth_efficiency: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Bandwidth efficiency factor (0.0-1.0) for network collectives. "
        "Accounts for protocol overhead, serialization, etc. Defaults to 0.85 (85% efficient) "
        "which is realistic for multi-node collective operations.",
    )


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


class TopologySpec(BaseModel):
    """Scale-out topology type and parameters."""

    type: TopologyType
    params: Optional[dict[str, Any]] = None


class ScaleUpSpec(BaseModel):
    """Intra-node (NVLink) network — one NVSwitch per node assumed."""

    switch: Optional[Union[str, SwitchSpec]] = None  # NVSwitch spec


class ScaleOutSpec(BaseModel):
    """Inter-node network."""

    nic: Optional[Union[str, NICSpec]] = None
    leaf_switch: Optional[Union[str, SwitchSpec]] = None
    spine_switch: Optional[Union[str, SwitchSpec]] = None
    topology: Optional[TopologySpec] = None


class NetworkSpec(BaseModel):
    """Top-level network block containing scale-up and scale-out sub-configs."""

    scale_up: Optional[ScaleUpSpec] = None
    scale_out: Optional[ScaleOutSpec] = None


# ---------------------------------------------------------------------------
# Top-level datacenter config
# ---------------------------------------------------------------------------


def _coerce_node(v):
    if isinstance(v, str):
        return {"from": v}
    return v


class DatacenterConfig(BaseModel):
    datacenter: DatacenterMeta
    cluster: ClusterSpec
    node: Annotated[NodeSpec, BeforeValidator(_coerce_node)]
    network: Optional[NetworkSpec] = None  # deprecated, use scale_out
    # New top-level scale-out, replacing network.scale_out.
    scale_out: Optional[ScaleOutSpec] = None
