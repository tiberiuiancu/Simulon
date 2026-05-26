from __future__ import annotations

import math
from dataclasses import dataclass

from simulon.config.common import CostField
from simulon.config.resolve import resolve_gpu_spec, resolve_node_spec
from simulon.energy import EnergyResult


def _resolve_cost(field: CostField | None) -> tuple[float, float | None, float | None] | None:
    """Return (value, min, max) from a CostField, or None if field is absent."""
    if field is None:
        return None
    if isinstance(field, int | float):
        return float(field), None, None
    return field.value, field.min, field.max


def _scale_cost(
    field: CostField | None, count: int
) -> tuple[float, float | None, float | None] | None:
    """Return (total, total_min, total_max) scaled by count, or None if no cost."""
    resolved = _resolve_cost(field)
    if resolved is None:
        return None
    value, mn, mx = resolved
    return (
        value * count,
        mn * count if mn is not None else None,
        mx * count if mx is not None else None,
    )


@dataclass
class CapexComponent:
    component: str
    total: float
    min: float | None
    max: float | None
    pct: float  # % of total capex


@dataclass
class CapexResult:
    total: float
    min: float | None
    max: float | None
    breakdown: list[CapexComponent]


@dataclass
class CostPerRun:
    total: float
    capex_component: float
    opex_component: float


@dataclass
class CostResult:
    capex: CapexResult
    opex_per_run: float
    cost_per_run: CostPerRun | None = None


def compute_cost(scenario, energy_result: EnergyResult) -> CostResult:
    """Compute CAPEX, OPEX per run, and optionally combined cost per run.

    Args:
        scenario: ScenarioConfig providing hardware costs and datacenter params.
        energy_result: EnergyResult from compute_energy(), supplying total_wh and
                       run_duration_hours.

    Returns:
        CostResult with CAPEX breakdown, OPEX per run, and (if
        datacenter_lifetime_years is set) combined cost per run.
    """
    dc = scenario.datacenter
    num_nodes = dc.cluster.num_nodes
    node = resolve_node_spec(dc)
    gpus_per_node = node.gpus_per_node
    if gpus_per_node is None:
        raise ValueError("node.gpus_per_node must be set after resolution")
    gpus_per_nic = node.gpus_per_nic
    nics_per_node = gpus_per_node // gpus_per_nic

    rack = dc.datacenter.rack
    if rack is not None and rack.nodes_per_rack is not None and rack.nodes_per_rack > 0:
        num_racks = math.ceil(num_nodes / rack.nodes_per_rack)
    else:
        num_racks = 1

    # --- Topology switch counts ---
    num_leaf_switches = 0
    num_spine_switches = 0
    network = dc.network
    if network is not None and network.scale_out is not None:
        topo = network.scale_out.topology
        if topo is not None and isinstance(topo.params, dict):
            num_leaf_switches = topo.params.get("num_leaf_switches", 0) or 0
            num_spine_switches = topo.params.get("num_spine_switches", 0) or 0

    # --- GPU spec (may need template resolution for cost) ---
    gpu_spec = resolve_gpu_spec(dc)

    # --- Collect component costs ---
    raw: list[tuple[str, tuple[float, float | None, float | None] | None]] = []

    # GPU
    raw.append(("gpu", _scale_cost(gpu_spec.cost, gpus_per_node * num_nodes)))

    # CPU
    cpu = dc.node.cpu
    if cpu is not None and not isinstance(cpu, str):
        raw.append(("cpu", _scale_cost(cpu.cost, cpu.sockets * num_nodes)))
        if cpu.memory_cost_per_gb is not None and cpu.memory_gb is not None:
            mem_cost_total = cpu.memory_cost_per_gb * cpu.memory_gb * num_nodes
            raw.append(("cpu_memory", (mem_cost_total, None, None)))

    # Node cooling
    node_cooling = dc.node.cooling
    if node_cooling is not None:
        raw.append(("node_cooling", _scale_cost(node_cooling.cost, num_nodes)))

    # NIC
    if network is not None and network.scale_out is not None:
        nic = network.scale_out.nic
        if nic is not None and not isinstance(nic, str):
            raw.append(("nic", _scale_cost(nic.cost, nics_per_node * num_nodes)))

    # NVSwitch (one per node)
    if network is not None and network.scale_up is not None:
        nvswitch = network.scale_up.switch
        if nvswitch is not None and not isinstance(nvswitch, str):
            raw.append(("nvswitch", _scale_cost(nvswitch.cost, num_nodes)))

    # Leaf switches
    if network is not None and network.scale_out is not None and num_leaf_switches > 0:
        leaf = network.scale_out.leaf_switch
        if leaf is not None and not isinstance(leaf, str):
            raw.append(("leaf_switches", _scale_cost(leaf.cost, num_leaf_switches)))

    # Spine switches
    if network is not None and network.scale_out is not None and num_spine_switches > 0:
        spine = network.scale_out.spine_switch
        if spine is not None and not isinstance(spine, str):
            raw.append(("spine_switches", _scale_cost(spine.cost, num_spine_switches)))

    # Rack hardware
    if rack is not None:
        raw.append(("rack", _scale_cost(rack.cost, num_racks)))
        if rack.cooling is not None:
            raw.append(("rack_cooling", _scale_cost(rack.cooling.cost, num_racks)))

    # Datacenter cooling
    dc_cooling = dc.datacenter.cooling
    if dc_cooling is not None:
        raw.append(("datacenter_cooling", _scale_cost(dc_cooling.cost, 1)))

    # --- Build CapexResult ---
    present = [(name, scaled) for name, scaled in raw if scaled is not None]
    capex_total = sum(v for _, (v, _, _) in present)

    total_min: float | None = None
    total_max: float | None = None
    has_range = any(mn is not None or mx is not None for _, (_, mn, mx) in present)
    if has_range:
        total_min = sum((mn if mn is not None else v) for _, (v, mn, _) in present)
        total_max = sum((mx if mx is not None else v) for _, (v, _, mx) in present)

    breakdown = [
        CapexComponent(
            component=name,
            total=v,
            min=mn,
            max=mx,
            pct=(v / capex_total * 100) if capex_total > 0 else 0.0,
        )
        for name, (v, mn, mx) in present
    ]

    capex = CapexResult(total=capex_total, min=total_min, max=total_max, breakdown=breakdown)

    # --- OPEX per run ---
    electricity_cost = dc.datacenter.electricity_cost_per_kwh or 0.0
    energy_kwh = energy_result.total_wh / 1000
    opex_per_run = energy_kwh * electricity_cost

    # --- Combined cost per run (only if datacenter_lifetime_years set) ---
    cost_per_run: CostPerRun | None = None
    lifetime_years = dc.datacenter.datacenter_lifetime_years
    if lifetime_years is not None and energy_result.run_duration_hours > 0:
        idle_fraction = dc.datacenter.idle_fraction
        runs_per_lifetime = math.floor(
            lifetime_years * 8760 * (1 - idle_fraction) / energy_result.run_duration_hours
        )
        if runs_per_lifetime > 0:
            capex_per_run = capex_total / runs_per_lifetime
            cost_per_run = CostPerRun(
                total=capex_per_run + opex_per_run,
                capex_component=capex_per_run,
                opex_component=opex_per_run,
            )

    return CostResult(capex=capex, opex_per_run=opex_per_run, cost_per_run=cost_per_run)
