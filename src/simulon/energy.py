from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from simulon.config.common import ConstantPowerModel, LinearPowerModel, PowerModel
from simulon.config.resolve import resolve_gpu_spec

logger = logging.getLogger(__name__)

# Seconds per millisecond, used for Wh conversion: Wh = W * ms / 3_600_000
_MS_TO_HOURS = 1 / 3_600_000


@dataclass
class ComponentEnergy:
    component: str
    wh: float
    pct: float  # percentage of hardware subtotal


@dataclass
class EnergyResult:
    total_wh: float
    hardware_subtotal_wh: float
    pue_overhead_wh: float
    avg_power_kw: float
    run_duration_hours: float  # used by compute_cost
    breakdown: list[ComponentEnergy]


def _power_w(model: PowerModel, utilisation: float) -> float:
    """Return instantaneous power draw in watts given a utilisation in [0, 1]."""
    if isinstance(model, ConstantPowerModel):
        return model.tdp_w
    elif isinstance(model, LinearPowerModel):
        return model.idle_power_w + utilisation * (model.tdp_w - model.idle_power_w)
    raise TypeError(f"Unknown power model type: {type(model)}")


def _component_wh(power_w: float, count: int, total_time_ms: float) -> float:
    return power_w * count * total_time_ms * _MS_TO_HOURS


def compute_energy(dag, scenario) -> EnergyResult | None:
    """Compute energy consumption for one training iteration from a replayed DAG.

    Args:
        dag: Fully timing-populated and replayed ExecutionDAG (nodes have start_ms/finish_ms).
        scenario: ScenarioConfig providing hardware specs and datacenter params.

    Returns:
        EnergyResult, or None (with a warning) if the GPU has no power_model set.
    """
    dc = scenario.datacenter
    gpu_spec = resolve_gpu_spec(dc)

    if gpu_spec.power_model is None:
        logger.warning(
            "Energy modeling requested but GPU %r has no power_model set. "
            "Skipping energy calculation.",
            gpu_spec.name or "unknown",
        )
        return None

    # --- Derive total iteration time from the replayed DAG ---
    finish_times = [
        n.finish_ms
        for n in (list(dag.compute_nodes) + list(dag.comm_nodes))
        if n.finish_ms is not None
    ]
    if not finish_times:
        logger.warning("No node finish times found in DAG; cannot compute energy.")
        return None

    total_time_ms = max(finish_times)
    run_duration_hours = total_time_ms * _MS_TO_HOURS

    # --- Cluster scale ---
    num_nodes = dc.cluster.num_nodes
    gpus_per_node = dc.node.gpus_per_node

    # --- GPU utilisation: active compute time averaged over ALL cluster GPUs ---
    # Ranks absent from the DAG (no compute nodes) contribute 0 active time.
    active_ms_by_rank: dict[int, float] = defaultdict(float)
    for node in dag.compute_nodes:
        if node.start_ms is not None and node.finish_ms is not None:
            active_ms_by_rank[node.gpu_rank] += node.finish_ms - node.start_ms

    num_gpus_total = num_nodes * gpus_per_node
    avg_active_ms = sum(active_ms_by_rank.values()) / num_gpus_total if num_gpus_total > 0 else 0.0
    utilisation = avg_active_ms / total_time_ms if total_time_ms > 0 else 0.0
    gpus_per_nic = dc.node.gpus_per_nic
    nics_per_node = gpus_per_node // gpus_per_nic

    # nodes_per_rack for rack count derivation
    rack = dc.datacenter.rack
    if rack is not None and rack.nodes_per_rack is not None and rack.nodes_per_rack > 0:
        num_racks = math.ceil(num_nodes / rack.nodes_per_rack)
    else:
        num_racks = 1

    # --- Topology switch counts (read from params dict if present) ---
    num_leaf_switches = 0
    num_spine_switches = 0
    network = dc.network
    if network is not None and network.scale_out is not None:
        topo = network.scale_out.topology
        if topo is not None and isinstance(topo.params, dict):
            num_leaf_switches = topo.params.get("num_leaf_switches", 0) or 0
            num_spine_switches = topo.params.get("num_spine_switches", 0) or 0

    # --- Accumulate per-component energy ---
    components: list[tuple[str, float]] = []  # (name, wh)

    # GPU
    gpu_power = _power_w(gpu_spec.power_model, utilisation)
    num_gpus = gpus_per_node * num_nodes
    components.append(("gpu", _component_wh(gpu_power, num_gpus, total_time_ms)))

    # CPU
    if dc.node.cpu is not None:
        from simulon.config.dc import CPUSpec
        cpu = dc.node.cpu
        if isinstance(cpu, str):
            cpu = None  # can't resolve string refs here without profile loading
        if cpu is not None and cpu.power_model is not None:
            cpu_power = _power_w(cpu.power_model, 0.0)  # constant model only
            components.append(("cpu", _component_wh(cpu_power, cpu.sockets * num_nodes, total_time_ms)))

    # NIC
    if network is not None and network.scale_out is not None:
        nic = network.scale_out.nic
        if nic is not None and not isinstance(nic, str) and nic.power_model is not None:
            nic_power = _power_w(nic.power_model, 0.0)
            components.append(("nic", _component_wh(nic_power, nics_per_node * num_nodes, total_time_ms)))

    # NVSwitch (one per node)
    if network is not None and network.scale_up is not None:
        nvswitch = network.scale_up.switch
        if nvswitch is not None and not isinstance(nvswitch, str) and nvswitch.power_model is not None:
            nvswitch_power = _power_w(nvswitch.power_model, 0.0)
            components.append(("nvswitch", _component_wh(nvswitch_power, num_nodes, total_time_ms)))

    # Leaf switches
    if network is not None and network.scale_out is not None and num_leaf_switches > 0:
        leaf = network.scale_out.leaf_switch
        if leaf is not None and not isinstance(leaf, str) and leaf.power_model is not None:
            leaf_power = _power_w(leaf.power_model, 0.0)
            components.append(("leaf_switches", _component_wh(leaf_power, num_leaf_switches, total_time_ms)))

    # Spine switches
    if network is not None and network.scale_out is not None and num_spine_switches > 0:
        spine = network.scale_out.spine_switch
        if spine is not None and not isinstance(spine, str) and spine.power_model is not None:
            spine_power = _power_w(spine.power_model, 0.0)
            components.append(("spine_switches", _component_wh(spine_power, num_spine_switches, total_time_ms)))

    # Node-level cooling (flat tdp_w, no power_model)
    node_cooling = dc.node.cooling
    if node_cooling is not None and node_cooling.tdp_w is not None:
        components.append(("node_cooling", _component_wh(node_cooling.tdp_w, num_nodes, total_time_ms)))

    # Rack cooling
    if rack is not None and rack.cooling is not None and rack.cooling.tdp_w is not None:
        components.append(("rack_cooling", _component_wh(rack.cooling.tdp_w, num_racks, total_time_ms)))

    # Datacenter cooling
    dc_cooling = dc.datacenter.cooling
    if dc_cooling is not None and dc_cooling.tdp_w is not None:
        components.append(("datacenter_cooling", _component_wh(dc_cooling.tdp_w, 1, total_time_ms)))

    hardware_subtotal_wh = sum(wh for _, wh in components)
    pue = dc.datacenter.pue
    total_wh = hardware_subtotal_wh * pue
    pue_overhead_wh = total_wh - hardware_subtotal_wh

    avg_power_kw = (total_wh / run_duration_hours / 1000) if run_duration_hours > 0 else 0.0

    breakdown = [
        ComponentEnergy(
            component=name,
            wh=wh,
            pct=(wh / hardware_subtotal_wh * 100) if hardware_subtotal_wh > 0 else 0.0,
        )
        for name, wh in components
    ]

    return EnergyResult(
        total_wh=total_wh,
        hardware_subtotal_wh=hardware_subtotal_wh,
        pue_overhead_wh=pue_overhead_wh,
        avg_power_kw=avg_power_kw,
        run_duration_hours=run_duration_hours,
        breakdown=breakdown,
    )
