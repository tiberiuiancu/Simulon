from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from simulon.config.common import ConstantPowerModel, LinearPowerModel, PowerModel
from simulon.config.resolve import resolve_gpu_spec, resolve_node_spec

logger = logging.getLogger(__name__)

_MS_TO_HOURS = 1 / 3_600_000


@dataclass
class ComponentEnergy:
    component: str
    wh: float
    pct: float


@dataclass
class EnergyResult:
    total_wh: float
    hardware_subtotal_wh: float
    pue_overhead_wh: float
    avg_power_kw: float
    run_duration_hours: float
    breakdown: list[ComponentEnergy]
    co2eq_g: float | None = None
    source: Literal["measured", "estimated"] = "estimated"


def _power_w(model: PowerModel, utilisation: float) -> float:
    if isinstance(model, ConstantPowerModel):
        return model.tdp_w
    elif isinstance(model, LinearPowerModel):
        return model.idle_power_w + utilisation * (model.tdp_w - model.idle_power_w)
    raise TypeError(f"Unknown power model type: {type(model)}")


def _component_wh(power_w: float, count: int, total_time_ms: float) -> float:
    return power_w * count * total_time_ms * _MS_TO_HOURS


def _total_time_ms(dag) -> float:
    finish_times = [
        n.finish_ms for n in dag.compute_nodes + dag.comm_nodes if n.finish_ms is not None
    ]
    if not finish_times:
        return 0.0
    return max(finish_times)


def _active_ms_by_rank(dag) -> dict[int, float]:
    active: dict[int, float] = defaultdict(float)
    for node in dag.compute_nodes:
        if node.start_ms is not None and node.finish_ms is not None:
            active[node.gpu_rank] += node.finish_ms - node.start_ms
    return active


def _idle_energy_wh(
    total_time_ms: float,
    active_ms_by_rank: dict[int, float],
    num_gpus_total: int,
    idle_power_w: float,
) -> float:
    total_idle_wh = 0.0
    for rank in range(num_gpus_total):
        idle_ms = total_time_ms - active_ms_by_rank.get(rank, 0.0)
        if idle_ms > 0:
            total_idle_wh += idle_power_w * idle_ms * _MS_TO_HOURS
    return total_idle_wh


def _other_component_energies(dc, total_time_ms: float) -> list[tuple[str, float]]:
    components: list[tuple[str, float]] = []
    node = resolve_node_spec(dc)
    num_nodes = dc.num_nodes
    nics_per_node = node.nics_per_node or 0

    rack = dc.datacenter.rack if dc.datacenter is not None else None
    if rack is not None and rack.nodes_per_rack is not None and rack.nodes_per_rack > 0:
        num_racks = math.ceil(num_nodes / rack.nodes_per_rack)
    else:
        num_racks = 1

    num_leaf_switches = 0
    num_spine_switches = 0
    if node.scale_out is not None:
        topo = node.scale_out.topology
        if topo is not None and isinstance(topo.params, dict):
            num_leaf_switches = topo.params.get("num_leaf_switches", 0) or 0
            num_spine_switches = topo.params.get("num_spine_switches", 0) or 0

    if dc.node.cpu is not None:
        cpu = dc.node.cpu
        if isinstance(cpu, str):
            cpu = None
        if cpu is not None and cpu.power_model is not None:
            cpu_power = _power_w(cpu.power_model, 0.0)
            components.append(
                ("cpu", _component_wh(cpu_power, cpu.sockets * num_nodes, total_time_ms))
            )

    if node.scale_out is not None:
        nic = node.scale_out.nic
        if nic is not None and not isinstance(nic, str) and nic.power_model is not None:
            nic_power = _power_w(nic.power_model, 0.0)
            components.append(
                ("nic", _component_wh(nic_power, nics_per_node * num_nodes, total_time_ms))
            )

    if node.scale_up is not None:
        nvswitch = node.scale_up.switch
        if (
            nvswitch is not None
            and not isinstance(nvswitch, str)
            and nvswitch.power_model is not None
        ):
            nvswitch_power = _power_w(nvswitch.power_model, 0.0)
            components.append(("nvswitch", _component_wh(nvswitch_power, num_nodes, total_time_ms)))

    if node.scale_out is not None and num_leaf_switches > 0:
        leaf = node.scale_out.leaf_switch
        if leaf is not None and not isinstance(leaf, str) and leaf.power_model is not None:
            leaf_power = _power_w(leaf.power_model, 0.0)
            components.append(
                ("leaf_switches", _component_wh(leaf_power, num_leaf_switches, total_time_ms))
            )

    if node.scale_out is not None and num_spine_switches > 0:
        spine = node.scale_out.spine_switch
        if spine is not None and not isinstance(spine, str) and spine.power_model is not None:
            spine_power = _power_w(spine.power_model, 0.0)
            components.append(
                ("spine_switches", _component_wh(spine_power, num_spine_switches, total_time_ms))
            )

    node_cooling = dc.node.cooling
    if node_cooling is not None and node_cooling.tdp_w is not None:
        components.append(
            ("node_cooling", _component_wh(node_cooling.tdp_w, num_nodes, total_time_ms))
        )

    if rack is not None and rack.cooling is not None and rack.cooling.tdp_w is not None:
        components.append(
            ("rack_cooling", _component_wh(rack.cooling.tdp_w, num_racks, total_time_ms))
        )

    dc_cooling = dc.datacenter.cooling if dc.datacenter is not None else None
    if dc_cooling is not None and dc_cooling.tdp_w is not None:
        components.append(("datacenter_cooling", _component_wh(dc_cooling.tdp_w, 1, total_time_ms)))

    return components


def _build_result(
    components: list[tuple[str, float]],
    total_time_ms: float,
    co2eq_g: float | None,
    source: Literal["measured", "estimated"],
) -> EnergyResult:
    hardware_subtotal_wh = sum(wh for _, wh in components)
    total_wh = hardware_subtotal_wh
    pue_overhead_wh = 0.0

    run_duration_hours = total_time_ms * _MS_TO_HOURS
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
        co2eq_g=co2eq_g,
        source=source,
    )


def compute_energy(dag, scenario) -> EnergyResult | None:
    total_time_ms = _total_time_ms(dag)
    if total_time_ms <= 0:
        logger.warning("No node finish times found in DAG; cannot compute energy.")
        return None

    dc = scenario.datacenter
    resolved_node = resolve_node_spec(dc)
    gpus_per_node = resolved_node.gpus_per_node
    if gpus_per_node is None:
        raise ValueError("node.gpus_per_node must be set after resolution")
    num_gpus_total = dc.num_nodes * gpus_per_node

    active_ms = _active_ms_by_rank(dag)
    gpu_spec = resolve_gpu_spec(dc)

    if gpu_spec.power_model is None:
        logger.warning(
            "Energy modeling requested but GPU %r has no power_model set. Skipping.",
            gpu_spec.name or "unknown",
        )
        return None

    idle_power_w = 0.0
    if isinstance(gpu_spec.power_model, LinearPowerModel):
        idle_power_w = gpu_spec.power_model.idle_power_w
    elif isinstance(gpu_spec.power_model, ConstantPowerModel):
        idle_power_w = gpu_spec.power_model.tdp_w

    other_components = _other_component_energies(dc, total_time_ms)

    if dag.energy_kwh is not None and dag.co2eq_kg is not None:
        measured_compute_wh = dag.energy_kwh * 1000.0
        idle_wh = _idle_energy_wh(total_time_ms, active_ms, num_gpus_total, idle_power_w)

        components = [("measured_compute", measured_compute_wh)]
        if idle_wh > 0:
            components.append(("idle_energy", idle_wh))
        components.extend(other_components)

        return _build_result(components, total_time_ms, dag.co2eq_kg * 1000.0, "measured")

    num_gpus = gpus_per_node * dc.num_nodes
    avg_active_ms = sum(active_ms.values()) / num_gpus_total if num_gpus_total > 0 else 0.0
    utilisation = avg_active_ms / total_time_ms if total_time_ms > 0 else 0.0
    gpu_power = _power_w(gpu_spec.power_model, utilisation)

    components = [("gpu", _component_wh(gpu_power, num_gpus, total_time_ms))]
    components.extend(other_components)

    return _build_result(components, total_time_ms, None, "estimated")
