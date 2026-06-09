from __future__ import annotations

import json
from typing import TYPE_CHECKING

from simulon.config.dc import DatacenterConfig
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import CollectiveWorkload, MegatronWorkload

if TYPE_CHECKING:
    from simulon.backend.dag.replayer import SimulationResult


def _flatten_dict(prefix: str, data: dict[str, object]) -> dict[str, str | int | float | bool]:
    """Recursively flatten a dict into dot-separated keys with scalar values."""
    out: dict[str, str | int | float | bool] = {}
    for key, val in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            out.update(_flatten_dict(full_key, val))
        elif isinstance(val, list):
            # Serialize lists to JSON strings for logging
            out[full_key] = json.dumps(val)
        elif isinstance(val, bool):
            out[full_key] = val
        elif isinstance(val, (int, float)):
            out[full_key] = val
        elif val is not None:
            out[full_key] = str(val)
    return out


from simulon.config.resolve import resolve_node_spec


def extract_params(scenario: ScenarioConfig) -> dict[str, str | int | float | bool]:
    params: dict[str, str | int | float | bool] = {}

    # Flatten collective config
    params.update(_flatten_dict("collective", scenario.collective.model_dump(mode="json")))

    # Flatten workload config
    wl = scenario.workload
    if isinstance(wl, MegatronWorkload):
        params["workload.framework"] = wl.framework
        params.update(_flatten_dict("workload.config", wl.config))
    elif isinstance(wl, CollectiveWorkload):
        params["workload.framework"] = wl.framework
        params["workload.collective_type"] = wl.collective_type.value
        params["workload.message_size_bytes"] = wl.message_size_bytes

    # Flatten datacenter config (with resolved node so template refs are expanded)
    dc = scenario.datacenter
    if isinstance(dc, DatacenterConfig):
        dc_dict = dc.model_dump(mode="json")
        if dc_dict.get("node") is not None:
            resolved_node = resolve_node_spec(dc)
            dc_dict["node"] = resolved_node.model_dump(mode="json")
        params.update(_flatten_dict("datacenter", dc_dict))

    return params


def extract_metrics(result: SimulationResult) -> dict[str, float]:
    metrics: dict[str, float] = {
        "total_time_ms": result.total_time_ms,
        "compute_ms": result.compute_ms,
        "exposed_comm_ms": result.exposed_comm_ms,
        "bubble_ms": result.bubble_ms,
        "overlapped_comm_ms": result.overlapped_comm_ms,
    }
    for ctype, ms in result.exposed_comm_by_type.items():
        metrics[f"exposed_comm.{ctype}_ms"] = ms
    if result.total_flops is not None:
        metrics["total_flops"] = result.total_flops
    return metrics
