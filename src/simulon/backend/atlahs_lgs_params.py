"""LogGOPSim parameter mapping for ATLAHS.

Derivation rules:
- ``-L`` is the link latency in nanoseconds.
- ``-G`` is the gap per byte in nanoseconds/byte, computed as ``1 / bandwidth_bytes_per_ns``.
- ``-o``/``-g``/``-O``/``-S`` default to the ATLAHS demo values unless overridden.

The mapper prefers inter-node ``scale_out.nic`` values and falls back to
intra-node ``scale_up.switch`` values when scale-out specs are missing.
"""

from __future__ import annotations

import re
import warnings

from simulon.config.dc import DatacenterConfig, NICSpec, SwitchSpec  # pyright: ignore[reportMissingTypeStubs]
from simulon.config.resolve import resolve_node_spec, resolve_scale_out  # pyright: ignore[reportMissingTypeStubs]
from simulon.config.scenario import LogGOPSimConfig, ScenarioConfig  # pyright: ignore[reportMissingTypeStubs]

_DEFAULT_LOGGOPSIM_PARAMS: dict[str, float] = {
    "-L": 3700.0,
    "-G": 0.04,
    "-o": 200.0,
    "-g": 5.0,
    "-O": 0.0,
    "-S": 0.0,
}

_SPEED_FACTORS = {
    "": 1.0,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
}


def _parse_latency_ns(value: str | int | float) -> float:
    """Parse a latency string into nanoseconds."""

    if isinstance(value, (int, float)):
        return float(value)
    m = re.fullmatch(r"([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)\s*(ms|us|ns)", value.strip())
    if not m:
        raise ValueError(f"Cannot parse latency: {value!r}")
    amount = float(m.group(1))
    unit = m.group(2)
    if unit == "ms":
        return amount * 1_000_000.0
    if unit == "us":
        return amount * 1_000.0
    return amount


def _parse_speed_bytes_per_ns(value: str | int | float) -> float:
    """Parse a bandwidth string into bytes/ns.

    Supported forms include ``400Gbps``, ``50MBps``, ``1.25TBps``.
    """

    if isinstance(value, (int, float)):
        return float(value)
    m = re.fullmatch(r"([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)\s*([KMGTP]?)\s*([bB])ps", value.strip())
    if not m:
        raise ValueError(f"Cannot parse bandwidth: {value!r}")
    amount = float(m.group(1))
    factor = _SPEED_FACTORS[m.group(2)]
    unit = m.group(3)
    bytes_per_sec = amount * factor if unit == "B" else (amount * factor) / 8.0
    return bytes_per_sec / 1_000_000_000.0


def _pick_scale_out(dc: DatacenterConfig):
    scale_out = resolve_scale_out(dc)
    if scale_out and scale_out.nic:
        return scale_out.nic
    return None


def _pick_scale_up_switch(dc: DatacenterConfig):
    node = resolve_node_spec(dc)
    if node.scale_up and node.scale_up.switch:
        return node.scale_up.switch
    if dc.network and dc.network.scale_up and dc.network.scale_up.switch:
        warnings.warn(
            "datacenter.network.scale_up is deprecated. Move scale_up to the node spec.",
            DeprecationWarning,
            stacklevel=2,
        )
        return dc.network.scale_up.switch
    return None


def _resolve_latency_ns(dc: DatacenterConfig) -> float:
    nic = _pick_scale_out(dc)
    if isinstance(nic, NICSpec) and nic.latency:
        return _parse_latency_ns(nic.latency)
    switch = _pick_scale_up_switch(dc)
    if isinstance(switch, SwitchSpec) and switch.latency:
        return _parse_latency_ns(switch.latency)
    return _DEFAULT_LOGGOPSIM_PARAMS["-L"]


def _resolve_gap_per_byte_ns(dc: DatacenterConfig) -> float:
    nic = _pick_scale_out(dc)
    if isinstance(nic, NICSpec) and nic.speed:
        bw = _parse_speed_bytes_per_ns(nic.speed)
        return 1.0 / bw if bw > 0 else _DEFAULT_LOGGOPSIM_PARAMS["-G"]
    switch = _pick_scale_up_switch(dc)
    if isinstance(switch, SwitchSpec) and switch.port_speed:
        bw = _parse_speed_bytes_per_ns(switch.port_speed)
        return 1.0 / bw if bw > 0 else _DEFAULT_LOGGOPSIM_PARAMS["-G"]
    return _DEFAULT_LOGGOPSIM_PARAMS["-G"]


def map_datacenter_to_loggopsim(dc: DatacenterConfig) -> dict[str, float]:
    """Map a datacenter config to LogGOPSim CLI parameters.

    Formulas:
    - ``-L`` = network latency in ns.
    - ``-G`` = ``1 / bytes_per_ns``.
    - ``-o``, ``-g``, ``-O``, ``-S`` default to the ATLAHS demo values.
    """

    params = dict(_DEFAULT_LOGGOPSIM_PARAMS)
    params["-L"] = _resolve_latency_ns(dc)
    params["-G"] = _resolve_gap_per_byte_ns(dc)
    return params


def map_scenario_to_loggopsim(scenario: ScenarioConfig) -> dict[str, float]:
    """Map a scenario to LogGOPSim parameters, applying overrides if present."""

    if not isinstance(scenario.datacenter, DatacenterConfig):
        raise TypeError("ScenarioConfig.datacenter must be a DatacenterConfig to map LogGOPSim params")

    params = map_datacenter_to_loggopsim(scenario.datacenter)
    overrides: LogGOPSimConfig | None = scenario.loggopsim
    if overrides is None:
        return params

    override_map = {
        "-L": overrides.latency_ns,
        "-G": overrides.gap_per_byte_ns,
        "-o": overrides.overhead_ns,
        "-g": overrides.gap_ns,
        "-O": overrides.overhead_per_byte_ns,
        "-S": overrides.eager_threshold_bytes,
    }
    for key, value in override_map.items():
        if value is not None:
            params[key] = float(value)
    return params
