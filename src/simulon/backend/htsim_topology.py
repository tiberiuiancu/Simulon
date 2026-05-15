"""Generate htsim leaf-spine topology files from a :class:`DatacenterConfig`."""

from __future__ import annotations

import math
import re
from pathlib import Path

from typing import Any

from simulon.config.dc import DatacenterConfig, NICSpec, ScaleOutSpec
from simulon.config.resolve import resolve_node_spec, resolve_scale_out

_SPEED_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(Gbps|Mbps|Kbps|bps)$", re.IGNORECASE)
_DEFAULT_SPEED_GBPS = 200


def _parse_speed_gbps(speed: str | None) -> int:
    if speed is None:
        return _DEFAULT_SPEED_GBPS
    m = _SPEED_RE.match(speed.strip())
    if not m:
        raise ValueError(f"Cannot parse NIC speed: {speed!r}")
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "gbps":
        return int(val)
    if unit == "mbps":
        return int(val / 1_000)
    if unit == "kbps":
        return int(val / 1_000_000)
    return int(val / 1_000_000_000)


def _nearest_divisor(n: int, target: int) -> int:
    best = 1
    best_dist = abs(target - 1)
    limit = int(math.isqrt(n))
    for i in range(1, limit + 1):
        if n % i != 0:
            continue
        for d in (i, n // i):
            dist = abs(d - target)
            if dist < best_dist or (dist == best_dist and d > best):
                best_dist = dist
                best = d
    return best


def _get_scale_out_nic(scale_out: ScaleOutSpec | None) -> NICSpec | None:
    if scale_out is None:
        return None
    nic = scale_out.nic
    if isinstance(nic, NICSpec):
        return nic
    return None


def generate_topology(dc: DatacenterConfig) -> str:
    """Generate an htsim leaf-spine ``.topo`` file from *dc*.

    The topology is always 2-tier (leaf-spine).  Parameters are derived from
    ``dc.scale_out`` (or the deprecated ``dc.network.scale_out``) when present,
    with sensible defaults filled in automatically.

    Args:
        dc: A fully-populated datacenter configuration.

    Returns:
        The contents of the ``.topo`` file as a single string.

    Raises:
        ValueError: If the configuration cannot be mapped to a valid leaf-spine
            topology (e.g. missing *gpus_per_node*, degenerate node count, or
            parameter constraints that cannot be satisfied).
    """
    node = resolve_node_spec(dc)
    if node.gpus_per_node is None:
        raise ValueError("datacenter.node.gpus_per_node is required to compute topology size")

    num_nodes = dc.cluster.num_nodes
    if num_nodes <= 0:
        raise ValueError(f"datacenter.cluster.num_nodes must be > 0, got {num_nodes}")

    nodes = num_nodes * node.gpus_per_node
    if nodes < 2:
        raise ValueError(
            f"Total node count ({nodes} = {num_nodes} * {node.gpus_per_node}) must be >= 2 for a leaf-spine topology"
        )

    scale_out = resolve_scale_out(dc)
    nic = _get_scale_out_nic(scale_out)

    params: dict[str, Any] = {}
    if scale_out is not None and scale_out.topology is not None and scale_out.topology.params is not None:
        params = dict(scale_out.topology.params)

    speed_gbps = _DEFAULT_SPEED_GBPS
    if "speed_gbps" in params:
        speed_gbps = int(params["speed_gbps"])
    elif nic is not None and nic.speed is not None:
        speed_gbps = _parse_speed_gbps(nic.speed)

    downlink_latency_ns = int(params.get("downlink_latency_ns", 1))
    switch_latency_ns = int(params.get("switch_latency_ns", 0))
    oversubscription = int(params.get("oversubscription", params.get("oversubscribed", 1)))
    if oversubscription < 1:
        raise ValueError(f"oversubscription must be >= 1, got {oversubscription}")

    podsize_raw = params.get("podsize")
    podsize: int = int(podsize_raw) if podsize_raw is not None else (min(nodes, 32) if nodes > 32 else nodes)

    radix_down_leaf: int | None = (
        params.get("radix_down")
        or params.get("radix_down_leaf")
        or params.get("nics_per_leaf")
    )
    if radix_down_leaf is None:
        radix_down_leaf = _nearest_divisor(nodes, int(math.isqrt(nodes)))
    if radix_down_leaf <= 0:
        raise ValueError(f"leaf Radix_Down must be > 0, got {radix_down_leaf}")

    radix_up_leaf: int | None = params.get("radix_up") or params.get("radix_up_leaf")
    if radix_up_leaf is None:
        if radix_down_leaf % oversubscription != 0:
            raise ValueError(
                f"leaf Radix_Down ({radix_down_leaf}) must be divisible by oversubscription ({oversubscription})"
            )
        radix_up_leaf = radix_down_leaf // oversubscription
    if radix_up_leaf <= 0:
        raise ValueError(f"leaf Radix_Up must be > 0, got {radix_up_leaf}")

    if nodes % podsize != 0:
        podsize = nodes
    if podsize % radix_down_leaf != 0:
        podsize = nodes
        if podsize % radix_down_leaf != 0:
            raise ValueError(
                f"podsize ({podsize}) must be divisible by leaf Radix_Down ({radix_down_leaf})"
            )

    if nodes % oversubscription != 0:
        raise ValueError(
            f"total nodes ({nodes}) must be divisible by oversubscription ({oversubscription})"
        )

    no_of_pods = nodes // podsize
    no_of_tor_uplinks = nodes // oversubscription
    denominator = no_of_pods * radix_up_leaf
    if no_of_tor_uplinks % denominator != 0:
        raise ValueError(
            f"cannot derive an integer spine Radix_Down: {no_of_tor_uplinks} % {denominator} != 0"
        )
    radix_down_spine = no_of_tor_uplinks // denominator
    if radix_down_spine <= 0:
        raise ValueError(
            f"derived spine Radix_Down ({radix_down_spine}) must be > 0; podsize ({podsize}) may be too small relative to leaf radix ({radix_down_leaf})"
        )

    lines: list[str] = [
        f"Nodes {nodes}",
        "Tiers 2",
        f"Podsize {podsize}",
        "",
        "Tier 0",
        f"Downlink_speed_Gbps {speed_gbps}",
        f"Radix_Down {radix_down_leaf}",
        f"Radix_Up {radix_up_leaf}",
        f"Downlink_Latency_ns {downlink_latency_ns}",
        f"Switch_Latency_ns {switch_latency_ns}",
    ]
    if oversubscription != 1:
        lines.append(f"Oversubscribed {oversubscription}")
    lines.extend([
        "",
        "Tier 1",
        f"Downlink_speed_Gbps {speed_gbps}",
        f"Radix_Down {radix_down_spine}",
        f"Downlink_Latency_ns {downlink_latency_ns}",
        f"Switch_Latency_ns {switch_latency_ns}",
    ])

    return "\n".join(lines) + "\n"


def write_topology(dc: DatacenterConfig, path: str | Path) -> None:
    """Generate and write an htsim topology file to *path*."""
    _ = Path(path).write_text(generate_topology(dc))
