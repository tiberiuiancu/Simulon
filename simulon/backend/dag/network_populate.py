from __future__ import annotations

import functools
import logging
import re

from simulon.backend.dag.nodes import ExecutionDAG
from simulon.config.dc import DatacenterConfig
from simulon.config.resolve import resolve_node_spec

logger = logging.getLogger(__name__)


_default_intra_bw = 0.0
_default_intra_latency = 0.0
_default_inter_bw = 0.0
_default_inter_latency = 0.0


def _parse_speed(s: str) -> float:
    """Parse a bandwidth string to bytes per millisecond.

    Handles: Gbps, Mbps, GBps, MBps
    """
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(G|M)(b|B)ps", s.strip())
    if not m:
        raise ValueError(f"Cannot parse bandwidth: {s!r}")
    value = float(m.group(1))
    magnitude = m.group(2)
    unit = m.group(3)
    if unit == "b":
        bits_per_sec = value * (1e9 if magnitude == "G" else 1e6)
    else:  # "B" -> bytes -> bits
        bits_per_sec = value * 8 * (1e9 if magnitude == "G" else 1e6)
    bytes_per_ms = bits_per_sec / 8 / 1000
    return bytes_per_ms


def _parse_latency(s: str) -> float:
    """Parse a latency string to milliseconds.

    Handles: ms, us, ns
    """
    m = re.fullmatch(r"([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)\s*(ms|us|ns)", s.strip())
    if not m:
        raise ValueError(f"Cannot parse latency: {s!r}")
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "ms":
        return value
    elif unit == "us":
        return value / 1000
    else:  # ns
        return value / 1_000_000


def _ensure_defaults() -> None:
    global _default_intra_bw, _default_intra_latency, _default_inter_bw, _default_inter_latency
    if _default_intra_bw == 0.0:
        _default_intra_bw = _parse_speed("2880Gbps")
    if _default_inter_bw == 0.0:
        _default_inter_bw = _parse_speed("400Gbps") * 0.85


@functools.cache
def _get_link_params(
    src_gpu: int,
    dst_gpu: int,
    gpus_per_node: int,
    intra_bw_bytes_per_ms: float,
    intra_latency_ms: float,
    inter_bw_bytes_per_ms: float,
    inter_latency_ms: float,
) -> tuple[float, float]:
    is_intra = (src_gpu // gpus_per_node) == (dst_gpu // gpus_per_node)
    if is_intra:
        return intra_bw_bytes_per_ms, intra_latency_ms
    return inter_bw_bytes_per_ms, inter_latency_ms


def populate_network(
    dag: ExecutionDAG,
    datacenter: DatacenterConfig,
    per_step_latency_ms: float = 0.0,
    bw_override_bytes_per_ms: float | None = None,
    inter_bw_override_bytes_per_ms: float | None = None,
) -> ExecutionDAG:
    """Fill CommNode.duration_ms using the analytical network model (latency + bytes/bandwidth).

    No congestion is modeled -- each flow's duration is a fixed function of its
    transfer size and the link spec between src_gpu and dst_gpu.

    per_step_latency_ms is no longer used -- NCCL kernel launch overhead is captured
    in the effective BW from calbusbw (nccl-tests measurements already include it).

    bw_override_bytes_per_ms, when set, replaces the intra-node bandwidth from the
    datacenter spec. Used to apply per-collective effective NVLink bandwidth calibration.

    inter_bw_override_bytes_per_ms, when set, replaces the inter-node (NIC) bandwidth.
    Used to apply per-collective NIC efficiency from calbusbw.

    Mutates nodes in-place and returns the dag.
    """
    _ensure_defaults()

    node = resolve_node_spec(datacenter)
    gpus_per_node = node.gpus_per_node
    if gpus_per_node is None:
        raise ValueError("node.gpus_per_node must be set after resolution")

    if node.scale_up and node.scale_up.switch:
        sw = node.scale_up.switch
        if not isinstance(sw, str):
            intra_bw = _parse_speed(sw.port_speed) if sw.port_speed else _default_intra_bw
            intra_latency = _parse_latency(sw.latency) if sw.latency else _default_intra_latency
        else:
            raise ValueError(
                f"node.scale_up.switch is a string reference {sw!r} -- string switch templates "
                "not yet supported. Specify inline or use a node template with inline switch spec."
            )
    else:
        intra_bw, intra_latency = _default_intra_bw, _default_intra_latency

    if node.scale_out and node.scale_out.nic:
        nic = node.scale_out.nic
        if not isinstance(nic, str):
            inter_bw = (
                _parse_speed(nic.speed) * nic.bandwidth_efficiency
                if nic.speed
                else _default_inter_bw
            )
            inter_latency = _parse_latency(nic.latency) if nic.latency else _default_inter_latency
        else:
            raise ValueError(
                f"node.scale_out.nic is a string reference {nic!r} -- string NIC templates "
                "not yet supported. Specify inline or use a node template with inline NIC spec."
            )
    else:
        inter_bw, inter_latency = _default_inter_bw, _default_inter_latency

    if bw_override_bytes_per_ms is not None:
        intra_bw = bw_override_bytes_per_ms
    if inter_bw_override_bytes_per_ms is not None:
        inter_bw = inter_bw_override_bytes_per_ms

    for comm_node in dag.comm_nodes:
        bw, latency_ms = _get_link_params(
            comm_node.src_gpu,
            comm_node.dst_gpu,
            gpus_per_node,
            intra_bw,
            intra_latency,
            inter_bw,
            inter_latency,
        )
        comm_node.duration_ms = (
            latency_ms + per_step_latency_ms + (comm_node.bytes / bw if bw > 0 else 0.0)
        )

    return dag
