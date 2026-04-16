from __future__ import annotations

import logging
import re
import warnings

from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.nodes import ExecutionDAG
from simulon.config.dc import DatacenterConfig, GPUSpec, NICSpec, SwitchSpec
from simulon.config.resolve import resolve_node_spec, resolve_scale_out
from simulon.config.workload import MegatronWorkload
from simulon.profiling.lookup import lookup_kernel_time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network helpers (shared with replayer)
# ---------------------------------------------------------------------------


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
    else:  # "B" → bytes → bits
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


def _get_link_params(
    src_gpu: int,
    dst_gpu: int,
    datacenter: DatacenterConfig,
) -> tuple[float, float]:
    """Return (bandwidth_bytes_per_ms, latency_ms) for the logical link src_gpu→dst_gpu."""
    node = resolve_node_spec(datacenter)
    gpus_per_node = node.gpus_per_node
    if gpus_per_node is None:
        raise ValueError("node.gpus_per_node must be set after resolution")
    is_intra = (src_gpu // gpus_per_node) == (dst_gpu // gpus_per_node)

    if is_intra:
        switch_spec: SwitchSpec | None = None
        # Prefer node.scale_up (new location), fall back to network.scale_up (deprecated).
        if node.scale_up and node.scale_up.switch:
            sw = node.scale_up.switch
            if isinstance(sw, SwitchSpec):
                switch_spec = sw
            else:
                raise ValueError(
                    f"node.scale_up.switch is a string reference {sw!r} — "
                    "string switch templates are not yet supported. "
                    "Specify the switch inline or use a node template with an inline switch spec."
                )
        elif datacenter.network and datacenter.network.scale_up and datacenter.network.scale_up.switch:
            warnings.warn(
                "network.scale_up is deprecated. Move scale_up into the node spec.",
                DeprecationWarning,
                stacklevel=2,
            )
            sw = datacenter.network.scale_up.switch
            if isinstance(sw, SwitchSpec):
                switch_spec = sw
            else:
                raise ValueError(
                    f"network.scale_up.switch is a string reference {sw!r} — "
                    "string switch templates are not yet supported. "
                    "Specify the switch inline or use a node template with an inline switch spec."
                )
        bw = _parse_speed(switch_spec.port_speed) if (switch_spec and switch_spec.port_speed) else _parse_speed("2880Gbps")
        latency_ms = _parse_latency(switch_spec.latency) if (switch_spec and switch_spec.latency) else 0.0
    else:
        nic_spec: NICSpec | None = None
        scale_out = resolve_scale_out(datacenter)
        if scale_out and scale_out.nic:
            nic = scale_out.nic
            if isinstance(nic, NICSpec):
                nic_spec = nic
        if nic_spec and nic_spec.speed:
            bw = _parse_speed(nic_spec.speed) * nic_spec.bandwidth_efficiency
        else:
            bw = _parse_speed("400Gbps") * 0.85
        latency_ms = _parse_latency(nic_spec.latency) if (nic_spec and nic_spec.latency) else 0.0

    return bw, latency_ms


# ---------------------------------------------------------------------------
# Populate functions
# ---------------------------------------------------------------------------


def populate_dag(
    dag: ExecutionDAG,
    workload: MegatronWorkload,
    gpu_spec: GPUSpec,
) -> ExecutionDAG:
    """Fill ComputeNode.duration_ms by looking up kernel times in gpu_spec.

    Mutates nodes in-place and returns the dag.
    """
    t = workload.training
    p = workload.parallelism

    match_params = {
        "hidden_size": _model_hidden_size(workload),
        "seq_len": t.sequence_length,
        "batch_size": t.micro_batch_size,
        "dtype": t.dtype.value,
        "tp": p.tp,
    }

    adamw_match_params = {"num_params": None, "dtype": t.dtype.value}

    with log_progress("  resolving compute", len(dag.compute_nodes), logger) as advance:
        for node in dag.compute_nodes:
            if node.kernel == "adamw":
                mp = {**adamw_match_params, "num_params": node.extra_params.get("num_params")}
                node.duration_ms = lookup_kernel_time("adamw", mp, gpu_spec)
            elif node.fused_kernels:
                times = [lookup_kernel_time(k, match_params, gpu_spec) for k in node.fused_kernels]
                node.duration_ms = None if any(t is None for t in times) else sum(times)
            else:
                node.duration_ms = lookup_kernel_time(node.kernel, match_params, gpu_spec)
            advance()

    return dag


def populate_network(
    dag: ExecutionDAG,
    datacenter: DatacenterConfig,
    per_step_latency_ms: float = 0.0,
    bw_override_bytes_per_ms: float | None = None,
    inter_bw_override_bytes_per_ms: float | None = None,
) -> ExecutionDAG:
    """Fill CommNode.duration_ms using the analytical network model (latency + bytes/bandwidth).

    No congestion is modeled — each flow's duration is a fixed function of its
    transfer size and the link spec between src_gpu and dst_gpu.

    per_step_latency_ms is no longer used — NCCL kernel launch overhead is captured
    in the effective BW from calbusbw (nccl-tests measurements already include it).

    bw_override_bytes_per_ms, when set, replaces the intra-node bandwidth from the
    datacenter spec. Used to apply per-collective effective NVLink bandwidth calibration.

    inter_bw_override_bytes_per_ms, when set, replaces the inter-node (NIC) bandwidth.
    Used to apply per-collective NIC efficiency from calbusbw.

    Mutates nodes in-place and returns the dag.
    """
    resolved_node = resolve_node_spec(datacenter)
    gpus_per_node = resolved_node.gpus_per_node
    if gpus_per_node is None:
        raise ValueError("node.gpus_per_node must be set after resolution")
    for comm_node in dag.comm_nodes:
        bw, latency_ms = _get_link_params(comm_node.src_gpu, comm_node.dst_gpu, datacenter)
        is_intra = (comm_node.src_gpu // gpus_per_node) == (comm_node.dst_gpu // gpus_per_node)
        if is_intra and bw_override_bytes_per_ms is not None:
            bw = bw_override_bytes_per_ms
        elif not is_intra and inter_bw_override_bytes_per_ms is not None:
            bw = inter_bw_override_bytes_per_ms
        comm_node.duration_ms = latency_ms + per_step_latency_ms + (comm_node.bytes / bw if bw > 0 else 0.0)

    return dag


def _model_hidden_size(workload: MegatronWorkload) -> int | None:
    from simulon.profiling.models import _resolve_model

    model = _resolve_model(workload.model)
    return model.hidden_size
