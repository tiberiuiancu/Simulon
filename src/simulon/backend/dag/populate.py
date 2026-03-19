from __future__ import annotations

import logging
import re

from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.nodes import ExecutionDAG
from simulon.config.dc import DatacenterConfig, GPUSpec, NICSpec, SwitchSpec
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
    gpus_per_node = datacenter.node.gpus_per_node
    is_intra = (src_gpu // gpus_per_node) == (dst_gpu // gpus_per_node)

    network = datacenter.network

    if is_intra:
        switch_spec: SwitchSpec | None = None
        if network and network.scale_up and network.scale_up.switch:
            sw = network.scale_up.switch
            if isinstance(sw, SwitchSpec):
                switch_spec = sw
        bw = _parse_speed(switch_spec.port_speed) if (switch_spec and switch_spec.port_speed) else _parse_speed("2880Gbps")
        latency_ms = _parse_latency(switch_spec.latency) if (switch_spec and switch_spec.latency) else 0.0
    else:
        nic_spec: NICSpec | None = None
        if network and network.scale_out and network.scale_out.nic:
            nic = network.scale_out.nic
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

    with log_progress("  resolving compute", len(dag.compute_nodes), logger) as advance:
        for node in dag.compute_nodes:
            node.duration_ms = lookup_kernel_time(node.kernel, match_params, gpu_spec)
            advance()

    return dag


def populate_network(
    dag: ExecutionDAG,
    datacenter: DatacenterConfig,
) -> ExecutionDAG:
    """Fill CommNode.duration_ms using the analytical network model (latency + bytes/bandwidth).

    No congestion is modeled — each flow's duration is a fixed function of its
    transfer size and the link spec between src_gpu and dst_gpu.

    Mutates nodes in-place and returns the dag.
    """
    for node in dag.comm_nodes:
        bw, latency_ms = _get_link_params(node.src_gpu, node.dst_gpu, datacenter)
        node.duration_ms = latency_ms + (node.bytes / bw if bw > 0 else 0.0)

    return dag


def _model_hidden_size(workload: MegatronWorkload) -> int | None:
    from simulon.profiling.models import _resolve_model

    model = _resolve_model(workload.model)
    return model.hidden_size
