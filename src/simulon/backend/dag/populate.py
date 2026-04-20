from __future__ import annotations

import logging
import re
import warnings

from simulon.backend.dag._progress import log_progress
from simulon.backend.dag.nodes import ExecutionDAG
from simulon.config.dc import DatacenterConfig, GPUSpec, NICSpec, SwitchSpec
from simulon.config.resolve import resolve_node_spec, resolve_scale_out
from simulon.config.workload import MegatronWorkload
from simulon.profiling.lookup import is_kernel_oom, lookup_kernel_time

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
        elif (
            datacenter.network
            and datacenter.network.scale_up
            and datacenter.network.scale_up.switch
        ):
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
        bw = (
            _parse_speed(switch_spec.port_speed)
            if (switch_spec and switch_spec.port_speed)
            else _parse_speed("2880Gbps")
        )
        latency_ms = (
            _parse_latency(switch_spec.latency)
            if (switch_spec and switch_spec.latency)
            else 0.0
        )
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
        latency_ms = (
            _parse_latency(nic_spec.latency) if (nic_spec and nic_spec.latency) else 0.0
        )

    return bw, latency_ms


# ---------------------------------------------------------------------------
# Populate functions
# ---------------------------------------------------------------------------


def _handle_missing(
    kernel: str,
    params: dict,
    gpu_spec: GPUSpec,
    ignore_oom: bool,
    ignore_missing: bool,
) -> None:
    """Called when lookup_kernel_time returns None.

    - If the kernel+params match a known OOM profiling entry: raise RuntimeError
      (unless ignore_oom=True, in which case silently proceed with None timing).
    - Otherwise: raise RuntimeError for missing profiling data (unless
      ignore_missing=True, in which case emit a warning and proceed with None).
    """
    if is_kernel_oom(kernel, params, gpu_spec):
        if not ignore_oom:
            raise RuntimeError(
                f"Kernel '{kernel}' matches a known OOM profiling entry. "
                "Pass --ignore-oom to suppress this error and simulate anyway."
            )
    else:
        if not ignore_missing:
            raise RuntimeError(
                f"No profiling data found for kernel '{kernel}' with params {params}. "
                "Reprofile the GPU or pass --ignore-missing to proceed with 0 timing."
            )
        warnings.warn(
            f"No profiling data found for kernel '{kernel}'. "
            "Timing will be None — results may be incomplete.",
            UserWarning,
            stacklevel=4,
        )


def populate_dag(
    dag: ExecutionDAG,
    workload: MegatronWorkload,
    gpu_spec: GPUSpec,
    ignore_oom: bool = False,
    ignore_missing: bool = False,
) -> ExecutionDAG:
    """Fill ComputeNode.duration_ms by looking up kernel times in gpu_spec.

    For each kernel node, if no timing data is found and the kernel+params
    combination matches a known OOM profiling entry in gpu_spec.oom_kernel_runs,
    a RuntimeError is raised unless ignore_oom=True.

    Sets ComputeNode.is_extrapolated=True when the duration was obtained via
    linear extrapolation rather than an exact or partial profile match.

    Mutates nodes in-place and returns the dag.
    """
    from simulon.profiling.models import _resolve_model

    t = workload.training
    p = workload.parallelism
    model = _resolve_model(workload.model)

    # Build a comprehensive params dict covering all kernel types.
    # lookup_kernel_time will filter to only the params relevant to each kernel.
    all_params: dict = {
        "hidden_size": model.hidden_size,
        "num_heads": model.num_heads,
        "ffn_hidden_size": model.ffn_hidden_size,
        "vocab_size": model.vocab_size,
        "seq_len": t.sequence_length,
        "batch_size": t.micro_batch_size,
        "dtype": t.dtype.value,
        "tp": p.tp,
        "ep": p.ep,
    }
    if model.num_experts is not None:
        all_params["num_experts"] = model.num_experts
    if model.top_k is not None:
        all_params["top_k"] = model.top_k

    adamw_base = {"dtype": t.dtype.value}

    with log_progress("  resolving compute", len(dag.compute_nodes), logger) as advance:
        for node in dag.compute_nodes:
            if node.kernel == "adamw":
                mp = {**adamw_base, "num_params": node.extra_params.get("num_params")}
                time_ms, extrap = lookup_kernel_time("adamw", mp, gpu_spec)
                if time_ms is None:
                    _handle_missing("adamw", mp, gpu_spec, ignore_oom, ignore_missing)
            elif node.fused_kernels:
                fused_results = [(k, lookup_kernel_time(k, all_params, gpu_spec)) for k in node.fused_kernels]
                times = [r[0] for _, r in fused_results]
                time_ms = None if any(t is None for t in times) else sum(times)  # type: ignore[arg-type]
                extrap = any(r[1] for _, r in fused_results)
                if time_ms is None:
                    for k, (t, _) in fused_results:
                        if t is None:
                            _handle_missing(k, all_params, gpu_spec, ignore_oom, ignore_missing)
            else:
                time_ms, extrap = lookup_kernel_time(node.kernel, all_params, gpu_spec)
                if time_ms is None:
                    _handle_missing(node.kernel, all_params, gpu_spec, ignore_oom, ignore_missing)

            node.duration_ms = time_ms
            node.is_extrapolated = extrap
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
        bw, latency_ms = _get_link_params(
            comm_node.src_gpu, comm_node.dst_gpu, datacenter
        )
        is_intra = (comm_node.src_gpu // gpus_per_node) == (
            comm_node.dst_gpu // gpus_per_node
        )
        if is_intra and bw_override_bytes_per_ms is not None:
            bw = bw_override_bytes_per_ms
        elif not is_intra and inter_bw_override_bytes_per_ms is not None:
            bw = inter_bw_override_bytes_per_ms
        comm_node.duration_ms = (
            latency_ms + per_step_latency_ms + (comm_node.bytes / bw if bw > 0 else 0.0)
        )

    return dag


def _model_hidden_size(workload: MegatronWorkload) -> int | None:
    from simulon.profiling.models import _resolve_model

    model = _resolve_model(workload.model)
    return model.hidden_size
