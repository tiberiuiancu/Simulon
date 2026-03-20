"""Chrome Trace (Perfetto/chrome://tracing) export for a populated ExecutionDAG.

Usage:
    dag, result = backend.simulate(scenario)
    trace = to_chrome_trace(dag, tp=2, pp=2, dp=1)
    with open("trace.json", "w") as f:
        json.dump(trace, f)

Requires replay() to have been called so that start_ms/finish_ms/duration_ms
are populated on all nodes.

Layout:
  - One pid per GPU (pid = 1000 + gpu_rank).
  - PIDs are sorted by (dp_rank, pp_stage, tp_rank) — so within each DP replica
    you see PP stages in order, and within each PP stage you see TP ranks together.
  - Three tids per GPU:
      tid 1000 — Compute
      tid 1001 — Comm (Send)   [CommNode as src_gpu]
      tid 1002 — Comm (Recv)   [CommNode as dst_gpu]
  - Timestamps and durations are in microseconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simulon.backend.dag.nodes import CommNode, ComputeNode, ExecutionDAG

_TID_COMPUTE   = 1000
_TID_COLL_SEND = 1001   # AllGather / ReduceScatter / AllReduce send
_TID_COLL_RECV = 1002   # AllGather / ReduceScatter / AllReduce recv
_TID_PP_SEND   = 1003   # PP_Send (inter-stage point-to-point, send side)
_TID_PP_RECV   = 1004   # PP_Send recv side


def _decode_rank(gpu_rank: int, tp: int, pp: int, ep: int = 1) -> tuple[int, int, int, int]:
    """Return (dp_rank, pp_stage, ep_rank, tp_rank) for a gpu_rank."""
    tp_rank = gpu_rank % tp
    ep_rank = (gpu_rank // tp) % ep
    pp_stage = (gpu_rank // (tp * ep)) % pp
    dp_rank = gpu_rank // (tp * ep * pp)
    return dp_rank, pp_stage, ep_rank, tp_rank


def to_chrome_trace(dag: ExecutionDAG, tp: int, pp: int, dp: int, ep: int = 1) -> dict[str, Any]:
    """Build a Chrome Trace dict from a timing-populated ExecutionDAG.

    Args:
        dag:  ExecutionDAG after replay() has been called.
        tp:   Tensor parallelism degree.
        pp:   Pipeline parallelism degree.
        dp:   Data parallelism degree.

    Returns:
        Dict with "traceEvents" list, ready for json.dump().

    Compute event args include: kernel, phase, layer_id, microbatch_id,
    pipeline_stage, duration_ms. For compacted nodes (fused sequential kernels),
    a ``fused_kernels`` arg is also present with the comma-joined original kernel names.
    """
    events: list[dict[str, Any]] = []

    # Collect all GPU ranks present in the DAG
    all_gpus: set[int] = set()
    for n in dag.compute_nodes:
        all_gpus.add(n.gpu_rank)
    for n in dag.comm_nodes:
        all_gpus.add(n.src_gpu)
        all_gpus.add(n.dst_gpu)

    # Emit process/thread metadata sorted by (dp, pp, tp) = natural gpu_rank order
    for gpu in sorted(all_gpus):
        pid = 1000 + gpu
        dp_rank, pp_stage, ep_rank, tp_rank = _decode_rank(gpu, tp, pp, ep)

        # Group label: DP replicas together, PP stages within, EP/TP ranks innermost
        proc_name = f"GPU {gpu} | DP={dp_rank} PP={pp_stage} EP={ep_rank} TP={tp_rank}"
        sort_idx = dp_rank * (pp * ep * tp) + pp_stage * (ep * tp) + ep_rank * tp + tp_rank

        events += [
            {"name": "process_name",       "ph": "M", "pid": pid, "tid": 0,
             "args": {"name": proc_name}},
            {"name": "process_sort_index", "ph": "M", "pid": pid, "tid": 0,
             "args": {"sort_index": sort_idx}},
            {"name": "thread_name",       "ph": "M", "pid": pid, "tid": _TID_COMPUTE,
             "args": {"name": "Compute"}},
            {"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": _TID_COMPUTE,
             "args": {"sort_index": 0}},
            {"name": "thread_name",       "ph": "M", "pid": pid, "tid": _TID_COLL_SEND,
             "args": {"name": "Coll Send (collective)"}},
            {"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": _TID_COLL_SEND,
             "args": {"sort_index": 1}},
            {"name": "thread_name",       "ph": "M", "pid": pid, "tid": _TID_COLL_RECV,
             "args": {"name": "Coll Recv (collective)"}},
            {"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": _TID_COLL_RECV,
             "args": {"sort_index": 2}},
            {"name": "thread_name",       "ph": "M", "pid": pid, "tid": _TID_PP_SEND,
             "args": {"name": "PP Send"}},
            {"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": _TID_PP_SEND,
             "args": {"sort_index": 3}},
            {"name": "thread_name",       "ph": "M", "pid": pid, "tid": _TID_PP_RECV,
             "args": {"name": "PP Recv"}},
            {"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": _TID_PP_RECV,
             "args": {"sort_index": 4}},
        ]

    # Compute events
    for n in dag.compute_nodes:
        if n.start_ms is None or n.finish_ms is None:
            continue
        args: dict[str, Any] = {
            "kernel":          n.kernel,
            "phase":           n.phase,
            "layer_id":        n.layer_id,
            "microbatch_id":   n.microbatch_id,
            "pipeline_stage":  n.pipeline_stage,
            "duration_ms":     n.duration_ms,
        }
        if n.fused_kernels:
            args["fused_kernels"] = ", ".join(n.fused_kernels)
        events.append({
            "name": n.kernel,
            "ph": "X",
            "pid": 1000 + n.gpu_rank,
            "tid": _TID_COMPUTE,
            "ts":  n.start_ms * 1_000,
            "dur": (n.duration_ms or 0.0) * 1_000,
            "args": args,
        })

    # Comm events — one send event on src, one recv event on dst.
    # The tracer creates one CommNode per GPU participating in a collective, so the
    # same physical P2P transfer (src, dst, start_ms, bytes) may appear multiple times.
    # Deduplicate by physical identity before emitting.
    seen_transfers: set[tuple[int, int, float, int]] = set()  # (src, dst, start_ms, bytes)

    for n in dag.comm_nodes:
        if n.start_ms is None or n.finish_ms is None:
            continue

        key = (n.src_gpu, n.dst_gpu, n.start_ms, n.bytes)
        if key in seen_transfers:
            continue
        seen_transfers.add(key)

        ts_us  = n.start_ms * 1_000
        dur_us = (n.duration_ms or 0.0) * 1_000
        args = {
            "collective_type": n.collective_type,
            "phase":           n.phase,
            "layer_id":        n.layer_id,
            "bytes":           n.bytes,
            "duration_ms":     n.duration_ms,
            "src_gpu":         n.src_gpu,
            "dst_gpu":         n.dst_gpu,
            "flow_id":         n.flow_id,
        }
        is_pp = n.collective_type == "PP_Send"
        tid_send = _TID_PP_SEND   if is_pp else _TID_COLL_SEND
        tid_recv = _TID_PP_RECV   if is_pp else _TID_COLL_RECV

        events.append({
            "name": f"{n.collective_type} \u2192 GPU{n.dst_gpu}",
            "ph": "X",
            "pid": 1000 + n.src_gpu,
            "tid": tid_send,
            "ts":  ts_us,
            "dur": dur_us,
            "args": args,
        })
        events.append({
            "name": f"{n.collective_type} \u2190 GPU{n.src_gpu}",
            "ph": "X",
            "pid": 1000 + n.dst_gpu,
            "tid": tid_recv,
            "ts":  ts_us,
            "dur": dur_us,
            "args": args,
        })

    return {"traceEvents": events}


def write_chrome_trace(
    dag: ExecutionDAG,
    tp: int,
    pp: int,
    dp: int,
    path: str | Path,
    ep: int = 1,
) -> None:
    """Write a Chrome Trace JSON file from a populated ExecutionDAG."""
    import json
    trace = to_chrome_trace(dag, tp=tp, pp=pp, dp=dp, ep=ep)
    with open(path, "w") as f:
        json.dump(trace, f)
