"""Chrome Trace (Perfetto/chrome://tracing) export for a populated ExecutionDAG.

Usage:
    dag, result = backend.simulate(scenario)
    from simulon.backend.dag.trace_tracer import ParallelConfig
    config = ParallelConfig.from_workload(workload)
    trace = to_chrome_trace(dag, config=config)
    with open("trace.json", "w") as f:
        json.dump(trace, f)

Requires replay() to have been called so that start_ms/finish_ms/duration_ms
are populated on all nodes.

Layout:
  - One pid per GPU (pid = 1000 + gpu_rank).
  - PIDs are sorted by proper MoE/non-MoE parallel folding (dp, pp, ep, tp, edp, etp).
  - GPU labels show both attention and expert coordinates for clarity.
  - Three tids per GPU:
      tid 1000 — Compute
      tid 1001 — Comm (Send)   [CommNode as src_gpu]
      tid 1002 — Comm (Recv)   [CommNode as dst_gpu]
  - Timestamps and durations are in microseconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simulon.backend.dag.nodes import ExecutionDAG
from simulon.backend.dag.trace_tracer import ParallelConfig, _decompose_rank

_TID_COMPUTE = 1000
_TID_COLL_SEND = 1001  # AllGather / ReduceScatter / AllReduce send
_TID_COLL_RECV = 1002  # AllGather / ReduceScatter / AllReduce recv
_TID_PP_SEND = 1003  # PP_Send (inter-stage point-to-point, send side)
_TID_PP_RECV = 1004  # PP_Send recv side


def _decode_rank(gpu_rank: int, config: ParallelConfig) -> tuple[int, ...]:
    """Return decomposed coordinates for a gpu_rank using proper MoE/non-MoE folding."""
    coords = _decompose_rank(gpu_rank, config)
    return coords.dp, coords.pp, coords.ep, coords.tp, coords.edp, coords.etp


def to_chrome_trace(
    dag: ExecutionDAG, config: ParallelConfig, *, only_profiled: bool = False
) -> dict[str, Any]:
    """Build a Chrome Trace dict from a timing-populated ExecutionDAG.

    Args:
        dag:  ExecutionDAG after replay() has been called.
        config: ParallelConfig with tp/cp/ep/dp/pp/etp/edp dimensions.
        only_profiled: If True, only emit events for ranks that had exact trace files.

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
    for cn in dag.collective_nodes.values():
        all_gpus.update(cn.group_ranks)

    if only_profiled:
        all_gpus &= dag.profiled_ranks

    # Emit process/thread metadata sorted by (dp, pp, ep, tp, edp, etp) = natural gpu_rank order
    for gpu in sorted(all_gpus):
        pid = 1000 + gpu
        dp_rank, pp_stage, ep_rank, tp_rank, edp_rank, etp_rank = _decode_rank(gpu, config)

        proc_name = f"GPU {gpu} | DP={dp_rank} PP={pp_stage} TP={tp_rank}"
        if config.ep > 1 or config.etp != config.tp:
            proc_name += f" | EDP={edp_rank} EP={ep_rank} ETP={etp_rank}"
        sort_idx = (
            dp_rank * (config.pp * config.edp * config.etp)
            + pp_stage * (config.edp * config.etp)
            + edp_rank * config.etp
            + etp_rank
        )

        events += [
            {"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": proc_name}},
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"sort_index": sort_idx},
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": _TID_COMPUTE,
                "args": {"name": "Compute"},
            },
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": _TID_COMPUTE,
                "args": {"sort_index": 0},
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": _TID_COLL_SEND,
                "args": {"name": "Coll Send (collective)"},
            },
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": _TID_COLL_SEND,
                "args": {"sort_index": 1},
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": _TID_COLL_RECV,
                "args": {"name": "Coll Recv (collective)"},
            },
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": _TID_COLL_RECV,
                "args": {"sort_index": 2},
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": _TID_PP_SEND,
                "args": {"name": "PP Send"},
            },
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": _TID_PP_SEND,
                "args": {"sort_index": 3},
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": _TID_PP_RECV,
                "args": {"name": "PP Recv"},
            },
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": _TID_PP_RECV,
                "args": {"sort_index": 4},
            },
        ]

    # Compute events
    for n in dag.compute_nodes:
        if n.start_ms is None or n.finish_ms is None:
            continue
        if only_profiled and n.gpu_rank not in dag.profiled_ranks:
            continue
        args: dict[str, Any] = {
            "kernel": n.kernel,
            "phase": n.phase,
            "layer_id": n.layer_id,
            "microbatch_id": n.microbatch_id,
            "pipeline_stage": n.pipeline_stage,
            "duration_ms": n.duration_ms,
            "is_extrapolated": n.is_extrapolated,
        }
        if n.fused_kernels:
            args["fused_kernels"] = ", ".join(n.fused_kernels)
        base_name = f"{n.phase} {n.kernel}" if n.phase else n.kernel
        event_name = ("! " + base_name) if n.is_extrapolated else base_name
        entry: dict[str, Any] = {
            "name": event_name,
            "ph": "X",
            "pid": 1000 + n.gpu_rank,
            "tid": _TID_COMPUTE,
            "ts": n.start_ms * 1_000,
            "dur": (n.duration_ms or 0.0) * 1_000,
            "args": args,
        }
        events.append(entry)

    # Comm events — one send event on src, one recv event on dst.
    # The tracer creates one CommNode per GPU participating in a collective, so the
    # same physical P2P transfer (src, dst, start_ms, bytes) may appear multiple times.
    # Deduplicate by physical identity before emitting.
    seen_transfers: set[tuple[int, int, float, float, int]] = (
        set()
    )  # (src, dst, start_ms, finish_ms, bytes)

    for n in dag.comm_nodes:
        if n.start_ms is None or n.finish_ms is None:
            continue
        if only_profiled and (
            n.src_gpu not in dag.profiled_ranks and n.dst_gpu not in dag.profiled_ranks
        ):
            continue

        key = (n.src_gpu, n.dst_gpu, n.start_ms, n.finish_ms, n.bytes)
        if key in seen_transfers:
            continue
        seen_transfers.add(key)

        ts_us = n.start_ms * 1_000
        dur_us = (n.duration_ms or 0.0) * 1_000
        args = {
            "collective_type": n.collective_type,
            "phase": n.phase,
            "layer_id": n.layer_id,
            "bytes": n.bytes,
            "duration_ms": n.duration_ms,
            "src_gpu": n.src_gpu,
            "dst_gpu": n.dst_gpu,
            "flow_id": n.flow_id,
        }
        is_pp = n.collective_type == "PP_Send"
        tid_send = _TID_PP_SEND if is_pp else _TID_COLL_SEND
        tid_recv = _TID_PP_RECV if is_pp else _TID_COLL_RECV

        if not only_profiled or n.src_gpu in dag.profiled_ranks:
            events.append(
                {
                    "name": f"{n.collective_type} \u2192 GPU{n.dst_gpu}",
                    "ph": "X",
                    "pid": 1000 + n.src_gpu,
                    "tid": tid_send,
                    "ts": ts_us,
                    "dur": dur_us,
                    "args": args,
                }
            )
        if not only_profiled or n.dst_gpu in dag.profiled_ranks:
            events.append(
                {
                    "name": f"{n.collective_type} \u2190 GPU{n.src_gpu}",
                    "ph": "X",
                    "pid": 1000 + n.dst_gpu,
                    "tid": tid_recv,
                    "ts": ts_us,
                    "dur": dur_us,
                    "args": args,
                }
            )

    # Collective events — emitted on all participating GPUs when CollectiveNodes
    # are kept intact (collective-level network simulation).
    for cn in dag.collective_nodes.values():
        if cn.start_ms is None or cn.finish_ms is None:
            continue
        ts_us = cn.start_ms * 1_000
        dur_us = (cn.duration_ms or 0.0) * 1_000
        args = {
            "collective_type": cn.collective_type,
            "phase": cn.phase,
            "layer_id": cn.layer_id,
            "data_size": cn.data_size,
            "duration_ms": cn.duration_ms,
            "algorithm": cn.algorithm,
            "num_channels": cn.num_channels,
            "group_ranks": cn.group_ranks,
        }
        for gpu in cn.group_ranks:
            if only_profiled and gpu not in dag.profiled_ranks:
                continue
            events.append(
                {
                    "name": f"{cn.collective_type}",
                    "ph": "X",
                    "pid": 1000 + gpu,
                    "tid": _TID_COLL_SEND,
                    "ts": ts_us,
                    "dur": dur_us,
                    "args": args,
                }
            )

    return {"traceEvents": events}


def write_chrome_trace(
    dag: ExecutionDAG, config: ParallelConfig, path: str | Path, *, only_profiled: bool = False
) -> None:
    """Write a Chrome Trace JSON file from a populated ExecutionDAG."""
    import json

    trace = to_chrome_trace(dag, config=config, only_profiled=only_profiled)
    with open(path, "w") as f:
        json.dump(trace, f)
