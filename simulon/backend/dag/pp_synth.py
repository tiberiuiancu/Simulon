"""Derive a low-pipeline-parallel trace from a high-PP trace of the same (TP, SP).

Per-layer compute is **independent of the pipeline degree**: tensor-parallel GEMM
shapes, the activation sizes and the TP collectives a transformer layer performs
depend on TP / hidden / seq / mbs — not on how many pipeline stages the model is
sliced into.  PP only decides *which layer range* lives on a device and adds the
inter-stage P2P + the pipeline bubble, both of which Simulon already models from the
per-rank slot order and the PP_Send/PP_Recv events.

So a target-PP stage can be assembled from a higher-PP trace by concatenating the
per-microbatch slot interiors of the source stages whose layers it owns (forward in
stage order, backward in reverse), then laying those slots out under the target PP's
1F1B schedule with synthesized P2P transfers.  ``_slot_streams`` drops the source's
own PP transfers; we re-emit the target's.

This exists because low-PP traces of a large model OOM during ``trace generate``
(fake process group => no DP peers => the full optimizer for a 1-/2-way-split shard
lands on one GPU).  The high-PP trace of the same (TP, SP) config fits and PP is
derivable, so the low-PP trace never has to be captured on hardware.

The synthesized trace is fed to the *unmodified* simulator, which computes the
pipeline bubble itself from the DAG — nothing here hand-fakes the schedule timing.

Validation: the derivation must be gated against build-consistent MEASURED runs
(experiments/cross_validation/validate_pp_derive.py); the pre-existing on-disk
low-PP traces are build-inconsistent and cannot serve as a gate.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from simulon.backend.dag.schedule_synth import CollectiveItem, ComputeItem, _slot_streams
from simulon.backend.dag.trace_parser import TraceEvent, TraceFile, TraceFileParser


def _representative_stream(streams: dict, direction: str) -> list:
    """Pick a steady-state microbatch's item stream for one direction.

    Prefer the stream with the most collectives (full per-layer structure); break
    ties toward the median microbatch id (avoids 1F1B warmup/cooldown slots that
    can be split into partial backward passes).
    """
    cands = [(mb, s) for (mb, d), s in streams.items() if d == direction]
    if not cands:
        return []
    mb_ids = [mb for mb, _ in cands if isinstance(mb, int)]
    median_mb = statistics.median(mb_ids) if mb_ids else 0

    def n_coll(s: list) -> int:
        return sum(1 for it in s if isinstance(it, CollectiveItem))

    best_coll = max(n_coll(s) for _, s in cands)
    full = [(mb, s) for mb, s in cands if n_coll(s) == best_coll]
    full.sort(key=lambda ms: abs((ms[0] if isinstance(ms[0], int) else 0) - median_mb))
    return full[0][1]


def _pp_transfer_bytes(stage_traces: list[TraceFile]) -> int:
    """The activation-tensor size at a PP boundary (same shape for any PP degree)."""
    for tr in stage_traces:
        for ev in tr.events:
            if ev.type == "collective" and ev.metadata.get("collective_type") in (
                "PP_Send", "PP_Recv"
            ):
                b = int(ev.metadata.get("bytes", 0) or 0)
                if b > 0:
                    return b
    return 0


def _oneF1B_actions(stage: int, pp: int, n_mb: int) -> list[tuple[str, int]]:
    """Per-stage 1F1B action order: warmup forwards, steady (B,F) pairs, cooldown.

    num_warmup = pp - stage (clamped to n_mb), confirmed against measured traces
    (PP=2: stage0=2, stage1=1; PP=4: stage0=4 ... stage3=1).
    """
    nw = max(0, min(pp - stage, n_mb))
    actions: list[tuple[str, int]] = [("fwd", i) for i in range(nw)]
    for i in range(n_mb - nw):
        actions.append(("bwd", i))
        actions.append(("fwd", nw + i))
    for i in range(n_mb - nw, n_mb):
        actions.append(("bwd", i))
    return actions


def _stage_interiors(stage_traces: list[TraceFile], src_pp: int, target_pp: int):
    """Per target stage, the concatenated representative fwd/bwd item streams.

    group = src_pp // target_pp source stages per target stage.  Forward keeps stage
    order (embedding lands on stage 0); backward reverses it (last layer first, loss
    on the final stage).
    """
    group = src_pp // target_pp
    reps = [
        {d: _representative_stream(_slot_streams(tr), d) for d in ("fwd", "bwd")}
        for tr in stage_traces
    ]
    fwd, bwd = [], []
    for t in range(target_pp):
        srcs = list(range(t * group, (t + 1) * group))
        fwd.append([it for s in srcs for it in reps[s]["fwd"]])
        bwd.append([it for s in reversed(srcs) for it in reps[s]["bwd"]])
    return fwd, bwd


def _emit_slot(events: list, t: float, items: list, mb: int, direction: str,
               stage: int) -> float:
    """Append slot_begin .. collectives .. slot_end, reconstructing gaps from compute."""
    events.append(TraceEvent("slot_begin", t, {
        "microbatch_id": mb, "direction": direction, "pipeline_stage": stage}))
    for it in items:
        if isinstance(it, ComputeItem):
            t += it.duration_ms
        elif isinstance(it, CollectiveItem):
            md = dict(it.metadata)
            md["microbatch_id"] = None
            md["direction"] = None
            events.append(TraceEvent("collective", t, md))
    events.append(TraceEvent("slot_end", t, {
        "microbatch_id": mb, "direction": direction, "pipeline_stage": stage}))
    return t


def synthesize_pp_trace(stage_traces: list[TraceFile], target_pp: int,
                        num_microbatches: int) -> dict[int, TraceFile]:
    """Build target-PP stage traces (one per stage) from ordered high-PP stage traces.

    Returns ``{global_rank: TraceFile}`` keyed by the target stage's representative
    rank (stage * world // target_pp), ready to write as trace_rank_{rank}.json.
    """
    src_pp = len(stage_traces)
    if src_pp % target_pp != 0:
        raise ValueError(f"target_pp={target_pp} must divide src_pp={src_pp}")
    world = stage_traces[0].world_size
    rps = world // target_pp           # ranks per target stage
    pp_bytes = _pp_transfer_bytes(stage_traces)
    fwd_items, bwd_items = _stage_interiors(stage_traces, src_pp, target_pp)

    out: dict[int, TraceFile] = {}
    for stage in range(target_pp):
        rank = stage * rps
        lo, hi = rank, rank + rps                  # this stage's rank span
        prev_rank, next_rank = rank - rps, rank + rps
        events: list[TraceEvent] = []
        t = 0.0
        spacer = 0.001
        for direction, mb in _oneF1B_actions(stage, target_pp, num_microbatches):
            # P2P before the slot
            if direction == "fwd" and stage > 0:
                events.append(TraceEvent("collective", t, {
                    "name": "recv_forward", "collective_type": "PP_Recv",
                    "bytes": pp_bytes, "group_ranks": [prev_rank, rank],
                    "microbatch_id": mb, "direction": "fwd"}))
            elif direction == "bwd" and stage < target_pp - 1:
                events.append(TraceEvent("collective", t, {
                    "name": "recv_backward", "collective_type": "PP_Recv",
                    "bytes": pp_bytes, "group_ranks": [rank, next_rank],
                    "microbatch_id": mb, "direction": "bwd"}))
            t += spacer
            items = (fwd_items if direction == "fwd" else bwd_items)[stage]
            t = _emit_slot(events, t, items, mb, direction, stage) + spacer
            # P2P after the slot
            if direction == "fwd" and stage < target_pp - 1:
                events.append(TraceEvent("collective", t, {
                    "name": "send_forward", "collective_type": "PP_Send",
                    "bytes": pp_bytes, "group_ranks": [rank, next_rank],
                    "microbatch_id": mb, "direction": "fwd"}))
            elif direction == "bwd" and stage > 0:
                events.append(TraceEvent("collective", t, {
                    "name": "send_backward", "collective_type": "PP_Send",
                    "bytes": pp_bytes, "group_ranks": [prev_rank, rank],
                    "microbatch_id": mb, "direction": "bwd"}))
            t += spacer
        src = stage_traces[min(stage * (src_pp // target_pp), src_pp - 1)]
        out[rank] = TraceFile(
            trace_format_version=src.trace_format_version,
            rank=rank, world_size=world, pipeline_stage=stage,
            events=events, total_flops=src.total_flops,
            energy_kwh=src.energy_kwh, co2eq_kg=src.co2eq_kg,
        )
    return out


def synthesize_pp1_trace(stage_traces: list[TraceFile], num_microbatches: int) -> TraceFile:
    """PP=1 convenience wrapper (single stage, no P2P)."""
    return synthesize_pp_trace(stage_traces, 1, num_microbatches)[0]


def _write_workload(source_dir: Path, dest_dir: Path, target_pp: int) -> None:
    import yaml
    src_wl = source_dir / "workload.yaml"
    if not src_wl.exists():
        return
    data = yaml.safe_load(src_wl.read_text())
    if isinstance(data, dict) and isinstance(data.get("config"), dict):
        data["config"]["pipeline-model-parallel-size"] = target_pp
        (dest_dir / "workload.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def derive_pp_from_dir(source_dir: Path, dest_dir: Path, src_pp: int, target_pp: int,
                       num_microbatches: int) -> Path:
    """Read a high-PP trace dir, write a derived target-PP trace dir, return dest_dir."""
    rank0 = TraceFileParser.parse(str(source_dir / "trace_rank_0.json"))
    rps_src = rank0.world_size // src_pp
    stage_traces = [
        TraceFileParser.parse(str(source_dir / f"trace_rank_{s * rps_src}.json"))
        for s in range(src_pp)
    ]
    derived = synthesize_pp_trace(stage_traces, target_pp, num_microbatches)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rank, tf in derived.items():
        (dest_dir / f"trace_rank_{rank}.json").write_text(tf.to_json(indent=None))
    _write_workload(source_dir, dest_dir, target_pp)
    return dest_dir


def derive_pp1_from_dir(source_dir: Path, dest_dir: Path, src_pp: int,
                        num_microbatches: int) -> Path:
    """PP=1 convenience wrapper (kept for the validation gate)."""
    return derive_pp_from_dir(source_dir, dest_dir, src_pp, 1, num_microbatches)
