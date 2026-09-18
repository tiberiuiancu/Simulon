"""Trimming microbatches must not depend on the micro-batch size changing.

A trace captured at M=64 microbatches per DP rank has to serve a run whose DP group is
bigger or whose GBS is smaller (M=8, same mbs). Until 2026-09-16 the trim lived only inside
the mbs-extrapolation branch, so with equal mbs all 64 microbatches were replayed and
gbs=64 returned the same iteration time as gbs=256 (experiments/ground.py: 16n TP4/PP2 at
gbs64 came out +389%). The tracer now calls `_extrapolate_trace_for_mbs` with scale 1.0 in
that case; this pins the function's trim-only behaviour.
"""
from __future__ import annotations

from simulon.backend.dag.trace_parser import TraceEvent, TraceFile
from simulon.backend.dag.trace_tracer import _extrapolate_trace_for_mbs


def _trace(n_mb: int) -> TraceFile:
    ev: list[TraceEvent] = []
    t = 0.0
    for mb in range(n_mb):
        for direction in ("fwd", "bwd"):
            ev.append(TraceEvent("slot_begin", t, {"microbatch_id": mb, "direction": direction,
                                                   "pipeline_stage": 0, "kernel_device_ms": 7.0,
                                                   "host_ms": 9.0, "host_ops": 100}))
            t += 10.0
            ev.append(TraceEvent("slot_end", t, {"microbatch_id": mb, "direction": direction,
                                                 "pipeline_stage": 0}))
            t += 1.0
    ev.append(TraceEvent("slot_begin", t, {"microbatch_id": None, "direction": "step", "pipeline_stage": 0}))
    ev.append(TraceEvent("slot_end", t + 5.0, {"microbatch_id": None, "direction": "step", "pipeline_stage": 0}))
    return TraceFile(trace_format_version="1.0", rank=0, world_size=1, pipeline_stage=0,
                     events=ev, total_flops=None, energy_kwh=None, co2eq_kg=None)


def _mb_ids(tf: TraceFile) -> set:
    return {e.metadata.get("microbatch_id") for e in tf.events
            if e.type == "slot_begin" and e.metadata.get("direction") != "step"}


def test_trim_only_drops_microbatches_and_keeps_timing():
    src = _trace(8)
    out = _extrapolate_trace_for_mbs(src, 1, 1, 3, keep_step=True)      # same mbs -> scale 1.0
    assert _mb_ids(out) == {0, 1, 2}
    # the optimizer step is per iteration and must survive a trim
    assert any(e.type == "slot_begin" and e.metadata.get("direction") == "step" for e in out.events)
    kept = [e for e in out.events if e.type == "slot_begin" and e.metadata.get("direction") != "step"]
    assert all(e.metadata["kernel_device_ms"] == 7.0 for e in kept), "scale 1.0 must not touch durations"
    assert all(e.metadata["host_ops"] == 100 for e in kept)
    # timestamps unscaled: the first backward slot still begins at 11.0
    first_bwd = next(e for e in kept if e.metadata["direction"] == "bwd")
    assert first_bwd.timestamp_ms == 11.0


def test_no_trim_when_counts_match():
    src = _trace(4)
    out = _extrapolate_trace_for_mbs(src, 1, 1, 4)
    assert _mb_ids(out) == {0, 1, 2, 3}


def test_mbs_scaling_path_still_drops_step_by_default():
    """Unchanged legacy behaviour for the validated mbs-extrapolation path."""
    out = _extrapolate_trace_for_mbs(_trace(4), 1, 2, 2)
    assert not any(e.type == "slot_begin" and e.metadata.get("direction") == "step" for e in out.events)
