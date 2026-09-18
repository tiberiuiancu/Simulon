"""pp_synth must carry per-slot metrics (kernel_device_ms, host_ms, host_ops, ...) through
schedule recomposition, summing them over the source stages a target stage absorbs.

Before 2026-09-16 `synthesize_pp_trace` refused any trace carrying these fields, which made
PP derivation unusable for exactly the traces the host model needs. Dropping them silently
would have been worse: kernel_device_ms absent => replay falls back to wall-clock spans;
host_ms/host_ops absent => zero modelled host time. So the derived slot must carry the SUM
of its source slots' scalars -- the layer-linearity measured on Qwen3-32B (job 1836105:
per-stage kernel time tracks the layer count to ~2%).
"""
from __future__ import annotations

from simulon.backend.dag.pp_synth import _SLOT_METRICS, synthesize_pp_trace
from simulon.backend.dag.trace_parser import TraceEvent, TraceFile


def _stage(rank: int, stage: int, world: int, kernel: float, host: float, ops: int) -> TraceFile:
    """A PP=2 stage trace with 2 microbatches (fwd+bwd each) and a PP transfer."""
    ev: list[TraceEvent] = []
    t = 0.0
    for mb in range(2):
        for direction in ("fwd", "bwd"):
            ev.append(TraceEvent("slot_begin", t, {
                "microbatch_id": mb, "direction": direction, "pipeline_stage": stage,
                "kernel_device_ms": kernel, "host_ms": host, "host_ops": ops,
                "launch_count": ops + 1, "kernel_count": ops // 2,
            }))
            t += 10.0
            ev.append(TraceEvent("slot_end", t, {
                "microbatch_id": mb, "direction": direction, "pipeline_stage": stage}))
            t += 0.5
            ev.append(TraceEvent("collective", t, {
                "name": "send_forward" if direction == "fwd" else "send_backward",
                "collective_type": "PP_Send", "bytes": 4096,
                "group_ranks": [rank, rank + 1], "microbatch_id": mb, "direction": direction,
            }))
            t += 0.5
    return TraceFile(trace_format_version="1.0", rank=rank, world_size=world,
                     pipeline_stage=stage, events=ev, total_flops=1.0,
                     energy_kwh=None, co2eq_kg=None)


def test_derived_slot_metrics_are_summed_over_absorbed_stages():
    src = [_stage(0, 0, 2, kernel=30.0, host=40.0, ops=1000),
           _stage(1, 1, 2, kernel=50.0, host=60.0, ops=3000)]
    out = synthesize_pp_trace(src, target_pp=1, num_microbatches=2)
    (tf,) = out.values()
    begins = [e for e in tf.events if e.type == "slot_begin"]
    assert begins, "derived trace has no slots"
    for e in begins:
        for k in _SLOT_METRICS:
            assert k in e.metadata, f"{k} dropped from the derived slot"
        assert e.metadata["kernel_device_ms"] == 80.0      # 30 + 50
        assert e.metadata["host_ms"] == 100.0              # 40 + 60
        assert e.metadata["host_ops"] == 4000              # 1000 + 3000


def test_same_pp_derivation_keeps_metrics_unchanged():
    """target_pp == src_pp absorbs exactly one source stage per target stage."""
    src = [_stage(0, 0, 2, kernel=30.0, host=40.0, ops=1000),
           _stage(1, 1, 2, kernel=50.0, host=60.0, ops=3000)]
    out = synthesize_pp_trace(src, target_pp=2, num_microbatches=2)
    k = {tf.pipeline_stage: {e.metadata["kernel_device_ms"] for e in tf.events if e.type == "slot_begin"}
         for tf in out.values()}
    assert k == {0: {30.0}, 1: {50.0}}


def test_span_only_source_yields_no_metric_keys():
    """No metrics on the source => none invented on the target (the replay then uses spans,
    which is the correct behaviour for a span-only trace)."""
    src = [_stage(0, 0, 2, 0.0, 0.0, 0), _stage(1, 1, 2, 0.0, 0.0, 0)]
    for tr in src:
        for e in tr.events:
            if e.type == "slot_begin":
                for m in _SLOT_METRICS:
                    e.metadata.pop(m, None)
    (tf,) = synthesize_pp_trace(src, target_pp=1, num_microbatches=2).values()
    for e in tf.events:
        if e.type == "slot_begin":
            assert not any(m in e.metadata for m in _SLOT_METRICS)
