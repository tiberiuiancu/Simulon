"""Tests for steady-state microbatch replication (decoupling trace GBS from sim GBS)."""

from __future__ import annotations

from simulon.backend.dag.trace_parser import TraceEvent, TraceFile
from simulon.backend.dag.trace_tracer import (
    _replicate_trace_microbatches,
    _slot_direction,
    _split_trace_items,
)


def _slot(events: list, t0: float, mb: int, direction: str, step: float = 1.0):
    """Emit slot_begin, one compute event, slot_end for one microbatch."""
    meta = {"microbatch_id": mb, "direction": direction}
    return [
        TraceEvent(type="slot_begin", timestamp_ms=t0, metadata=dict(meta)),
        TraceEvent(type="compute", timestamp_ms=t0 + step, metadata=dict(meta)),
        TraceEvent(type="slot_end", timestamp_ms=t0 + 2 * step, metadata=dict(meta)),
    ]


def _make_trace(order: list[tuple[int, str]], stage: int = 0, pp: int = 1) -> TraceFile:
    """Build a trace from an explicit (microbatch_id, direction) execution order."""
    events: list = []
    t = 0.0
    for mb, direction in order:
        events.extend(_slot(events, t, mb, direction))
        t += 3.0
    return TraceFile(
        trace_format_version=1,
        rank=0,
        world_size=pp,
        pipeline_stage=stage,
        events=events,
        total_flops=1000,
        energy_kwh=None,
        co2eq_kg=None,
    )


def _schedule(trace: TraceFile) -> list[tuple[int, str]]:
    """Recover the (microbatch_id, direction) execution order from a trace."""
    out = []
    for kind, payload in _split_trace_items(sorted(trace.events, key=lambda e: e.timestamp_ms)):
        if kind != "slot":
            continue
        d = _slot_direction(payload[0])
        if d in ("step", ""):
            continue
        out.append((payload[0].metadata["microbatch_id"], d))
    return out


def _pp1_order(n: int) -> list[tuple[int, str]]:
    return [(i, d) for i in range(n) for d in ("fwd", "bwd")]


def _1f1b_order(n: int, pp: int, stage: int) -> list[tuple[int, str]]:
    """Canonical 1F1B order for a stage: warmup fwds, alternating steady, cooldown bwds."""
    warmup = min(pp - 1 - stage, n)
    order = [(i, "fwd") for i in range(warmup)]
    f, b = warmup, 0
    while f < n:
        order.append((f, "fwd"))
        order.append((b, "bwd"))
        f += 1
        b += 1
    order.extend((i, "bwd") for i in range(b, n))
    return order


class TestReplicationPP1:
    def test_reaches_target_microbatch_count(self):
        tr = _replicate_trace_microbatches(_make_trace(_pp1_order(4)), 16)
        fwd = [s for s in _schedule(tr) if s[1] == "fwd"]
        bwd = [s for s in _schedule(tr) if s[1] == "bwd"]
        assert len(fwd) == 16
        assert len(bwd) == 16

    def test_microbatch_ids_are_contiguous_per_direction(self):
        tr = _replicate_trace_microbatches(_make_trace(_pp1_order(4)), 16)
        sched = _schedule(tr)
        assert [m for m, d in sched if d == "fwd"] == list(range(16))
        assert [m for m, d in sched if d == "bwd"] == list(range(16))

    def test_schedule_matches_canonical_pp1(self):
        tr = _replicate_trace_microbatches(_make_trace(_pp1_order(4)), 16)
        assert _schedule(tr) == _pp1_order(16)

    def test_timestamps_strictly_ordered(self):
        tr = _replicate_trace_microbatches(_make_trace(_pp1_order(4)), 32)
        ts = [e.timestamp_ms for e in tr.events]
        assert ts == sorted(ts)

    def test_per_microbatch_span_preserved(self):
        """Replication must not stretch or compress per-microbatch cost."""
        base = _make_trace(_pp1_order(4))
        rep = _replicate_trace_microbatches(base, 16)
        def span(t):
            ts = [e.timestamp_ms for e in t.events]
            return max(ts) - min(ts)
        # 4 mb over span S ⇒ 16 mb should span ~4S (within one unit).
        assert abs(span(rep) / span(base) - 4.0) < 0.5

    def test_flops_scale_with_microbatch_count(self):
        rep = _replicate_trace_microbatches(_make_trace(_pp1_order(4)), 16)
        assert rep.total_flops == 4000  # 1000 * 16/4


class TestReplicationPP4:
    """PP>1 is where replication can go subtly wrong: warmup depth and cooldown
    drain set the pipeline bubble, so they must survive untouched."""

    @staticmethod
    def _leading(sched, direction):
        n = 0
        for _, d in sched:
            if d != direction:
                break
            n += 1
        return n

    def test_warmup_depth_preserved(self):
        """Deeper stages fill less; replication must not change the fill depth."""
        for stage in range(4):
            base = _make_trace(_1f1b_order(8, pp=4, stage=stage), stage=stage, pp=4)
            rep = _replicate_trace_microbatches(base, 32)
            expected = self._leading(_1f1b_order(32, pp=4, stage=stage), "fwd")
            assert self._leading(_schedule(rep), "fwd") == expected, f"stage {stage}"
            # and it must genuinely shrink with stage depth, not be constant
            assert expected == 4 - stage, f"stage {stage}"

    def test_schedule_matches_canonical_1f1b(self):
        for stage in range(4):
            base = _make_trace(_1f1b_order(8, pp=4, stage=stage), stage=stage, pp=4)
            rep = _replicate_trace_microbatches(base, 32)
            assert _schedule(rep) == _1f1b_order(32, pp=4, stage=stage), f"stage {stage}"

    def test_cooldown_drain_preserved(self):
        """The trailing backward drain is the other half of the pipeline bubble."""
        for stage in range(4):
            base = _make_trace(_1f1b_order(8, pp=4, stage=stage), stage=stage, pp=4)
            rep = _replicate_trace_microbatches(base, 32)
            expected = self._leading(
                list(reversed(_1f1b_order(32, pp=4, stage=stage))), "bwd"
            )
            got = self._leading(list(reversed(_schedule(rep))), "bwd")
            assert got == expected, f"stage {stage}"


class TestReplicationNoOp:
    def test_target_equal_is_noop(self):
        base = _make_trace(_pp1_order(8))
        assert _replicate_trace_microbatches(base, 8) is base

    def test_target_smaller_is_noop(self):
        """Shrinking is the trimming path's job, not replication's."""
        base = _make_trace(_pp1_order(8))
        assert _replicate_trace_microbatches(base, 4) is base


class TestPP1BubbleInvariant:
    """The pipeline bubble is fill+drain — it must not grow with microbatch count.

    This is the invariant the pp>1 path currently violates (see the guard in
    trace_tracer). Kept here as the regression target for the fix: once PP>1
    replication is correct, the guard is removed and this is extended to pp>1.
    """

    def test_replication_adds_no_idle_between_microbatches(self):
        """Replicated slots must abut exactly as the originals did (pp=1)."""
        base = _make_trace(_pp1_order(4))
        rep = _replicate_trace_microbatches(base, 16)
        items = _split_trace_items(sorted(rep.events, key=lambda e: e.timestamp_ms))
        starts = [p[0].timestamp_ms for k, p in items if k == "slot"]
        gaps = [round(b - a, 6) for a, b in zip(starts, starts[1:])]
        assert len(set(gaps)) == 1, f"uneven slot spacing after replication: {set(gaps)}"


def _pp_event(t: float, ct: str, mb: int, direction: str) -> TraceEvent:
    return TraceEvent(
        type="collective",
        timestamp_ms=t,
        metadata={"collective_type": ct, "microbatch_id": mb, "direction": direction,
                  "bytes": 1024, "group_ranks": [0, 1], "name": ct},
    )


def _make_pp2_stage0_trace(n: int) -> TraceFile:
    """Reproduce the real stage-0 pp=2 layout: PP_Send trails its fwd slot,
    PP_Recv *leads* its bwd slot (verified against a measured qwen3-32b trace)."""
    events: list = []
    t = 0.0
    order = _1f1b_order(n, pp=2, stage=0)
    for mb, direction in order:
        events.extend(_slot(events, t, mb, direction))
        t += 2.0
        if direction == "fwd":
            events.append(_pp_event(t, "PP_Send", mb, "fwd"))
            t += 0.5
            if mb >= 1:  # the matching PP_Recv leads the next bwd slot
                events.append(_pp_event(t, "PP_Recv", mb - 1, "bwd"))
                t += 0.5
        t += 0.5
    return TraceFile(trace_format_version=1, rank=0, world_size=2, pipeline_stage=0,
                     events=events, total_flops=1000, energy_kwh=None, co2eq_kg=None)


class TestPPEventAttribution:
    """A PP_Recv leads its backward slot; binding it to the last-seen slot instead
    shifts it a microbatch early and serialises the pipeline."""

    @staticmethod
    def _pp_pairs(trace):
        out = []
        for e in sorted(trace.events, key=lambda x: x.timestamp_ms):
            ct = e.metadata.get("collective_type")
            if ct in ("PP_Send", "PP_Recv"):
                out.append((ct, e.metadata["microbatch_id"], e.metadata["direction"]))
        return out

    def test_recv_binds_to_following_bwd_slot(self):
        """After replication each PP_Recv must still name the bwd slot that follows it."""
        rep = _replicate_trace_microbatches(_make_pp2_stage0_trace(6), 24)
        items = _split_trace_items(sorted(rep.events, key=lambda e: e.timestamp_ms))
        pending = None
        for kind, payload in items:
            if kind == "event":
                m = payload.metadata
                if m.get("collective_type") == "PP_Recv":
                    pending = m["microbatch_id"]
            elif _slot_direction(payload[0]).startswith("b") and pending is not None:
                assert payload[0].metadata["microbatch_id"] == pending, (
                    f"PP_Recv(mb={pending}) does not match the bwd slot it precedes "
                    f"(mb={payload[0].metadata['microbatch_id']})"
                )
                pending = None

    def test_send_binds_to_preceding_fwd_slot(self):
        rep = _replicate_trace_microbatches(_make_pp2_stage0_trace(6), 24)
        items = _split_trace_items(sorted(rep.events, key=lambda e: e.timestamp_ms))
        last_fwd = None
        for kind, payload in items:
            if kind == "slot" and _slot_direction(payload[0]).startswith("f"):
                last_fwd = payload[0].metadata["microbatch_id"]
            elif kind == "event" and payload.metadata.get("collective_type") == "PP_Send":
                assert payload.metadata["microbatch_id"] == last_fwd

    def test_pp_ids_stay_in_range(self):
        rep = _replicate_trace_microbatches(_make_pp2_stage0_trace(6), 24)
        for ct, mb, _d in self._pp_pairs(rep):
            assert 0 <= mb < 24, f"{ct} has out-of-range microbatch_id {mb}"

    def test_no_duplicate_recv_ids(self):
        """A duplicated PP_Recv id means two bwd slots wait on the same transfer."""
        rep = _replicate_trace_microbatches(_make_pp2_stage0_trace(6), 24)
        recvs = [mb for ct, mb, _ in self._pp_pairs(rep) if ct == "PP_Recv"]
        assert len(recvs) == len(set(recvs)), f"duplicate PP_Recv ids: {recvs}"
