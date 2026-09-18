#!/usr/bin/env python3
"""Measure the per-collective rank-skew constant DIRECTLY from an nsys profile.

This is the calibration procedure for `NcclProfile.launch_latency_ms` /
`by_nranks[k].launch_latency_ms`. Run it once per cluster, per communicator size you
intend to sweep. It is a MACHINE property (noise, interconnect, scheduler), not a model
property, so it does not repeat per model — but see the CAVEAT at the bottom.

WHY THIS EXISTS -- and why it is not a residual fit
---------------------------------------------------
A collective cannot complete until every participant arrives. In real training the ranks
are not synchronised, so each sync point exposes the spread in arrival times. That wait
shows up as GPU idle on early ranks and as inflated NCCL kernel time on late ones (NCCL
spin-waits), which is why neither "idle" nor "NCCL time" alone reveals it.

The constant was FIRST obtained by residual — (physical iteration - modelled compute -
modelled comm) / n_collectives — which is a fit, not a measurement. This script measures
the same physical quantity directly, from kernel start timestamps only:

    skew_i = max(start_i over ranks) - min(start_i over ranks)      for collective i
    k      = mean_i(skew_i)

No compute model, no comm model, no physical-vs-simulated comparison. If this disagrees
with the calibrated value in the node template, believe THIS and re-derive.

The raw spread is an UPPER BOUND on the exposed cost: part of the wait overlaps the
collective's own transfer. Expect the template value to land somewhat below this number.

METHOD NOTES (both were needed to get a sane answer)
----------------------------------------------------
1. ANCHOR ON THE ITERATION MARKER. `ncclDevKernel_AllReduce_Sum_u32_RING_LL` fires exactly
   once per training step on every rank, so it gives a wall-clock instant that is identical
   across devices. Collectives are then paired BY INDEX from that anchor. Without it, the
   per-device capture start differs and index pairing silently compares unrelated
   collectives (observed: a nonsensical 63 ms "skew").
2. REJECT DEVICES WITH DROPPED RECORDS. CUPTI drops buffers under load; in practice one
   device reports ~half the kernels. Index pairing across a device that dropped records is
   meaningless, so groups with unequal counts are reported and skipped, not averaged in.

    python experiments/measure_skew.py <export.sqlite> [--groups 0,1 2,3]

With no --groups, every device is paired against every other in one group (i.e. the whole
node treated as a single communicator).
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MARKER = "ncclDevKernel_AllReduce_Sum_u32_RING_LL"
COLLECTIVE_MATCH = "AllReduce_Sum_bf16"  # the TP collective that dominates the count
DROP_TOLERANCE = 0.02  # >2% count mismatch within a group => dropped records


def iteration_bounds_per_device(rows) -> dict[int, tuple[int, int]]:
    """Per-device [start, end) of one iteration, from the once-per-step marker collective.

    PER DEVICE, deliberately. An earlier version collapsed markers within 1 ms into a single
    wall-clock instant and used one window for every device — but the rank skew being measured
    is itself of order 1 ms at 4 ranks, so each device's marker fell outside the collapse
    threshold and got counted as a separate iteration (symptom: "window 1 ms, 0 devices").
    Each rank runs the same collective sequence within its own iteration, so bounding every
    device by ITS OWN markers keeps index pairing valid without assuming the ranks are
    synchronised — which is precisely the assumption under test.
    """
    per_dev: dict[int, list[int]] = defaultdict(list)
    for start, dev, name in rows:
        if name == MARKER:
            per_dev[dev].append(start)
    if not per_dev:
        raise SystemExit(f"no {MARKER} in the capture — cannot anchor; widen the window")
    bounds = {}
    for dev, marks in per_dev.items():
        marks.sort()
        if len(marks) >= 2:
            bounds[dev] = (marks[0], marks[1])
        else:
            bounds[dev] = (marks[0], max(s for s, d, _n in rows if d == dev))
    return bounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("--groups", nargs="*", default=None,
                    help="comma-separated device ids per communicator, e.g. 0,1 2,3")
    args = ap.parse_args()
    if not args.sqlite.exists():
        sys.exit(f"missing: {args.sqlite}")

    con = sqlite3.connect(str(args.sqlite))
    rows = con.execute("""
        SELECT k.start, k.deviceId, COALESCE(s.value, '')
        FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k.shortName
    """).fetchall()
    if not rows:
        sys.exit("no kernels captured")

    bounds = iteration_bounds_per_device(rows)
    per: dict[int, list[int]] = defaultdict(list)
    for start, dev, name in rows:
        if COLLECTIVE_MATCH in name and dev in bounds:
            lo, hi = bounds[dev]
            if lo <= start < hi:
                per[dev].append(start)
    for dev in per:
        per[dev].sort()

    groups = ([tuple(int(x) for x in g.split(",")) for g in args.groups]
              if args.groups else [tuple(sorted(per))])

    spans = [(hi - lo) / 1e6 for lo, hi in bounds.values()]
    print(f"one marker-bounded iteration per device: "
          f"{min(spans):.0f}-{max(spans):.0f} ms, {len(per)} device(s)")
    print(f"{'communicator':>18}{'n':>8}{'mean us':>10}{'median':>9}{'p90':>9}{'p99':>10}")
    for grp in groups:
        seqs = [per[d] for d in grp if d in per]
        if len(seqs) != len(grp):
            print(f"{str(grp):>18}   device missing from capture — skipped")
            continue
        counts = [len(s) for s in seqs]
        n = min(counts)
        if n == 0 or 1 - n / max(counts) > DROP_TOLERANCE:
            print(f"{str(grp):>18}   counts={counts} unequal — CUPTI dropped records, "
                  f"index pairing invalid; skipped")
            continue
        sk = sorted((max(s[i] for s in seqs) - min(s[i] for s in seqs)) / 1e3
                    for i in range(n))
        print(f"{str(grp):>18}{n:>8}{statistics.mean(sk):>10.1f}"
              f"{statistics.median(sk):>9.1f}{sk[int(0.9 * n)]:>9.1f}{sk[int(0.99 * n)]:>10.1f}")

    print("\nUse the MEAN as the upper bound for launch_latency_ms at that communicator size.")
    print("CAVEAT: this is measured at ONE model/config. Skew plausibly depends on how much")
    print("work sits between sync points, which is a model property — re-measure before")
    print("trusting it across a large change in model size or sequence length.")


if __name__ == "__main__":
    main()
