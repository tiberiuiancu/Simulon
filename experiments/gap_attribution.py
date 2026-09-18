#!/usr/bin/env python3
"""Attribute the straggler rank's inter-kernel GPU idle to what the host was doing.

WHY THIS EXISTS
---------------
Four nsys profiles of Qwen3-1.7B on 4x GH200 show the iteration time is set by one rank
whose GPU sits idle 7.6-8.6 s per iteration in gaps *between* compute kernels, while the
other ranks' idle is absorbed inside NCCL spin-wait. That is enough to know the missing
physics is host-side; it is not enough to know what the simulator should COUNT, and the
candidates need different counters out of the tracer:

  H1  PER-OP HOST COST.  Every op costs the host a roughly fixed amount; the GPU starves
      when that exceeds kernel duration. Denominator = op count.
  H2  PERIODIC BLOCKING STALLS.  Most ops are free and a minority of host calls block for a
      long time -- syncs, D2H copies, allocator calls, dataloader, GC.
      Denominator = stall count, which scales differently in mbs and TP.

PRE-REGISTERED PREDICTION (written before running; see git history)
-------------------------------------------------------------------
"The gap distribution on rank 0 of tp4pp1-mbs2 is median 4.8 us but mean 40.5 us -- a ratio
of 8.4x. A uniform per-op cost cannot produce that [...] So if >50% of total gap time sits
in gaps whose covering host call is a SYNC / MEMCPY / allocator / non-launch API, H1 IS
REFUTED."

RESULT: THE PREDICTION WAS WRONG. H2 IS REFUTED, H1 STANDS -- with one refinement that
changes which counter the tracer needs. Measured over one marker-bounded iteration:

    tp4pp1-mbs2 dev0   gaps 7605 ms   in-driver 1289 ms (16.9%)   in-framework 6316 ms
    tp2pp1-mbs1 dev0   gaps 8635 ms   in-driver 1551 ms (18.0%)   in-framework 7084 ms
    tp2pp2-mbs2 dev0   gaps 2832 ms   in-driver  594 ms (21.0%)   in-framework 2238 ms

Blocking non-launch calls (cudaStreamSynchronize, D2H memcpy, cudaFree) account for under
5% of gap time in every profile, so H2 is dead. But only ~17-21% of the idle is spent
inside the CUDA driver at all: **79-83% is host time in framework code (python / dispatcher
/ autograd) with no CUDA call in flight.** The cost is per-op, but it is a FRAMEWORK cost,
not a driver cost, so `cudaLaunchKernel` count is the wrong denominator -- the tracer must
count aten ops as well as launches, and the per-op constant must be measured from a source
that can see host-side framework time (a torch.profiler kernel-timing trace), not from a
cuda-only nsys capture, which is blind to it.

The heavy tail that motivated the prediction is real but was misread: an earlier version of
this script attributed each gap to the API call with the largest overlap and then charged
the WHOLE gap to it, which reported "~88 us per cudaLaunchKernel". The launch calls
themselves have a median duration of 11.6 us; they merely happen to sit inside much longer
framework-bound gaps. Coverage, not nearest-neighbour attribution, is the correct measure,
so this script reports both and leads with coverage.

SYMMETRY: the per-launch driver cost is the same on every rank (median 10-13 us, all
devices, all profiles). Ranks differ only in whether their GPU work HIDES the host time --
the exposed rank is whichever one collective ordering leaves waiting, which is why the
straggler is consistently TP-rank-0 of each group rather than a rank with more work. So the
simulator's host cost should be symmetric across ranks and the idle should emerge from the
schedule, not be assigned to a chosen rank.

    python experiments/gap_attribution.py <export.sqlite> [<export.sqlite> ...]
"""
from __future__ import annotations

import bisect
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

MARKER = "ncclDevKernel_AllReduce_Sum_u32_RING_LL"
GAP_CAP_NS = 200_000  # above this a gap is a collective wait, not a host stall
TOP_N = 8


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals = sorted(intervals)
    out: list[tuple[int, int]] = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            out.append((cs, ce))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    out.append((cs, ce))
    return out


def _iteration_windows(con: sqlite3.Connection) -> dict[int, tuple[int, int]]:
    """Per-device [start, end) of one iteration, bounded by its own marker collectives.

    Per device deliberately: the rank skew under study is itself of order 1 ms, so a single
    shared wall-clock window would mis-bound every rank but one.
    """
    per_dev: dict[int, list[int]] = defaultdict(list)
    for dev, start in con.execute(
        """SELECT k.deviceId, k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON s.id = k.shortName WHERE s.value = ?""",
        (MARKER,),
    ):
        per_dev[dev].append(start)
    out = {}
    for dev, marks in per_dev.items():
        marks.sort()
        if len(marks) >= 2:
            out[dev] = (marks[0], marks[1])
    return out


def _compute_kernels(con, dev, lo, hi):
    return con.execute(
        """SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON s.id = k.shortName
           WHERE k.deviceId = ? AND k.start >= ? AND k.start < ?
             AND s.value NOT LIKE '%nccl%' ORDER BY k.start""",
        (dev, lo, hi),
    ).fetchall()


def _api(con, pid, lo, hi):
    # nsys packs the pid into the high bits of both globalPid and globalTid.
    return con.execute(
        """SELECT r.start, r.end, COALESCE(s.value, '?') FROM CUPTI_ACTIVITY_KIND_RUNTIME r
           LEFT JOIN StringIds s ON s.id = r.nameId
           WHERE r.start < ? AND r.end > ? AND (r.globalTid >> 24) = (? >> 24)""",
        (hi, lo, pid),
    ).fetchall()


def analyse(path: Path) -> None:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    windows = _iteration_windows(con)
    if not windows:
        print(f"\n== {path.name}: <2 markers per device -- iteration not bounded, skipped")
        return

    print(f"\n{'='*86}\n== {path.name}")
    print(f"{'dev':>4}{'gaps ms':>9}{'inDriver':>10}{'inFramewk':>11}{'drv%':>7}"
          f"{'kernels':>10}{'mean gap':>10}{'median':>9}")
    rows = []
    for dev, (lo, hi) in sorted(windows.items()):
        pid = con.execute(
            "SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=? AND start>=? LIMIT 1",
            (dev, lo),
        ).fetchone()[0]
        ks = _compute_kernels(con, dev, lo, hi)
        api = _api(con, pid, lo, hi)
        gaps = [(ks[i - 1][1], ks[i][0]) for i in range(1, len(ks))
                if 0 < ks[i][0] - ks[i - 1][1] < GAP_CAP_NS]
        if not gaps or not api:
            continue
        merged = _merge([(s, e) for s, e, _ in api])
        total = sum(b - a for a, b in gaps)
        # Both lists are sorted, so coverage is one linear sweep -- an interval-per-gap
        # scan is O(gaps x api) and does not finish on these captures.
        drv, j = 0, 0
        for a, b in gaps:
            while j < len(merged) and merged[j][1] <= a:
                j += 1
            k = j
            while k < len(merged) and merged[k][0] < b:
                drv += min(merged[k][1], b) - max(merged[k][0], a)
                k += 1
        sizes = sorted(b - a for a, b in gaps)
        print(f"{dev:>4}{total/1e6:>9.0f}{drv/1e6:>10.0f}{(total-drv)/1e6:>11.0f}"
              f"{100*drv/total:>6.1f}%{len(ks):>10}"
              f"{statistics.mean(sizes)/1e3:>9.1f}us{sizes[len(sizes)//2]/1e3:>8.1f}us")
        rows.append((total, dev, gaps, api))

    if not rows:
        return
    # Name breakdown for the straggler only -- the rank whose idle sets the iteration.
    total, dev, gaps, api = max(rows)
    starts = sorted(api)
    st = [a[0] for a in starts]
    named: Counter[str] = Counter()
    for a, b in gaps:
        j = bisect.bisect_right(st, b)
        best, best_ov = None, 0
        k = j - 1
        while k >= 0 and starts[k][0] > a - 50_000_000:
            s, e, name = starts[k]
            ov = min(e, b) - max(s, a)
            if ov > best_ov:
                best, best_ov = name, ov
            k -= 1
        named[best or "<no CUDA API in flight>"] += best_ov

    print(f"\n   straggler = device {dev}. Which host calls the in-driver time belongs to")
    print(f"   (this is COVERED time only -- it does not charge whole gaps to a call):")
    covered = sum(named.values())
    for name, ns in named.most_common(TOP_N):
        print(f"     {name[:46]:<48}{ns/1e6:>8.0f} ms{100*ns/covered:>7.1f}%")
    blocking = sum(ns for nm, ns in named.items()
                   if any(t in nm for t in ("Synchronize", "Memcpy", "Free", "Malloc")))
    print(f"\n   VERDICT: framework (no CUDA call) {100*(total-covered)/total:.0f}% | "
          f"driver {100*covered/total:.0f}% | of which blocking-stall {100*blocking/covered:.0f}%")
    if blocking > 0.5 * covered:
        print("   -> H2: count STALLS.")
    else:
        print("   -> H1: cost is per-op. Dominated by FRAMEWORK time, so the denominator is")
        print("      the aten-op count, not the launch count; measure it from a torch.profiler")
        print("      trace, which sees host-side op time. A cuda-only capture cannot.")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyse(Path(p))
