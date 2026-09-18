#!/usr/bin/env python3
"""Decompose an nsys SQLite export into COMPUTE / NCCL / IDLE per GPU.

This answers the one open question in the campaign. At tp2pp1-mbs1 the physical iteration is
21010 ms, while the single-GPU fake-PG trace records only 9384 ms of kernel time for
identical FLOPs. Communication is NOT the explanation — bench_overhead (job 1140405) measured
a blocking collective in a real dependency chain at exactly its nccl-tests cost, and the
per-kernel launch gap at ~0. So the missing ~9000 ms is either real compute the 1-GPU trace
cannot see, or GPU idle.

  compute >> 9384 ms/iter  -> fake-PG single-GPU tracing UNDER-REPRESENTS multi-GPU TP
                              compute; single-GPU traces cannot support TP sweeps.
  compute ~= 9384 ms/iter  -> the missing time is IDLE; the GPUs are waiting, and the DAG
                              replay could in principle model it.

WHY NOT `nsys stats --report cuda_gpu_kern_sum`: it sums kernel durations, which (a) cannot
yield idle, and (b) double-counts concurrent kernels on different streams. This walks the raw
kernel intervals and takes their UNION per device, which is true GPU busy time.

    python experiments/analyze_nsys.py <export.sqlite>
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

TRACE_KERNEL_MS = float(__import__("os").environ.get("NSYS_TRACE_KERNEL_MS", 9384.0))  # per-cell, overridable
PHYS_ITER_MS = float(__import__("os").environ.get("NSYS_PHYS_ITER_MS", 21010.0))  # per-cell, overridable


def union_ms(intervals: list[tuple[int, int]]) -> float:
    """Total wall time covered by the union of [start,end) ns intervals -> ms."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total / 1e6


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    db = Path(sys.argv[1])
    if not db.exists():
        print(f"missing: {db}")
        sys.exit(1)
    con = sqlite3.connect(str(db))
    cur = con.cursor()

    tables = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    ktab = "CUPTI_ACTIVITY_KIND_KERNEL"
    if ktab not in tables:
        print(f"no {ktab} table — was the capture window inside the run?")
        print("tables:", sorted(t for t in tables if "KERNEL" in t or "NVTX" in t)[:10])
        sys.exit(1)

    rows = cur.execute(f"""
        SELECT k.start, k.end, k.deviceId, COALESCE(s.value, '')
        FROM {ktab} k LEFT JOIN StringIds s ON s.id = k.shortName
    """).fetchall()
    if not rows:
        print("no kernels captured")
        sys.exit(1)

    by_dev: dict[int, list] = {}
    for st, en, dev, name in rows:
        by_dev.setdefault(dev, []).append((st, en, name))

    t0 = min(r[0] for r in rows)
    t1 = max(r[1] for r in rows)
    window_ms = (t1 - t0) / 1e6

    print("=" * 78)
    print(f"nsys decomposition — capture window {window_ms:.0f} ms, {len(rows)} kernels, "
          f"{len(by_dev)} GPU(s)")
    print("=" * 78)
    print(f"{'GPU':>4}{'busy':>10}{'compute':>10}{'NCCL':>9}{'idle':>9}{'idle%':>8}  "
          f"{'kernels':>8}")

    agg = {}
    for dev in sorted(by_dev):
        ivs = by_dev[dev]
        nccl = [(s, e) for s, e, n in ivs if "nccl" in n.lower()]
        comp = [(s, e) for s, e, n in ivs if "nccl" not in n.lower()]
        busy = union_ms([(s, e) for s, e, _ in ivs])
        nccl_ms = union_ms(nccl)
        comp_ms = union_ms(comp)
        idle = window_ms - busy
        agg[dev] = (busy, comp_ms, nccl_ms, idle)
        print(f"{dev:>4}{busy:>10.0f}{comp_ms:>10.0f}{nccl_ms:>9.0f}{idle:>9.0f}"
              f"{idle/window_ms*100:>7.1f}%{len(ivs):>9}")

    # Per-iteration figures. The window was sized for ~2 iterations of ~21 s; scale by the
    # measured physical iteration rather than assuming an exact count.
    print()
    d0 = sorted(agg)[0]
    busy, comp_ms, nccl_ms, idle = agg[d0]
    iters = window_ms / PHYS_ITER_MS
    print(f"window covers ~{iters:.2f} physical iterations ({PHYS_ITER_MS:.0f} ms each)")
    if iters > 0:
        print(f"  per iteration (GPU {d0}): compute {comp_ms/iters:8.0f} ms | "
              f"NCCL {nccl_ms/iters:7.0f} ms | idle {idle/iters:7.0f} ms")
        print(f"  1-GPU fake-PG trace records : {TRACE_KERNEL_MS:8.0f} ms of kernel time")
        ratio = (comp_ms / iters) / TRACE_KERNEL_MS
        print(f"  real compute / traced compute = {ratio:.2f}x")
        print()
        if ratio > 1.5:
            print("  VERDICT: the fake-PG single-GPU trace UNDER-REPRESENTS real multi-GPU TP")
            print("           compute. Single-GPU tracing cannot support TP sweeps as-is.")
        elif idle / iters > 0.25 * PHYS_ITER_MS:
            print("  VERDICT: the missing time is IDLE — the GPUs are waiting. Find what on:")
            print("           rank skew, dependency stalls, or exposed non-NCCL sync.")
        else:
            print("  VERDICT: compute and NCCL roughly account for the iteration; re-check the")
            print("           accounting — the gap may be outside the captured window.")

    # Top kernels by total time, to see what dominates.
    print("\n  top kernels by total time:")
    tot: dict[str, float] = {}
    for st, en, _dev, name in rows:
        tot[name] = tot.get(name, 0.0) + (en - st) / 1e6
    for name, ms in sorted(tot.items(), key=lambda x: -x[1])[:8]:
        tag = "NCCL" if "nccl" in name.lower() else "    "
        print(f"    {tag} {ms:9.0f} ms  {name[:64]}")


if __name__ == "__main__":
    main()
