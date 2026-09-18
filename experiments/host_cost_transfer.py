#!/usr/bin/env python3
"""Is host_cost_us a machine constant, or a per-configuration fit?

`host_cost_us` (per-op host cost, simulon/backend/dag/replayer.py) is the ONE calibrated
scalar left in the compute path. Everything else -- kernel times, collective sizes, NCCL
curves, schedule -- is measured or derived from the architecture. A calibrated scalar is
only legitimate if it is a property of the MACHINE, so this solves for the value each
capture independently requires and reports the spread. If they cluster, it transfers and a
new model or cluster needs one calibration; if they scatter with the parallelism strategy,
it is a fit and every strategy would need its own -- which would make the simulator useless
for choosing between strategies, the thing it exists to do.

Each self-anchor is a trace replayed against its OWN capture run's wall time, so this needs
no cluster time at all.

    PYTHONPATH=. python experiments/host_cost_transfer.py [--registry ...] [--node ...]
"""
from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
logging.disable(logging.INFO)

from experiments.ground import (  # noqa: E402
    DEFAULT_REGISTRY,
    Plan,
    anchor_measurement,
    scan_registry,
    simulate,
)


def solve(m, plan, node: str, lo: float = 0.0, hi: float = 80.0, tol: float = 0.3) -> float | None:
    """host_cost_us at which sim == measured for this one anchor (bisection; monotone)."""
    def delta(h: float) -> float:
        return simulate(m, plan, node, h).total_time_ms / m.med_ms - 1
    d_lo, d_hi = delta(lo), delta(hi)
    if d_lo > 0 or d_hi < 0:
        return None                      # measured outside the bracket: no solution
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if delta(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", nargs="*", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--node", default="jupiter-gh200-4g")
    ap.add_argument("--model", default="qwen3-32b")
    a = ap.parse_args()

    reg = [t for t in scan_registry(a.registry) if t.model == a.model]
    print(f"{'capture':<28}{'pp':>3}{'dp':>3}{'fp8':>5}{'layout':>7}{'meas ms':>9}"
          f"{'implied host_cost_us':>22}")
    got: list[tuple[str, float, int, int]] = []
    for t in sorted(reg, key=lambda t: (t.pp, t.dp, t.fp8, t.interleaved)):
        if t.ubo:            # userbuffer kernels are booked as compute -- not a clean anchor
            continue
        m = anchor_measurement(t)
        if m is None:
            continue
        plan = Plan("self-anchor", trace=t, comm="measured")
        h = solve(m, plan, a.node)
        got.append((t.path.name, h, t.pp, t.dp)) if h is not None else None
        print(f"{t.path.name:<28}{t.pp:>3}{t.dp:>3}{str(t.fp8):>5}{str(t.interleaved):>7}"
              f"{m.med_ms:>9.0f}{(f'{h:.1f}' if h is not None else 'no solution'):>22}")
    vals = [h for _, h, _, _ in got]
    if len(vals) > 1:
        print(f"\nspread over {len(vals)} captures: {min(vals):.1f} - {max(vals):.1f} us/op "
              f"(mean {statistics.mean(vals):.1f}, sd {statistics.stdev(vals):.1f} "
              f"= {statistics.stdev(vals)/statistics.mean(vals)*100:.0f}% of mean)")
        for label, sel in (("PP", lambda r: r[2]), ("DP", lambda r: r[3])):
            by: dict[int, list[float]] = {}
            for r in got:
                by.setdefault(sel(r), []).append(r[1])
            if len(by) > 1:
                print(f"  by {label}: " + "   ".join(
                    f"{label}{k}: {statistics.mean(v):.1f}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    main()
