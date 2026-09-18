#!/usr/bin/env python3
"""Attribute the per-sequence overhead (k_seq ~= 49 ms/seq) to a Megatron component.

The kernel-time decomposition ([[fakepg-overhead-vs-microbatch]]) showed the sim's real
per-iteration overhead = physical - kernel_compute - comm - bubble scales with SEQUENCES per
GPU (gbs/dp) at ~49 ms/seq, NOT with launches or microbatches. This script reads the
per-component Megatron timers (timing_log_level=2) from the profiling runs and finds WHICH
component (batch-generator / forward / backward / optimizer / ...) accounts for it and whether
that component scales per-sequence — turning k_seq from a measured invariant into an
attributed, principled term.

Profiling runs (aux.experiment_group=qwen3-1p7b-profile), timing_log_level=2:
  tp1pp1-mbs1 (64 seq/GPU), tp1pp1-mbs2 (64 seq/GPU), tp2pp1-mbs1 (128 seq/GPU).
A per-SEQUENCE component has the SAME value at tp1/mbs1 and tp1/mbs2 (both 64 seq/GPU) and
2x at tp2/mbs1 (128 seq/GPU). A per-MICROBATCH component would instead halve mbs1->mbs2.

    PYTHONPATH=. python experiments/analyze_1p7b_timers.py
"""
from __future__ import annotations

import glob
import re
import statistics
from pathlib import Path

PHYS = Path("/e/project1/e-sta-openeurollm/vanosch1/oellm-autoexp/output/qwen3-1p7b-profile")
WARMUP = 20

# label -> (job-dir glob, seq/GPU = gbs/dp)
RUNS = {
    "tp1pp1-mbs1 (64 seq)": ("*pp1_tp1_mbs1_*", 64),
    "tp1pp1-mbs2 (64 seq)": ("*pp1_tp1_mbs2_*", 64),
    "tp2pp1-mbs1 (128 seq)": ("*pp1_tp2_mbs1_*", 128),
}
TIMER = re.compile(r"^\s*(?:\[default\d+\]:)?\s*([a-zA-Z0-9/_-]+) \.{2,}: \(([0-9.]+), ([0-9.]+)\)")


def component_means(logfile: str) -> dict[str, float]:
    """Mean (over iters 21..40) of each per-iteration component timer's MAX-rank value."""
    series: dict[str, list[float]] = {}
    for line in open(logfile, errors="ignore"):
        m = TIMER.match(line)
        if not m:
            continue
        name, _mn, mx = m.group(1), float(m.group(2)), float(m.group(3))
        series.setdefault(name, []).append(mx)
    out = {}
    for name, vals in series.items():
        # keep only components that recur per-iteration (drop one-off setup timers)
        if len(vals) >= 30:
            out[name] = statistics.mean(vals[WARMUP:40])
    return out


def main() -> None:
    data: dict[str, dict[str, float]] = {}
    for label, (patt, _seq) in RUNS.items():
        logs = glob.glob(str(PHYS / patt / "slurm-*.log"))
        if not logs:
            print(f"{label}: NO LOG ({PHYS}/{patt})")
            continue
        # pick the MOST COMPLETE log (cancelled duplicates leave partial logs with higher
        # job ids, so newest-wins is wrong); rank by number of completed iterations.
        best = max(logs, key=lambda f: sum(
            1 for _ in re.finditer(r"forward-backward \.{2,}: \(", open(f, errors="ignore").read())))
        data[label] = component_means(best)

    if not data:
        print("No profiling logs found. Run STEP 1 in RUNBOOK_1p7b_kerneltime.md first.")
        return

    comps = sorted({c for d in data.values() for c in d}, key=lambda c: -max(
        d.get(c, 0) for d in data.values()))
    labels = list(data)
    print(f"{'component':32}" + "".join(f"{l.split(' (')[0]:>16}" for l in labels))
    for c in comps:
        row = "".join(f"{data[l].get(c, 0):>16.1f}" for l in labels)
        print(f"{c:32}{row}")

    print("\nPER-SEQUENCE TEST (a per-sequence component: ~equal at the two 64-seq runs, ~2x at 128-seq):")
    a, b, cc = labels[0], labels[1], labels[2] if len(labels) > 2 else None
    print(f"{'component':32}{'mbs1/mbs2 ratio':>18}{'tp2/tp1 ratio':>16}  verdict")
    for c in comps:
        v1, v2 = data[a].get(c, 0), data[b].get(c, 0)
        v3 = data[cc].get(c, 0) if cc else 0
        r_mbs = (v2 / v1) if v1 else 0
        r_tp = (v3 / v1) if (v1 and cc) else 0
        # per-sequence => r_mbs ~ 1.0 and r_tp ~ 2.0
        verdict = "PER-SEQUENCE" if (0.85 <= r_mbs <= 1.15 and 1.7 <= r_tp <= 2.3) else ""
        if v1 > 20 or v2 > 20:  # only show material components
            print(f"{c:32}{r_mbs:>18.2f}{r_tp:>16.2f}  {verdict}")

    print("\nTarget to attribute: real overhead ~3047 ms @ tp1pp1-mbs1 (49.3 ms/seq * 64).")
    print("The component(s) tagged PER-SEQUENCE whose tp1pp1-mbs1 value sums to ~3000 ms is k_seq.")


if __name__ == "__main__":
    main()
