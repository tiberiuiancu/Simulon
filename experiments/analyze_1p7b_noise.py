#!/usr/bin/env python3
"""Run-to-run noise floor for the Qwen3-1.7B grid — the resolution limit of every verdict.

Reads the repeat groups (qwen3-1p7b-rep1/2/3, from submit_1p7b_repeats.sh) and reports,
per cell, the spread across identical re-runs. That number is the smallest difference the
scorecard can legitimately claim. It also re-states the existing verdicts against it, so a
"GOOD -2.9%" that sits inside the noise band is called out as unresolved rather than good.

Also folds in the original val40 run of each cell as a 4th sample where available (it is the
same configuration; only the experiment_group tag differs).

    PYTHONPATH=. python experiments/analyze_1p7b_noise.py
"""
from __future__ import annotations

import glob
import re
import statistics
from pathlib import Path

OUT = Path("/e/project1/e-sta-openeurollm/vanosch1/oellm-autoexp/output")
WARMUP = 20
REPS = ("qwen3-1p7b-rep1", "qwen3-1p7b-rep2", "qwen3-1p7b-rep3")
BASE_GROUP = "qwen3-1p7b"

REPO = Path(__file__).resolve().parents[1]
# label -> (output-dir glob, trace-cell name used in the scorecard tables)
CELLS = {
    "tp1pp1-mbs1 (anchor)": ("*pp1_tp1_mbs1_rcFalse*", "tp1pp1-mbs1"),
    "tp4pp1-mbs1":          ("*pp1_tp4_mbs1_rcFalse*", "tp4pp1-mbs1"),
    # present only if the optional third cell was submitted
    "tp1pp1-mbs2":          ("*pp1_tp1_mbs2_rcFalse*", "tp1pp1-mbs2"),
}
# Deltas are READ from the saved scorecards, never pasted here: re-capturing traces changes
# them, and a stale literal would silently compare the new noise floor against old results.
SCORECARDS = {"span": REPO / "experiments/results_1p7b.txt",
              "refounded": REPO / "experiments/results_1p7b_refounded.txt"}


def live_deltas(cell: str) -> dict[str, float]:
    """Current delta for `cell` from each saved scorecard ({} if not generated yet)."""
    out: dict[str, float] = {}
    for model, path in SCORECARDS.items():
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            # scorecard rows start with the cell name and carry a signed percentage
            if line.strip().startswith(cell):
                m = re.search(r"([+-][0-9]+\.[0-9])%", line)
                if m:
                    out[model] = float(m.group(1))
                    break
    return out


def iters(logfile: str) -> list[float]:
    return [float(x) for x in re.findall(
        r"elapsed time per iteration \(ms\): ([0-9.]+)",
        open(logfile, errors="ignore").read())]


def cell_samples(patt: str) -> list[tuple[str, float]]:
    """One sample per COMPLETE log across all repeat groups (+ the original val40 run)."""
    out = []
    for grp in (*REPS, BASE_GROUP):
        for f in sorted(glob.glob(str(OUT / grp / patt / "slurm-*.log"))):
            v = iters(f)
            if len(v) >= 40:
                out.append((f"{grp}/{Path(f).name}", statistics.mean(v[WARMUP:40])))
    return out


def main() -> None:
    print("=" * 78)
    print("RUN-TO-RUN NOISE FLOOR (mean of iters 21-40, identical configs)")
    print("=" * 78)
    any_data = False
    noise = {}
    for label, (patt, _cell) in CELLS.items():
        s = cell_samples(patt)
        if not s:
            print(f"{label:22} no complete runs found")
            continue
        any_data = True
        vals = [v for _, v in s]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        spread = (max(vals) / min(vals) - 1) * 100 if len(vals) > 1 else 0.0
        noise[label] = spread
        print(f"\n{label}   n={len(vals)}")
        for src, v in s:
            print(f"    {v:9.0f} ms   {src}")
        print(f"    mean={mean:.0f}  sd={sd:.0f} ({sd/mean*100:.1f}%)  "
              f"peak-to-peak spread={spread:.1f}%")

    if not any_data:
        print("\nNo repeat runs yet. Submit them with:")
        print("  bash experiments/submit_1p7b_repeats.sh")
        return

    if noise:
        worst = max(noise.values())
        print("\n" + "=" * 78)
        print(f"RESOLUTION: worst observed spread = {worst:.1f}%")
        print("=" * 78)
        print("Re-reading the current verdicts against that floor:")
        for label, (_patt, cell) in CELLS.items():
            if label not in noise:
                continue
            band = max(noise[label], 0.1)
            deltas = live_deltas(cell)
            if not deltas:
                print(f"  {label:22} (no saved scorecard yet — run analyze_1p7b*.py first)")
                continue
            for model, d in deltas.items():
                verdict = "WITHIN NOISE (unresolved)" if abs(d) <= band else "resolved"
                print(f"  {label:22} {model:11} {d:+6.1f}%  vs +/-{band:.1f}% -> {verdict}")
        print("\nIf a delta sits inside the band, the scorecard cannot distinguish it from")
        print("a re-run of the same config — report it as unresolved, not as agreement.")


if __name__ == "__main__":
    main()
