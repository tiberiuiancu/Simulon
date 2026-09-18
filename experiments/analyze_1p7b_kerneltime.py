#!/usr/bin/env python3
"""Kernel-time decomposition for the Qwen3-1.7B mbs axis — the foundation diagnostic.

Pairs the non-profiler SPAN traces (templates/gpu/gh200_jupiter-1p7b/traces) with the
KERNEL-TIMING traces (templates/gpu/gh200_jupiter-1p7b-kerneltime/traces, from
trace_thomas_1p7b_kerneltime.sbatch). For each cell it reports:

    span_compute   = compute_ms replaying the SPAN trace (what analyze_1p7b native uses)
    kernel_compute = compute_ms replaying the KT trace (pure GPU device time, idle removed)
    overhead       = span_compute - kernel_compute      (the fake-PG CPU-launch idle)
    launches       = number of compute slots in the replay
    overhead/launch, kernel/launch, kernel/microbatch

THE HYPOTHESIS the refounding rests on: is overhead PER LAUNCH config-invariant across mbs?
  * If YES -> refound compute as kernel_compute + (global overhead/launch)*launches, a
    reference-FREE, no-fitted-constant model that should kill both the mbs=1 anchor
    inflation and the mbs-extrapolation failure at once.
  * If NO (overhead/launch varies with mbs) -> the idle gap is size-dependent and the
    refounding needs a per-op model, not a global per-launch constant. Either way this
    tells us which.

It also prints, per cell, the REFOUNDED native total using a single global overhead/launch
(mean across the mbs=1 anchors), vs physical — a first read on whether the refounding helps.

    PYTHONPATH=. python experiments/analyze_1p7b_kerneltime.py
"""
from __future__ import annotations

import glob
import os
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import logging  # noqa: E402

logging.disable(logging.INFO)

import yaml  # noqa: E402

from simulon.backend.analytical import simulate as run_simulation  # noqa: E402
from simulon.config.dc import DatacenterConfig  # noqa: E402
from simulon.config.scenario import ScenarioConfig  # noqa: E402
from simulon.config.workload import MegatronWorkload  # noqa: E402

NODE = "jupiter-gh200-4g"
SPAN = REPO / "templates/gpu/gh200_jupiter-1p7b/traces"
KT = REPO / "templates/gpu/gh200_jupiter-1p7b-kerneltime/traces"
PHYS = Path("/e/project1/e-sta-openeurollm/vanosch1/oellm-autoexp/output/qwen3-1p7b")
WARMUP = 20
CANON_CFG = yaml.safe_load(
    (REPO / "experiments/thomas_1p7b_workloads/tp1pp1-mbs1.yaml").read_text())["config"]

# cell -> (physical dir, tp, pp, mbs)
CELLS = {
    "tp1pp1-mbs1": ("v40_qwen3_1p7b_1_pp1_tp1_mbs1_rcFalse_vppNone_spFalse_gbs256", 1, 1, 1),
    "tp1pp1-mbs2": ("v40_qwen3_1p7b_1_pp1_tp1_mbs2_rcFalse_vppNone_spFalse_gbs256", 1, 1, 2),
    "tp1pp1-mbs4": ("v40_qwen3_1p7b_1_pp1_tp1_mbs4_rcFalse_vppNone_spFalse_gbs256", 1, 1, 4),
    "tp2pp1-mbs1": ("v40_qwen3_1p7b_1_pp1_tp2_mbs1_rcFalse_vppNone_spFalse_gbs256", 2, 1, 1),
    "tp2pp1-mbs8": ("v40_qwen3_1p7b_1_pp1_tp2_mbs8_rcFalse_vppNone_spFalse_gbs256", 2, 1, 8),
}


def measure(dirname: str) -> float | None:
    for f in reversed(sorted(glob.glob(str(PHYS / dirname / "slurm-*.log")))):
        v = [float(x) for x in re.findall(
            r"elapsed time per iteration \(ms\): ([0-9.]+)",
            open(f, errors="ignore").read())]
        if len(v) >= 40:
            return statistics.mean(v[WARMUP:40])
    return None


def replay(trace_dir: Path, tp: int, pp: int, mbs: int):
    """Return (total_ms, compute_ms, exposed_comm_ms, bubble_ms, n_launches) or None."""
    if not (trace_dir / "trace_rank_0.json").exists():
        return None
    cfg = dict(CANON_CFG)
    cfg["tensor-model-parallel-size"] = tp
    cfg["pipeline-model-parallel-size"] = pp
    cfg["micro-batch-size"] = mbs
    cfg["num-gpus"] = 4
    cfg.setdefault("global-batch-size", 256)
    try:
        dc = DatacenterConfig.model_validate({"num_nodes": 1, "node": NODE})
        wl = MegatronWorkload.model_validate(
            {"framework": "megatron", "config": cfg, "traces_dir": str(trace_dir)})
        dag, r = run_simulation(ScenarioConfig(datacenter=dc, workload=wl))
        return (r.total_time_ms, r.compute_ms, r.exposed_comm_ms, r.bubble_ms,
                len(dag.compute_nodes))
    except Exception as exc:  # noqa: BLE001
        print(f"    replay error {trace_dir.name}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    rows = {}
    print("=" * 100)
    print("KERNEL-TIME DECOMPOSITION (span trace vs kernel-timing trace), per cell")
    print("=" * 100)
    print(f"{'cell':14}{'phys':>8}{'span_cmp':>9}{'kern_cmp':>9}{'ovhd':>8}"
          f"{'launch':>8}{'ovhd/lau':>9}{'kern/lau':>9}{'kern/mb':>9}")
    for cell, (pdir, tp, pp, mbs) in CELLS.items():
        m = measure(pdir)
        span = replay(SPAN / cell, tp, pp, mbs)
        kt = replay(KT / cell, tp, pp, mbs)
        if span is None or kt is None:
            print(f"{cell:14}{'--':>8}  (span={'ok' if span else 'MISSING'} "
                  f"kt={'ok' if kt else 'MISSING'})")
            continue
        span_cmp, kern_cmp, launches = span[1], kt[1], span[4]
        ovhd = span_cmp - kern_cmp
        dp = 4 // (tp * pp)
        n_mb = 256 // (mbs * dp)
        opl = ovhd / launches if launches else 0.0
        kpl = kern_cmp / launches if launches else 0.0
        kpmb = kern_cmp / n_mb if n_mb else 0.0
        rows[cell] = dict(phys=m, tp=tp, pp=pp, mbs=mbs, span=span, kt=kt,
                          span_cmp=span_cmp, kern_cmp=kern_cmp, ovhd=ovhd,
                          launches=launches, opl=opl, n_mb=n_mb)
        print(f"{cell:14}{(f'{m:.0f}' if m else '-'):>8}{span_cmp:>9.0f}{kern_cmp:>9.0f}"
              f"{ovhd:>8.0f}{launches:>8d}{opl:>9.3f}{kpl:>9.3f}{kpmb:>9.1f}")

    # Config-invariance of overhead/launch across the tp1pp1 mbs sweep.
    inv = [rows[c]["opl"] for c in ("tp1pp1-mbs1", "tp1pp1-mbs2", "tp1pp1-mbs4") if c in rows]
    print()
    if len(inv) >= 2:
        spread = (max(inv) / min(inv) - 1) * 100
        print(f"overhead/launch across tp1pp1 mbs{{1,2,4}}: {[round(x,3) for x in inv]} "
              f"spread={spread:.1f}%  ->  {'CONFIG-INVARIANT (refounding viable)' if spread < 15 else 'mbs-DEPENDENT (needs per-op model)'}")

    # Refounded native total using a single global overhead/launch (mean of mbs=1 anchors).
    anchors = [rows[c]["opl"] for c in ("tp1pp1-mbs1", "tp2pp1-mbs1") if c in rows]
    if anchors and rows:
        g_opl = statistics.mean(anchors)
        print(f"\nREFOUNDED native: compute = kernel_compute + {g_opl:.3f} ms/launch * launches")
        print(f"{'cell':14}{'phys':>8}{'span Δ':>9}{'refnd Δ':>10}  (span vs refounded native error)")
        for cell, r in rows.items():
            if r["phys"] is None:
                continue
            span_total = r["span"][0]
            refnd_compute = r["kern_cmp"] + g_opl * r["launches"]
            refnd_total = span_total - r["span_cmp"] + refnd_compute
            sd = (span_total / r["phys"] - 1) * 100
            rd = (refnd_total / r["phys"] - 1) * 100
            print(f"{cell:14}{r['phys']:>8.0f}{sd:>+8.1f}%{rd:>+9.1f}%")


if __name__ == "__main__":
    main()
