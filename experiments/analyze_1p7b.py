#!/usr/bin/env python3
"""Qwen3-1.7B methodology scorecard: native accuracy + mbs-extrapolation-from-base.

The 1.7B grid exists to test, on real hardware, the axes that OOM at 32B (mbs>=4, TP=1).
Every cell was BOTH run physically (val40: mean iters 21-40) AND traced on one GPU. This
script scores the simulator against the physical ground truth in two modes:

  (A) NATIVE      — each cell simulated from ITS OWN trace. "Given the right trace, is the
                    trace-driven sim accurate?" This is the plumbing/comm/bubble check.
  (B) EXTRAPOLATE — the mbs axis predicted from an mbs=1 anchor at the same (tp,pp): the
                    trace loader (trace_tracer.py:_extrapolate_trace_for_mbs) scales the
                    mbs=1 trace to the requested mbs. "Can we trace mbs1 and predict mbs?"

SCOPE NOTE (important for the end goal). The trace-driven extrapolation path only spans the
MBS axis. There is NO cross-TP extrapolation (each TP needs its own trace) and PP only
derives DOWNWARD via pp_synth (pp4 -> pp2/pp1), not up. So "trace ONE config, sweep all"
is not what the code does: the minimal trace set is one mbs=1 trace per (TP, highest-PP),
then mbs extrapolated up and PP synthesised down. This script validates the mbs leg
directly; TP/PP are scored NATIVE only (their own traces).

Thresholds (frozen): |delta| <= 5% GOOD, <= 10% ACCEPTABLE, else FAIL.

    PYTHONPATH=. python experiments/analyze_1p7b.py
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
# Which trace registry to score. Overridable so the same scorecard can be pointed at the
# span registry and the kernel-timing registry without editing the script:
#   SIMULON_1P7B_REGISTRY=gh200_jupiter-1p7b-unified-kt python experiments/analyze_1p7b.py
_REGISTRY = os.environ.get("SIMULON_1P7B_REGISTRY", "gh200_jupiter-1p7b")
TRACES = REPO / f"templates/gpu/{_REGISTRY}/traces"
# Enables the host-resource term (see replayer.replay). Requires a registry whose traces
# carry host_ms, i.e. one captured after the tracer change:
#   SIMULON_1P7B_REGISTRY=gh200_jupiter-1p7b-host SIMULON_HOST_COST_US=20 python ...
# Microseconds of framework time per aten op; ~19-21 us measured across eight mbs=1 cells
# on this cluster. Unset = old model, for the side-by-side.
_HOST_COST_US = (float(os.environ["SIMULON_HOST_COST_US"])
                 if os.environ.get("SIMULON_HOST_COST_US") else None)
PHYS = Path("/e/project1/e-sta-openeurollm/vanosch1/oellm-autoexp/output/qwen3-1p7b")
WARMUP = 20  # analyse iterations 21..40
# Canonical 1.7B body config. The model dims are identical across every cell (only the
# parallel/mbs/rc knobs differ, and those we override), so one source serves all cells —
# and it sidesteps trace dirs whose workload.yaml is missing (e.g. the cancelled-job rc
# trace, which has trace_rank_0.json but no workload.yaml).
CANON_CFG = yaml.safe_load(
    (REPO / "experiments/thomas_1p7b_workloads/tp1pp1-mbs1.yaml").read_text())["config"]

# label -> (physical output dir, trace cell, tp, pp, mbs, rc)
CELLS: dict[str, tuple[str, str, int, int, int, bool]] = {
    "tp1 pp1 mbs1":    ("v40_qwen3_1p7b_1_pp1_tp1_mbs1_rcFalse_vppNone_spFalse_gbs256", "tp1pp1-mbs1", 1, 1, 1, False),
    "tp1 pp1 mbs2":    ("v40_qwen3_1p7b_1_pp1_tp1_mbs2_rcFalse_vppNone_spFalse_gbs256", "tp1pp1-mbs2", 1, 1, 2, False),
    "tp1 pp1 mbs4":    ("v40_qwen3_1p7b_1_pp1_tp1_mbs4_rcFalse_vppNone_spFalse_gbs256", "tp1pp1-mbs4", 1, 1, 4, False),
    "tp2 pp1 mbs1":    ("v40_qwen3_1p7b_1_pp1_tp2_mbs1_rcFalse_vppNone_spFalse_gbs256", "tp2pp1-mbs1", 2, 1, 1, False),
    "tp2 pp1 mbs8":    ("v40_qwen3_1p7b_1_pp1_tp2_mbs8_rcFalse_vppNone_spFalse_gbs256", "tp2pp1-mbs8", 2, 1, 8, False),
    "tp4 pp1 mbs1":    ("v40_qwen3_1p7b_1_pp1_tp4_mbs1_rcFalse_vppNone_spFalse_gbs256", "tp4pp1-mbs1", 4, 1, 1, False),
    # OUT-OF-SAMPLE test of the 4-rank skew constant: communicator size stays 4 while the
    # collective count halves (28928 -> 14464). Skipped until the physical run and the trace
    # both exist. Pre-registered prediction: 26.9-28.3 s/iteration.
    "tp4 pp1 mbs2":    ("v40_qwen3_1p7b_1_pp1_tp4_mbs2_rcFalse_vppNone_spFalse_gbs256", "tp4pp1-mbs2", 4, 1, 2, False),
    "tp1 pp2 mbs1":    ("v40_qwen3_1p7b_1_pp2_tp1_mbs1_rcFalse_vppNone_spFalse_gbs256", "tp1pp2-mbs1", 1, 2, 1, False),
    "tp1 pp4 mbs1":    ("v40_qwen3_1p7b_1_pp4_tp1_mbs1_rcFalse_vppNone_spFalse_gbs256", "tp1pp4-mbs1", 1, 4, 1, False),
    "tp1 pp1 mbs1 rc": ("v40_qwen3_1p7b_1_pp1_tp1_mbs1_rcTrue_vppNone_spFalse_gbs256",  "tp1pp1-mbs1-rc", 1, 1, 1, True),
    "tp2 pp2 mbs2":    ("v40_qwen3_1p7b_1_pp2_tp2_mbs2_rcFalse_vppNone_spFalse_gbs256", "tp2pp2-mbs2", 2, 2, 2, False),
    # --- experiments/close_rc_and_pp.sbatch; MISSING until that job lands ---
    # Second host-bound recompute cell. Discriminates a two-class per-op cost (19.0 for
    # normal ops, ~8.2 for ops replayed under no_grad) from a uniform one: the two predict
    # 22.2 s and 24.4 s respectively, a 10% separation against a ~2% noise floor at TP=2.
    "tp2 pp1 mbs1 rc": ("v40_qwen3_1p7b_1_pp1_tp2_mbs1_rcTrue_vppNone_spFalse_gbs256", "tp2pp1-mbs1-rc", 2, 1, 1, True),
    # PP=2 at dp=1, on 2 GPUs with gbs=128 so the microbatch count stays 128 -- an
    # identical per-rank program to `tp1 pp2 mbs1`, with ONLY the data-parallel dimension
    # changed. Separates "PP depth is mismodelled" from "the residual is DP-coupled": of
    # the three PP cells the two at dp=1 are acceptable and the only dp=2 one fails.
    "tp1 pp2 mbs1 dp1": ("v40_qwen3_1p7b_1_pp2_tp1_mbs1_rcFalse_vppNone_spFalse_gbs128_ngpu2", "tp1pp2-mbs1-dp1", 1, 2, 1, False),
}

# label -> config overrides for cells that deviate from "4 GPUs, gbs 256".
CELL_OVERRIDES: dict[str, dict] = {
    "tp1 pp2 mbs1 dp1": {"num-gpus": 2, "global-batch-size": 128},
}

# mbs-extrapolation legs: predict `target` from the mbs=1 `anchor` trace (same tp,pp).
EXTRAP = [
    ("mbs 1->2 @tp1pp1", "tp1 pp1 mbs2", "tp1pp1-mbs1", 2),
    ("mbs 1->4 @tp1pp1", "tp1 pp1 mbs4", "tp1pp1-mbs1", 4),
    ("mbs 1->8 @tp2pp1", "tp2 pp1 mbs8", "tp2pp1-mbs1", 8),
]

# axis-isolated NATIVE pairs (ratios cancel common-mode bias)
PAIRS = [
    ("MBS 1->2 @tp1pp1", "tp1 pp1 mbs2", "tp1 pp1 mbs1"),
    ("MBS 1->4 @tp1pp1", "tp1 pp1 mbs4", "tp1 pp1 mbs1"),
    ("TP  1->2 @pp1mbs1", "tp2 pp1 mbs1", "tp1 pp1 mbs1"),
    ("TP  2->4 @pp1mbs1", "tp4 pp1 mbs1", "tp2 pp1 mbs1"),
    ("PP  1->2 @tp1mbs1", "tp1 pp2 mbs1", "tp1 pp1 mbs1"),
    ("PP  2->4 @tp1mbs1", "tp1 pp4 mbs1", "tp1 pp2 mbs1"),
    ("RC  off->on @tp1pp1", "tp1 pp1 mbs1 rc", "tp1 pp1 mbs1"),
]


# Physical val40 measurements, kept as a SAFETY NET only -- `measure()` reads the real
# logs and this is used solely when a log cannot be found.
#
# History worth keeping: on 2026-08-21 these logs vanished from
# /e/home/jusers/vanosch1/jupiter/oellm-autoexp/output and I concluded they had been purged,
# recovering the values second-hand from the `phys ms` column of experiments/results_1p7b*.txt.
# They had in fact been MOVED to /e/project1/e-sta-openeurollm/vanosch1/oellm-autoexp, and
# PHYS now points there. Every one of the 11 recovered values then recomputed EXACTLY from
# the real logs (40 iterations each), so the fallback is verified rather than merely
# plausible -- but prefer the logs, which also carry the 2-4 repeat runs per cell that the
# noise floor needs and a bare mean cannot provide.
PHYS_FALLBACK: dict[str, float] = {
    "tp1 pp1 mbs1": 9921, "tp1 pp1 mbs2": 9542, "tp1 pp1 mbs4": 9072,
    "tp2 pp1 mbs1": 21010, "tp2 pp1 mbs8": 11124,
    "tp4 pp1 mbs1": 41449, "tp4 pp1 mbs2": 21314,
    "tp1 pp2 mbs1": 14479, "tp1 pp4 mbs1": 20106,
    "tp1 pp1 mbs1 rc": 10741, "tp2 pp2 mbs2": 16002,
}


def measure(dirname: str) -> tuple[float | None, str]:
    """Mean of iters 21..40 per the val40 protocol; newest complete log wins."""
    logs = sorted(glob.glob(str(PHYS / dirname / "slurm-*.log")))
    for f in reversed(logs):
        v = [float(x) for x in re.findall(
            r"elapsed time per iteration \(ms\): ([0-9.]+)",
            open(f, errors="ignore").read())]
        if len(v) >= 40:
            return statistics.mean(v[WARMUP:40]), Path(f).name
        if v:
            return None, f"{Path(f).name}: only {len(v)} iters"
    return None, "no logs"


def sim_ms(trace_cell: str, tp: int, pp: int, mbs: int, rc: bool,
           overrides: dict | None = None) -> tuple[float | None, str]:
    """Trace-driven iteration time. Requesting mbs != the trace's own mbs triggers the
    loader's mbs-extrapolation, so this serves both native and extrapolation modes."""
    tdir = TRACES / trace_cell
    if not (tdir / "trace_rank_0.json").exists():
        return None, "no trace"
    cfg = dict(CANON_CFG)
    cfg["tensor-model-parallel-size"] = tp
    cfg["pipeline-model-parallel-size"] = pp
    cfg["micro-batch-size"] = mbs
    cfg["recompute-activations"] = rc
    cfg.setdefault("global-batch-size", 256)
    cfg["num-gpus"] = 4
    # Applied last so a cell can deviate from the "4 GPUs, gbs 256" campaign default
    # (e.g. the dp=1 PP probe, which runs on 2 GPUs at gbs=128 to hold the microbatch
    # count fixed while removing data parallelism).
    cfg.update(overrides or {})
    try:
        node: dict | str = NODE
        if _HOST_COST_US is not None:
            # Turn on the two-resource (host + GPU) replay for this scorecard only, without
            # editing the shared node template -- the span registries must keep scoring
            # under the old model so the two can be compared side by side.
            node = {"from": NODE, "host_cost_us": _HOST_COST_US}
        dc = DatacenterConfig.model_validate({"num_nodes": 1, "node": node})
        wl = MegatronWorkload.model_validate(
            {"framework": "megatron", "config": cfg, "traces_dir": str(tdir)})
        _dag, result = run_simulation(ScenarioConfig(datacenter=dc, workload=wl))
        return result.total_time_ms, "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"error: {exc}"


def flag(d: float) -> str:
    a = abs(d)
    return "GOOD" if a <= 5 else ("ACCEPTABLE" if a <= 10 else "FAIL")


def main() -> None:
    meas: dict[str, float] = {}
    native: dict[str, float] = {}

    print("=" * 82)
    print("(A) NATIVE: physical (val40 iters 21-40) vs sim from each cell's OWN trace")
    print("=" * 82)
    print(f"{'cell':17}{'phys ms':>9}{'sim ms':>9}{'delta':>8}  verdict")
    for label, (pdir, cell, tp, pp, mbs, rc) in CELLS.items():
        m, msrc = measure(pdir)
        if m is None and label in PHYS_FALLBACK:
            m, msrc = PHYS_FALLBACK[label], "RECOVERED (raw log purged)"
        s, ssrc = sim_ms(cell, tp, pp, mbs, rc, CELL_OVERRIDES.get(label))
        if m is None:
            print(f"{label:17}{'MISSING':>9}          ({msrc})")
            continue
        if s is None:
            print(f"{label:17}{m:9.0f}{'--':>9}          (sim {ssrc})")
            continue
        meas[label], native[label] = m, s
        d = (s / m - 1) * 100
        print(f"{label:17}{m:9.0f}{s:9.0f}{d:+7.1f}%  {flag(d)}")

    print()
    print("=" * 82)
    print("(B) EXTRAPOLATE mbs from an mbs=1 anchor trace vs physical")
    print("=" * 82)
    print(f"{'leg':20}{'phys ms':>9}{'sim ms':>9}{'delta':>8}  verdict")
    for leg, target, anchor, mbs in EXTRAP:
        _p, cell, tp, pp, _m, rc = CELLS[target]
        m = meas.get(target)
        s, ssrc = sim_ms(anchor, tp, pp, mbs, rc)  # anchor = mbs1 trace -> scaled to mbs
        if m is None or s is None:
            print(f"{leg:20}{'--':>9}{'--':>9}          ({'no phys' if m is None else ssrc})")
            continue
        d = (s / m - 1) * 100
        print(f"{leg:20}{m:9.0f}{s:9.0f}{d:+7.1f}%  {flag(d)}")

    print()
    print("=" * 82)
    print("AXIS-ISOLATED (NATIVE): measured effect vs simulated effect (ratios)")
    print("=" * 82)
    print(f"{'axis':22}{'meas':>9}{'sim':>9}{'error':>9}  verdict")
    for axis, num, den in PAIRS:
        if num not in meas or den not in meas:
            print(f"{axis:22}  (missing)")
            continue
        me = (meas[num] / meas[den] - 1) * 100
        se = (native[num] / native[den] - 1) * 100
        print(f"{axis:22}{me:+8.1f}%{se:+8.1f}%{se - me:+8.1f}pp  {flag(se - me)}")


if __name__ == "__main__":
    main()
