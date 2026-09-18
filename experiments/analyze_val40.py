"""Analyze the v40 validation campaign — protocol FROZEN before the data lands.

Usage (from Simulon repo root, after the v40_* jobs and trace_unified job finish):

    PYTHONPATH=. python experiments/analyze_val40.py

Protocol (pre-registered 2026-07-20, before any v40 job ran):
  * measurement = mean of ITERATIONS 21-40 of each run (train_iters=40; the first
    ~20 are warmup — 10-iteration runs never plateaued, which is what invalidated
    every earlier verdict);
  * one log file = one run; logs are NEVER concatenated across files;
  * simulation uses the production-matched traces (TRACE_REGISTRY_UNIFIED, repointed
    2026-07-22 to gh200_jupiter-prod: nemo_26.04.sif + fusions ON, job 1012359);
  * absolute check per config: |delta| <= 5% good, <= 10% acceptable, else FAIL;
  * axis-isolated check per pair (ratio of measured effects vs simulated effects):
    same thresholds. Axis ratios cancel residual common-mode bias, so they are the
    primary verdict; absolute deltas are reported alongside.
"""
from __future__ import annotations

import glob
import re
import statistics
import sys

sys.path.insert(0, "experiments")
import logging  # noqa: E402

logging.disable(logging.INFO)
import sweep_qwen3_32b as S  # noqa: E402

BASE = "/e/project1/e-sta-openeurollm/vanosch1/oellm-autoexp/output/qwen3-32b/"
WARMUP_ITERS = 20  # analysis uses iterations 21..40 only

# label -> (output dir, tp, pp, mbs, rc)
RUNS = {
    "tp2 pp4 mbs1": ("v40_qwen3_32b_16_pp4_tp2_mbs1_rcFalse_spFalse_gbs256", 2, 4, 1, False),
    "tp4 pp4 mbs1": ("v40_qwen3_32b_16_pp4_tp4_mbs1_rcFalse_spFalse_gbs256", 4, 4, 1, False),
    "tp4 pp1 mbs1": ("v40_qwen3_32b_16_pp1_tp4_mbs1_rcFalse_spFalse_gbs256", 4, 1, 1, False),
    "tp4 pp1 mbs2": ("v40_qwen3_32b_16_pp1_tp4_mbs2_rcFalse_spFalse_gbs256", 4, 1, 2, False),
    "tp4 pp2 mbs1": ("v40_qwen3_32b_16_pp2_tp4_mbs1_rcFalse_spFalse_gbs256", 4, 2, 1, False),
    "tp4 pp2 mbs2": ("v40_qwen3_32b_16_pp2_tp4_mbs2_rcFalse_spFalse_gbs256", 4, 2, 2, False),
    "tp4 pp4 mbs2": ("v40_qwen3_32b_16_pp4_tp4_mbs2_rcFalse_spFalse_gbs256", 4, 4, 2, False),
    "tp4 pp2 rc":   ("v40_qwen3_32b_16_pp2_tp4_mbs1_rcTrue_spFalse_gbs256", 4, 2, 1, True),
    "tp4 pp4 rc":   ("v40_qwen3_32b_16_pp4_tp4_mbs1_rcTrue_spFalse_gbs256", 4, 4, 1, True),
}

# VPP axis (val40 protocol, job qwen3_32b_val40_vpp). Baseline = tp4 pp4 mbs1 vpp=off
# (already in RUNS). Each entry: (dir, tp, pp, mbs, num_layers_per_virtual_pipeline_stage).
# interleave v = 64 / (pp * NLVPS). Populated once the runs land.
VPP_RUNS = {
    "tp4 pp4 mbs1 v=4": ("v40vpp_qwen3_32b_16_pp4_tp4_mbs1_nlvps4_spFalse_gbs256", 4, 4, 1, 4),
    "tp4 pp4 mbs1 v=8": ("v40vpp_qwen3_32b_16_pp4_tp4_mbs1_nlvps2_spFalse_gbs256", 4, 4, 1, 2),
}

# (axis, numerator, denominator) — the isolated single-parameter effects
PAIRS = [
    ("TP 4->2   @pp4",  "tp2 pp4 mbs1", "tp4 pp4 mbs1"),
    ("MBS 1->2  @pp1",  "tp4 pp1 mbs2", "tp4 pp1 mbs1"),
    ("MBS 1->2  @pp2",  "tp4 pp2 mbs2", "tp4 pp2 mbs1"),
    ("MBS 1->2  @pp4",  "tp4 pp4 mbs2", "tp4 pp4 mbs1"),
    ("PP 1->2   @mbs1", "tp4 pp2 mbs1", "tp4 pp1 mbs1"),
    ("PP 2->4   @mbs1", "tp4 pp4 mbs1", "tp4 pp2 mbs1"),
    ("PP 2->4   @mbs2", "tp4 pp4 mbs2", "tp4 pp2 mbs2"),
    ("RC off->on @pp2", "tp4 pp2 rc",   "tp4 pp2 mbs1"),
    ("RC off->on @pp4", "tp4 pp4 rc",   "tp4 pp4 mbs1"),
]


def measure(dirname: str) -> tuple[float | None, int, str]:
    """Mean of iterations 21..40, per the frozen protocol. Newest complete log wins."""
    logs = sorted(glob.glob(BASE + dirname + "/slurm-*.log"))
    for f in reversed(logs):
        v = [float(x) for x in re.findall(
            r"elapsed time per iteration \(ms\): ([0-9.]+)", open(f, errors="ignore").read())]
        if len(v) >= 40:
            return statistics.mean(v[WARMUP_ITERS:40]), len(v), f.rsplit("/", 1)[-1]
        if len(v) > 0:
            note = f"{f.rsplit('/', 1)[-1]}: only {len(v)} iters"
            return None, len(v), note
    return None, 0, "no logs"


def flag(d: float) -> str:
    a = abs(d)
    return "GOOD (<=5%)" if a <= 5 else ("ACCEPTABLE (<=10%)" if a <= 10 else "FAIL (>10%)")


def main() -> None:
    meas: dict[str, float] = {}
    sim: dict[str, float] = {}

    print("=" * 86)
    print("ABSOLUTE: measured (mean iters 21-40) vs simulated (production-matched traces)")
    print("=" * 86)
    print(f"{'config':14}{'measured':>10}{'sim':>10}{'delta':>9}  {'trace':32} verdict")
    for label, (d, tp, pp, mbs, rc) in RUNS.items():
        m, n, src = measure(d)
        r = S.simulate_point(S.SweepPoint(tp=tp, pp=pp, sp=False, recompute=rc,
                                          mbs=mbs, vpp=None, num_nodes=16))
        if m is None:
            print(f"{label:14}{'MISSING':>10}{'':>10}{'':>9}  ({src})")
            continue
        if r.status != "ok":
            print(f"{label:14}{m:10.0f}  sim status={r.status} {r.error}")
            continue
        meas[label], sim[label] = m, r.total_ms
        delta = (r.total_ms / m - 1) * 100
        tr = (r.traces_dir or "").rsplit("traces/", 1)[-1]
        prod = "-prod" in (r.traces_dir or "")
        print(f"{label:14}{m:10.0f}{r.total_ms:10.0f}{delta:+8.1f}%  "
              f"{tr[:30]:32} {flag(delta)}{'' if prod else '  [NON-PROD TRACE!]'}")

    print()
    print("=" * 86)
    print("AXIS-ISOLATED: single-parameter effects, measured vs simulated")
    print("=" * 86)
    print(f"{'axis':18}{'measured':>10}{'sim':>10}{'error':>9}  verdict")
    for axis, num, den in PAIRS:
        if num not in meas or den not in meas:
            print(f"{axis:18}   (missing runs)")
            continue
        me = (meas[num] / meas[den] - 1) * 100
        se = (sim[num] / sim[den] - 1) * 100
        err = se - me
        print(f"{axis:18}{me:+9.1f}%{se:+9.1f}%{err:+8.1f}pp  {flag(err)}")

    print()
    print("Cross-checks:")
    if "tp4 pp2 mbs1" in meas and "tp4 pp4 mbs1" in meas:
        print(f"  * measured PP2 vs PP4 @mbs1: {meas['tp4 pp2 mbs1']:.0f} vs "
              f"{meas['tp4 pp4 mbs1']:.0f} — at 10 iters PP2 measured SLOWER than PP4 "
              "(mechanically backwards); if that persists at steady state it is real "
              "physics the model misses, otherwise it was warmup contamination.")
    # VPP axis: interleaving effect vs the vpp=off baseline (tp4 pp4 mbs1), measured
    # and simulated. The sim model is structural (_simulate_vpp_point, no fitted const).
    print()
    print("=" * 86)
    print("VPP AXIS: interleaving vs vpp=off (baseline = tp4 pp4 mbs1)")
    print("=" * 86)
    base_m = meas.get("tp4 pp4 mbs1")
    base_s = sim.get("tp4 pp4 mbs1")
    if base_m is None or base_s is None:
        print("  baseline tp4 pp4 mbs1 missing — cannot score VPP")
    else:
        print(f"  baseline vpp=off: measured={base_m:.0f}  sim={base_s:.0f}")
        print(f"{'config':20}{'measured':>10}{'sim':>9}{'meas_eff':>10}{'sim_eff':>9}{'error':>8}  verdict")
        for label, (d, tp, pp, mbs, nlvps) in VPP_RUNS.items():
            m, n, src = measure(d)
            r = S.simulate_point(S.SweepPoint(tp=tp, pp=pp, sp=False, recompute=False,
                                              mbs=mbs, vpp=nlvps, num_nodes=16))
            if m is None:
                print(f"{label:20}{'MISSING':>10}{'':>9}{'':>10}{'':>9}{'':>8}  ({src})")
                continue
            meff = (m / base_m - 1) * 100
            seff = (r.total_ms / base_s - 1) * 100
            err = seff - meff
            print(f"{label:20}{m:>10.0f}{r.total_ms:>9.0f}{meff:>+9.1f}%{seff:>+8.1f}%"
                  f"{err:>+7.1f}pp  {flag(err)}")

    # Trace-side invariant: total compute must be PP-independent per mbs (unified
    # traces were captured back-to-back precisely to enforce this; verify it held).
    try:
        for mbs in (1, 2):
            tots = {}
            for pp in (1, 2, 4):
                key = (4, pp, False, mbs, False)
                path = S.TRACE_REGISTRY_UNIFIED.get(key)
                if path is None or not S._has_trace(path):
                    continue
                dag, _ = S.run_simulation(S.build_scenario(
                    S.SweepPoint(tp=4, pp=pp, sp=False, recompute=False, mbs=mbs,
                                 vpp=None, num_nodes=16), path))
                tots[pp] = sum(n.duration_ms for n in dag.compute_nodes)
            if len(tots) >= 2:
                vals = list(tots.values())
                spread = (max(vals) / min(vals) - 1) * 100
                print(f"  * unified-trace PP-invariance @mbs{mbs}: total compute "
                      f"{ {k: round(v) for k, v in tots.items()} } spread={spread:.1f}% "
                      f"(legacy cross-session spread was 6.4%)")
    except Exception as exc:  # keep the report usable even if traces are partial
        print(f"  * PP-invariance check skipped: {exc}")


if __name__ == "__main__":
    main()
