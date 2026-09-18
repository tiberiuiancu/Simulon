#!/usr/bin/env python3
"""Validation gate for the PP trace derivation (simulon/backend/dag/pp_synth.py).

Covers PP=1 and PP=2 (both derivable from the build-consistent tp{tp}pp4 trace).

The derivation can't be gated against the on-disk traces (the pre-existing low-PP
trace is build-inconsistent with the freshly generated PP4 traces — see
sweep_validation notes / the lowpp-trace-oom memory).  The clean, build-consistent
ground truth is the **measured** wall-clock from the physical sweep.

This script finds measured PP=1 runs in an oellm-autoexp output dir, derives the
matching PP=1 trace from the (TP, SP)-matched PP4 source trace, simulates it, and
reports the error vs measured.  If the errors are within --threshold, the derivation
is validated and can be wired into the sweep (add the derived dirs to TRACE_REGISTRY,
or call derive_pp1_from_dir at load time).

    python experiments/cross_validation/validate_pp_derive.py \
        --oellm-dir <out>/qwen3-32b --threshold 5

Runs PP=1 and PP=2 only (the derivable, OOM-blocked configs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from megatron_run import parse_run  # noqa: E402



def _pp4_source(tp: int, sp: bool) -> Path:
    tag = f"tp{tp}pp4sp{'true' if sp else 'false'}"
    return REPO_ROOT / "templates/gpu" / f"gh200_jupiter-{tag}" / "traces" / "gbs256"


def _simulate_derived(measured) -> float:
    """Simulate the measured config through the production sweep path.

    Routes via sweep.simulate_point with DERIVE_LOW_PP enabled, so PP derivation,
    selective-recompute (the OOM-avoiding PP1 anchors run recompute=true) and
    mbs-extrapolation are all applied exactly as the sweep would.
    """
    import sweep_qwen3_32b as sw

    src = _pp4_source(measured.tp, measured.sp)
    if not (src / "trace_rank_0.json").exists():
        raise FileNotFoundError(f"PP4 source trace missing: {src}")

    sw.DERIVE_LOW_PP = True
    sw.GLOBAL_BATCH_SIZE = measured.gbs or sw.GLOBAL_BATCH_SIZE
    pt = sw.SweepPoint(tp=measured.tp, pp=measured.pp, sp=measured.sp,
                       recompute=bool(measured.recompute), mbs=measured.mbs,
                       vpp=None, num_nodes=(measured.num_gpus // 4))
    res = sw.simulate_point(pt)
    if res.status != "ok":
        raise RuntimeError(res.error or res.status)
    return res.total_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oellm-dir", required=True, type=Path,
                    help="oellm-autoexp output dir holding qwen3-32b run subdirs")
    ap.add_argument("--threshold", type=float, default=5.0, help="pass band, percent")
    args = ap.parse_args()

    run_dirs = [d for d in args.oellm_dir.iterdir() if d.is_dir()]
    rows = []
    for d in sorted(run_dirs):
        try:
            m = parse_run(d)
        except Exception:
            continue
        if m.iter_time_ms is None or m.pp not in (1, 2) or m.vpp:
            continue  # derivable PP (1 or 2), no-VPP. recompute IS allowed (PP1 needs it).
        if m.tp * 4 > m.num_gpus:  # need a tp{tp}pp4 source (src_pp=4)
            continue
        try:
            sim_ms = _simulate_derived(m)
        except Exception as exc:
            rows.append((m, None, str(exc)))
            continue
        rows.append((m, sim_ms, None))

    if not rows:
        print("No PP=1 measured runs found under", args.oellm_dir)
        return 1

    print(f"\n{'tp':>2} {'pp':>2} {'sp':>5} {'mbs':>3} | {'measured':>9} {'derived':>9} {'err%':>7}  result")
    print("-" * 60)
    worst = 0.0
    for m, sim_ms, err in sorted(rows, key=lambda r: (r[0].tp, r[0].pp, r[0].sp, r[0].mbs)):
        if sim_ms is None:
            print(f"{m.tp:>2} {m.pp:>2} {str(m.sp):>5} {m.mbs:>3} |  (sim error: {err})")
            continue
        e = (sim_ms - m.iter_time_ms) / m.iter_time_ms * 100
        worst = max(worst, abs(e))
        ok = "PASS" if abs(e) <= args.threshold else "FAIL"
        print(f"{m.tp:>2} {m.pp:>2} {str(m.sp):>5} {m.mbs:>3} | "
              f"{m.iter_time_ms:>9.1f} {sim_ms:>9.1f} {e:>+6.1f}%  {ok}")

    verdict = "VALIDATED" if worst <= args.threshold else "NOT validated"
    print(f"\nworst |err| = {worst:.1f}%  (threshold {args.threshold:.1f}%)  ->  derivation {verdict}")
    return 0 if worst <= args.threshold else 2


if __name__ == "__main__":
    raise SystemExit(main())
