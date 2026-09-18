#!/usr/bin/env python3
"""Refounded Qwen3-1.7B scorecard: kernel-fraction model, f calibrated from ONE anchor.

The kernel-time work ([[fakepg-overhead-vs-microbatch]]) established:
  iteration = kernel_compute*(1 + f) + comm + bubble
where kernel_compute is pure GPU device time (KT trace, efficiency-correct, profiler-
independent), comm is busbw-modeled, bubble is the pipeline schedule, and f is the launch-
bound overhead FRACTION of compute. Step-1 test: f is NOT per-launch (decisively rejected);
it is a per-model measured constant ~0.48, validated across the TP and MBS axes to 13%.

HONEST STRUCTURE (this is the whole point): f is calibrated from a SINGLE physical anchor
(tp1pp1-mbs1: f = (phys - kernel - comm - bubble)/kernel). Every other cell is then PREDICTED
from its KT trace + that one f, with NO further physical input. So the model needs one short
profiled run per model, then extrapolates the sweep. This scorecard measures how well that holds.

    PYTHONPATH=. python experiments/analyze_1p7b_refounded.py
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


def _pick(*cands: str) -> Path:
    """Prefer the single-session UNIFIED traces; fall back to the legacy multi-session set.

    The legacy set was captured across >=4 jobs and mixes a ~9% cross-session thermal/clock
    drift into every span-minus-kernel decomposition — the same size as the signal. The
    unified capture (trace_1p7b_unified.sbatch) removes that confound, so use it when present.
    """
    for c in cands:
        p = REPO / c
        if p.is_dir() and any(p.glob("*/trace_rank_0.json")):
            return p
    return REPO / cands[-1]


SPAN = _pick("templates/gpu/gh200_jupiter-1p7b-unified/traces",
             "templates/gpu/gh200_jupiter-1p7b/traces")
KT = _pick("templates/gpu/gh200_jupiter-1p7b-unified-kt/traces",
           "templates/gpu/gh200_jupiter-1p7b-kerneltime/traces")
PHYS = Path("/e/project1/e-sta-openeurollm/vanosch1/oellm-autoexp/output/qwen3-1p7b")
WARMUP = 20
CANON = yaml.safe_load((REPO / "experiments/thomas_1p7b_workloads/tp1pp1-mbs1.yaml").read_text())["config"]
ANCHOR = "tp1pp1-mbs1"

# cell -> (phys dir, tp, pp, mbs, axis)  [KT-under-captured mbs4/mbs8 flagged, not used for f]
CELLS = {
    "tp1pp1-mbs1": ("v40_qwen3_1p7b_1_pp1_tp1_mbs1_rcFalse_vppNone_spFalse_gbs256", 1, 1, 1, "anchor"),
    "tp1pp1-mbs2": ("v40_qwen3_1p7b_1_pp1_tp1_mbs2_rcFalse_vppNone_spFalse_gbs256", 1, 1, 2, "mbs"),
    "tp2pp1-mbs1": ("v40_qwen3_1p7b_1_pp1_tp2_mbs1_rcFalse_vppNone_spFalse_gbs256", 2, 1, 1, "TP"),
    "tp4pp1-mbs1": ("v40_qwen3_1p7b_1_pp1_tp4_mbs1_rcFalse_vppNone_spFalse_gbs256", 4, 1, 1, "TP"),
    "tp1pp2-mbs1": ("v40_qwen3_1p7b_1_pp2_tp1_mbs1_rcFalse_vppNone_spFalse_gbs256", 1, 2, 1, "PP"),
    "tp1pp4-mbs1": ("v40_qwen3_1p7b_1_pp4_tp1_mbs1_rcFalse_vppNone_spFalse_gbs256", 1, 4, 1, "PP"),
    "tp1pp1-mbs4": ("v40_qwen3_1p7b_1_pp1_tp1_mbs4_rcFalse_vppNone_spFalse_gbs256", 1, 1, 4, "mbs*"),
    "tp2pp1-mbs8": ("v40_qwen3_1p7b_1_pp1_tp2_mbs8_rcFalse_vppNone_spFalse_gbs256", 2, 1, 8, "mbs*"),
}


def measure(d: str):
    for f in reversed(sorted(glob.glob(str(PHYS / d / "slurm-*.log")))):
        v = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\): ([0-9.]+)",
                                          open(f, errors="ignore").read())]
        if len(v) >= 40:
            return statistics.mean(v[WARMUP:40])
    return None


def replay(reg: Path, cell: str, tp: int, pp: int, mbs: int):
    if not (reg / cell / "trace_rank_0.json").exists():
        return None
    cfg = dict(CANON)
    cfg.update({"tensor-model-parallel-size": tp, "pipeline-model-parallel-size": pp,
                "micro-batch-size": mbs, "num-gpus": 4})
    cfg.setdefault("global-batch-size", 256)
    dc = DatacenterConfig.model_validate({"num_nodes": 1, "node": NODE})
    wl = MegatronWorkload.model_validate({"framework": "megatron", "config": cfg,
                                          "traces_dir": str(reg / cell)})
    try:
        _, r = run_simulation(ScenarioConfig(datacenter=dc, workload=wl))
        return r
    except Exception:  # partial/mid-write trace (KT job still running) -> treat as missing
        return None


def flag(d):
    a = abs(d)
    return "GOOD" if a <= 5 else ("ACCEPTABLE" if a <= 10 else "FAIL")


def kt_bwd_is_broken(cell: str) -> bool:
    """True if this KT trace books zero kernel time on backward slots.

    Pre-fix traces attribute every kernel to the forward slots (autograd links backward
    kernels to the forward op, and record_function scopes are thread-local). Totals stay
    ~right, so PP=1 numbers survive, but backward compute replays as ZERO — which makes
    every bubble/schedule-dependent result meaningless. Flag it rather than report it.
    """
    import json
    p = KT / cell / "trace_rank_0.json"
    if not p.exists():
        return False
    bwd = sum(e["metadata"]["kernel_device_ms"]
              for e in json.loads(p.read_text()).get("events", [])
              if e.get("type") == "slot_begin"
              and "kernel_device_ms" in (e.get("metadata") or {})
              and (e["metadata"].get("direction") == "bwd"))
    return bwd <= 0.0


def main() -> None:
    # 1) calibrate f from the single physical anchor
    ap, atp, app_, ambs, _ = CELLS[ANCHOR]
    am = measure(ap)
    aspan = replay(SPAN, ANCHOR, atp, app_, ambs)
    akt = replay(KT, ANCHOR, atp, app_, ambs)
    if None in (am, aspan, akt):
        print("anchor data missing — cannot calibrate f"); return
    a_comm, a_bub, a_kern = aspan.exposed_comm_ms, aspan.bubble_ms, akt.compute_ms
    f = (am - a_kern - a_comm - a_bub) / a_kern
    unified = "unified" in SPAN.as_posix() and "unified" in KT.as_posix()
    print(f"TRACES span={SPAN.parent.name}  kernel={KT.parent.name}"
          f"{'' if unified else '   [LEGACY multi-session: ~9% cross-session drift confound]'}")
    print(f"CALIBRATION anchor={ANCHOR}: phys={am:.0f} kernel={a_kern:.0f} comm={a_comm:.0f} "
          f"bubble={a_bub:.0f}  ->  f = {f:.3f}")
    print(f"  (f = launch-bound overhead fraction of compute; one physical run pins it)\n")

    # 2) predict every cell from its KT trace + comm/bubble + f; compare to physical
    broken = kt_bwd_is_broken(ANCHOR)
    if broken:
        print("!! KT traces have ZERO backward kernel time (pre-fix attribution bug).")
        print("!! Totals are ~right so PP=1 rows are meaningful, but every PP>1 row below")
        print("!! is an ARTIFACT — backward replays as zero, so the bubble is fiction.")
        print("!! Re-capture with trace_1p7b_unified.sbatch before trusting the PP axis.\n")
    print(f"REFOUNDED: iter = kernel*(1+{f:.3f}) + comm + bubble    [* = KT under-captured]")
    print(f"{'cell':13}{'axis':7}{'phys':>7}{'refnd':>7}{'Δ':>7}  {'f_cell':>7}  verdict")
    fcells = []
    for cell, (pd, tp, pp, mbs, axis) in CELLS.items():
        m = measure(pd); span = replay(SPAN, cell, tp, pp, mbs); kt = replay(KT, cell, tp, pp, mbs)
        if None in (m, span, kt):
            print(f"{cell:13}{axis:7}{(f'{m:.0f}' if m else '-'):>7}  (KT {'ok' if kt else 'MISSING'})")
            continue
        comm, bub, kern = span.exposed_comm_ms, span.bubble_ms, kt.compute_ms
        refnd = kern * (1 + f) + comm + bub
        d = (refnd / m - 1) * 100
        f_cell = (m - kern - comm - bub) / kern  # this cell's own implied f
        # With the pre-fix traces, PP>1 rows are schedule-dependent => artifact, not physics.
        suspect = broken and pp > 1
        if axis in ("anchor", "mbs", "TP", "PP") and not suspect:
            fcells.append((cell, f_cell))
        note = "  <-- ARTIFACT (bwd=0)" if suspect else ""
        print(f"{cell:13}{axis:7}{m:>7.0f}{refnd:>7.0f}{d:>+6.1f}%  {f_cell:>7.3f}  "
              f"{flag(d)}{note}")

    # 3) does f hold across axes? (the all-axes confirmation)
    print()
    fs = [x for _, x in fcells]
    if len(fs) >= 2:
        print(f"per-cell f across clean axis cells: "
              f"{{{', '.join(f'{c.split(chr(45))[0]}{c.split(chr(45))[1]}={v:.2f}' for c, v in fcells)}}}")
        print(f"  f mean={statistics.mean(fs):.3f}  spread={(max(fs)/min(fs)-1)*100:.0f}%  "
              f"-> {'f is axis-INVARIANT (refounding validated all-axes)' if (max(fs)/min(fs)-1) < 0.25 else 'f DRIFTS across an axis (scope it)'}")


if __name__ == "__main__":
    main()
