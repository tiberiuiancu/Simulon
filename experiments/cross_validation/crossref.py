#!/usr/bin/env python3
"""Cross-reference oellm-autoexp measurements against Simulon simulations.

For every measured Megatron run discovered under an oellm-autoexp output
directory, this builds the *same* config in Simulon, simulates it, and prints a
side-by-side table of measured vs simulated metrics with the relative error —
the same comparison the per-scenario ``reference.yaml`` files capture, but
generated automatically and in bulk.

    python experiments/cross_validation/crossref.py \
        --oellm-dir /e/home/.../oellm-autoexp/output/qwen3-32b

    # restrict to the canonical sweep batch size and emit a markdown table
    python experiments/cross_validation/crossref.py --gbs 256 --md out/crossval.md

A Simulon row is only produced when a trace (or trace synthesis) exists for the
config; otherwise the simulated columns show ``—`` so the gap between what was
measured and what Simulon can currently reproduce is explicit.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the sweep's Simulon scenario builder, trace lookup and simulation path
# so measured runs are simulated through exactly the same code as the sweep.
from experiments.sweep_qwen3_32b import (  # noqa: E402
    GPUS_PER_NODE,
    SweepPoint,
    simulate_point,
)
from experiments.cross_validation.megatron_run import (  # noqa: E402
    MeasuredRun,
    discover_runs,
    parse_run,
)


@dataclass
class JoinedRow:
    measured: MeasuredRun
    sim_status: str                       # "ok" | "no_trace" | "error" | "skip_gbs"
    sim_iter_ms: float | None = None
    sim_per_gpu_tps: float | None = None
    sim_per_gpu_tflops: float | None = None
    sim_mfu_pct: float | None = None
    sim_note: str | None = None


def _measured_to_point(m: MeasuredRun) -> SweepPoint:
    return SweepPoint(
        tp=m.tp,
        pp=m.pp,
        sp=m.sp,
        recompute=m.recompute,
        mbs=m.mbs,
        vpp=m.vpp,
        num_nodes=(m.num_gpus // GPUS_PER_NODE),
    )


def join_run(m: MeasuredRun, sweep_gbs: int = 256) -> JoinedRow:
    """Simulate the Simulon config matching a measured run and pair the metrics."""
    if m.iter_time_ms is None or m.num_gpus is None:
        return JoinedRow(measured=m, sim_status="error", sim_note="no measured data")

    # The Simulon sweep fixes global-batch-size; only compare matching gbs.
    if m.gbs != sweep_gbs:
        return JoinedRow(measured=m, sim_status="skip_gbs",
                         sim_note=f"gbs={m.gbs}≠{sweep_gbs}")

    point = _measured_to_point(m)
    sim = simulate_point(point)
    if sim.status != "ok":
        return JoinedRow(measured=m, sim_status=sim.status, sim_note=sim.error)

    return JoinedRow(
        measured=m,
        sim_status="ok",
        sim_iter_ms=sim.total_ms,
        sim_per_gpu_tps=sim.per_gpu_tps,
        sim_per_gpu_tflops=sim.per_gpu_tflops,
        sim_mfu_pct=sim.mfu_pct,
        # ok rows may still carry a caveat (e.g. uncalibrated recompute factor).
        sim_note=sim.error,
    )


def _err(sim: float | None, meas: float | None) -> str:
    if sim is None or meas in (None, 0):
        return "—"
    e = (sim - meas) / meas * 100
    return f"{'+' if e >= 0 else ''}{e:.1f}%"


# Peak bf16 for the Jupiter GH200 (TFLOP/s); used to express measured MFU on the
# same basis Simulon uses. Kept here as a constant so the parser stays free of
# hardware assumptions.
_GH200_PEAK_TFLOPS_BF16 = 989.0


def _measured_mfu(m: MeasuredRun) -> float | None:
    if m.per_gpu_tflops is None:
        return None
    return m.per_gpu_tflops / _GH200_PEAK_TFLOPS_BF16 * 100


def print_table(rows: list[JoinedRow]) -> None:
    cols = (
        f"{'config':<40}  {'metric':<10}  {'measured':>11}  {'simulated':>11}  {'err':>7}"
    )
    sep = "─" * len(cols)
    print(f"\n{'Cross-validation — oellm-autoexp (measured) vs Simulon (simulated)':^{len(cols)}}")
    print(sep)
    print(cols)
    print(sep)

    n_ok = 0
    for row in rows:
        m = row.measured
        # Few logged iterations means the "fastest" is still warmup/compile
        # polluted — flag it so a large error isn't mistaken for a model defect.
        low = " ⚠few-iters" if m.num_iters_seen and m.num_iters_seen < 3 else ""
        label = m.label() + low
        if row.sim_status == "ok":
            n_ok += 1
            meas_mfu = _measured_mfu(m)
            metric_rows = [
                ("iter ms", m.iter_time_ms, row.sim_iter_ms, "{:,.0f}"),
                ("tps/GPU", m.per_gpu_tps, row.sim_per_gpu_tps, "{:,.0f}"),
                ("TF/s/GPU", m.per_gpu_tflops, row.sim_per_gpu_tflops, "{:,.1f}"),
                ("MFU %", meas_mfu, row.sim_mfu_pct, "{:.2f}"),
            ]
            for i, (name, meas, sim, fmt) in enumerate(metric_rows):
                head = label if i == 0 else ""
                meas_s = fmt.format(meas) if meas is not None else "—"
                sim_s = fmt.format(sim) if sim is not None else "—"
                print(f"{head:<40}  {name:<10}  {meas_s:>11}  {sim_s:>11}  {_err(sim, meas):>7}")
            if row.sim_note:
                print(f"{'  ⚠ ' + row.sim_note:<40}")
            print(sep)
        else:
            note = row.sim_note or row.sim_status
            meas_s = f"{m.iter_time_ms:,.0f}" if m.iter_time_ms else "—"
            print(f"{label:<40}  {'iter ms':<10}  {meas_s:>11}  "
                  f"{'(' + row.sim_status + ')':>11}  {note or '':>7}")
            print(sep)

    paired = [r for r in rows if r.sim_status == "ok"]
    print(f"  {len(rows)} runs, {n_ok} paired with a Simulon simulation, "
          f"{len(rows) - n_ok} unpaired\n")

    if paired:
        # Headline accuracy summary on iteration time.
        errs = [
            abs((r.sim_iter_ms - r.measured.iter_time_ms) / r.measured.iter_time_ms * 100)
            for r in paired
        ]
        print(f"  Iteration-time error (paired): mean {sum(errs) / len(errs):.1f}%  "
              f"max {max(errs):.1f}%\n")


def write_csv(rows: list[JoinedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "config", "tp", "pp", "sp", "recompute", "mbs", "gbs", "vpp", "num_gpus",
            "sim_status",
            "meas_iter_ms", "sim_iter_ms", "iter_err_pct",
            "meas_tps_gpu", "sim_tps_gpu",
            "meas_tflops_gpu", "sim_tflops_gpu",
            "meas_mfu_pct", "sim_mfu_pct",
        ])
        for r in rows:
            m = r.measured
            meas_mfu = _measured_mfu(m)
            iter_err = (
                (r.sim_iter_ms - m.iter_time_ms) / m.iter_time_ms * 100
                if r.sim_iter_ms and m.iter_time_ms else None
            )
            w.writerow([
                m.label(), m.tp, m.pp, m.sp, m.recompute, m.mbs, m.gbs, m.vpp, m.num_gpus,
                r.sim_status,
                m.iter_time_ms, r.sim_iter_ms, iter_err,
                m.per_gpu_tps, r.sim_per_gpu_tps,
                m.per_gpu_tflops, r.sim_per_gpu_tflops,
                meas_mfu, r.sim_mfu_pct,
            ])
    print(f"CSV written to {path}")


def write_markdown(rows: list[JoinedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| config | measured iter ms | sim iter ms | err | meas MFU% | sim MFU% | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        m = r.measured
        meas_mfu = _measured_mfu(m)
        meas_iter = f"{m.iter_time_ms:,.0f}" if m.iter_time_ms else "—"
        sim_iter = f"{r.sim_iter_ms:,.0f}" if r.sim_iter_ms else "—"
        err = _err(r.sim_iter_ms, m.iter_time_ms)
        mm = f"{meas_mfu:.2f}" if meas_mfu is not None else "—"
        sm = f"{r.sim_mfu_pct:.2f}" if r.sim_mfu_pct is not None else "—"
        lines.append(
            f"| {m.label()} | {meas_iter} | {sim_iter} | {err} | {mm} | {sm} | {r.sim_status} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"Markdown written to {path}")


def parse_args() -> argparse.Namespace:
    default_oellm = Path(
        os.environ.get("OELLM_OUTPUT_DIR", "")
    ) if os.environ.get("OELLM_OUTPUT_DIR") else None
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oellm-dir", type=Path, default=default_oellm, required=default_oellm is None,
                    help="oellm-autoexp output dir to discover runs under "
                         "(or set OELLM_OUTPUT_DIR)")
    ap.add_argument("--gbs", type=int, default=256,
                    help="Only cross-compare runs with this global-batch-size (default 256)")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup iters to skip (default 1)")
    ap.add_argument("--csv", type=Path, default=None, help="Write joined rows to CSV")
    ap.add_argument("--md", type=Path, default=None, help="Write a markdown table")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = discover_runs(args.oellm_dir)
    if not run_dirs:
        print(f"No runs found under {args.oellm_dir}")
        return

    rows: list[JoinedRow] = []
    for rd in run_dirs:
        m = parse_run(rd, warmup_iters=args.warmup)
        rows.append(join_run(m, sweep_gbs=args.gbs))

    # Paired rows first, sorted by iteration-time error magnitude.
    rows.sort(key=lambda r: (
        r.sim_status != "ok",
        abs((r.sim_iter_ms - r.measured.iter_time_ms) / r.measured.iter_time_ms)
        if r.sim_status == "ok" and r.measured.iter_time_ms else 0,
    ))

    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
    if args.md:
        write_markdown(rows, args.md)


if __name__ == "__main__":
    main()
