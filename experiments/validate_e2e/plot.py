#!/usr/bin/env python3
"""Plot real vs simulated iteration time for validate_e2e experiments.

Pulls baseline iteration-time metrics from W&B (per-iteration, iterations >= 4)
and simulation total_time_ms from W&B, then plots a grouped bar chart comparing
the two for each config.

Usage (from repo root):
    uv run python experiments/validate_e2e/plot.py
    uv run python experiments/validate_e2e/plot.py --output results.pdf
    uv run python experiments/validate_e2e/plot.py --use-csv
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from _plot_utils import label_for_model, make_figure, setup_latex_style

_WARMUP_ITERS = 3


def _find_configs(base_dir: Path) -> list[str]:
    configs_dir = base_dir / "configs"
    search_dir = configs_dir if configs_dir.is_dir() else base_dir
    return sorted(
        item.name
        for item in search_dir.iterdir()
        if item.is_dir() and (item / "scenario.yaml").exists() and "3nic" not in item.name
    )


def _load_csv(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Warning: failed to read {csv_path}: {exc}", file=sys.stderr)
        return None


def _save_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


def _pull_baseline_iteration_times(config: str) -> list[float]:
    import os

    import wandb

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "simulon")
    api = wandb.Api()

    prefix = f"validate-e2e-baseline-{config}-"
    filters: dict[str, object] = {"state": "finished", "display_name": {"$regex": f"^{prefix}"}}
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)

    values: list[float] = []
    for run in runs:
        rest = run.display_name[len(prefix) :]
        if not rest or rest == "local":
            continue
        try:
            int(rest)
        except ValueError:
            continue
        history = run.history(samples=10000)
        if "iteration-time" not in history.columns:
            continue
        for _, row in history.iterrows():
            step = row.get("_step", row.get("iteration", 0))
            if step <= _WARMUP_ITERS:
                continue
            val = row["iteration-time"]
            if pd.notna(val):
                values.append(float(val) * 1000.0)
    return values


def _pull_simulated_iteration_time(config: str) -> float | None:
    import os

    import wandb

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "simulon")
    api = wandb.Api()

    filters: dict[str, object] = {"state": "finished", "display_name": f"validate-e2e-{config}"}
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
    for run in runs:
        if run.display_name != f"validate-e2e-{config}":
            continue
        summary = dict(run.summary)
        val = summary.get("total_time_ms")
        if val is not None:
            return float(val)
    return None


def _gather_results(base_dir: Path) -> pd.DataFrame:
    configs = _find_configs(base_dir)
    if not configs:
        print("No config sub-folders found.", file=sys.stderr)
        sys.exit(1)

    from simulon.tracking.env import load_cascading_tracking_env

    first_scenario = base_dir / "configs" / configs[0] / "scenario.yaml"
    if not first_scenario.exists():
        first_scenario = base_dir / configs[0] / "scenario.yaml"
    load_cascading_tracking_env(str(first_scenario))

    rows: list[dict[str, float | str | None]] = []
    for config in configs:
        baseline_values = _pull_baseline_iteration_times(config)
        sim_ms = _pull_simulated_iteration_time(config)

        if baseline_values:
            baseline_median = float(np.median(baseline_values))
            baseline_min = float(np.min(baseline_values))
            baseline_max = float(np.max(baseline_values))
        else:
            baseline_median = None
            baseline_min = None
            baseline_max = None

        if baseline_median is None:
            print(f"Skipping {config}: no baseline iteration-time data from W&B.", file=sys.stderr)
        if sim_ms is None:
            print(f"Skipping {config}: no simulation total_time_ms from W&B.", file=sys.stderr)

        rows.append(
            {
                "model": config,
                "baseline_median_ms": baseline_median,
                "baseline_min_ms": baseline_min,
                "baseline_max_ms": baseline_max,
                "simulated_ms": sim_ms,
            }
        )

    complete = [
        r for r in rows if r["baseline_median_ms"] is not None and r["simulated_ms"] is not None
    ]
    if not complete:
        print("No complete results from W&B. Exiting.", file=sys.stderr)
        sys.exit(1)

    return pd.DataFrame(complete)


def _plot(df: pd.DataFrame, output: Path | None) -> None:
    setup_latex_style()

    df = df.copy()
    df["label"] = df["model"].map(label_for_model)

    fig, axes = make_figure("E2E Validation: Iteration Time", width_in=5.5)
    ax = axes[0] if isinstance(axes, list) else axes
    ax.set_axisbelow(True)

    x = np.arange(len(df))
    bar_width = 0.35

    baseline_vals = df["baseline_median_ms"].values.astype(float)
    baseline_lows = df["baseline_min_ms"].fillna(0).values.astype(float)
    baseline_highs = df["baseline_max_ms"].fillna(0).values.astype(float)
    sim_vals = df["simulated_ms"].values.astype(float)

    yerr = np.vstack([baseline_vals - baseline_lows, baseline_highs - baseline_vals])

    ax.bar(
        x - bar_width / 2,
        baseline_vals,
        bar_width,
        yerr=yerr,
        label="Baseline",
        color="#4c72b0",
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + bar_width / 2,
        sim_vals,
        bar_width,
        label="Simulated",
        color="#dd8452",
        edgecolor="white",
        linewidth=0.5,
    )

    y_max = max(float(np.max(baseline_highs)), float(np.max(sim_vals)))
    for i, (real, sim) in enumerate(zip(baseline_vals, sim_vals, strict=False)):
        if real > 0:
            pct = (sim - real) / real * 100
            ax.text(
                x[i] + bar_width / 2,
                sim + 0.03 * y_max,
                f"{pct:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color="red",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=8)
    ax.set_ylabel("Iteration time (ms)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    ax.set_ylim(0, y_max * 1.15)

    fig.tight_layout(rect=[0, 0, 1, 1.02])

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def plot_real_vs_simulated(output: Path | None, base_dir: Path, use_csv: bool = False) -> None:
    csv_path = base_dir / "results.csv"

    if use_csv:
        df = _load_csv(csv_path)
        if df is None:
            print(f"--use-csv requested but {csv_path} not found.", file=sys.stderr)
            sys.exit(1)
    else:
        df = _gather_results(base_dir)
        _save_csv(df, csv_path)

    _plot(df, output)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write chart to file (PDF/PNG/SVG). Omit to display.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing configs/ sub-folders with scenario YAMLs.",
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Plot from the local results.csv instead of pulling from W&B.",
    )
    args = parser.parse_args()
    plot_real_vs_simulated(args.output, args.base_dir, use_csv=args.use_csv)


if __name__ == "__main__":
    main()
