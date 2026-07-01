#!/usr/bin/env python3
"""Plot real vs simulated iteration time for validate_e2e experiments.

Pulls baseline iteration-time metrics from W&B (per-iteration, iterations > 3)
and simulation total_time_ms from W&B (multiple runs), then plots violin plots
comparing the two distributions for each config.

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
from _plot_utils import label_for_model, setup_latex_style

_WARMUP_ITERS = 3


def _find_configs(base_dir: Path) -> list[str]:
    configs_dir = base_dir / "configs"
    search_dir = configs_dir if configs_dir.is_dir() else base_dir
    return sorted(
        item.name
        for item in search_dir.iterdir()
        if item.is_dir() and (item / "scenario.yaml").exists()
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


def _pull_simulated_iteration_times(config: str) -> list[float]:
    import os

    import wandb

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "simulon")
    api = wandb.Api()

    filters: dict[str, object] = {"state": "finished", "display_name": f"validate-e2e-{config}"}
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)

    values: list[float] = []
    for run in runs:
        if run.display_name != f"validate-e2e-{config}":
            continue
        summary = dict(run.summary)
        val = summary.get("total_time_ms")
        if val is not None:
            values.append(float(val))
    return values


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

    records: list[dict[str, float | str]] = []
    for config in configs:
        baseline_values = _pull_baseline_iteration_times(config)
        sim_values = _pull_simulated_iteration_times(config)

        if not baseline_values:
            print(f"Skipping {config}: no baseline iteration-time data from W&B.", file=sys.stderr)
        if not sim_values:
            print(f"Skipping {config}: no simulation total_time_ms from W&B.", file=sys.stderr)

        for v in baseline_values:
            records.append({"model": config, "source": "Baseline", "value": v})
        for v in sim_values:
            records.append({"model": config, "source": "Simulated", "value": v})

    if not records:
        print("No results from W&B. Exiting.", file=sys.stderr)
        sys.exit(1)

    return pd.DataFrame(records)


def _plot(df: pd.DataFrame, output: Path | None) -> None:
    setup_latex_style()

    df = df.copy()
    df["label"] = df["model"].map(label_for_model)

    models = list(df["model"].unique())
    labels = [label_for_model(m) for m in models]
    n = len(models)

    fig, ax = plt.subplots(figsize=(max(5.5, 1.2 * n), 3.0))

    positions_baseline = np.arange(n) * 2.0
    positions_sim = positions_baseline + 1.0

    baseline_data = [
        df[(df["model"] == m) & (df["source"] == "Baseline")]["value"].values.astype(float)
        for m in models
    ]
    sim_data = [
        df[(df["model"] == m) & (df["source"] == "Simulated")]["value"].values.astype(float)
        for m in models
    ]

    color_baseline = "#4c72b0"
    color_sim = "#dd8452"

    for i, (bd, sd) in enumerate(zip(baseline_data, sim_data, strict=False)):
        pos_b = positions_baseline[i]
        pos_s = positions_sim[i]

        if len(bd) > 0:
            parts_b = ax.violinplot(bd, positions=[pos_b], widths=0.8, showmedians=True)
            for pc in parts_b["bodies"]:
                pc.set_facecolor(color_baseline)
                pc.set_alpha(0.7)
            parts_b["cmedians"].set_color("white")
            parts_b["cmins"].set_color(color_baseline)
            parts_b["cmaxes"].set_color(color_baseline)

        if len(sd) > 0:
            parts_s = ax.violinplot(sd, positions=[pos_s], widths=0.8, showmedians=True)
            for pc in parts_s["bodies"]:
                pc.set_facecolor(color_sim)
                pc.set_alpha(0.7)
            parts_s["cmedians"].set_color("white")
            parts_s["cmins"].set_color(color_sim)
            parts_s["cmaxes"].set_color(color_sim)

        if len(bd) > 0 and len(sd) > 0:
            real_median = float(np.median(bd))
            sim_median = float(np.median(sd))
            if real_median > 0:
                pct = (sim_median - real_median) / real_median * 100
                y_top = max(float(np.max(bd)), float(np.max(sd)))
                ax.text(
                    pos_s,
                    y_top * 1.03,
                    f"{pct:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="red",
                )

    all_positions = np.concatenate([positions_baseline, positions_sim])
    ax.set_xticks(np.sort(all_positions))
    tick_labels = []
    for i in range(n):
        tick_labels.append(f"{labels[i]}\n(Base)")
        tick_labels.append(f"{labels[i]}\n(Sim)")
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("Iteration time (ms)")
    ax.set_title("E2E Validation: Iteration Time", fontsize=10, fontweight="bold")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_baseline, alpha=0.7, label="Baseline"),
        Patch(facecolor=color_sim, alpha=0.7, label="Simulated"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    ax.tick_params(axis="x", rotation=0)

    all_vals = df["value"].values.astype(float)
    y_max = float(np.max(all_vals)) if len(all_vals) else 1.0
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
