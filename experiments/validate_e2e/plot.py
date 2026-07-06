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


_PLOT_ORDER = [
    ("gptoss-bf16", "GPT-OSS 20B\n(EP8 PP2)"),
    ("gptoss-bf16-pp4-ep4", "GPT-OSS 20B\n(EP4 PP4)"),
    ("gptoss-bf16", "GPT-OSS 20B\n(100% BW)"),
    ("gptoss-bf16-3nic", "GPT-OSS 20B\n(75% BW)"),
    ("gptoss-bf16-2nic", "GPT-OSS 20B\n(50% BW)"),
    ("gptoss-bf16-1nic", "GPT-OSS 20B\n(25% BW)"),
    ("gptoss-bf16", "GPT-OSS 20B\n(BF16)"),
    ("gptoss-fp8", "GPT-OSS 20B\n(FP8)"),
    ("qwen3-32b", "Qwen3-32B\n(TP4 PP1)"),
    ("qwen3-32b-tp4-pp2-mbs2-vpp8", "Qwen3-32B\n(TP4 PP2 VPP8)"),
    ("qwen3-32b-tp2-pp4-mbs1-vpp1", "Qwen3-32B\n(TP2 PP4)"),
]

_DIVIDERS_AFTER_IDX = {1, 5, 7}


def _find_configs(base_dir: Path) -> list[str]:
    configs_dir = base_dir / "configs"
    search_dir = configs_dir if configs_dir.is_dir() else base_dir
    found = {
        item.name
        for item in search_dir.iterdir()
        if item.is_dir() and (item / "scenario.yaml").exists()
    }
    ordered = [model for model, _label in _PLOT_ORDER if model in found]
    extras = sorted(found - {model for model, _label in _PLOT_ORDER})
    return ordered + extras


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


_QWEN_MODELS = {"qwen3-32b", "qwen3-32b-tp4-pp2-mbs2-vpp8", "qwen3-32b-tp2-pp4-mbs1-vpp1"}


def _plot(df: pd.DataFrame, output: Path | None) -> None:
    setup_latex_style()

    df = df.copy()
    available = set(df["model"].unique())
    models = [model for model, _label in _PLOT_ORDER if model in available]
    labels = [label for model, label in _PLOT_ORDER if model in available]
    known = {model for model, _label in _PLOT_ORDER}
    extras = sorted(available - known)
    models.extend(extras)
    labels.extend(label_for_model(m) for m in extras)
    n = len(models)

    baseline_min = []
    baseline_max = []
    sim_mean = []
    sim_std = []
    pct_labels = []

    for m in models:
        bd = df[(df["model"] == m) & (df["source"] == "Baseline")]["value"].values.astype(float)
        sd = df[(df["model"] == m) & (df["source"] == "Simulated")]["value"].values.astype(float)

        b_min = float(np.min(bd)) if len(bd) else 0.0
        b_max = float(np.max(bd)) if len(bd) else 0.0
        s_mean = float(np.mean(sd)) if len(sd) else 0.0
        s_std = float(np.std(sd)) if len(sd) else 0.0

        baseline_min.append(b_min)
        baseline_max.append(b_max)
        sim_mean.append(s_mean)
        sim_std.append(s_std)

        if b_min > 0 and s_mean > 0:
            pct_labels.append(f"{(s_mean - b_min) / b_min * 100:+.1f}%")
        else:
            pct_labels.append("")

    gptoss_idx = [i for i, m in enumerate(models) if m not in _QWEN_MODELS]
    qwen_idx = [i for i, m in enumerate(models) if m in _QWEN_MODELS]

    n_gptoss = len(gptoss_idx)
    n_qwen = len(qwen_idx)

    if n_qwen == 0:
        fig, ax = plt.subplots(1, 1, figsize=(max(7.0, 1.1 * n), 3.5))
        axes = [ax]
        section_indices = [(list(range(n)), models, labels)]
    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(max(7.0, 1.1 * n_gptoss) + max(4.0, 1.1 * n_qwen), 3.5),
            gridspec_kw={"width_ratios": [max(1, n_gptoss), max(1, n_qwen)]},
        )
        axes = list(axes)
        section_indices = [
            (gptoss_idx, [models[i] for i in gptoss_idx], [labels[i] for i in gptoss_idx]),
            (qwen_idx, [models[i] for i in qwen_idx], [labels[i] for i in qwen_idx]),
        ]

    color_baseline = "#4c72b0"
    color_sim = "#dd8452"

    for section, (idxs, _sec_models, sec_labels) in enumerate(section_indices):
        ax = axes[section]
        sec_n = len(idxs)
        sec_baseline_min = [baseline_min[i] for i in idxs]
        sec_baseline_max = [baseline_max[i] for i in idxs]
        sec_sim_mean = [sim_mean[i] for i in idxs]
        sec_sim_std = [sim_std[i] for i in idxs]
        sec_pct = [pct_labels[i] for i in idxs]

        x = np.arange(sec_n)
        width = 0.35

        ax.bar(
            x - width / 2,
            sec_baseline_min,
            width,
            yerr=[np.zeros(sec_n), np.array(sec_baseline_max) - np.array(sec_baseline_min)],
            capsize=3,
            color=color_baseline,
            alpha=0.8,
            label="Baseline (min)",
            error_kw={"ecolor": "#333333", "elinewidth": 1.0},
        )
        ax.bar(
            x + width / 2,
            sec_sim_mean,
            width,
            yerr=sec_sim_std,
            capsize=3,
            color=color_sim,
            alpha=0.8,
            label="Simulated (mean)",
            error_kw={"ecolor": "#333333", "elinewidth": 1.0},
        )

        sec_y_max = (
            max(
                max(sec_baseline_max) if sec_baseline_max else 0,
                max(np.array(sec_sim_mean) + np.array(sec_sim_std)) if sec_sim_mean else 0,
            )
            if sec_n
            else 1.0
        )
        for i, label in enumerate(sec_pct):
            if label:
                ax.text(
                    x[i] + width / 2,
                    sec_y_max * 1.04,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="red",
                )

        sec_dividers = {1, 5, 7}
        for i in range(sec_n):
            if i in sec_dividers and i < sec_n - 1:
                ax.axvline(x[i] + 0.5, color="#999999", linestyle=":", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(sec_labels, fontsize=7)
        if section == 0:
            ax.set_ylabel("Iteration time (ms)")
        ax.set_ylim(0, sec_y_max * 1.15)
        if section == 0:
            ax.set_title("GPT-OSS 20B", fontsize=10, fontweight="bold")
        else:
            ax.set_title("Qwen3-32B", fontsize=10, fontweight="bold")

    axes[-1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )

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
        if "source" not in df.columns:
            print(
                f"{csv_path} is in an incompatible format (missing 'source' column).\n"
                "Delete it and re-run without --use-csv to pull fresh data from W&B.",
                file=sys.stderr,
            )
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
