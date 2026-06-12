#!/usr/bin/env python3
"""Plot real vs simulated metrics for validate_e2e experiments.

Usage (from repo root):
    uv run python experiments/validate_e2e/plot.py
    uv run python experiments/validate_e2e/plot.py --output results.pdf
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from simulon.tracking import get_trackers


def _find_models(base_dir: Path) -> list[Path]:
    models = []
    for item in sorted(base_dir.iterdir()):
        if (
            item.is_dir()
            and (item / "reference.yaml").exists()
            and (item / "scenario.yaml").exists()
        ):
            models.append(item)
    return models


def _read_reference(ref_path: Path) -> dict:
    with open(ref_path) as f:
        return yaml.safe_load(f) or {}


def plot_real_vs_simulated(output: Path | None, base_dir: Path) -> None:
    models = _find_models(base_dir)
    if not models:
        print("No model sub-folders found.", file=sys.stderr)
        sys.exit(1)

    trackers = get_trackers(models[0] / "scenario.yaml")
    if not trackers:
        print("No active trackers found.", file=sys.stderr)
        sys.exit(1)

    records: list[dict[str, float | str]] = []

    for model_dir in models:
        model = model_dir.name
        ref = _read_reference(model_dir / "reference.yaml")

        sim_metrics = None
        for tracker in trackers:
            sim_metrics = tracker.pull_metrics(run_name_prefix=f"validate-e2e-{model}")
            if sim_metrics is not None:
                break

        metrics = [
            ("iter_time_ms", "Iteration Time (ms)"),
            ("throughput_tps", "Throughput (t/s)"),
            ("per_gpu_tps", "Per-GPU Throughput (t/s)"),
            ("per_gpu_tflops", "Per-GPU TFLOPs/s"),
            ("mfu_pct", "MFU (%)"),
        ]

        for key, label in metrics:
            real_val = ref.get(key)
            sim_val = sim_metrics.get(key) if sim_metrics else None

            if real_val is not None:
                records.append(
                    {"model": model, "metric": label, "source": "Real", "value": float(real_val)}
                )
            if sim_val is not None:
                records.append(
                    {
                        "model": model,
                        "metric": label,
                        "source": "Simulated",
                        "value": float(sim_val),
                    }
                )

    if not records:
        print("No data found for any model. Exiting.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    sns.set_theme(style="whitegrid")

    metric_labels = df["metric"].unique()
    n_metrics = len(metric_labels)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics + 1, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric_label in zip(axes, metric_labels, strict=False):
        sub = df[df["metric"] == metric_label]
        sns.barplot(
            data=sub,
            x="model",
            y="value",
            hue="source",
            hue_order=["Real", "Simulated"],
            palette={"Real": "#4c72b0", "Simulated": "#dd8452"},
            ax=ax,
        )
        ax.set_ylabel(metric_label)
        ax.set_xlabel("")
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.legend(title="", loc="upper right")
        sns.despine(ax=ax, top=True, right=True)

    fig.suptitle("Real vs Simulated — validate_e2e", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {output}")
    else:
        plt.show()


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
        help="Directory containing model sub-folders with reference and scenario YAMLs.",
    )
    args = parser.parse_args()
    plot_real_vs_simulated(args.output, args.base_dir)


if __name__ == "__main__":
    main()
