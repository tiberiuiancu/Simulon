#!/usr/bin/env python3
"""Plot MFU for node-size experiments from Weights \u0026 Biases.

Usage (from repo root):
    uv run python experiments/usecase_node_size/plot.py
    uv run python experiments/usecase_node_size/plot.py --output results.pdf
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from simulon.config.resolve import resolve_workload, workload_hash
from simulon.config.scenario import ScenarioConfig
from simulon.tracking import get_trackers


def _find_scenarios(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("scenario*.yaml"))


def _model_from_scenario(path: Path) -> str:
    return path.parent.name


def _node_size_from_scenario(path: Path) -> int:
    return int(path.stem.replace("scenario", ""))


def _compute_workload_hash(scenario: Path) -> str:
    import yaml

    with open(scenario) as f:
        raw: dict = yaml.safe_load(f)
    sc = ScenarioConfig.model_validate(raw)
    if isinstance(sc.workload, Path):
        sc.workload = resolve_workload(sc.workload)
    return workload_hash(sc.workload)


def plot_mfu_from_wandb(output: Path | None, base_dir: Path) -> None:
    scenarios = _find_scenarios(base_dir)
    if not scenarios:
        print("No scenario files found.", file=sys.stderr)
        sys.exit(1)

    # get_trackers will load the cascading tracking env and instantiate trackers
    trackers = get_trackers(scenarios[0])
    if not trackers:
        print("No active trackers found.", file=sys.stderr)
        sys.exit(1)

    records: list[dict[str, float | str]] = []

    for sc_path in scenarios:
        model = _model_from_scenario(sc_path)
        node_size = _node_size_from_scenario(sc_path)
        wl_hash = _compute_workload_hash(sc_path)

        for tracker in trackers:
            metrics = tracker.pull_metrics(wl_hash)
            if metrics is None:
                continue
            mfu = metrics.get("mfu_pct")
            if mfu is None:
                continue
            records.append({
                "model": model,
                "node_size": f"{node_size} GPU/node",
                "mfu_pct": float(mfu),
            })

    if not records:
        print("No wandb data found for any scenario. Exiting.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    order = sorted(df["node_size"].unique())
    sns.barplot(
        data=df,
        x="model",
        y="mfu_pct",
        hue="node_size",
        order=sorted(df["model"].unique()),
        hue_order=order,
        palette="deep",
        ax=ax,
    )

    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("")
    ax.set_title("MFU by Node Size", fontsize=14, fontweight="bold")
    ax.legend(title="Node config", loc="upper right")
    sns.despine(fig=fig, top=True, right=True)
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
        help="Directory containing model sub-folders with scenario YAMLs.",
    )
    args = parser.parse_args()
    plot_mfu_from_wandb(args.output, args.base_dir)


if __name__ == "__main__":
    main()
