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

from simulon.tracking import get_trackers


def _find_scenarios(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("scenario*.yaml"))


def _model_from_scenario(path: Path) -> str:
    return path.parent.name


def _node_size_from_scenario(path: Path) -> int:
    return int(path.stem.replace("scenario", ""))


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


def plot_mfu_from_wandb(output: Path | None, base_dir: Path, use_csv: bool = False) -> None:
    csv_path = base_dir / "results.csv"
    records: list[dict[str, float | str]] = []

    if use_csv:
        df = _load_csv(csv_path)
        if df is None:
            print(f"--use-csv requested but {csv_path} not found.", file=sys.stderr)
            sys.exit(1)
    else:
        scenarios = _find_scenarios(base_dir)
        if not scenarios:
            print("No scenario files found.", file=sys.stderr)
            sys.exit(1)

        # get_trackers will load the cascading tracking env and instantiate trackers
        trackers = get_trackers(scenarios[0])

        for sc_path in scenarios:
            model = _model_from_scenario(sc_path)
            node_size = _node_size_from_scenario(sc_path)
            # The run script sets WANDB_RUN_NAME="node-size-{model}-node{size}"
            run_prefix = f"node-size-{model}-node{node_size}"

            mfu: float | None = None
            if trackers:
                for tracker in trackers:
                    metrics = tracker.pull_metrics(run_name_prefix=run_prefix)
                    if metrics is not None:
                        mfu = metrics.get("mfu_pct")
                        if mfu is not None:
                            break

            if mfu is None:
                print(f"No wandb data for {model} node{node_size}; skipping.", file=sys.stderr)
                continue

            records.append(
                {"model": model, "node_size": f"{node_size} GPU/node", "mfu_pct": float(mfu)}
            )

        if records:
            df = pd.DataFrame(records)
            _save_csv(df, csv_path)
        else:
            print("No wandb data found; falling back to local CSV.", file=sys.stderr)
            df = _load_csv(csv_path)
            if df is None:
                print("No cached results.csv available. Exiting.", file=sys.stderr)
                sys.exit(1)

    required_cols = {"model", "node_size", "mfu_pct"}
    if not required_cols.issubset(df.columns):
        print(
            f"CSV {csv_path} is missing required columns: {sorted(required_cols - set(df.columns))}",
            file=sys.stderr,
        )
        sys.exit(1)
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
    ax.legend(title="Node config", loc="upper left")
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
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Plot from the local results.csv instead of pulling from wandb.",
    )
    args = parser.parse_args()
    plot_mfu_from_wandb(args.output, args.base_dir, use_csv=args.use_csv)


if __name__ == "__main__":
    main()
