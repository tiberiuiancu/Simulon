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

sys.path.insert(0, str(Path(__file__).parent.parent))
from _plot_utils import label_for_model, setup_latex_style


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
        else:
            print("No wandb data found; falling back to local CSV.", file=sys.stderr)
            df = _load_csv(csv_path)
            if df is None:
                print("No cached results.csv available. Exiting.", file=sys.stderr)
                sys.exit(1)

    _save_csv(df, csv_path)

    required_cols = {"model", "node_size", "mfu_pct"}
    if not required_cols.issubset(df.columns):
        print(
            f"CSV {csv_path} is missing required columns: {sorted(required_cols - set(df.columns))}",
            file=sys.stderr,
        )
        sys.exit(1)

    setup_latex_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    df["model_label"] = df["model"].map(label_for_model)
    order = sorted(str(x) for x in df["node_size"].unique())
    models = sorted(str(x) for x in df["model_label"].unique())
    sns.barplot(
        data=df,
        x="model_label",
        y="mfu_pct",
        hue="node_size",
        order=models,
        hue_order=order,
        palette="deep",
        ax=ax,
    )

    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("")
    ax.set_title("MFU by Node Size", fontweight="bold")
    ax.legend(
        title="Node config",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=max(1, len(order)),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    ax.tick_params(axis="x", rotation=0)
    sns.despine(ax=ax, top=True, right=True)

    # Add percentage-difference labels above the second (non-baseline) bar in each group.
    y_max = float(df["mfu_pct"].max())
    baseline_label = order[0] if order else None
    x_by_label = {
        label.get_text(): float(pos)
        for label, pos in zip(ax.get_xticklabels(), ax.get_xticks(), strict=False)
    }
    if baseline_label is not None and len(order) > 1:
        baseline_by_model = {
            model: float(
                df.loc[
                    (df["model_label"] == model) & (df["node_size"] == baseline_label), "mfu_pct"
                ].values[0]
            )
            for model in models
            if len(
                df.loc[
                    (df["model_label"] == model) & (df["node_size"] == baseline_label), "mfu_pct"
                ]
            )
            > 0
        }
        for model_name in models:
            x = x_by_label.get(model_name)
            baseline = baseline_by_model.get(model_name)
            if x is None or baseline is None or baseline == 0:
                continue
            for node_size_label in order[1:]:
                subset = df.loc[
                    (df["model_label"] == model_name) & (df["node_size"] == node_size_label),
                    "mfu_pct",
                ]
                if len(subset) == 0:
                    continue
                value = float(subset.values[0])
                pct = (value - baseline) / baseline * 100
                ax.text(
                    x,
                    value + 0.02 * y_max,
                    f"{pct:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="red",
                )

    fig.tight_layout(rect=(0, 0.05, 1, 1.02))

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=300)
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
