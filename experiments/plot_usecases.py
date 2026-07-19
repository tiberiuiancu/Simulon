#!/usr/bin/env python3
"""Combined node-size and link-bandwidth usecase plots side by side.

Usage (from repo root):
    uv run python experiments/plot_usecases.py
    uv run python experiments/plot_usecases.py --output usecases.pdf
    uv run python experiments/plot_usecases.py --line --output usecases_line.pdf
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from simulon.tracking import get_trackers

sys.path.insert(0, str(Path(__file__).parent))
from _plot_utils import label_for_model, setup_latex_style
from usecase_link_bw.plot import _load_csv as _load_link_csv
from usecase_link_bw.plot import _merge_link_bw_frames, _model_bw_from_run
from usecase_link_bw.plot import _save_csv as _save_link_csv
from usecase_node_size.plot import (
    _find_scenarios,
    _is_overlap_scenario,
    _merge_node_size_frames,
    _model_from_scenario,
    _node_size_from_scenario,
)
from usecase_node_size.plot import _load_csv as _load_node_csv
from usecase_node_size.plot import _save_csv as _save_node_csv


def _load_node_size_data(base_dir: Path, use_csv: bool) -> pd.DataFrame:
    csv_path = base_dir / "results.csv"
    records: list[dict[str, float | str]] = []

    if use_csv:
        df = _load_node_csv(csv_path)
        if df is None:
            print(f"--use-csv requested but {csv_path} not found.", file=sys.stderr)  # noqa: T201
            sys.exit(1)
        return df

    scenarios = _find_scenarios(base_dir)
    if not scenarios:
        print("No scenario files found.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    trackers = get_trackers(scenarios[0])
    for sc_path in scenarios:
        model = _model_from_scenario(sc_path)
        node_size = _node_size_from_scenario(sc_path)
        overlap = _is_overlap_scenario(sc_path)
        run_prefix = f"node-size-{model}-node{node_size}"
        if overlap:
            run_prefix = f"{run_prefix}-overlap"
            model = f"{model}-overlap"

        mfu: float | None = None
        if trackers:
            for tracker in trackers:
                metrics = tracker.pull_metrics(run_name_prefix=run_prefix)
                if metrics is not None:
                    mfu = metrics.get("mfu_pct")
                    if mfu is not None:
                        break

        if mfu is None:
            print(f"No wandb data for {model} node{node_size}; skipping.", file=sys.stderr)  # noqa: T201
            continue

        records.append(
            {"model": model, "node_size": f"{node_size} GPU/node", "mfu_pct": float(mfu)}
        )

    cached_df = _load_node_csv(csv_path)
    if records:
        fresh_df = pd.DataFrame(records)
        df = _merge_node_size_frames(cached_df, fresh_df) if cached_df is not None else fresh_df
    elif cached_df is not None:
        print("No wandb data found; falling back to local CSV.", file=sys.stderr)  # noqa: T201
        df = cached_df
    else:
        print("No wandb data and no cached results.csv available. Exiting.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    _save_node_csv(df, csv_path)
    return df


def _load_link_bw_data(base_dir: Path, use_csv: bool) -> pd.DataFrame:
    csv_path = base_dir / "results.csv"
    records: list[dict[str, float | str]] = []

    if use_csv:
        df = _load_link_csv(csv_path)
        if df is None:
            print(f"--use-csv requested but {csv_path} not found.", file=sys.stderr)  # noqa: T201
            sys.exit(1)
        return df

    env_file = base_dir / ".tracking.env"
    trackers = get_trackers(env_file if env_file.exists() else base_dir)
    if trackers:
        for tracker in trackers:
            runs = tracker.fetch_runs(prefix="link-bw-")
            for run in runs:
                model, bw = _model_bw_from_run(run)
                mfu = run["summary"].get("mfu_pct")
                if mfu is None:
                    continue
                records.append(
                    {"model": model, "bw_gbps": bw, "bw_label": f"{bw} Gbps", "mfu_pct": float(mfu)}
                )

    cached_df = _load_link_csv(csv_path)
    if records:
        fresh_df = pd.DataFrame(records)
        df = _merge_link_bw_frames(cached_df, fresh_df) if cached_df is not None else fresh_df
    elif cached_df is not None:
        print("No wandb data found; falling back to local CSV.", file=sys.stderr)  # noqa: T201
        df = cached_df
    else:
        print("No wandb data and no cached results.csv available. Exiting.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    _save_link_csv(df, csv_path)
    return df


def _plot_node_size_bar(ax, df: pd.DataFrame) -> None:
    df = df.copy()
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
        width=0.70,
        ax=ax,
    )

    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("")
    ax.set_title("MFU by Node Size", fontweight="bold")
    ax.legend(
        title="Node config",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
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
                    fontsize=5,
                    color="red",
                )


def _plot_link_bw_bar(ax, df: pd.DataFrame) -> None:
    df = df.copy()
    df["model_label"] = df["model"].map(label_for_model)
    models = sorted(str(x) for x in df["model_label"].unique())
    order = sorted(
        df["bw_label"].unique(),
        key=lambda s: int(str(s).split()[0]) if str(s).split()[0].isdigit() else 0,
    )

    sns.barplot(
        data=df,
        x="model_label",
        y="mfu_pct",
        hue="bw_label",
        order=models,
        hue_order=order,
        palette="deep",
        width=0.70,
        ax=ax,
    )

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("MFU by Link Bandwidth", fontweight="bold")
    ax.legend(
        title="NIC speed",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=max(1, len(order)),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    sns.despine(ax=ax, top=True, right=True)

    # Annotate each non-baseline bar with its percentage difference vs. 400 Gbps.
    baseline_label = "400 Gbps" if "400 Gbps" in order else (order[0] if order else None)
    if baseline_label is not None and len(ax.containers) == len(order):
        baseline_idx = order.index(baseline_label)
        baseline_container = ax.containers[baseline_idx]
        for idx, _model_name in enumerate(models):
            baseline_bar = baseline_container[idx]
            baseline = float(baseline_bar.get_height())
            if baseline == 0:
                continue
            for hidx, bw_label in enumerate(order):
                if bw_label == baseline_label:
                    continue
                bar = ax.containers[hidx][idx]
                value = float(bar.get_height())
                pct = (value - baseline) / baseline * 100
                x = float(bar.get_x() + bar.get_width() / 2)
                y = float(bar.get_y() + bar.get_height())
                ax.text(
                    x,
                    y + 0.02 * y if y != 0 else 0.02,
                    f"{pct:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    color="red",
                )


def _plot_link_bw_line(ax, df: pd.DataFrame) -> None:
    df = df.copy()
    df["model_label"] = df["model"].map(label_for_model)
    models = sorted(str(x) for x in df["model_label"].unique())

    for model_name in models:
        sub = df.loc[df["model_label"] == model_name, ["bw_gbps", "mfu_pct"]].sort_values("bw_gbps")
        ax.plot(
            sub["bw_gbps"].to_numpy(),
            sub["mfu_pct"].to_numpy(),
            marker="o",
            label=model_name,
            linewidth=1.2,
            markersize=4,
        )

    ax.set_xlabel("Link bandwidth (Gbps)")
    ax.set_ylabel("")
    ax.set_title("MFU vs Link Bandwidth", fontweight="bold")
    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=max(1, len(models)),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    sns.despine(ax=ax, top=True, right=True)


def plot_usecases(
    output: Path | None, base_dir: Path, line: bool = False, use_csv: bool = False
) -> None:
    node_dir = base_dir / "usecase_node_size"
    link_dir = base_dir / "usecase_link_bw"

    df_node = _load_node_size_data(node_dir, use_csv)
    df_link = _load_link_bw_data(link_dir, use_csv)

    for df, name, required in [
        (df_node, "node_size", {"model", "node_size", "mfu_pct"}),
        (df_link, "link_bw", {"model", "bw_gbps", "bw_label", "mfu_pct"}),
    ]:
        missing = sorted(required - set(df.columns))
        if missing:
            print(f"{name} results.csv is missing required columns: {missing}", file=sys.stderr)  # noqa: T201
            sys.exit(1)

    setup_latex_style()
    fig = plt.figure(figsize=(7.0, 2.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 3])
    ax_node = fig.add_subplot(gs[0])
    ax_link = fig.add_subplot(gs[1], sharey=ax_node)

    y_max = max(float(df_node["mfu_pct"].max()), float(df_link["mfu_pct"].max()))
    y_top = min(100.0, y_max * 1.15)

    _plot_node_size_bar(ax_node, df_node)
    if line:
        _plot_link_bw_line(ax_link, df_link)
    else:
        _plot_link_bw_bar(ax_link, df_link)

    for ax in (ax_node, ax_link):
        ax.set_ylim(0, y_top)
        ax.tick_params(axis="x", rotation=0)

    ax_node.set_yticks([0, 20, 40, 60])
    ax_node.set_yticks(range(0, int(y_top) + 1, 10), minor=True)

    fig.tight_layout(rect=(0, 0.05, 1, 1.02))

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {output}")  # noqa: T201
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
        help="Directory containing usecase_node_size and usecase_link_bw subdirectories.",
    )
    parser.add_argument(
        "--line", action="store_true", help="Render link bandwidth as a line plot (default: bar)."
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Plot from the local results.csv files instead of pulling from wandb.",
    )
    args = parser.parse_args()
    plot_usecases(args.output, args.base_dir, line=args.line, use_csv=args.use_csv)


if __name__ == "__main__":
    main()
