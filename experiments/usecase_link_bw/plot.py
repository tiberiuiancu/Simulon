#!/usr/bin/env python3
"""Plot MFU vs link bandwidth for usecase_link_bw experiments.

Usage (from repo root):
    uv run python experiments/usecase_link_bw/plot.py
    uv run python experiments/usecase_link_bw/plot.py --chart line --output line.pdf
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


def _bw_from_scenario(path: Path) -> int:
    return int(path.stem.replace("scenario", ""))


def _config_filter_from_scenario(path: Path) -> dict[str, str | int | float | bool]:
    bw = _bw_from_scenario(path)
    return {"datacenter.node.scale_out.nic.speed": f"{bw}Gbps"}


def _compute_workload_hash(scenario: Path) -> str:
    import yaml

    with open(scenario) as f:
        raw: dict[str, object] = yaml.safe_load(f)
    sc = ScenarioConfig.model_validate(raw)
    if isinstance(sc.workload, Path):
        sc.workload = resolve_workload(sc.workload)
    from simulon.config.workload import MegatronWorkload
    if not isinstance(sc.workload, MegatronWorkload):
        raise ValueError("Expected MegatronWorkload")
    return workload_hash(sc.workload)


def plot_mfu_from_wandb(
    output: Path | None,
    base_dir: Path,
    chart: str = "both",
) -> None:
    scenarios = _find_scenarios(base_dir)
    if not scenarios:
        print("No scenario files found.", file=sys.stderr)
        sys.exit(1)

    trackers = get_trackers(scenarios[0])
    if not trackers:
        print("No active trackers found.", file=sys.stderr)
        sys.exit(1)

    records: list[dict[str, float | str]] = []

    for sc_path in scenarios:
        model = _model_from_scenario(sc_path)
        bw = _bw_from_scenario(sc_path)
        wl_hash = _compute_workload_hash(sc_path)
        cfg_filter = _config_filter_from_scenario(sc_path)

        for tracker in trackers:
            metrics = tracker.pull_metrics(wl_hash, config_filters=cfg_filter)
            if metrics is None:
                continue
            mfu = metrics.get("mfu_pct")
            if mfu is None:
                continue
            records.append({
                "model": model,
                "bw_gbps": bw,
                "bw_label": f"{bw} Gbps",
                "mfu_pct": float(mfu),
            })

    if not records:
        print("No wandb data found for any scenario. Exiting.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    sns.set_theme(style="whitegrid")

    show_bar = chart in ("both", "bar")
    show_line = chart in ("both", "line")

    n_plots = (1 if show_bar else 0) + (1 if show_line else 0)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5 * n_plots, 5), squeeze=False
    )
    ax_idx = 0

    if show_bar:
        ax = axes[0, ax_idx]
        ax_idx += 1
        order = sorted(df["bw_label"].unique(), key=lambda s: int(s.split()[0]))
        sns.barplot(
            data=df,
            x="model",
            y="mfu_pct",
            hue="bw_label",
            order=sorted(df["model"].unique()),
            hue_order=order,
            palette="deep",
            ax=ax,
        )
        ax.set_ylabel("MFU (%)")
        ax.set_xlabel("")
        ax.set_title("MFU by Link Bandwidth", fontsize=14, fontweight="bold")
        ax.legend(title="NIC speed", loc="upper right")
        sns.despine(ax=ax, top=True, right=True)

    if show_line:
        ax = axes[0, ax_idx]
        ax_idx += 1
        for model_name in sorted(df["model"].unique()):
            sub = df[df["model"] == model_name].sort_values(["bw_gbps"])
            ax.plot(
                sub["bw_gbps"],
                sub["mfu_pct"],
                marker="o",
                label=model_name,
                linewidth=2,
                markersize=8,
            )
        ax.set_xlabel("Link bandwidth (Gbps)")
        ax.set_ylabel("MFU (%)")
        ax.set_title("MFU vs Link Bandwidth", fontsize=14, fontweight="bold")
        ax.legend(title="Model", loc="upper right")
        sns.despine(ax=ax, top=True, right=True)

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
        "--chart",
        choices=["bar", "line", "both"],
        default="both",
        help="Which chart(s) to render (default: both).",
    )
    args = parser.parse_args()
    plot_mfu_from_wandb(args.output, args.base_dir, chart=args.chart)


if __name__ == "__main__":
    main()
