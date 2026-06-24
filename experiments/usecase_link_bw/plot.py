#!/usr/bin/env python3
"""Plot MFU vs link bandwidth for usecase_link_bw experiments.

Usage (from repo root):
    uv run python experiments/usecase_link_bw/plot.py
    uv run python experiments/usecase_link_bw/plot.py --line --output line.pdf
"""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from simulon.tracking import get_trackers

sys.path.insert(0, str(Path(__file__).parent.parent))
from _plot_utils import label_for_model, setup_latex_style


def _extract_bw(run: dict[str, Any]) -> int:
    cfg = run.get("config", {})
    for key, val in cfg.items():
        if key.endswith(".scale_out.nic.speed") and isinstance(val, str) and "Gbps" in val:
            return int(val.replace("Gbps", ""))
        if key == "datacenter.node.from_" and isinstance(val, str):
            m = re.search(r"node_bw(\d+)", val)
            if m:
                return int(m.group(1))
    return 0


def _model_bw_from_run(run: dict[str, Any]) -> tuple[str, int]:
    name = run["display_name"]
    m = re.match(r"link-bw-(.+)-bw(\d+)(?:-overlap)?$", name)
    if m:
        model = m.group(1)
        if name.endswith("-overlap"):
            model = f"{model}-overlap"
        return model, int(m.group(2))
    parts = name.split("-")
    if len(parts) >= 3:
        return "-".join(parts[2:]), _extract_bw(run)
    return name, _extract_bw(run)


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


def plot_mfu_from_wandb(
    output: Path | None, base_dir: Path, line: bool = False, use_csv: bool = False
) -> None:
    csv_path = base_dir / "results.csv"
    records: list[dict[str, float | str]] = []

    if use_csv:
        df = _load_csv(csv_path)
        if df is None:
            print(f"--use-csv requested but {csv_path} not found.", file=sys.stderr)
            sys.exit(1)
    else:
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
                        {
                            "model": model,
                            "bw_gbps": bw,
                            "bw_label": f"{bw} Gbps",
                            "mfu_pct": float(mfu),
                        }
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

    required_cols = {"model", "bw_gbps", "bw_label", "mfu_pct"}
    if not required_cols.issubset(df.columns):
        print(
            f"CSV {csv_path} is missing required columns: {sorted(required_cols - set(df.columns))}",
            file=sys.stderr,
        )
        sys.exit(1)

    setup_latex_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    df["model_label"] = df["model"].map(label_for_model)
    models = sorted(str(x) for x in df["model_label"].unique())

    if not line:
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
            width=0.9,
            ax=ax,
        )
        ax.set_ylabel("MFU (%)")
        ax.set_xlabel("")
        ax.set_title("MFU by Link Bandwidth", fontweight="bold")
        y_max = float(df["mfu_pct"].max())
        ax.set_ylim(0, min(100, y_max * 1.15))
        ax.set_yticks(range(0, int(min(100, y_max * 1.15)) + 1, 10))
        ax.legend(
            title="NIC speed",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
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
                        fontsize=4,
                        color="red",
                    )

    else:
        for model_name in models:
            sub = df.loc[df["model_label"] == model_name, ["bw_gbps", "mfu_pct"]].sort_values(
                "bw_gbps"
            )
            ax.plot(
                sub["bw_gbps"].to_numpy(),
                sub["mfu_pct"].to_numpy(),
                marker="o",
                label=model_name,
                linewidth=1.2,
                markersize=4,
            )
        ax.set_xlabel("Link bandwidth (Gbps)")
        ax.set_ylabel("MFU (%)")
        ax.set_title("MFU vs Link Bandwidth", fontweight="bold")
        ax.legend(
            title="",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            ncol=max(1, len(models)),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
        )
        sns.despine(ax=ax, top=True, right=True)

    ax.tick_params(axis="x", rotation=0)
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
        "--line", action="store_true", help="Render line plot instead of bar plot (default: bar)."
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Plot from the local results.csv instead of pulling from wandb.",
    )
    args = parser.parse_args()
    plot_mfu_from_wandb(args.output, args.base_dir, line=args.line, use_csv=args.use_csv)


if __name__ == "__main__":
    main()
