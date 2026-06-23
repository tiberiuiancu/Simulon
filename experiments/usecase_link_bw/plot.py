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
    m = re.match(r"link-bw-(.+)-bw(\d+)", name)
    if m:
        return m.group(1), int(m.group(2))
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
            _save_csv(df, csv_path)
        else:
            print("No wandb data found; falling back to local CSV.", file=sys.stderr)
            df = _load_csv(csv_path)
            if df is None:
                print("No cached results.csv available. Exiting.", file=sys.stderr)
                sys.exit(1)

    required_cols = {"model", "bw_gbps", "bw_label", "mfu_pct"}
    if not required_cols.issubset(df.columns):
        print(
            f"CSV {csv_path} is missing required columns: {sorted(required_cols - set(df.columns))}",
            file=sys.stderr,
        )
        sys.exit(1)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))

    if not line:
        order = sorted(
            df["bw_label"].unique(),
            key=lambda s: int(s.split()[0]) if s.split()[0].isdigit() else 0,
        )
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
        ax.legend(title="NIC speed", loc="upper left")
        sns.despine(ax=ax, top=True, right=True)

    else:
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
        ax.legend(title="Model", loc="upper left")
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
