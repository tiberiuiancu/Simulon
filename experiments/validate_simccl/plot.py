#!/usr/bin/env python3
"""Plot simulon vs nccl-tests bus bandwidth: 3×3 grid (rows=GPU count, cols=collective).

Loads JSON files from a results directory:
  sim_<collective>_<config>_<cluster>.json      — simulon predictions (from sim_ccl.py)
  nccl_<collective>_<config>_<cluster>.json     — measured nccl-tests output (from run_nccl*.sh)
  simai_analytical_<collective>_<config>.json   — SimAI analytical mode
  simai_ns3_<collective>_<config>.json          — SimAI NS3 mode

All files are optional; missing ones are silently skipped.

Usage (from repo root):
    uv run python experiments/validate_simccl/plot.py
    uv run python experiments/validate_simccl/plot.py --cluster jupiter
    uv run python experiments/validate_simccl/plot.py --results-dir /path/to/results --output plot.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Layout ─────────────────────────────────────────────────────────────────

COLLECTIVES = ["AllReduce", "AllGather", "ReduceScatter", "AllToAll"]
CONFIGS = [
    {"label": "1n4g", "ngpus": 4, "title": "4 GPUs (1 node)"},
    {"label": "2n4g", "ngpus": 8, "title": "8 GPUs (2 nodes)"},
    {"label": "4n4g", "ngpus": 16, "title": "16 GPUs (4 nodes)"},
]


# ── JSON loading ───────────────────────────────────────────────────────────


def _load(path: Path) -> tuple[list[float], list[float]] | None:
    """Return (sizes_MB, bus_bw_GBps) or None if file is missing/empty."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    results = data.get("results", [])
    if not results:
        return None
    sizes_MB = [r["size"] / (1024**2) for r in results]
    bus_bw = [r["out_of_place"]["bus_bw"] for r in results]
    return sizes_MB, bus_bw


# ── Plotting ───────────────────────────────────────────────────────────────


def plot(results_dir: Path, output: Path | None, cluster: str = "snellius") -> None:
    fig, axes = plt.subplots(
        nrows=len(CONFIGS), ncols=len(COLLECTIVES), figsize=(18, 10), sharex=True
    )

    # Track which series appear (for a shared legend)
    legend_handles: dict[str, object] = {}

    for row, cfg in enumerate(CONFIGS):
        for col, collective in enumerate(COLLECTIVES):
            ax = axes[row][col]
            label = cfg["label"]
            cname_lower = collective.lower()

            series = [
                ("nccl-tests", f"nccl_{cname_lower}_{label}_{cluster}.json", "#ff7f0e", "s", "--"),
                ("simulon", f"sim_{cname_lower}_{label}_{cluster}.json", "#1f77b4", "o", "-"),
                (
                    "SimAI analytical",
                    f"simai_analytical_{cname_lower}_{label}.json",
                    "#2ca02c",
                    "^",
                    "-",
                ),
                ("SimAI NS3", f"simai_ns3_{cname_lower}_{label}.json", "#9467bd", "D", "-."),
            ]

            for sname, fname, color, marker, ls in series:
                data = _load(results_dir / fname)
                if data is None:
                    continue
                sizes, bws = data
                (line,) = ax.plot(
                    sizes,
                    bws,
                    marker=marker,
                    markersize=4,
                    linewidth=1.5,
                    color=color,
                    linestyle=ls,
                    label=sname,
                )
                if sname not in legend_handles:
                    legend_handles[sname] = line

            # Axes formatting
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda x, _: f"{round(x)}MB" if x < 1024 else f"{round(x / 1024)}GB"
                )
            )
            ax.set_xlim(left=8, right=8192)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, which="major", linestyle=":", alpha=0.5)
            ax.set_ylim(bottom=0)

            # Titles and labels
            if row == 0:
                ax.set_title(collective, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{cfg['title']}\nBus BW (GB/s)", fontsize=9)
            if row == len(CONFIGS) - 1:
                ax.set_xlabel("Message size", fontsize=9)

    cluster_labels = {
        "snellius": "H100 · NVSwitch 4 · quad-rail NDR200",
        "jupiter": "GH200 · NVLink 4 · quad-rail HDR",
    }
    cluster_desc = cluster_labels.get(cluster, cluster)
    fig.suptitle(
        f"Collective Bus Bandwidth: simulon vs nccl-tests vs SimAI\n({cluster_desc})",
        fontsize=13,
        y=1.01,
    )

    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc="upper left",
            ncol=1,
            fontsize=9,
            frameon=True,
        )

    fig.tight_layout()

    if output is not None:
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved: {output}")  # noqa: T201
    else:
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cluster",
        default="snellius",
        help="Cluster name used as filename suffix (default: snellius)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/validate_simccl/results"),
        help="Directory containing sim_*.json and nccl_*.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save figure to file (PDF/PNG/SVG). If omitted, show interactively.",
    )
    args = parser.parse_args()
    plot(args.results_dir, args.output, args.cluster)


if __name__ == "__main__":
    main()
