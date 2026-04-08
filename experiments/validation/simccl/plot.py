#!/usr/bin/env python3
"""Plot simulon vs nccl-tests bus bandwidth: 3×3 grid (rows=GPU count, cols=collective).

Loads JSON files from a results directory:
  sim_<collective>_<config>.json   — simulon predictions (from sim_ccl.py)
  nccl_<collective>_<config>.json  — measured nccl-tests output (from run_nccl.sh)

Measured files are optional; missing ones are silently skipped.

Usage (from repo root):
    uv run python experiments/validation/simccl/plot.py
    uv run python experiments/validation/simccl/plot.py --results-dir /path/to/results --output plot.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Layout ─────────────────────────────────────────────────────────────────

COLLECTIVES = ["AllReduce", "AllGather", "ReduceScatter", "AllToAll"]
CONFIGS = [
    {"label": "1n4g", "ngpus": 4,  "title": "4 GPUs (1 node)"},
    {"label": "2n4g", "ngpus": 8,  "title": "8 GPUs (2 nodes)"},
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
    sizes_MB = [r["size"] / (1024 ** 2) for r in results]
    bus_bw = [r["out_of_place"]["bus_bw"] for r in results]
    return sizes_MB, bus_bw


# ── Plotting ───────────────────────────────────────────────────────────────

def plot(results_dir: Path, output: Path | None) -> None:
    fig, axes = plt.subplots(
        nrows=len(CONFIGS),
        ncols=len(COLLECTIVES),
        figsize=(18, 10),
        sharex=True,
    )

    for row, cfg in enumerate(CONFIGS):
        for col, collective in enumerate(COLLECTIVES):
            ax = axes[row][col]
            label = cfg["label"]
            cname_lower = collective.lower()

            sim_data = _load(results_dir / f"sim_{cname_lower}_{label}.json")
            meas_data = _load(results_dir / f"nccl_{cname_lower}_{label}.json")

            if sim_data is not None:
                sizes, bws = sim_data
                ax.plot(sizes, bws, marker="o", markersize=4, linewidth=1.5,
                        color="#1f77b4", label="simulon")

            if meas_data is not None:
                sizes, bws = meas_data
                ax.plot(sizes, bws, marker="s", markersize=4, linewidth=1.5,
                        color="#ff7f0e", linestyle="--", label="nccl-tests")

            # Axes formatting
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{round(x)}MB" if x < 1024 else f"{round(x / 1024)}GB")
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

            # Legend only in first subplot
            if row == 0 and col == 0 and (sim_data is not None or meas_data is not None):
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Collective Bus Bandwidth: simulon vs nccl-tests\n(H100 · NVSwitch 4 · IB HDR100)",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved: {output}")
    else:
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/validation/simccl/results"),
        help="Directory containing sim_*.json and nccl_*.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save figure to file (PDF/PNG/SVG). If omitted, show interactively.",
    )
    args = parser.parse_args()
    plot(args.results_dir, args.output)


if __name__ == "__main__":
    main()
