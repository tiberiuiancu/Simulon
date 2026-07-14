#!/usr/bin/env python3
"""Plot simulon vs. nccl-tests bus bandwidth: 3×4 grid (rows=GPU count, cols=collective).

Loads JSON files from a results directory:
  sim_<collective>_<config>_<cluster>.json      — simulon predictions (from sim_ccl.py)
  nccl_<collective>_<config>_<cluster>.json     — measured nccl-tests output (from run_nccl*.sh)
  simai_analytical_<collective>_<config>.json   — SimAI analytical mode
  simai_ns3_<collective>_<config>.json          — SimAI NS3 mode

All files are optional; missing ones are silently skipped.

Usage (from repo root):
    uv run python experiments/validate_simccl/plot.py --output plot.pdf
    uv run python experiments/validate_simccl/plot.py --cluster jupiter --output plot.pdf
    uv run python experiments/validate_simccl/plot.py --simai --output plot.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent.parent))
from _plot_utils import setup_latex_style

# ── Layout ─────────────────────────────────────────────────────────────────

COLLECTIVES = ["AllReduce", "AllGather", "ReduceScatter", "AllToAll"]
CONFIGS = [
    {"label": "1n4g", "ngpus": 4, "title": "4 GPUs\n(1 node)"},
    {"label": "2n4g", "ngpus": 8, "title": "8 GPUs\n(2 nodes)"},
    {"label": "4n4g", "ngpus": 16, "title": "16 GPUs\n(4 nodes)"},
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


def _load_with_errors(
    path: Path,
) -> tuple[list[float], list[float], list[float], list[float]] | None:
    """Return (sizes_MB, bus_bw_GBps, bus_bw_min, bus_bw_max) or None.

    Loads per-run files (<path stem>_run1.json .. _run20.json) and computes
    the mean, min, and max bus bandwidth across runs for each message size.
    Falls back to the single-file avg (path) when per-run files are absent.
    """
    if not path.exists() and not path.with_name(f"{path.stem}_run1{path.suffix}").exists():
        return None

    stem = path.stem
    suffix = path.suffix

    run_files = sorted(path.parent.glob(f"{stem}_run*{suffix}"))
    if not run_files:
        if path.exists():
            sizes, bws = _load(path)
            return sizes, bws, bws, bws
        return None

    all_results: list[list[dict]] = []
    for rf in run_files:
        data = json.loads(rf.read_text())
        results = data.get("results", [])
        if results:
            all_results.append(results)

    if not all_results:
        return None

    sizes_MB = []
    bus_bw = []
    bw_min = []
    bw_max = []
    n_runs = len(all_results)
    for i in range(len(all_results[0])):
        size = all_results[0][i]["size"]
        sizes_MB.append(size / (1024**2))
        bws = [all_results[r][i]["out_of_place"]["bus_bw"] for r in range(n_runs)]
        bus_bw.append(sum(bws) / n_runs)
        bw_min.append(min(bws))
        bw_max.append(max(bws))

    return sizes_MB, bus_bw, bw_min, bw_max


# ── Plotting ───────────────────────────────────────────────────────────────


def plot(
    results_dir: Path, output: Path | None, cluster: str = "snellius", simai: bool = False
) -> None:
    setup_latex_style()

    fig, axes = plt.subplots(
        nrows=len(CONFIGS), ncols=len(COLLECTIVES), figsize=(7, 4.5), sharex=True
    )

    # Track which series appear (for a shared legend)
    legend_handles: dict[str, object] = {}

    for row, cfg in enumerate(CONFIGS):
        for col, collective in enumerate(COLLECTIVES):
            ax = axes[row][col]
            label = cfg["label"]
            cname_lower = collective.lower()

            series = [
                ("simulon", f"sim_{cname_lower}_{label}_{cluster}.json", "#1f77b4", "o", "-"),
                ("nccl-tests", f"nccl_{cname_lower}_{label}_{cluster}.json", "#ff7f0e", "s", "--"),
            ]
            if simai:
                series.extend(
                    [
                        (
                            "SimAI analytical",
                            f"simai_analytical_{cname_lower}_{label}.json",
                            "#2ca02c",
                            "^",
                            "-",
                        ),
                        (
                            "SimAI NS3",
                            f"simai_ns3_{cname_lower}_{label}.json",
                            "#9467bd",
                            "D",
                            "-.",
                        ),
                    ]
                )

            for sname, fname, color, marker, ls in series:
                if sname == "nccl-tests":
                    data = _load_with_errors(results_dir / fname)
                else:
                    data = _load(results_dir / fname)
                if data is None:
                    continue
                if sname == "nccl-tests":
                    sizes, bws, bw_min, bw_max = data
                else:
                    sizes, bws = data
                lw = 1.5 if ls == "--" else 1.0
                ms = 4 if ls == "--" else 3
                (line,) = ax.plot(
                    sizes,
                    bws,
                    marker=marker,
                    markersize=ms,
                    linewidth=lw,
                    color=color,
                    linestyle=ls,
                    label=sname,
                )
                if sname == "nccl-tests":
                    ax.fill_between(
                        sizes, bw_min, bw_max, color=color, alpha=0.15, linewidth=0, step="mid"
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
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, which="major", linestyle=":", alpha=0.5)
            ax.set_ylim(0, 200 if row > 0 else 350)

            # Titles and labels
            if row == 0:
                ax.set_title(collective, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{cfg['title']}\nBus BW (GB/s)", fontsize=8)

    fig.text(0.5, 0.02, "Message size", ha="center", fontsize=8)

    cluster_labels = {
        "snellius": "H100 · NVSwitch 4 · quad-rail NDR200",
        "jupiter": "GH200 · NVLink 4 · quad-rail HDR",
    }
    cluster_desc = cluster_labels.get(cluster, cluster)
    title = "Collective Bus Bandwidth: simulon vs. nccl-tests"
    if simai:
        title += " vs. SimAI"
    fig.suptitle(f"{title}\n({cluster_desc})", fontsize=9, y=0.99)

    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=len(legend_handles),
            fontsize=7,
            frameon=False,
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=1.0,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.91))

    if output is not None:
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved: {output}")  # noqa: T201
    else:
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
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
    parser.add_argument(
        "--simai", action="store_true", help="Include SimAI analytical and NS3 results in the plot."
    )
    args = parser.parse_args()
    plot(args.results_dir, args.output, args.cluster, simai=args.simai)


if __name__ == "__main__":
    main()
