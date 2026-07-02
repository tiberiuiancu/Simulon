#!/usr/bin/env python3
"""Plot Qwen3-32B workload-tuning grid-search results as an MFU heatmap.

Usage (from repo root):
    uv run python experiments/usecase_workload_tuning/plot.py
    uv run python experiments/usecase_workload_tuning/plot.py --output workload_tuning.pdf
    uv run python experiments/usecase_workload_tuning/plot.py --use-csv
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from simulon.tracking import get_trackers

sys.path.insert(0, str(Path(__file__).parent.parent))
from _plot_utils import setup_latex_style

_RUN_PREFIX = "usecase-workload-tuning-"

_CSV_COLS = [
    "name",
    "tp",
    "pp",
    "mbs",
    "vpp",
    "status",
    "oom",
    "skipped",
    "invalid",
    "error",
    "simulated",
    "throughput_tps",
    "mfu_pct",
    "iteration_time_ms",
    "invalid_reason",
    "error_detail",
]

_NUM_GPUS = 16
_NUM_LAYERS = 64

_ROWS: list[tuple[int, int, str]] = [
    (1, 1, "TP=1, MBS=1"),
    (2, 1, "TP=2, MBS=1"),
    (4, 1, "TP=4, MBS=1"),
    (4, 2, "TP=4, MBS=2"),
]


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


def _truthy(row: pd.Series, key: str) -> bool:
    value = row.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return bool(value)


def _status_from_row(row: pd.Series) -> str:
    if _truthy(row, "oom"):
        return "oom"
    if _truthy(row, "skipped"):
        return "skipped"
    if _truthy(row, "invalid"):
        return "invalid"
    if _truthy(row, "error"):
        return "error"
    if _truthy(row, "simulated"):
        return "valid"
    status = row.get("status")
    if isinstance(status, str) and status in ("valid", "oom", "invalid", "skipped", "error"):
        return status
    return "error"


def _pull_wandb_metrics(base_dir: Path) -> dict[str, dict[str, Any]]:
    metrics_by_name: dict[str, dict[str, Any]] = {}
    trackers = get_trackers(base_dir / "scenarios" / "base_workload.yaml")
    if not trackers:
        return metrics_by_name

    for tracker in trackers:
        runs = tracker.fetch_runs(prefix=_RUN_PREFIX)
        for run in runs:
            name = run["display_name"].replace(_RUN_PREFIX, "")
            metrics_by_name[name] = {
                "throughput_tps": run["summary"].get("throughput_tps"),
                "mfu_pct": run["summary"].get("mfu_pct"),
                "iteration_time_ms": run["summary"].get("iteration_time_ms"),
            }
    return metrics_by_name


def _vpp_options(pp: int) -> list[int | None]:
    if pp == 1:
        return [None]
    layers_per_stage = _NUM_LAYERS // pp
    return sorted(i for i in range(1, layers_per_stage + 1) if layers_per_stage % i == 0)


def _scan_trace_statuses(names: list[str]) -> dict[str, str]:
    """Check trace dir for .OOM / .error.log / trace files.

    Reads ``traces_dir`` directly from each scenario YAML so it works
    with the human-readable paths (no hash computation needed).
    """
    statuses: dict[str, str] = {}
    for name in names:
        scenario_path = (
            Path("experiments/usecase_workload_tuning/scenarios") / name / "scenario.yaml"
        )
        if not scenario_path.exists():
            continue
        try:
            with open(scenario_path) as fh:
                sc = yaml.safe_load(fh)
            trace_dir_rel = sc.get("datacenter", {}).get("datacenter", {}).get("traces_dir")
            if not trace_dir_rel:
                continue
            trace_dir = Path(trace_dir_rel)
        except Exception:
            continue
        if not trace_dir.exists():
            continue
        if (trace_dir / ".OOM").exists():
            statuses[name] = "oom"
        elif (trace_dir / ".INVALID").exists():
            statuses[name] = "invalid"
        elif (trace_dir / ".error.log").exists():
            statuses[name] = "error"
        elif (trace_dir / "workload.yaml").exists() and list(trace_dir.glob("trace_rank_*.json")):
            statuses[name] = "traced"
    return statuses


def _build_frame(base_dir: Path, use_csv: bool) -> pd.DataFrame:
    csv_path = base_dir / "results.csv"
    cached_df = _load_csv(csv_path) if use_csv else None

    if use_csv and cached_df is None:
        print("--use-csv requested but results.csv not found.", file=sys.stderr)
        sys.exit(1)

    records: list[dict[str, Any]] = []
    for tp in [1, 2, 4]:
        for pp in [1, 2, 4]:
            if (_NUM_GPUS // tp // pp) == 0:
                continue
            for mbs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                for vpp in _vpp_options(pp):
                    name = f"tp{tp}_pp{pp}_mbs{mbs}"
                    if vpp is not None:
                        name = f"{name}_vpp{vpp}"
                    records.append({"name": name, "tp": tp, "pp": pp, "mbs": mbs, "vpp": vpp})

    df = pd.DataFrame(records)

    status_map: dict[str, str] = {}
    extra_map: dict[str, dict[str, str]] = {}
    metric_map: dict[str, dict[str, Any]] = {}
    trace_status_map: dict[str, str] = _scan_trace_statuses(df["name"].tolist())

    if cached_df is not None:
        for _, row in cached_df.iterrows():
            name = str(row["name"])
            csv_status = _status_from_row(row)
            trace_status = trace_status_map.get(name)
            if trace_status in ("oom", "invalid"):
                status_map[name] = trace_status
                df.loc[df["name"] == name, trace_status] = True
            elif trace_status == "error":
                status_map[name] = "error"
                df.loc[df["name"] == name, "error"] = True
            else:
                status_map[name] = csv_status
                for col in ("oom", "skipped", "invalid", "error", "simulated"):
                    if _truthy(row, col):
                        df.loc[df["name"] == name, col] = True
            invalid_reason = row.get("invalid_reason")
            error_detail = row.get("error_detail")
            extra_map[name] = {
                "invalid_reason": ""
                if invalid_reason is None or str(invalid_reason) == "nan"
                else str(invalid_reason),
                "error_detail": ""
                if error_detail is None or str(error_detail) == "nan"
                else str(error_detail),
            }
            metric_map[name] = {
                "throughput_tps": row.get("throughput_tps"),
                "mfu_pct": row.get("mfu_pct"),
                "iteration_time_ms": row.get("iteration_time_ms"),
            }

    if not use_csv:
        wandb_metrics = _pull_wandb_metrics(base_dir)
        for name, metrics in wandb_metrics.items():
            metric_map[name] = metrics
            if trace_status_map.get(name) not in ("oom", "invalid", "error"):
                status_map[name] = "valid"

    for name, ts in trace_status_map.items():
        if name not in status_map:
            if ts == "traced":
                status_map[name] = "valid"
            elif ts in ("oom", "invalid", "error"):
                status_map[name] = ts

    df["status"] = df["name"].map(lambda n: status_map.get(n, "error"))
    df["invalid_reason"] = df["name"].map(lambda n: extra_map.get(n, {}).get("invalid_reason", ""))
    df["error_detail"] = df["name"].map(lambda n: extra_map.get(n, {}).get("error_detail", ""))

    def _get_metric(name: str, key: str) -> Any:
        return metric_map.get(name, {}).get(key)

    for metric_name in ("throughput_tps", "mfu_pct", "iteration_time_ms"):
        df[metric_name] = df["name"].map(lambda n, key=metric_name: _get_metric(n, key))

    df["vpp"] = df["vpp"].apply(lambda x: int(x) if pd.notna(x) else pd.NA).astype("Int64")

    for col in _CSV_COLS:
        if col not in df.columns:
            df[col] = None if col in ("throughput_tps", "mfu_pct", "iteration_time_ms") else ""

    df = df[_CSV_COLS]
    _save_csv(df, csv_path)
    return df


def _col_label(pp: int, vpp: int | None) -> str:
    if vpp is None:
        return "PP1"
    return f"PP{pp}\nVPP{vpp}"


def _build_columns() -> list[tuple[int, int | None, str]]:
    """X-axis columns in display order: PP1, PP2 VPP 1..32, PP4 VPP 1..16.

    VPP is ordered numerically, not alphabetically.
    """
    cols: list[tuple[int, int | None, str]] = []
    cols.append((1, None, _col_label(1, None)))
    for vpp in [1, 2, 4, 8, 16, 32]:
        cols.append((2, vpp, _col_label(2, vpp)))
    for vpp in [1, 2, 4, 8, 16]:
        cols.append((4, vpp, _col_label(4, vpp)))
    return cols


def _find_config(df: pd.DataFrame, tp: int, mbs: int, pp: int, vpp: int | None) -> pd.Series | None:
    mask = (df["tp"] == tp) & (df["mbs"] == mbs) & (df["pp"] == pp)
    mask = mask & df["vpp"].isna() if vpp is None else mask & (df["vpp"] == vpp)
    matches = df[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def _render_heatmap(ax, df: pd.DataFrame) -> None:
    all_cols = _build_columns()
    all_rows = _ROWS

    n_rows_all = len(all_rows)
    n_cols_all = len(all_cols)

    mfu_full = np.full((n_rows_all, n_cols_all), np.nan)
    status_full: list[str | None] = [None] * (n_rows_all * n_cols_all)

    for i, (tp, mbs, _label) in enumerate(all_rows):
        for j, (pp, vpp, _col_label) in enumerate(all_cols):
            row = _find_config(df, tp, mbs, pp, vpp)
            if row is None:
                continue
            status = str(row["status"])
            status_full[i * n_cols_all + j] = status
            if status == "valid":
                mfu = row.get("mfu_pct")
                if mfu is not None and not (isinstance(mfu, float) and np.isnan(mfu)):
                    mfu_full[i, j] = float(mfu)

    def _is_gray(status: str | None) -> bool:
        return status in ("oom", "error", "invalid", None)

    keep_rows = [
        i
        for i in range(n_rows_all)
        if not all(_is_gray(status_full[i * n_cols_all + j]) for j in range(n_cols_all))
    ]
    keep_cols = [
        j
        for j in range(n_cols_all)
        if not all(_is_gray(status_full[i * n_cols_all + j]) for i in range(n_rows_all))
    ]

    rows = [all_rows[i] for i in keep_rows]
    cols = [all_cols[j] for j in keep_cols]
    n_rows = len(rows)
    n_cols = len(cols)

    mfu_matrix = np.full((n_rows, n_cols), np.nan)
    status_matrix: list[str | None] = [None] * (n_rows * n_cols)
    for di, i in enumerate(keep_rows):
        for dj, j in enumerate(keep_cols):
            status_matrix[di * n_cols + dj] = status_full[i * n_cols_all + j]
            mfu_matrix[di, dj] = mfu_full[i, j]

    max_mfu = np.nanmax(mfu_matrix) if np.any(np.isfinite(mfu_matrix)) else 1.0
    if max_mfu <= 0:
        max_mfu = 1.0
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_mfu", ["#e8f5e9", "#a5d6a7", "#66bb6a", "#388e3c"], N=256
    )
    norm = mcolors.Normalize(vmin=0, vmax=max_mfu)

    ax.set_facecolor("#f5f5f5")
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="#f5f5f5",
                    edgecolor="#cccccc",
                    linewidth=0.5,
                )
            )

    for i in range(n_rows):
        for j in range(n_cols):
            status = status_matrix[i * n_cols + j]
            if status == "valid" and np.isfinite(mfu_matrix[i, j]):
                color = cmap(norm(mfu_matrix[i, j]))
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="#999999",
                        linewidth=0.5,
                    )
                )
            elif status in ("oom", "error", "invalid"):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#e0e0e0",
                        edgecolor="#bdbdbd",
                        linewidth=0.5,
                    )
                )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.grid(False)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([c[2] for c in cols], fontsize=7)
    ax.set_yticklabels([r[2] for r in rows], fontsize=7)
    ax.set_xlabel("Pipeline / virtual pipeline", fontsize=8)
    ax.set_ylabel("Configuration", fontsize=8)
    ax.set_title("Qwen3-32B MFU heatmap", fontweight="bold", fontsize=9)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    max_mfu_val = np.nanmax(mfu_matrix) if np.any(np.isfinite(mfu_matrix)) else None

    for i in range(n_rows):
        for j in range(n_cols):
            status = status_matrix[i * n_cols + j]
            if status == "valid" and np.isfinite(mfu_matrix[i, j]):
                val = mfu_matrix[i, j]
                is_max = max_mfu_val is not None and abs(val - max_mfu_val) < 1e-9
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold" if is_max else "normal",
                    color="#1b5e20",
                )

    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="#bbbbbb", linewidth=0.4)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="#bbbbbb", linewidth=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("MFU (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def _render_plot(output: Path | None, df: pd.DataFrame) -> None:
    setup_latex_style()

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))
    _render_heatmap(ax, df)

    fig.tight_layout()

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
        help="Write figure to file (PDF/PNG/SVG). Omit to display.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing the usecase_workload_tuning results.csv.",
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Use only the local results.csv; do not pull metrics from wandb.",
    )
    args = parser.parse_args()

    df = _build_frame(args.base_dir, use_csv=args.use_csv)
    _render_plot(args.output, df)


if __name__ == "__main__":
    main()
