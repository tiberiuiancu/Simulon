#!/usr/bin/env python3
"""Plot Qwen3-32B workload-tuning grid-search results.

Usage (from repo root):
    uv run python experiments/usecase_workload_tuning/plot.py
    uv run python experiments/usecase_workload_tuning/plot.py --output workload_tuning.pdf
    uv run python experiments/usecase_workload_tuning/plot.py --use-csv
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulon.tracking import get_trackers

sys.path.insert(0, str(Path(__file__).parent.parent))
from _plot_utils import setup_latex_style

_RUN_PREFIX = "usecase-workload-tuning-"

_STATUS_COLORS = {
    "valid": "#2ca02c",
    "oom": "#d62728",
    "invalid": "#ff7f0e",
    "skipped": "#9467bd",
    "error": "#8c564b",
}

_STATUS_LABELS = {
    "valid": "Valid",
    "oom": "OOM",
    "invalid": "Invalid",
    "skipped": "Skipped",
    "error": "Error",
}

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
    # Fall back to the status column written by previous plot runs
    status = row.get("status")
    if isinstance(status, str) and status in _STATUS_COLORS:
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


def _vpp_options(pp: int) -> Sequence[int | None]:
    if pp == 1:
        return [None]
    layers_per_stage = _NUM_LAYERS // pp
    return sorted(i for i in range(1, layers_per_stage + 1) if layers_per_stage % i == 0)


def _scan_trace_statuses(names: list[str]) -> dict[str, str]:
    from simulon.config.resolve import resolve_gpu_spec, resolve_workload, workload_hash
    from simulon.config.scenario import ScenarioConfig

    statuses: dict[str, str] = {}
    for name in names:
        scenario_path = (
            Path("experiments/usecase_workload_tuning/scenarios") / name / "scenario.yaml"
        )
        if not scenario_path.exists():
            continue
        try:
            sc = ScenarioConfig.from_yaml(str(scenario_path))
            gpu_spec = resolve_gpu_spec(sc.datacenter)
            gpu_name = (gpu_spec.name or "default").lower().replace(" ", "-")
        except Exception:
            gpu_name = "default"
        try:
            wl = resolve_workload(str(scenario_path.parent / "workload.yaml"))
            h = workload_hash(wl)
        except Exception:
            continue
        trace_dir = Path("templates/gpu") / gpu_name / "traces" / h
        if (trace_dir / ".OOM").exists():
            statuses[name] = "oom"
        elif (trace_dir / ".INVALID").exists():
            statuses[name] = "invalid"
    return statuses


def _build_frame(base_dir: Path, use_csv: bool) -> pd.DataFrame:
    csv_path = base_dir / "results.csv"
    cached_df = _load_csv(csv_path)

    if use_csv and cached_df is None:
        print("--use-csv requested but results.csv not found.", file=sys.stderr)
        sys.exit(1)

    records: list[dict[str, Any]] = []
    # Generate the full grid explicitly so the status matrix is complete
    # even if results.csv is partial from an interrupted sweep.
    for tp in [1, 2, 4]:
        for pp in [1, 2, 4]:
            if _NUM_GPUS // tp // pp == 0:
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
            status_map[name] = "valid"

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


def _format_vpp(row: pd.Series) -> str:
    vpp = row.get("vpp")
    if vpp is None or vpp is pd.NA or (isinstance(vpp, float) and np.isnan(vpp)):
        return "no VPP"
    return f"VPP{int(vpp)}"


def _format_number(row: pd.Series, key: str, fmt: str) -> str:
    value = row.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return fmt.format(float(value))


def _render_valid_table(ax, df: pd.DataFrame) -> None:
    valid_df = df[df["status"] == "valid"].copy()
    valid_df = valid_df.sort_values(["tp", "pp", "mbs", "vpp"])

    columns = [
        "Config",
        "TP",
        "PP",
        "MBS",
        "VPP",
        "Throughput\n(tokens/s)",
        "MFU\n(%)",
        "Iteration\n(ms)",
    ]
    rows: list[list[str]] = []
    for _, row in valid_df.iterrows():
        label = f"TP{int(row['tp'])} PP{int(row['pp'])} MBS{int(row['mbs'])} {_format_vpp(row)}"
        rows.append(
            [
                label,
                str(int(row["tp"])),
                str(int(row["pp"])),
                str(int(row["mbs"])),
                str(int(row["vpp"])) if pd.notna(row.get("vpp")) else "—",
                _format_number(row, "throughput_tps", "{:,.0f}"),
                _format_number(row, "mfu_pct", "{:.1f}"),
                _format_number(row, "iteration_time_ms", "{:.1f}"),
            ]
        )

    ax.axis("off")
    if not rows:
        ax.text(
            0.5, 0.5, "No valid configurations", ha="center", va="center", transform=ax.transAxes
        )
        return

    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.6)

    for key, cell in table.get_celld().items():
        row, _col = key
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e6e6e6")
        cell.set_edgecolor("#666666")

    ax.set_title("Valid configurations (simulated)", fontweight="bold", fontsize=9)


def _pp_vpp_label(r: pd.Series) -> str:
    vpp = r.get("vpp")
    if vpp is None or vpp is pd.NA or (isinstance(vpp, float) and np.isnan(vpp)):
        vpp_label = "—"
    else:
        vpp_label = str(int(vpp))
    return f"PP{r['pp']}\nVPP{vpp_label}"


def _render_status_matrix(ax, df: pd.DataFrame) -> None:
    sub = df[df["mbs"] == 1].copy()
    sub["pp_vpp"] = sub.apply(_pp_vpp_label, axis=1)

    tps = sorted(int(x) for x in sub["tp"].unique())
    cols = sorted(
        {str(x) for x in sub["pp_vpp"].unique()},
        key=lambda s: (int(s.split("\n")[0].replace("PP", "")), s.split("\n")[1]),
    )

    status_to_num = {status: i for i, status in enumerate(_STATUS_COLORS)}
    matrix = np.full((len(tps), len(cols)), np.nan)
    for i, tp in enumerate(tps):
        for j, col in enumerate(cols):
            matches = sub[(sub["tp"] == tp) & (sub["pp_vpp"] == col)]
            if len(matches) > 0:
                matrix[i, j] = status_to_num[str(matches.iloc[0]["status"])]

    ax.imshow(
        matrix,
        cmap=mcolors.ListedColormap(list(_STATUS_COLORS.values())),
        vmin=-0.5,
        vmax=len(_STATUS_COLORS) - 0.5,
    )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(tps)))
    ax.set_xticklabels(cols, fontsize=7)
    ax.set_yticklabels([f"TP{tp}" for tp in tps], fontsize=7)
    ax.set_xlabel("Pipeline / virtual pipeline", fontsize=8)
    ax.set_ylabel("Tensor parallelism", fontsize=8)
    ax.set_title("Grid-search status (MBS=1)", fontweight="bold", fontsize=9)

    for i in range(len(tps)):
        for j in range(len(cols)):
            if not np.isnan(matrix[i, j]):
                status = list(_STATUS_COLORS)[int(matrix[i, j])]
                label = _STATUS_LABELS[status]
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                    fontweight="bold",
                )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color, edgecolor="black", label=_STATUS_LABELS[status])
        for status, color in _STATUS_COLORS.items()
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(_STATUS_COLORS),
        frameon=False,
        fontsize=7,
    )


def _render_plot(output: Path | None, df: pd.DataFrame) -> None:
    setup_latex_style()

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.0))
    _render_status_matrix(axes[0], df)
    _render_valid_table(axes[1], df)

    fig.suptitle("Qwen3-32B Workload Tuning", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 1.0))

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
