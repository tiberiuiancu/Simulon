#!/usr/bin/env python3
"""Plot real vs simulated metrics for validate_bridge experiments.

Usage (from repo root):
    uv run python experiments/validate_bridge/plot.py
    uv run python experiments/validate_bridge/plot.py --output results.pdf
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from simulon.tracking import get_trackers

sys.path.insert(0, str(Path(__file__).parent.parent))
from _plot_utils import label_for_model, make_figure, plot_metric_panel, setup_latex_style


def _find_models(base_dir: Path) -> list[Path]:
    models = []
    for item in sorted(base_dir.iterdir()):
        if (
            item.is_dir()
            and (item / "reference.yaml").exists()
            and (item / "scenario.yaml").exists()
        ):
            models.append(item)
    return models


def _read_reference(ref_path: Path) -> dict:
    with open(ref_path) as f:
        return yaml.safe_load(f) or {}


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


def _records_to_long(
    rows: list[dict[str, float | str | None]],
    metric_key: str,
    metric_label: str,
    model_label: str | None = None,
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for row in rows:
        if row is None:
            continue
        model = model_label if model_label is not None else row["model"]
        real_val = row.get("real")
        sim_val = row.get("simulated")
        overlap_val = row.get("simulated_overlap")
        if real_val is not None:
            records.append(
                {"model": model, "metric": metric_label, "source": "Real", "value": float(real_val)}
            )
        if sim_val is not None:
            records.append(
                {
                    "model": model,
                    "metric": metric_label,
                    "source": "Simulated",
                    "value": float(sim_val),
                }
            )
        if overlap_val is not None:
            records.append(
                {
                    "model": model,
                    "metric": metric_label,
                    "source": "Simulated (overlap)",
                    "value": float(overlap_val),
                }
            )
    return records


def _group_key(model: str) -> str:
    return "qwen3-30b" if model == "qwen3-30b-overlap" else model


def plot_real_vs_simulated(output: Path | None, base_dir: Path, use_csv: bool = False) -> None:
    csv_path = base_dir / "results.csv"
    metrics = [("per_gpu_tps", "Throughput", "tokens/s/GPU")]

    if use_csv:
        df = _load_csv(csv_path)
        if df is None:
            print(f"--use-csv requested but {csv_path} not found.", file=sys.stderr)
            sys.exit(1)
    else:
        models = _find_models(base_dir)
        if not models:
            print("No model sub-folders found.", file=sys.stderr)
            sys.exit(1)

        trackers = get_trackers(models[0] / "scenario.yaml")

        raw_rows: list[dict[str, float | str | None]] = []
        for model_dir in models:
            model = model_dir.name
            ref = _read_reference(model_dir / "reference.yaml")

            sim_metrics = None
            if trackers:
                for tracker in trackers:
                    sim_metrics = tracker.pull_metrics(run_name=f"validate-bridge-{model}")
                    if sim_metrics is not None:
                        break

            row: dict[str, float | str | None] = {"model": model}
            for key, _label, _unit in metrics:
                row[f"real_{key}"] = ref.get(key)
                row[f"sim_{key}"] = sim_metrics.get(key) if sim_metrics else None
            raw_rows.append(row)

        grouped: dict[str, dict[str, float | str | None]] = {}
        for row in raw_rows:
            model = str(row["model"])
            group = _group_key(model)
            if group not in grouped:
                grouped[group] = {"model": group}
            for key, _label, _unit in metrics:
                if model.endswith("-overlap"):
                    grouped[group][f"simulated_overlap_{key}"] = row[f"sim_{key}"]
                else:
                    grouped[group][f"real_{key}"] = row.get(f"real_{key}")
                    grouped[group][f"sim_{key}"] = row[f"sim_{key}"]

        complete_rows: list[dict[str, float | str | None]] = []
        for group, row in grouped.items():
            has_data = any(
                row.get(f"real_{key}") is not None
                or row.get(f"sim_{key}") is not None
                or row.get(f"simulated_overlap_{key}") is not None
                for key, _label, _unit in metrics
            )
            if has_data:
                complete_rows.append(row)
            else:
                print(f"Skipping {group}: no real or simulated results.", file=sys.stderr)

        records: list[dict[str, float | str]] = []
        for key, label, _unit in metrics:
            records.extend(
                _records_to_long(
                    [
                        {
                            "model": r["model"],
                            "real": r.get(f"real_{key}"),
                            "simulated": r.get(f"sim_{key}"),
                            "simulated_overlap": r.get(f"simulated_overlap_{key}"),
                        }
                        for r in complete_rows
                    ],
                    key,
                    label,
                )
            )

        if records:
            df = pd.DataFrame(records)
            df["model"] = df["model"].map(label_for_model)
            _save_csv(df, csv_path)
        else:
            print("No complete wandb data found; falling back to local CSV.", file=sys.stderr)
            df = _load_csv(csv_path)
            if df is None:
                print("No cached results.csv available. Exiting.", file=sys.stderr)
                sys.exit(1)

    setup_latex_style()

    metric_labels = df["metric"].unique()
    fig, axes = make_figure(
        "Megatron-Bridge Training Validation", width_in=3.5, n_panels=len(metric_labels)
    )

    for ax, metric_label in zip(axes, metric_labels, strict=False):
        sub = df[df["metric"] == metric_label]
        unit = next((unit for key, lbl, unit in metrics if lbl == metric_label), "")
        ylabel = f"{metric_label} ({unit})" if unit else metric_label
        plot_metric_panel(ax, sub, metric_label, ylabel)

    fig.tight_layout(rect=[0, 0, 1, 1.02])

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
        help="Directory containing model sub-folders with reference and scenario YAMLs.",
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Plot from the local results.csv instead of pulling from wandb.",
    )
    args = parser.parse_args()
    plot_real_vs_simulated(args.output, args.base_dir, use_csv=args.use_csv)


if __name__ == "__main__":
    main()
