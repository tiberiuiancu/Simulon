#!/usr/bin/env python3
"""Plot real vs simulated metrics for validate_e2e experiments.

Usage (from repo root):
    uv run python experiments/validate_e2e/plot.py
    uv run python experiments/validate_e2e/plot.py --output results.pdf
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from simulon.tracking import get_trackers


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
    rows: list[dict[str, float | str]], metric_key: str, metric_label: str
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for row in rows:
        model = row["model"]
        real_val = row.get("real")
        sim_val = row.get("simulated")
        if real_val is None or sim_val is None:
            continue
        records.append(
            {"model": model, "metric": metric_label, "source": "Real", "value": float(real_val)}
        )
        records.append(
            {"model": model, "metric": metric_label, "source": "Simulated", "value": float(sim_val)}
        )
    return records


def plot_real_vs_simulated(output: Path | None, base_dir: Path, use_csv: bool = False) -> None:
    csv_path = base_dir / "results.csv"
    metrics = [("mfu_pct", "MFU (%)")]

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

        rows: list[dict[str, float | str | None]] = []
        for model_dir in models:
            model = model_dir.name
            ref = _read_reference(model_dir / "reference.yaml")

            sim_metrics = None
            if trackers:
                for tracker in trackers:
                    sim_metrics = tracker.pull_metrics(run_name_prefix=f"validate-e2e-{model}")
                    if sim_metrics is not None:
                        break

            row: dict[str, float | str | None] = {"model": model}
            for key, _label in metrics:
                row[f"real_{key}"] = ref.get(key)
                row[f"sim_{key}"] = sim_metrics.get(key) if sim_metrics else None
            rows.append(row)

        def _has_both_results(row: dict[str, float | str | None]) -> bool:
            return all(
                row.get(f"real_{key}") is not None and row.get(f"sim_{key}") is not None
                for key, _ in metrics
            )

        complete_rows = []
        for row in rows:
            if _has_both_results(row):
                complete_rows.append(row)
            else:
                print(
                    f"Skipping {row['model']}: missing real or simulated results.", file=sys.stderr
                )

        records: list[dict[str, float | str]] = []
        for key, label in metrics:
            records.extend(
                _records_to_long(
                    [
                        {
                            "model": r["model"],
                            "real": r[f"real_{key}"],
                            "simulated": r[f"sim_{key}"],
                        }
                        for r in complete_rows
                    ],
                    key,
                    label,
                )
            )

        if records:
            df = pd.DataFrame(records)
            _save_csv(df, csv_path)
        else:
            print("No complete wandb data found; falling back to local CSV.", file=sys.stderr)
            df = _load_csv(csv_path)
            if df is None:
                print("No cached results.csv available. Exiting.", file=sys.stderr)
                sys.exit(1)
    sns.set_theme(style="whitegrid")

    metric_labels = df["metric"].unique()
    n_metrics = len(metric_labels)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics + 1, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric_label in zip(axes, metric_labels, strict=False):
        sub = df[df["metric"] == metric_label]
        sns.barplot(
            data=sub,
            x="model",
            y="value",
            hue="source",
            hue_order=["Real", "Simulated"],
            palette={"Real": "#4c72b0", "Simulated": "#dd8452"},
            ax=ax,
        )

        for model_name in sub["model"].unique():
            real_val = sub[(sub["model"] == model_name) & (sub["source"] == "Real")]["value"]
            sim_val = sub[(sub["model"] == model_name) & (sub["source"] == "Simulated")]["value"]
            if real_val.empty or sim_val.empty:
                continue
            real = float(real_val.iloc[0])
            sim = float(sim_val.iloc[0])
            if real == 0:
                continue
            pct = (sim - real) / real * 100
            ax.text(
                model_name, sim, f"{pct:+.1f}%", ha="center", va="bottom", fontsize=8, color="red"
            )

        ax.set_ylabel(metric_label)
        ax.set_xlabel("")
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="", loc="upper right")
        sns.despine(ax=ax, top=True, right=True)

    fig.suptitle("Real vs Simulated — validate_e2e", fontsize=14, fontweight="bold", y=1.02)
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
