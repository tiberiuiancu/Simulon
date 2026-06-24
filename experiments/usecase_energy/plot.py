"""Energy-usecase table: per-token and per-pretrain energy/CO2 for LLMs.

Usage (from repo root):
    uv run python experiments/usecase_energy/plot.py
    uv run python experiments/usecase_energy/plot.py --output energy.pdf
    uv run python experiments/usecase_energy/plot.py --use-csv
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from _plot_utils import label_for_model, setup_latex_style

from simulon.tracking import get_trackers

_PRETRAIN_TOKENS: dict[str, int] = {
    "deepseekv3": 14_800_000_000_000,
    "llama3-70b": 15_000_000_000_000,
    "gptoss-120b": 1_000_000_000_000,
}

_CSV_COLS = [
    "model",
    "energy_per_token_wh",
    "co2_per_token_g",
    "pretrain_tokens",
    "energy_per_pretrain_wh",
    "co2_per_pretrain_g",
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
    print(f"Saved results to {csv_path}")  # noqa: T201


def _parse_run_name(display_name: str) -> str | None:
    prefix = "node-size-"
    if not display_name.startswith(prefix):
        return None
    body = display_name[len(prefix) :]
    parts = body.split("-")
    if len(parts) < 2:
        return None
    size_idx = -2 if parts[-1] == "overlap" else -1
    if not parts[size_idx].startswith("node"):
        return None
    model = "-".join(parts[:size_idx])
    return model


def _pull_energy_metrics(base_dir: Path) -> pd.DataFrame:
    csv_path = base_dir / "results.csv"
    cached_df = _load_csv(csv_path)
    records: list[dict[str, Any]] = []

    trackers = get_trackers(base_dir)
    if trackers:
        for tracker in trackers:
            runs = tracker.fetch_runs(prefix="node-size-")
            for run in runs:
                name = run["display_name"]
                model = _parse_run_name(name)
                if model is None:
                    continue
                summary = run["summary"]
                energy_wh = summary.get("energy_wh")
                co2eq_g = summary.get("co2eq_g")
                throughput_tps = summary.get("throughput_tps")
                if energy_wh is None or throughput_tps is None or throughput_tps <= 0:
                    continue
                if "node8" not in name:
                    continue
                records.append(
                    {
                        "model": model,
                        "energy_per_token_wh": float(energy_wh) / float(throughput_tps),
                        "co2_per_token_g": float(co2eq_g or 0.0) / float(throughput_tps),
                    }
                )

    if records:
        fresh_df = pd.DataFrame(records)
        fresh_df = fresh_df.groupby("model", as_index=False).mean(numeric_only=True)
        df = _merge_frames(cached_df, fresh_df) if cached_df is not None else fresh_df
    elif cached_df is not None:
        print("No wandb data found; falling back to local CSV.", file=sys.stderr)  # noqa: T201
        df = cached_df
    else:
        print("No wandb data and no cached results.csv available. Exiting.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    _save_csv(df, csv_path)
    return df


def _merge_frames(cached_df: pd.DataFrame, fresh_df: pd.DataFrame) -> pd.DataFrame:
    df = cached_df.copy()
    for _, row in fresh_df.iterrows():
        mask = df["model"] == row["model"]
        if mask.any():
            for col in ("energy_per_token_wh", "co2_per_token_g"):
                df.loc[mask, col] = row[col]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def _compute_pretrain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pretrain_tokens"] = df["model"].map(_PRETRAIN_TOKENS)
    df["energy_per_pretrain_wh"] = df["energy_per_token_wh"] * df["pretrain_tokens"]
    df["co2_per_pretrain_g"] = df["co2_per_token_g"] * df["pretrain_tokens"]
    return df


def _format_cell(value: float, unit: str) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f} T{unit}"
    if value >= 1e9:
        return f"{value / 1e9:.2f} G{unit}"
    if value >= 1e6:
        return f"{value / 1e6:.2f} M{unit}"
    if value >= 1e3:
        return f"{value / 1e3:.2f} k{unit}"
    return f"{value:.3f} {unit}"


def _render_table(output: Path | None, df: pd.DataFrame) -> None:
    setup_latex_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.axis("off")

    labels = [str(label_for_model(model)) for model in df["model"]]
    rows = [
        [
            label,
            _format_cell(float(row["energy_per_token_wh"]), "Wh"),
            _format_cell(float(row["co2_per_token_g"]), "g"),
            _format_cell(float(row["pretrain_tokens"]), ""),
            _format_cell(float(row["energy_per_pretrain_wh"]), "Wh"),
            _format_cell(float(row["co2_per_pretrain_g"]), "g"),
        ]
        for label, (_, row) in zip(labels, df.iterrows(), strict=False)
    ]

    columns = [
        "Model",
        "Energy / token",
        "CO2 / token",
        "Pretrain tokens",
        "Energy / pretrain",
        "CO2 / pretrain",
    ]

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

    ax.set_title("Energy and CO2 estimates by model", fontweight="bold", fontsize=10)
    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {output}")  # noqa: T201
    else:
        plt.show()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write table to file (PDF/PNG/SVG). Omit to display.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing the usecase_energy results.csv.",
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Read from local results.csv instead of pulling from wandb.",
    )
    args = parser.parse_args()

    if args.use_csv:
        df = _load_csv(args.base_dir / "results.csv")
        if df is None:
            print("--use-csv requested but results.csv not found.", file=sys.stderr)  # noqa: T201
            sys.exit(1)
    else:
        df = _pull_energy_metrics(args.base_dir)

    df = _compute_pretrain(df)

    missing_cols = sorted(set(_CSV_COLS) - set(df.columns))
    if missing_cols:
        print(f"results.csv is missing required columns: {missing_cols}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    _render_table(args.output, df)


if __name__ == "__main__":
    main()
