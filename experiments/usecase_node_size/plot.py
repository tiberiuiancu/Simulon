#!/usr/bin/env python3
"""Plot MFU for node-size experiments from Weights \u0026 Biases.

Uses the same recursive `.tracking.env` loader used by `simulon simulate` so that
the WANDB_* environment variables are available (project, entity, run_name, etc).

Usage (from repo root):
    uv run python experiments/usecase_node_size/plot.py
    uv run python experiments/usecase_node_size/plot.py --output results.pdf
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
import yaml


def _load_tracking_env_file(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ[key] = val


def _load_cascading_tracking_env(scenario_path: Path) -> None:
    ENV_FILE_NAME = ".tracking.env"
    cwd = Path.cwd().resolve()
    scenario = scenario_path.resolve()

    try:
        rel = scenario.parent.relative_to(cwd)
    except ValueError:
        dirs = [scenario.parent]
    else:
        dirs = [cwd]
        cur = cwd
        for part in rel.parts:
            cur = cur / part
            dirs.append(cur)

    for d in dirs:
        env_file = d / ENV_FILE_NAME
        if env_file.is_file():
            _load_tracking_env_file(env_file)


def _find_scenarios(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("scenario*.yaml"))


def _model_from_scenario(path: Path) -> str:
    return path.parent.name


def _node_size_from_scenario(path: Path) -> int:
    return int(path.stem.replace("scenario", ""))


def _load_scenario_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _fetch_run(workload_hash: str) -> dict[str, Any] | None:
    project = os.environ.get("WANDB_PROJECT", "simulon")
    entity = os.environ.get("WANDB_ENTITY")
    run_name = os.environ.get("WANDB_RUN_NAME")
    api = wandb.Api()
    filters: dict[str, object] = {"state": "finished"}
    if run_name:
        filters["group"] = run_name
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
    for run in runs:
        if run.config.get("workload_hash") == workload_hash:
            return {"mfu_pct": run.summary.get("mfu_pct"), "run": run}
    return None


def plot_mfu_from_wandb(output: Path | None, base_dir: Path) -> None:
    scenarios = _find_scenarios(base_dir)
    if not scenarios:
        print("No scenario files found.", file=sys.stderr)
        sys.exit(1)

    _load_cascading_tracking_env(scenarios[0])

    records: list[dict[str, float | str]] = []

    for sc_path in scenarios:
        model = _model_from_scenario(sc_path)
        node_size = _node_size_from_scenario(sc_path)
        wl_hash = _compute_workload_hash(sc_path)
        run_data = _fetch_run(wl_hash)

        if run_data is None:
            print(f"  WARN: no W&B run for {sc_path}", file=sys.stderr)
            continue

        mfu = run_data["mfu_pct"]
        if mfu is None:
            print(f"  WARN: no MFU metric for {sc_path}", file=sys.stderr)
            continue

        records.append({
            "model": model,
            "node_size": f"{node_size} GPU/node",
            "mfu_pct": float(mfu),
        })

    if not records:
        print("No wandb data found for any scenario. Exiting.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    order = sorted(df["node_size"].unique())
    sns.barplot(
        data=df,
        x="model",
        y="mfu_pct",
        hue="node_size",
        order=sorted(df["model"].unique()),
        hue_order=order,
        palette="deep",
        ax=ax,
    )

    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("")
    ax.set_title("MFU by Node Size", fontsize=14, fontweight="bold")
    ax.legend(title="Node config", loc="upper right")
    sns.despine(fig=fig, top=True, right=True)
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
    args = parser.parse_args()
    plot_mfu_from_wandb(args.output, args.base_dir)


if __name__ == "__main__":
    main()
