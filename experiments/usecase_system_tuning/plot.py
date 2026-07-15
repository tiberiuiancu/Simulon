#!/usr/bin/env python3
"""System-tuning summary table: best MFU and energy per hardware config.

Pulls all simulation runs from W&B (or results.csv), finds the best MFU per
(node_size, link_bw) per model, computes a blended relative performance
score and a performance-per-dollar metric, and prints a table.

Usage (from repo root):
    uv run python experiments/usecase_system_tuning/plot.py
    uv run python experiments/usecase_system_tuning/plot.py --use-csv
"""

from __future__ import annotations

import csv
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from _plot_utils import setup_latex_style  # noqa: F401

_NODE_SIZES = [4, 8, 64]
_LINK_BWS = [100, 200, 400, 800]
_MODELS = ["gptoss-120b", "llama3-70b", "deepseek-v3"]

_NODE_COST_USD: dict[int, float] = {4: 120_000, 8: 250_000, 64: 3_000_000}
_NIC_COST_USD: dict[int, float] = {100: 1_000, 200: 2_000, 400: 4_000, 800: 10_000}
_CABLING_COST_USD: dict[int, float] = {
    100: 2_000_000,
    200: 3_000_000,
    400: 5_000_000,
    800: 10_000_000,
}
_TOTAL_GPUS = 1024

_CSV_PATH = Path(__file__).parent / "results.csv"

_RUN_NAME_RE = re.compile(
    r"^system-tuning-(?P<model>gptoss-120b|llama3-70b|deepseek-v3)-(?P<config>.+?)-node(?P<node_size>\d+)-bw(?P<link_bw>\d+)$"
)


def _load_csv() -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    if not _CSV_PATH.exists():
        return {}
    by_model: dict[str, dict[tuple[int, int], dict[str, float]]] = {m: {} for m in _MODELS}
    with open(_CSV_PATH) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model not in _MODELS:
                continue
            if row.get("oom") or row.get("error") or row.get("invalid"):
                continue
            key = (int(row["node_size"]), int(row["link_bw"]))
            mfu = float(row.get("mfu_pct", 0))
            energy_wh = float(row.get("energy_wh", 0)) if row.get("energy_wh") else 0
            co2eq_g = float(row.get("co2eq_g", 0)) if row.get("co2eq_g") else 0
            prev = by_model[model].get(key, {})
            if mfu > prev.get("mfu_pct", 0):
                by_model[model][key] = {"mfu_pct": mfu, "energy_wh": energy_wh, "co2eq_g": co2eq_g}
    return by_model


def _pull_wandb() -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    import wandb

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "simulon")
    api = wandb.Api()
    filters: dict[str, object] = {
        "state": "finished",
        "display_name": {"$regex": r"^system-tuning-"},
    }
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)

    by_model: dict[str, dict[tuple[int, int], dict[str, float]]] = {m: {} for m in _MODELS}
    all_rows: list[dict[str, object]] = []

    for run in runs:
        m = _RUN_NAME_RE.match(run.display_name)
        if not m:
            continue
        model = m.group("model")
        if model not in _MODELS:
            continue
        node_size = int(m.group("node_size"))
        link_bw = int(m.group("link_bw"))
        if node_size not in _NODE_SIZES or link_bw not in _LINK_BWS:
            continue

        summary = dict(run.summary)
        mfu = float(summary.get("mfu_pct", 0) or 0)
        if mfu <= 0:
            continue
        energy_wh = float(summary.get("energy_wh", 0) or 0)
        co2eq_g = float(summary.get("co2eq_g", 0) or 0)

        key = (node_size, link_bw)
        prev = by_model[model].get(key, {})
        row = {
            "model": model,
            "name": m.group("config"),
            "node_size": node_size,
            "link_bw": link_bw,
            "mfu_pct": mfu,
            "energy_wh": energy_wh,
            "co2eq_g": co2eq_g,
        }
        all_rows.append(row)
        if mfu > prev.get("mfu_pct", 0):
            by_model[model][key] = {"mfu_pct": mfu, "energy_wh": energy_wh, "co2eq_g": co2eq_g}

    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r})
        with open(_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {len(all_rows)} runs to {_CSV_PATH}")

    return by_model


def _print_table(results: dict[str, dict[tuple[int, int], dict[str, float]]]) -> None:
    max_mfu: dict[str, float] = {}
    for model in _MODELS:
        vals = [v["mfu_pct"] for v in results.get(model, {}).values() if v.get("mfu_pct", 0) > 0]
        max_mfu[model] = max(vals) if vals else 1.0

    headers = [
        "Node size",
        "Link BW",
        "System cost ($M)",
        "MFU GPT-OSS (%)",
        "MFU Llama (%)",
        "MFU DeepSeek (%)",
        "Rel. perf.",
        "Energy/iter (Wh)",
        "CO2/iter (g)",
        "Perf/$",
    ]
    rows: list[list[str]] = []
    raw_ppds: list[float] = []
    raw_rel_perfs: list[float] = []
    system_costs: list[float] = []

    for ns in _NODE_SIZES:
        for bw in _LINK_BWS:
            key = (ns, bw)
            mfu_gptoss = results.get("gptoss-120b", {}).get(key, {}).get("mfu_pct", 0.0)
            mfu_llama = results.get("llama3-70b", {}).get(key, {}).get("mfu_pct", 0.0)
            mfu_deepseek = results.get("deepseek-v3", {}).get(key, {}).get("mfu_pct", 0.0)

            rel_gptoss = mfu_gptoss / max_mfu["gptoss-120b"] if max_mfu["gptoss-120b"] > 0 else 0.0
            rel_llama = mfu_llama / max_mfu["llama3-70b"] if max_mfu["llama3-70b"] > 0 else 0.0
            rel_deepseek = (
                mfu_deepseek / max_mfu["deepseek-v3"] if max_mfu["deepseek-v3"] > 0 else 0.0
            )
            rel_perf = rel_gptoss + rel_llama + rel_deepseek
            raw_rel_perfs.append(rel_perf)

            num_nodes = _TOTAL_GPUS // ns
            system_cost = (
                num_nodes * _NODE_COST_USD[ns]
                + _TOTAL_GPUS * _NIC_COST_USD[bw]
                + _CABLING_COST_USD[bw]
            )
            system_costs.append(system_cost)

            best_energy = 0.0
            best_co2 = 0.0
            for model in _MODELS:
                data = results.get(model, {}).get(key, {})
                if data.get("energy_wh", 0) > best_energy:
                    best_energy = data["energy_wh"]
                    best_co2 = data.get("co2eq_g", 0.0)

            rows.append(
                [
                    f"{ns} GPU",
                    f"{bw} Gbps",
                    f"{system_cost / 1e6:.1f}M",
                    f"{mfu_gptoss:.1f}" if mfu_gptoss > 0 else "-",
                    f"{mfu_llama:.1f}" if mfu_llama > 0 else "-",
                    f"{mfu_deepseek:.1f}" if mfu_deepseek > 0 else "-",
                    "",
                    f"{best_energy:.0f}" if best_energy > 0 else "-",
                    f"{best_co2:.0f}" if best_co2 > 0 else "-",
                    "",
                ]
            )

    max_rel_perf = max(raw_rel_perfs) if raw_rel_perfs else 1.0
    for row, rel_perf, cost in zip(rows, raw_rel_perfs, system_costs, strict=False):
        norm_rel_perf = rel_perf / max_rel_perf if max_rel_perf > 0 else 0.0
        row[6] = f"{norm_rel_perf:.3f}"
        raw_ppd = norm_rel_perf / cost if cost > 0 else 0.0
        raw_ppds.append(raw_ppd)

    max_ppd = max(raw_ppds) if raw_ppds else 1.0
    for row, raw in zip(rows, raw_ppds, strict=False):
        row[-1] = f"{raw / max_ppd:.2f}" if max_ppd > 0 else "0.00"

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def _fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    print(_fmt_row(headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(_fmt_row(row))


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Use the local results.csv instead of pulling from W&B.",
    )
    args = parser.parse_args()

    if args.use_csv:
        results = _load_csv()
        if not results:
            print(f"No results.csv found at {_CSV_PATH}.", file=sys.stderr)
            sys.exit(1)
    else:
        results = _pull_wandb()
        if not any(results.values()):
            print(
                "No runs found on W&B. Set WANDB_API_KEY/WANDB_ENTITY or use --use-csv.",
                file=sys.stderr,
            )
            sys.exit(1)

    _print_table(results)


if __name__ == "__main__":
    main()
