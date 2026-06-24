#!/usr/bin/env python3
"""Grid-search TP/PP/MBS/VPP for Qwen3-32B dense on Snellius 4x4 H100.

Usage (from repo root):
    uv run python experiments/usecase_workload_tuning/grid_search.py
    uv run python experiments/usecase_workload_tuning/grid_search.py --dry-run
    uv run python experiments/usecase_workload_tuning/grid_search.py --trace-only
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import yaml

_BASE_DIR = Path(__file__).parent
_SCENARIOS_DIR = _BASE_DIR / "scenarios"
_BASE_WORKLOAD = _SCENARIOS_DIR / "base_workload.yaml"
_BASE_SCENARIO = _SCENARIOS_DIR / "base_scenario.yaml"
_NUM_LAYERS = 64
_NUM_GPUS = 16
_GBS = 1024


def _divisors(n: int) -> list[int]:
    divs = [i for i in range(1, n + 1) if n % i == 0]
    return sorted(divs)


def _mbs_values() -> list[int]:
    vals = []
    mbs = 1
    while mbs <= _GBS:
        vals.append(mbs)
        mbs *= 2
    return vals


def _load_base_workload() -> dict[str, Any]:
    with open(_BASE_WORKLOAD) as f:
        return yaml.safe_load(f)


def _make_config_dir(tp: int, pp: int, mbs: int, vpp: int | None) -> Path:
    name = f"tp{tp}_pp{pp}_mbs{mbs}"
    if vpp is not None:
        name = f"{name}_vpp{vpp}"
    path = _SCENARIOS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_workload(
    path: Path, cfg: dict[str, Any], tp: int, pp: int, mbs: int, vpp: int | None
) -> None:
    cfg = dict(cfg)
    cfg["config"] = dict(cfg.get("config", {}))
    cfg["config"]["tensor-model-parallel-size"] = tp
    cfg["config"]["pipeline-model-parallel-size"] = pp
    cfg["config"]["micro-batch-size"] = mbs
    if vpp is not None:
        cfg["config"]["num-virtual-stages-per-pipeline-rank"] = vpp
    else:
        cfg["config"].pop("num-virtual-stages-per-pipeline-rank", None)
        cfg["config"].pop("num-layers-per-virtual-pipeline-stage", None)
    with open(path / "workload.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def _write_scenario(path: Path) -> None:
    scenario = {
        "datacenter": {"num_nodes": 4, "node": "templates/node/snellius-h100-4g.yaml"},
        "workload": str(path / "workload.yaml"),
    }
    with open(path / "scenario.yaml", "w") as f:
        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)


def _trace_dir(path: Path) -> Path:
    return path / "traces"


def _run_trace(path: Path) -> bool:
    trace_dir = _trace_dir(path)
    trace_dir.mkdir(parents=True, exist_ok=True)
    oom_file = trace_dir / ".OOM"
    if oom_file.exists():
        return False
    if any(trace_dir.glob("trace_rank_*.json")):
        return True
    scenario = path / "scenario.yaml"
    cmd = [
        "uv",
        "run",
        "simulon",
        "trace",
        "generate",
        str(scenario),
        "--output-dir",
        str(trace_dir),
        "--force-regenerate",
    ]
    print(f"Tracing {path.name} ...")  # noqa: T201
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        if "CUDA out of memory" in (exc.stderr or "") or "OutOfMemoryError" in (exc.stderr or ""):
            oom_file.write_text(exc.stderr or "")
            print(f"  OOM: wrote {oom_file}")  # noqa: T201
            return False
        oom_file.write_text(exc.stderr or f"exit code {exc.returncode}")
        print(f"  Trace failed: {path.name} (exit {exc.returncode})")  # noqa: T201
        return False


def _run_simulate(path: Path) -> dict[str, Any] | None:
    scenario = path / "scenario.yaml"
    name = path.name
    cmd = [
        "uv",
        "run",
        "simulon",
        "simulate",
        str(scenario),
        "--skip-if-tracked",
    ]
    env = os.environ.copy()
    env["WANDB_RUN_NAME"] = f"usecase-workload-tuning-{name}"
    print(f"Simulating {name} ...")  # noqa: T201
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        metrics: dict[str, Any] = {}
        for line in result.stdout.splitlines():
            if "tokens/s" in line and "(" in line:
                with contextlib.suppress(Exception):
                    metrics["throughput_tps"] = float(line.split("(")[-1].split()[0].replace(",", ""))
            if "MFU:" in line:
                with contextlib.suppress(Exception):
                    metrics["mfu_pct"] = float(line.split("MFU:")[-1].replace("%", "").strip())
            if "wall time" in line and "ms" in line:
                with contextlib.suppress(Exception):
                    metrics["iteration_time_ms"] = float(line.split("ms")[0].split()[-1])
        return metrics
    except subprocess.CalledProcessError as exc:
        print(f"  Simulation failed: {name} (exit {exc.returncode})")  # noqa: T201
        return None



def _grid() -> list[tuple[int, int, int, int | None]]:
    combos: list[tuple[int, int, int, int | None]] = []
    tps = [1, 2, 4]
    pps = [1, 2, 4]
    mbss = _mbs_values()
    for tp in tps:
        for pp in pps:
            if (_NUM_GPUS // tp // pp) == 0:
                continue
            layers_per_stage = _NUM_LAYERS // pp
            vpps = [None] if pp == 1 else _divisors(layers_per_stage)
            for mbs in mbss:
                for vpp in vpps:
                    combos.append((tp, pp, mbs, vpp))
    return combos


def _generate_configs() -> list[Path]:
    base = _load_base_workload()
    paths: list[Path] = []
    for tp, pp, mbs, vpp in _grid():
        path = _make_config_dir(tp, pp, mbs, vpp)
        _write_workload(path, base, tp, pp, mbs, vpp)
        _write_scenario(path)
        paths.append(path)
    return paths


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print grid without tracing/simulating.",
    )
    parser.add_argument(
        "--trace-only", action="store_true", help="Trace all configs but do not simulate."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all generated config subdirectories before running.",
    )
    args = parser.parse_args()

    if args.clean:
        for path in _SCENARIOS_DIR.glob("tp*_pp*_mbs*"):
            if path.is_dir():
                shutil.rmtree(path)

    paths = _generate_configs()
    if args.dry_run:
        for _path in paths:
            pass
        return

    results: list[dict[str, Any]] = []
    for path in paths:
        oom = not _run_trace(path)
        if oom:
            results.append({"name": path.name, "oom": True})
            continue
        if args.trace_only:
            results.append({"name": path.name, "oom": False, "simulated": False})
            continue
        metrics = _run_simulate(path)
        row: dict[str, Any] = {"name": path.name, "oom": False, "simulated": metrics is not None}
        if metrics:
            row.update(metrics)
        results.append(row)

    csv_path = _BASE_DIR / "results.csv"
    import csv

    if results:
        fieldnames = sorted({k for r in results for k in r})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
