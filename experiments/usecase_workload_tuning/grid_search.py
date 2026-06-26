#!/usr/bin/env python3
"""Grid-search TP/PP/MBS/VPP for Qwen3-32B dense on Snellius 4x4 H100.

Usage (from repo root):
    uv run python experiments/usecase_workload_tuning/grid_search.py
    uv run python experiments/usecase_workload_tuning/grid_search.py --dry-run
    uv run python experiments/usecase_workload_tuning/grid_search.py --trace-only
"""

from __future__ import annotations

import contextlib
import csv
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
_NUM_LAYERS = 64
_NUM_GPUS = 16
_GBS = 1024


def _divisors(n: int) -> list[int]:
    return sorted(i for i in range(1, n + 1) if n % i == 0)


def _parse_config_name(name: str) -> tuple[int, int, int, int | None]:
    parts = name.split("_")
    tp = int(parts[0].replace("tp", ""))
    pp = int(parts[1].replace("pp", ""))
    mbs = int(parts[2].replace("mbs", ""))
    vpp: int | None = None
    for part in parts[3:]:
        if part.startswith("vpp"):
            vpp = int(part.replace("vpp", ""))
            break
    return tp, pp, mbs, vpp


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


def _default_trace_dir(path: Path) -> Path:
    from simulon.config.resolve import resolve_gpu_spec, resolve_workload, workload_hash
    from simulon.config.scenario import ScenarioConfig

    try:
        sc = ScenarioConfig.from_yaml(str(path / "scenario.yaml"))
        gpu_spec = resolve_gpu_spec(sc.datacenter)
        gpu_name = (gpu_spec.name or "default").lower().replace(" ", "-")
    except Exception:
        gpu_name = "default"
    try:
        wl = resolve_workload(str(path / "workload.yaml"))
        h = workload_hash(wl)
    except Exception:
        h = "unknown"
    return Path("templates/gpu") / gpu_name / "traces" / h


def _workload_hash(path: Path) -> str:
    from simulon.config.resolve import resolve_workload, workload_hash

    try:
        wl = resolve_workload(str(path / "workload.yaml"))
        return workload_hash(wl)[:12]
    except Exception:
        return "unknown"


def _trace_status(path: Path) -> str:
    trace_dir = _default_trace_dir(path)
    if (trace_dir / ".INVALID").exists():
        return "invalid"
    if (trace_dir / ".OOM").exists():
        return "OOM"
    if any(trace_dir.glob("trace_rank_*.json")):
        return "traced"
    return "pending"


def _write_invalid_marker(path: Path, reason: str) -> None:
    trace_dir = _default_trace_dir(path)
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / ".INVALID").write_text(reason)


def _run_trace(path: Path) -> tuple[str, str]:
    trace_dir = _default_trace_dir(path)
    trace_dir.mkdir(parents=True, exist_ok=True)
    oom_file = trace_dir / ".OOM"
    if oom_file.exists():
        return "OOM", "pre-existing .OOM marker"
    if any(trace_dir.glob("trace_rank_*.json")):
        return "traced", "trace files already present"
    scenario = path / "scenario.yaml"
    cmd = ["simulon", "trace", "generate", str(scenario), "--force-regenerate"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return "traced", ""
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").lower()
        stdout = (exc.stdout or "").lower()
        combined = stdout + "\n" + stderr
        oom_keywords = (
            "out of memory",
            "out-of-memory",
            "cuda oom",
            "runtimeerror: cuda",
            "torch.cuda.outofmemoryerror",
        )
        is_oom = oom_file.exists() or any(kw in combined for kw in oom_keywords)
        if is_oom:
            if not oom_file.exists():
                with contextlib.suppress(Exception):
                    oom_file.write_text(f"returncode={exc.returncode}\n{exc.stderr or ''}")
            return "OOM", f"Megatron OOM (return code {exc.returncode})"

        return (
            "error",
            f"non-OOM error (return code {exc.returncode}): {(exc.stderr or '').strip()[:200]}",
        )


def _run_simulate(path: Path) -> dict[str, Any] | None:
    scenario = path / "scenario.yaml"
    name = path.name
    cmd = ["simulon", "simulate", str(scenario), "--skip-if-tracked"]
    env = os.environ.copy()
    env["WANDB_RUN_NAME"] = f"usecase-workload-tuning-{name}"
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        metrics: dict[str, Any] = {}
        for line in result.stdout.splitlines():
            if "tokens/s" in line and "(" in line:
                with contextlib.suppress(Exception):
                    metrics["throughput_tps"] = float(
                        line.split("(")[-1].split()[0].replace(",", "")
                    )
            if "MFU:" in line:
                with contextlib.suppress(Exception):
                    metrics["mfu_pct"] = float(line.split("MFU:")[-1].replace("%", "").strip())
            if "wall time" in line and "ms" in line:
                with contextlib.suppress(Exception):
                    metrics["iteration_time_ms"] = float(line.split("ms")[0].split()[-1])
        return metrics
    except subprocess.CalledProcessError:
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


def _is_valid_config(tp: int, pp: int, mbs: int) -> tuple[bool, str]:
    dp = _NUM_GPUS // tp // pp
    if _GBS % (mbs * dp) != 0:
        return False, f"global_batch_size={_GBS} not divisible by mbs*dp={mbs * dp}"
    return True, ""


def _generate_configs() -> list[Path]:
    base = _load_base_workload()
    paths: list[Path] = []
    for tp, pp, mbs, vpp in _grid():
        path = _make_config_dir(tp, pp, mbs, vpp)
        _write_workload(path, base, tp, pp, mbs, vpp)
        _write_scenario(path)
        paths.append(path)
    return paths


def _sweep(paths: list[Path], trace_only: bool) -> list[dict[str, Any]]:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    console = Console()
    results: list[dict[str, Any]] = []
    oom_keys: set[tuple[int, int, int | None]] = set()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Workload tuning sweep", total=len(paths))
        for path in paths:
            name = path.name
            tp, pp, mbs, vpp = _parse_config_name(name)
            key = (tp, pp, vpp)
            trace_hash = _workload_hash(path)

            if key in oom_keys:
                trace_status = "skipped"
                trace_detail = f"inherited OOM from smaller MBS for tp={tp} pp={pp} vpp={vpp}"
                message = f"{name}: {trace_status} - {trace_detail}"
                results.append({"name": name, "oom": True, "skipped": True})
                console.log(message)
                progress.advance(task)
                continue

            valid, invalid_reason = _is_valid_config(tp, pp, mbs)
            if not valid:
                trace_status = "invalid"
                trace_detail = invalid_reason
                message = f"{name}: {trace_status} - {trace_detail}"
                _write_invalid_marker(path, invalid_reason)
                results.append({"name": name, "invalid": True, "invalid_reason": invalid_reason})
                console.log(message)
                progress.advance(task)
                continue

            status = _trace_status(path)
            trace_detail = ""
            if status in ("OOM", "traced", "invalid"):
                trace_status = status
                trace_detail = status
            else:
                trace_status, trace_detail = _run_trace(path)
            message = f"{name}: {trace_status} (traces at {trace_hash}) - {trace_detail}"
            if trace_status == "OOM":
                oom_keys.add(key)
                results.append({"name": name, "oom": True})
            elif trace_status == "invalid":
                results.append({"name": name, "invalid": True, "invalid_reason": trace_detail})
            elif trace_status == "error":
                results.append({"name": name, "error": True, "error_detail": trace_detail})
            elif trace_only:
                results.append({"name": name, "oom": False, "simulated": False})
            else:
                metrics = _run_simulate(path)
                row: dict[str, Any] = {"name": name, "oom": False, "simulated": metrics is not None}
                if metrics:
                    row.update(metrics)
                results.append(row)
            console.log(message)
            progress.advance(task)
    return results


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
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Only process the first N generated configs (for testing).",
    )
    parser.add_argument(
        "--clean-invalid-markers",
        action="store_true",
        help="Remove .INVALID markers so previously-invalid configs are re-evaluated.",
    )
    args = parser.parse_args()

    if args.clean:
        for path in _SCENARIOS_DIR.glob("tp*_pp*_mbs*"):
            if path.is_dir():
                shutil.rmtree(path)

    paths = _generate_configs()
    if args.clean_invalid_markers:
        for path in paths:
            invalid_file = _default_trace_dir(path) / ".INVALID"
            if invalid_file.exists():
                invalid_file.unlink()
    if args.max_runs is not None:
        paths = paths[: args.max_runs]
    if args.dry_run:
        for _path in paths:
            pass
        return

    results = _sweep(paths, trace_only=args.trace_only)

    csv_path = _BASE_DIR / "results.csv"
    if results:
        fieldnames = sorted({k for r in results for k in r})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
