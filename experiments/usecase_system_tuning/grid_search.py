#!/usr/bin/env python3
"""System-tuning sweep: node size × link bandwidth × workload config for GPT-OSS 120B and Llama 3 70B.

Usage (from repo root):
    uv run python experiments/usecase_system_tuning/grid_search.py
    uv run python experiments/usecase_system_tuning/grid_search.py --dry-run
    uv run python experiments/usecase_system_tuning/grid_search.py --trace-only
"""

from __future__ import annotations

import contextlib
import csv
import os
import shutil
import subprocess
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_BASE_DIR = Path(__file__).parent
_SCENARIOS_DIR = _BASE_DIR / "scenarios"

_NODE_SIZES = [4, 8]
_LINK_BWS = [100, 200, 400, 800]
_TOTAL_GPUS = 64

_NODE_TEMPLATE_FOR_SIZE: dict[int, str] = {
    4: "templates/node/snellius-h100-4g.yaml",
    8: "templates/node/dgx-h100.yaml",
}

_NODE_COST_USD: dict[int, float] = {4: 150_000, 8: 300_000}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    base_workload: Path
    num_layers: int
    gbs: int
    tp_options: list[int]
    pp_options: list[int]
    ep: int | None = None
    cp: int = 1


_MODELS: list[ModelSpec] = [
    ModelSpec(
        name="llama3-70b",
        base_workload=_SCENARIOS_DIR / "llama3-70b" / "base_workload.yaml",
        num_layers=80,
        gbs=512,
        tp_options=[1, 2, 4, 8],
        pp_options=[1, 2, 4, 8],
        cp=1,
    ),
    ModelSpec(
        name="gptoss-120b",
        base_workload=_SCENARIOS_DIR / "gptoss-120b" / "base_workload.yaml",
        num_layers=36,
        gbs=1280,
        tp_options=[1, 2],
        pp_options=[2, 4, 8],
        ep=None,
    ),
]


_GPTOSS_EP_OPTIONS = [4, 8, 16]


def _divisors(n: int) -> list[int]:
    return sorted(i for i in range(1, n + 1) if n % i == 0)


def _make_workload_dir(model: str, tp: int, pp: int, vpp: int | None, ep: int | None) -> Path:
    name = f"tp{tp}_pp{pp}"
    if vpp is not None:
        name = f"{name}_vpp{vpp}"
    if ep is not None:
        name = f"{name}_ep{ep}"
    path = _SCENARIOS_DIR / model / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_workload(
    path: Path, cfg: dict[str, Any], tp: int, pp: int, vpp: int | None, ep: int | None, cp: int
) -> None:
    cfg = dict(cfg)
    cfg["config"] = dict(cfg.get("config", {}))
    cfg["config"]["tensor-model-parallel-size"] = tp
    cfg["config"]["pipeline-model-parallel-size"] = pp
    cfg["config"]["micro-batch-size"] = 1
    cfg["config"]["context-parallel-size"] = cp
    if vpp is not None:
        cfg["config"]["num-virtual-stages-per-pipeline-rank"] = vpp
    else:
        cfg["config"].pop("num-virtual-stages-per-pipeline-rank", None)
        cfg["config"].pop("num-layers-per-virtual-pipeline-stage", None)
    if ep is not None:
        cfg["config"]["expert-model-parallel-size"] = ep
    else:
        cfg["config"].pop("expert-model-parallel-size", None)
    with open(path / "workload.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def _write_scenario(workload_path: Path, node_size: int, link_bw: int) -> Path:
    num_nodes = _TOTAL_GPUS // node_size
    model_name = workload_path.parent.name
    workload_name = workload_path.name
    scenario = {
        "datacenter": {
            "num_nodes": num_nodes,
            "node": {
                "from": _NODE_TEMPLATE_FOR_SIZE[node_size],
                "scale_out": {"nic": {"speed": f"{link_bw}Gbps"}},
                "cost": _NODE_COST_USD[node_size],
            },
            "datacenter": {
                "pue": 1.2,
                "electricity_cost_per_kwh": 0.15,
                "datacenter_lifetime_years": 5,
                "idle_fraction": 0.2,
                "traces_dir": f"templates/gpu/h100/traces/system-tuning-{model_name}-{workload_name}",
            },
        },
        "workload": str(workload_path / "workload.yaml"),
    }
    scenario_path = workload_path / f"node{node_size}_bw{link_bw}.yaml"
    with open(scenario_path, "w") as f:
        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)
    return scenario_path


def _scenario_trace_dir(workload_path: Path) -> Path | None:
    from simulon.config.scenario import ScenarioConfig

    for scenario_path in sorted(workload_path.glob("node*_bw*.yaml")):
        try:
            sc = ScenarioConfig.from_yaml(str(scenario_path))
            traces_dir = sc.datacenter.datacenter.traces_dir if sc.datacenter.datacenter else None
            if traces_dir:
                return Path(str(traces_dir))
        except Exception:
            continue
    return None


def _default_trace_dir(workload_path: Path) -> Path:
    from simulon.config.resolve import resolve_gpu_spec, resolve_workload, workload_hash
    from simulon.config.scenario import ScenarioConfig

    explicit = _scenario_trace_dir(workload_path)
    if explicit is not None:
        return explicit

    for scenario_path in sorted(workload_path.glob("node*_bw*.yaml")):
        if scenario_path.exists():
            break
    else:
        scenario_path = workload_path / f"node{_NODE_SIZES[0]}_bw{_LINK_BWS[0]}.yaml"
        _write_scenario(workload_path, _NODE_SIZES[0], _LINK_BWS[0])
    try:
        sc = ScenarioConfig.from_yaml(str(scenario_path))
        gpu_spec = resolve_gpu_spec(sc.datacenter)
        gpu_name = (gpu_spec.name or "default").lower().replace(" ", "-")
    except Exception:
        gpu_name = "default"
    try:
        wl = resolve_workload(str(workload_path / "workload.yaml"))
        h = workload_hash(wl)
    except Exception:
        h = "unknown"
    return Path("templates/gpu") / gpu_name / "traces" / h


def _trace_status(workload_path: Path) -> str:
    trace_dir = _default_trace_dir(workload_path)
    if (trace_dir / ".INVALID").exists():
        return "invalid"
    if (trace_dir / ".OOM").exists():
        return "OOM"
    if any(trace_dir.glob("trace_rank_*.json")):
        return "traced"
    return "pending"


def _write_invalid_marker(workload_path: Path, reason: str) -> None:
    trace_dir = _default_trace_dir(workload_path)
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / ".INVALID").write_text(reason)


def _run_trace(workload_path: Path) -> tuple[str, str]:
    trace_dir = _default_trace_dir(workload_path)
    trace_dir.mkdir(parents=True, exist_ok=True)
    oom_file = trace_dir / ".OOM"
    if oom_file.exists():
        return "OOM", "pre-existing .OOM marker"
    if any(trace_dir.glob("trace_rank_*.json")):
        return "traced", "trace files already present"

    scenario_path = workload_path / f"node{_NODE_SIZES[0]}_bw{_LINK_BWS[0]}.yaml"
    cmd = ["bash", "./scripts/apptainer-trace.sh", str(scenario_path)]
    env = os.environ.copy()
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=1800, env=env)
        return "traced", ""
    except subprocess.CalledProcessError as exc:
        if oom_file.exists():
            return "OOM", f"Megatron OOM (return code {exc.returncode})"

        error_log = trace_dir / ".error.log"
        combined = ""
        if error_log.exists():
            with contextlib.suppress(Exception):
                combined = error_log.read_text().lower()
        if not combined:
            stderr = (exc.stderr or "").lower()
            stdout = (exc.stdout or "").lower()
            combined = stdout + "\n" + stderr
        oom_keywords = (
            "out of memory",
            "out-of-memory",
            "cuda oom",
            "runtimeerror: cuda",
            "torch.cuda.outofmemoryerror",
            "torch.outofmemoryerror",
            "outofmemoryerror",
        )
        has_cuda_oom_phrase = "cuda" in combined and "memory" in combined and "allocate" in combined
        if any(kw in combined for kw in oom_keywords) or has_cuda_oom_phrase:
            with contextlib.suppress(Exception):
                oom_file.write_text(f"returncode={exc.returncode}\n{exc.stderr or ''}")
            return "OOM", f"Megatron OOM (return code {exc.returncode})"

        return ("error", f"non-OOM error (return code {exc.returncode})")
    except subprocess.TimeoutExpired:
        if oom_file.exists():
            return "OOM", "Megatron OOM (timeout, .OOM present)"
        return "error", "trace timeout after 1800s"


def _run_simulate(scenario_path: Path, run_name: str) -> dict[str, Any] | None:
    cmd = [
        "simulon",
        "simulate",
        str(scenario_path),
        "--network-simulation",
        "collective",
        "--skip-if-tracked",
        "--energy",
        "--cost",
    ]
    env = os.environ.copy()
    env["WANDB_RUN_NAME"] = run_name
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
            if "Energy per iteration:" in line:
                with contextlib.suppress(Exception):
                    metrics["energy_wh"] = float(line.split("Wh")[0].split()[-1])
            if "CO2eq:" in line:
                with contextlib.suppress(Exception):
                    metrics["co2eq_g"] = float(line.split("CO2eq:")[-1].split()[0])
            if "CAPEX total:" in line:
                with contextlib.suppress(Exception):
                    metrics["capex_total"] = float(line.split("$")[-1].split()[0].replace(",", ""))
            if "OPEX per run:" in line:
                with contextlib.suppress(Exception):
                    metrics["opex_per_run"] = float(line.split("$")[-1].split()[0].replace(",", ""))
            if "Cost per run:" in line:
                with contextlib.suppress(Exception):
                    metrics["cost_per_run"] = float(line.split("$")[-1].split()[0].replace(",", ""))
        return metrics
    except subprocess.CalledProcessError:
        return None


def _is_valid_llama(
    tp: int, pp: int, vpp: int | None, num_layers: int, gbs: int
) -> tuple[bool, str]:
    if num_layers % pp != 0:
        return False, f"num_layers={num_layers} not divisible by pp={pp}"
    if tp * pp > _TOTAL_GPUS:
        return False, f"tp*pp={tp * pp} exceeds total GPUs={_TOTAL_GPUS}"
    dp = _TOTAL_GPUS // (tp * pp)
    if gbs % (1 * dp) != 0:
        return False, f"global_batch_size={gbs} not divisible by dp={dp}"
    if pp > 1 and vpp is not None and (num_layers // pp) % vpp != 0:
        return False, f"vpp={vpp} does not divide layers_per_stage={num_layers // pp}"
    return True, ""


def _is_valid_gptoss(
    tp: int, pp: int, ep: int, vpp: int | None, num_layers: int, gbs: int
) -> tuple[bool, str]:
    if num_layers % pp != 0:
        return False, f"num_layers={num_layers} not divisible by pp={pp}"
    if tp * pp > _TOTAL_GPUS:
        return False, f"tp*pp={tp * pp} exceeds total GPUs={_TOTAL_GPUS}"
    if pp * ep > _TOTAL_GPUS:
        return False, f"pp*ep={pp * ep} exceeds total GPUs={_TOTAL_GPUS}"
    dp_dense = _TOTAL_GPUS // (tp * pp)
    dp_moe = _TOTAL_GPUS // (pp * ep)
    if gbs % (1 * dp_dense) != 0:
        return False, f"global_batch_size={gbs} not divisible by dense dp={dp_dense}"
    if gbs % (1 * dp_moe) != 0:
        return False, f"global_batch_size={gbs} not divisible by moe dp={dp_moe}"
    if pp > 1 and vpp is not None and (num_layers // pp) % vpp != 0:
        return False, f"vpp={vpp} does not divide layers_per_stage={num_layers // pp}"
    return True, ""


def _vpp_options(layers_per_stage: int) -> list[int]:
    candidates = [1, 2, 4]
    return sorted({vpp for vpp in candidates if layers_per_stage % vpp == 0})


def _grid_for_model(model: ModelSpec) -> list[tuple[int, int, int | None, int | None]]:
    combos: list[tuple[int, int, int | None, int | None]] = []
    valid_pp = [pp for pp in model.pp_options if model.num_layers % pp == 0]
    for tp in model.tp_options:
        for pp in valid_pp:
            if tp * pp > _TOTAL_GPUS:
                continue
            vpps = [None] if pp == 1 else _vpp_options(model.num_layers // pp)
            if model.name == "gptoss-120b":
                for ep in _GPTOSS_EP_OPTIONS:
                    if pp * ep > _TOTAL_GPUS:
                        continue
                    for vpp in vpps:
                        combos.append((tp, pp, vpp, ep))
            else:
                for vpp in vpps:
                    combos.append((tp, pp, vpp, None))
    return combos


def _generate_model_configs(model: ModelSpec) -> list[Path]:
    with open(model.base_workload) as f:
        base = yaml.safe_load(f)
    paths: list[Path] = []
    for tp, pp, vpp, ep in _grid_for_model(model):
        path = _make_workload_dir(model.name, tp, pp, vpp, ep)
        _write_workload(path, base, tp, pp, vpp, ep, model.cp)
        for node_size in _NODE_SIZES:
            for link_bw in _LINK_BWS:
                _write_scenario(path, node_size, link_bw)
        paths.append(path)
    return paths


def _generate_all_configs() -> dict[str, list[Path]]:
    return {model.name: _generate_model_configs(model) for model in _MODELS}


def _purge_traces(paths_by_model: dict[str, list[Path]]) -> None:
    for paths in paths_by_model.values():
        for path in paths:
            trace_dir = _default_trace_dir(path)
            if trace_dir.exists():
                shutil.rmtree(trace_dir)
                print(f"Purged {trace_dir}")


def _sweep_model(model: ModelSpec, paths: list[Path], trace_only: bool) -> list[dict[str, Any]]:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    console = Console()
    from simulon.config.resolve import resolve_workload, workload_hash

    results: list[dict[str, Any]] = []
    oom_keys: set[tuple[int, int, int | None, int | None]] = set()
    total_sims = len(paths) * len(_NODE_SIZES) * len(_LINK_BWS)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"{model.name} sweep", total=total_sims)

        for path in paths:
            name = path.name
            tp, pp = (
                int(name.split("_")[0].replace("tp", "")),
                int(name.split("_")[1].replace("pp", "")),
            )
            vpp: int | None = None
            ep: int | None = None
            for part in name.split("_")[2:]:
                if part.startswith("vpp"):
                    vpp = int(part.replace("vpp", ""))
                elif part.startswith("ep"):
                    ep = int(part.replace("ep", ""))
            key = (tp, pp, vpp, ep)

            try:
                wl = resolve_workload(str(path / "workload.yaml"))
                h = workload_hash(wl)
            except Exception:
                h = "unknown"

            if model.name == "gptoss-120b":
                valid, invalid_reason = _is_valid_gptoss(
                    tp, pp, ep or 8, vpp, model.num_layers, model.gbs
                )
            else:
                valid, invalid_reason = _is_valid_llama(tp, pp, vpp, model.num_layers, model.gbs)

            if not valid:
                _write_invalid_marker(path, invalid_reason)
                results.append(
                    {
                        "model": model.name,
                        "name": name,
                        "tp": tp,
                        "pp": pp,
                        "vpp": vpp,
                        "ep": ep,
                        "invalid": True,
                        "invalid_reason": invalid_reason,
                    }
                )
                console.log(f"{model.name}/{name}: invalid - {invalid_reason} [hash={h}]")
                progress.advance(task, advance=len(_NODE_SIZES) * len(_LINK_BWS))
                continue

            if key in oom_keys:
                console.log(f"{model.name}/{name}: skipped (inherited OOM) [hash={h}]")
                results.append(
                    {
                        "model": model.name,
                        "name": name,
                        "tp": tp,
                        "pp": pp,
                        "vpp": vpp,
                        "ep": ep,
                        "oom": True,
                        "skipped": True,
                    }
                )
                progress.advance(task, advance=len(_NODE_SIZES) * len(_LINK_BWS))
                continue

            status = _trace_status(path)
            if status == "pending":
                status, trace_detail = _run_trace(path)
            else:
                trace_detail = status

            if status == "OOM":
                oom_keys.add(key)
                console.log(f"{model.name}/{name}: OOM - {trace_detail} [hash={h}]")
                for node_size in _NODE_SIZES:
                    for link_bw in _LINK_BWS:
                        results.append(
                            {
                                "model": model.name,
                                "name": name,
                                "tp": tp,
                                "pp": pp,
                                "vpp": vpp,
                                "ep": ep,
                                "node_size": node_size,
                                "link_bw": link_bw,
                                "oom": True,
                            }
                        )
                        progress.advance(task)
                continue

            if status == "invalid":
                console.log(f"{model.name}/{name}: invalid - {trace_detail} [hash={h}]")
                for node_size in _NODE_SIZES:
                    for link_bw in _LINK_BWS:
                        results.append(
                            {
                                "model": model.name,
                                "name": name,
                                "tp": tp,
                                "pp": pp,
                                "vpp": vpp,
                                "ep": ep,
                                "node_size": node_size,
                                "link_bw": link_bw,
                                "invalid": True,
                                "invalid_reason": trace_detail,
                            }
                        )
                        progress.advance(task)
                continue

            if status == "error":
                console.log(f"{model.name}/{name}: error - {trace_detail} [hash={h}]")
                for node_size in _NODE_SIZES:
                    for link_bw in _LINK_BWS:
                        results.append(
                            {
                                "model": model.name,
                                "name": name,
                                "tp": tp,
                                "pp": pp,
                                "vpp": vpp,
                                "ep": ep,
                                "node_size": node_size,
                                "link_bw": link_bw,
                                "error": True,
                                "error_detail": trace_detail,
                            }
                        )
                        progress.advance(task)
                continue

            console.log(f"{model.name}/{name}: traced - {trace_detail} [hash={h}]")

            if trace_only:
                for node_size in _NODE_SIZES:
                    for link_bw in _LINK_BWS:
                        results.append(
                            {
                                "model": model.name,
                                "name": name,
                                "tp": tp,
                                "pp": pp,
                                "vpp": vpp,
                                "ep": ep,
                                "node_size": node_size,
                                "link_bw": link_bw,
                                "traced": True,
                            }
                        )
                        progress.advance(task)
                continue

            for node_size in _NODE_SIZES:
                for link_bw in _LINK_BWS:
                    scenario_path = path / f"node{node_size}_bw{link_bw}.yaml"
                    run_name = f"system-tuning-{model.name}-{name}-node{node_size}-bw{link_bw}"
                    metrics = _run_simulate(scenario_path, run_name)
                    row: dict[str, Any] = {
                        "model": model.name,
                        "name": name,
                        "tp": tp,
                        "pp": pp,
                        "vpp": vpp,
                        "ep": ep,
                        "node_size": node_size,
                        "link_bw": link_bw,
                        "simulated": metrics is not None,
                    }
                    if metrics:
                        row.update(metrics)
                    results.append(row)
                    console.log(
                        f"{model.name}/{name} node{node_size} bw{link_bw}: "
                        f"{'ok' if metrics else 'sim failed'}"
                    )
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
        "--trace-only", action="store_true", help="Trace all workload configs but do not simulate."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all generated config subdirectories before running.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to run (default: all).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Only process the first N generated configs per model (for testing).",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Remove all trace directories before running so every config is re-traced.",
    )
    args = parser.parse_args()

    if args.clean:
        for model_dir in _SCENARIOS_DIR.iterdir():
            if model_dir.is_dir():
                for path in model_dir.glob("tp*_pp*"):
                    if path.is_dir():
                        shutil.rmtree(path)

    models = _MODELS
    if args.models:
        selected = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m.name in selected]
        if not models:
            raise ValueError(f"No matching models in {selected}")

    paths_by_model = {model.name: _generate_model_configs(model) for model in models}

    if args.max_runs is not None:
        paths_by_model = {name: paths[: args.max_runs] for name, paths in paths_by_model.items()}

    if args.purge:
        _purge_traces(paths_by_model)

    if args.dry_run:
        for model in models:
            for path in paths_by_model[model.name]:
                print(f"Would process {model.name}/{path.name}")
        return

    all_results: list[dict[str, Any]] = []
    for model in models:
        all_results.extend(
            _sweep_model(model, paths_by_model[model.name], trace_only=args.trace_only)
        )

    csv_path = _BASE_DIR / "results.csv"
    if all_results:
        fieldnames = sorted({k for r in all_results for k in r})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
