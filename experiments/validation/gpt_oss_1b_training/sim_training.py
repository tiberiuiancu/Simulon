from __future__ import annotations

# pyright: reportUnknownVariableType=false

import copy
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from simulon.backend.analytical import AnalyticalBackend
from simulon.backend.dag import write_chrome_trace
from simulon.backend.dag.nodes import ExecutionDAG
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
SCENARIO_PATH = SCRIPT_DIR / "sim_training.yaml"
PROFILE_PATH = RESULTS_DIR / "h100_profile.yaml"
TRACE_PATH = RESULTS_DIR / "sim_trace.json"
def _load_yaml_mapping(path: Path) -> dict[str, object]:
    with open(path) as f:
        raw: dict[str, object] | object = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a mapping")
    data: dict[str, object] = raw
    return data


def _ensure_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _load_scenario_dict() -> dict[str, object]:
    data = _load_yaml_mapping(SCENARIO_PATH)

    if PROFILE_PATH.exists():
        profile = _load_yaml_mapping(PROFILE_PATH)
        datacenter = _ensure_dict(data.setdefault("datacenter", {}), "datacenter")
        node = _ensure_dict(datacenter.setdefault("node", {}), "datacenter.node")
        gpu_cfg = _ensure_dict(node.setdefault("gpu", {}), "datacenter.node.gpu")
        gpu_cfg.update(profile)
    else:
        print(f"warning: {PROFILE_PATH} not found; using templates/gpu/h100.yaml", file=sys.stderr)

    return data


def _print_breakdown(dag: ExecutionDAG) -> None:
    totals_ms: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for node in dag.compute_nodes:
        if node.duration_ms is None:
            continue
        totals_ms[node.kernel] += node.duration_ms
        counts[node.kernel] += 1

    if not totals_ms:
        return

    print("kernel breakdown:")
    for kernel in sorted(totals_ms):
        total = totals_ms[kernel]
        count = counts[kernel]
        print(f"  {kernel}: {total:.3f} ms total ({total / count:.3f} ms avg, n={count})")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scenario_dict = _load_scenario_dict()
    scenario = ScenarioConfig.model_validate(copy.deepcopy(scenario_dict))
    if not isinstance(scenario.workload, MegatronWorkload):
        raise TypeError("sim_training.py expects a Megatron workload")

    backend = AnalyticalBackend()
    ignore_missing = not PROFILE_PATH.exists()
    try:
        dag, result = backend.simulate(scenario, ignore_missing=ignore_missing)
    except RuntimeError as exc:
        if "No profiling data found" not in str(exc):
            raise
        print("warning: incomplete kernel profile; rerunning with ignore_missing=True", file=sys.stderr)
        dag, result = backend.simulate(scenario, ignore_missing=True)

    p = scenario.workload.parallelism
    dp = p.dp if p.dp is not None else scenario.workload.training.num_gpus // (p.tp * p.pp * p.ep)
    write_chrome_trace(dag, tp=p.tp, pp=p.pp, dp=dp, ep=p.ep, path=TRACE_PATH)

    print(f"total step time: {result.total_time_ms:.3f} ms")
    print(f"compute: {result.compute_ms:.3f} ms | exposed comm: {result.exposed_comm_ms:.3f} ms | bubble: {result.bubble_ms:.3f} ms")
    _print_breakdown(dag)
    print(f"trace written to: {TRACE_PATH}")


if __name__ == "__main__":
    main()
