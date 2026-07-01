#!/usr/bin/env python3
"""Run a real Megatron baseline from a Simulon scenario or workload YAML.

Usage (inside the apptainer container, from repo root):
    python3 experiments/validate_e2e/run_baseline.py \
        experiments/validate_e2e/gptoss-bf16/scenario.yaml \
        --warmup-iters 3 --train-iters 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from simulon.cli.trace import _build_megatron_args
from simulon.config.resolve import resolve_node_spec, resolve_workload
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload
from simulon.tracking.env import load_cascading_tracking_env

_MEGATRON_ENTRYPOINT = (
    Path(__file__).parents[2] / "vendor" / "Megatron-LM-traced" / "pretrain_gpt.py"
)


def _resolve_workload(input_path: str) -> tuple[MegatronWorkload, int, int, int | None]:
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist inside container: {path}")

    scenario = False
    workload: MegatronWorkload | None = None
    num_nodes = int(os.environ.get("NUM_NODES", 1))
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", 1))
    nics_per_node: int | None = None

    try:
        sc = ScenarioConfig.from_yaml(str(path))
        workload = sc.workload
        dc = sc.datacenter
        num_nodes = dc.num_nodes
        node_spec = resolve_node_spec(dc)
        if node_spec.gpus_per_node is not None:
            gpus_per_node = node_spec.gpus_per_node
        nics_per_node = node_spec.nics_per_node
        scenario = True
    except Exception:
        try:
            workload = resolve_workload(str(path))
        except Exception as workload_exc:
            raise RuntimeError(f"Could not parse {path} as scenario or workload") from workload_exc

    if not isinstance(workload, MegatronWorkload):
        raise ValueError("Baseline runner only supports Megatron workloads")
    if not scenario and (num_nodes == 1 and gpus_per_node == 1):
        raise ValueError(
            "Workload-only input requires NUM_NODES and GPUS_PER_NODE env vars to be set"
        )

    return workload, num_nodes, gpus_per_node, nics_per_node


def _build_torchrun_cmd(
    workload: MegatronWorkload,
    num_nodes: int,
    gpus_per_node: int,
    *,
    warmup_iters: int,
    train_iters: int,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    save_dir: str | None,
) -> list[str]:
    derived_args, _explicitly_set = _build_megatron_args(
        workload, dataset="mock", warmup=warmup_iters
    )

    total_iters = warmup_iters + train_iters
    derived_args["--train-iters"] = total_iters
    derived_args.pop("--trace-warmup-iters", None)
    derived_args["--log-interval"] = 1
    derived_args["--eval-iters"] = 0
    derived_args["--eval-interval"] = 1000000
    derived_args["--save-interval"] = 1000000
    derived_args["--log-throughput"] = True

    if wandb_project:
        derived_args["--wandb-project"] = wandb_project
    if wandb_entity:
        derived_args["--wandb-entity"] = wandb_entity
    if wandb_run_name:
        derived_args["--wandb-exp-name"] = wandb_run_name
        derived_args["--wandb-save-dir"] = os.path.join(save_dir or "./output", "wandb")

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "6000")

    rdzv_id = os.environ.get("SLURM_JOB_ID", "baseline")
    cmd: list[str] = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(num_nodes),
        "--nproc_per_node",
        str(gpus_per_node),
        "--rdzv_id",
        str(rdzv_id),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        f"{master_addr}:{master_port}",
        str(_MEGATRON_ENTRYPOINT),
    ]
    for flag, value in derived_args.items():
        if value is True:
            cmd.append(flag)
        elif value is False:
            continue
        elif isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.append(flag)
            cmd.append(str(value))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to scenario.yaml or workload.yaml")
    parser.add_argument("--warmup-iters", type=int, default=3, help="Iterations treated as warmup")
    parser.add_argument(
        "--train-iters", type=int, default=30, help="Iterations after warmup to run"
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    load_cascading_tracking_env(args.input)

    workload, num_nodes, gpus_per_node, nics_per_node = _resolve_workload(args.input)

    model_name = Path(args.input).parent.name
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    default_run_name = f"validate-e2e-baseline-{model_name}"
    if job_id:
        default_run_name = f"{default_run_name}-{job_id}"
    wandb_run_name = args.wandb_run_name or default_run_name
    save_dir = (
        args.save_dir or f"/scratch-shared/tiberiui/baseline-{model_name}-{job_id or 'local'}"
    )

    cmd = _build_torchrun_cmd(
        workload,
        num_nodes,
        gpus_per_node,
        warmup_iters=args.warmup_iters,
        train_iters=args.train_iters,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        save_dir=save_dir,
    )

    print(f"Running baseline for {model_name}")  # noqa: T201
    print(f"  nodes={num_nodes} gpus_per_node={gpus_per_node} nics={nics_per_node}")  # noqa: T201
    print(  # noqa: T201
        f"  warmup={args.warmup_iters} train={args.train_iters} total={args.warmup_iters + args.train_iters}"
    )
    print(
        f"  wandb_project={wandb_project} wandb_entity={wandb_entity} wandb_run_name={wandb_run_name}"
    )  # noqa: T201
    print(f"  command: {' '.join(cmd)}")  # noqa: T201

    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
            print(f"Removed local output directory: {save_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
