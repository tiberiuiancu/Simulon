#!/usr/bin/env python3
"""Run a profiled Megatron baseline to emit a Chrome trace and timer breakdown.

Produces:
  - <save-dir>/torch_profile/rank-*.json.gz  (PyTorch profiler Chrome traces)
  - stdout log with per-phase timers (forward-compute, backward-compute, optimizer, etc.)

Usage (inside the apptainer container, from repo root):
    python3 experiments/validate_e2e/run_profile.py \
        experiments/validate_e2e/configs/qwen3-32b-tp2-pp4-mbs1-vpp1/scenario.yaml \
        /scratch-shared/tiberiui/profile-qwen3-tp2
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from simulon.cli.trace import _build_megatron_args
from simulon.config.resolve import resolve_node_spec
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload
from simulon.tracking.env import load_cascading_tracking_env

_MEGATRON_ENTRYPOINT = (
    Path(__file__).parents[2] / "vendor" / "Megatron-LM-traced" / "pretrain_gpt.py"
)


def _resolve_workload(input_path: str):
    path = Path(input_path).resolve()
    workload = None
    num_nodes = int(os.environ.get("NUM_NODES", 1))
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", 1))

    try:
        sc = ScenarioConfig.from_yaml(str(path))
        workload = sc.workload
        dc = sc.datacenter
        num_nodes = dc.num_nodes
        node_spec = resolve_node_spec(dc)
        if node_spec.gpus_per_node is not None:
            gpus_per_node = node_spec.gpus_per_node
    except Exception:
        from simulon.config.resolve import resolve_workload as _rw

        workload = _rw(str(path))

    if not isinstance(workload, MegatronWorkload):
        raise ValueError("Profile runner only supports Megatron workloads")

    return workload, num_nodes, gpus_per_node


def _build_torchrun_cmd(
    workload: MegatronWorkload,
    num_nodes: int,
    gpus_per_node: int,
    *,
    warmup_iters: int,
    train_iters: int,
    save_dir: str,
) -> list[str]:
    derived_args, _ = _build_megatron_args(workload, dataset="mock", warmup=warmup_iters)

    total_iters = warmup_iters + train_iters
    derived_args["--train-iters"] = total_iters
    derived_args.pop("--trace-warmup-iters", None)
    derived_args["--log-interval"] = 1
    derived_args["--eval-iters"] = 0
    derived_args["--eval-interval"] = 1000000
    derived_args["--save-interval"] = 1000000
    derived_args["--log-throughput"] = True
    derived_args["--log-timers-to-tensorboard"] = True
    derived_args["--timing-log-level"] = 2
    derived_args["--save"] = save_dir
    derived_args["--use-pytorch-profiler"] = True
    derived_args["--profile-step-start"] = warmup_iters + 1
    derived_args["--profile-step-end"] = warmup_iters + 2

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "6000")
    rdzv_id = os.environ.get("SLURM_JOB_ID", "profile")

    cmd = [
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
    parser.add_argument("save_dir", help="Directory for Chrome traces and checkpoints")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--train-iters", type=int, default=5)
    args = parser.parse_args()

    load_cascading_tracking_env(args.input)

    workload, num_nodes, gpus_per_node = _resolve_workload(args.input)
    model_name = Path(args.input).parent.name

    save_dir = args.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cmd = _build_torchrun_cmd(
        workload,
        num_nodes,
        gpus_per_node,
        warmup_iters=args.warmup_iters,
        train_iters=args.train_iters,
        save_dir=save_dir,
    )

    print(f"Running profiled baseline for {model_name}")
    print(f"  nodes={num_nodes} gpus_per_node={gpus_per_node}")
    print(f"  warmup={args.warmup_iters} train={args.train_iters}")
    print(f"  save_dir={save_dir}")
    print(f"  Chrome traces will be at {save_dir}/torch_profile/")
    print(f"  command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    finally:
        # Clean up checkpoints but keep the torch_profile directory
        ckpt = Path(save_dir) / "checkpoints"
        if ckpt.exists():
            shutil.rmtree(ckpt, ignore_errors=True)
            print(f"Removed checkpoints: {ckpt}")
        print(f"Chrome traces saved at: {save_dir}/torch_profile/")


if __name__ == "__main__":
    main()
