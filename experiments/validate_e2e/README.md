# validate_e2e

These experiments validate Simulon against real Megatron-LM training runs.

Each config under `configs/` contains:
- `scenario.yaml` — Simulon datacenter + workload definition
- `workload.yaml` — Megatron workload overrides

## Running

From this directory, submit all jobs:

```bash
./run.sh
```

This submits:
- one baseline Slurm job per config (via `run_baseline.slurm`) that runs real Megatron training (30 iterations) and logs per-iteration `iteration-time` to W&B
- 30 simulation Slurm jobs (via `run_sim.slurm`) that trace each config with `simulon trace generate` and simulate it with `simulon simulate`, each producing an independent data point for violin plots

## NIC variants

The `gptoss-bf16` config family tests network sensitivity:
- `gptoss-bf16` — 4 NICs (default, no NCCL_IB_HCA override)
- `gptoss-bf16-3nic` — 3 NICs (`mlx5_2,mlx5_3,mlx5_4`)
- `gptoss-bf16-2nic` — 2 NICs (`mlx5_3,mlx5_4`)
- `gptoss-bf16-1nic` — 1 NIC (`mlx5_4`)

## Plotting

```bash
uv run python plot.py
```

Pulls baseline per-iteration metrics and simulation results from W&B, then plots violin plots comparing the two distributions per config. Use `--use-csv` to plot from the cached `results.csv` instead.
