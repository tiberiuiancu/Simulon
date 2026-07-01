# validate_e2e

These experiments validate Simulon against real Megatron-LM training runs.

Each config under `configs/` contains:
- `scenario.yaml` — Simulon datacenter + workload definition
- `workload.yaml` — Megatron workload overrides
- `reference.yaml` — real measured metrics from a baseline run

## Running

From this directory, submit all jobs:

```bash
./run.sh
```

This submits:
- one baseline Slurm job per config (via `run_baseline.slurm`) that runs real Megatron training and logs to W&B
- one simulation Slurm job (via `run_sim.slurm`) that traces each config with `simulon trace generate` and then simulates it with `simulon simulate`

`qwen3-32b` is excluded from the real baseline jobs because it requires 16 nodes × 4 GPUs; it is still traced and simulated in the simulation sweep.

## Plotting

```bash
uv run python plot.py
```

Pulls metrics from W&B (or `--use-csv` for the local `results.csv`) and plots real vs. simulated results.
