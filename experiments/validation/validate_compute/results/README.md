## Validation Experiment: Compute Validation (MoE Training)

### Model Architecture
- Name: `validation-model`
- Layers: `12`, Hidden: `2048`, Heads: `32`, FFN: `8192`
- Experts: `8`, Top-k: `2`, Vocab: `32000`, Seq: `8192`

### Artifacts

Available now:
- `results/sim_trace.json` — Simulon Chrome trace for the validation step; contains `126 traceEvents`.

Pending manual execution on Snellius:
- `results/h100_profile.yaml` — H100 kernel profile override from T4.
- Megatron synthetic training traces/logs from T7.
- Megatron real C4 training traces/logs from T8.

### Simulon Prediction
- `results/sim_trace.json` is available and valid.
- Predicted total step time: `60.611 ms` (for old 1B model; will be updated after re-simulation with 4B model).

### Real Training (Pending Manual Execution)
T4, T7, and T8 must be run on Snellius with SLURM.

```bash
# On Snellius:
cd experiments/validation/validate_compute

# Profile H100 kernels
sbatch profile_h100.sh

# Run synthetic training
sbatch run_megatron_synthetic.sh
# OR: python run_megatron.py --mode synthetic

# Run real C4 training
sbatch run_megatron_real.sh
# OR: python run_megatron.py --mode real
```

### Manual Comparison
Load traces in `chrome://tracing` or https://ui.perfetto.dev.

Compare:
- `experiments/validation/validate_compute/results/sim_trace.json`
- Megatron-generated Chrome traces from the Snellius runs, typically under the job `tensorboard_dir` / `torch_profile/rank-*.json.gz`
