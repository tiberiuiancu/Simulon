#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=validate-e2e-sim
#SBATCH --output=validate-e2e-sim-%x-%j.out
#SBATCH --error=validate-e2e-sim-%x-%j.err

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

JOB_ID="${SLURM_JOB_ID:-local}"
CONFIGS_DIR="$REPO_ROOT/experiments/validate_e2e/configs"

for dir in "$CONFIGS_DIR"/*/; do
    model="$(basename "$dir")"
    scenario="$dir/scenario.yaml"
    scenario_rel="${scenario#$REPO_ROOT/}"
    run_name="validate-e2e-$model"
    trace_dir="templates/gpu/h100/traces/validate-e2e-$model-$JOB_ID"

    # Reuse existing traces from a previous job if available
    trace_base="templates/gpu/h100/traces/validate-e2e-$model"
    existing_trace_dir=""
    if [ -d "$trace_base" ]; then
        existing_trace_dir="$trace_base"
    else
        for d in "$REPO_ROOT/templates/gpu/h100/traces/"validate-e2e-"$model"-*/; do
            if [ -d "$d" ]; then
                existing_trace_dir="${d%/}"
                break
            fi
        done
    fi

    echo ""
    echo "============================================================"
    echo "Config: $model"
    echo "Scenario: $scenario_rel"
    echo "Trace dir: $trace_dir"
    echo "Run name: $run_name"
    echo "============================================================"

    if uv run python3 -c "
import os, sys, wandb
api = wandb.Api()
entity = os.environ.get('WANDB_ENTITY')
project = os.environ.get('WANDB_PROJECT', 'simulon')
path = f'{entity}/{project}' if entity else project
runs = api.runs(path, filters={'state': 'finished', 'display_name': '${run_name}'})
for r in runs:
    if r.display_name == '${run_name}':
        sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
        echo "[skip] $model: already tracked in W&B as ${run_name}"
        continue
    fi

    if [ -n "$existing_trace_dir" ] && ls "$existing_trace_dir"/trace_rank_*.json >/dev/null 2>&1; then
        echo "[trace] $model: reusing existing traces from $existing_trace_dir"
        trace_dir="$existing_trace_dir"
    else
        echo "[trace] $model"
        if SIMULON_TRACES_DIR="$trace_dir" bash "$REPO_ROOT/scripts/apptainer-trace.sh" "$scenario_rel"; then
            echo "[trace] $model: OK"
        else
            echo "[trace] $model: FAILED (exit code $?)" >&2
        fi
    fi

    echo "[simulate] $model"
    if SIMULON_TRACES_DIR="$trace_dir" WANDB_RUN_NAME="$run_name" uv run simulon simulate \
        --network-simulation collective \
        --energy \
        "$scenario_rel"; then
        echo "[simulate] $model: OK"
    else
        echo "[simulate] $model: FAILED (exit code $?)" >&2
    fi
done

echo ""
echo "Validate-e2e simulation sweep complete (job $JOB_ID)."
