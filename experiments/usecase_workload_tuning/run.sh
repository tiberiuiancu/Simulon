#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=1:45:00
#SBATCH --gpus=1
#SBATCH --job-name=qwen32b-workload-tuning
#SBATCH --output=experiments/usecase_workload_tuning/logs/%j.out
#SBATCH --error=experiments/usecase_workload_tuning/logs/%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$SCRIPT_DIR/logs"

export PYTHONUNBUFFERED=1
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-simulon}"

uv run python "$SCRIPT_DIR/grid_search.py"
