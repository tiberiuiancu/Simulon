#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=1:45:00
#SBATCH --gpus=1
#SBATCH --job-name=qwen32b-workload-tuning
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

SCRIPT_DIR="$REPO_ROOT/experiments/usecase_workload_tuning"

export PYTHONUNBUFFERED=1

uv run python "$SCRIPT_DIR/grid_search.py"
