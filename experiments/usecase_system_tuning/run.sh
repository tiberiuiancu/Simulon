#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --job-name=system-tuning
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/experiments/usecase_system_tuning/grid_search.py" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

uv run python3 "$REPO_ROOT/experiments/usecase_system_tuning/grid_search.py"
