#!/usr/bin/env bash
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=192
#SBATCH --mem=192G
#SBATCH --job-name=deepseek-sims
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/simulate_deepseek.sh" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
    SCRIPT_DIR="$REPO_ROOT/scripts"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

SCENARIO="${1:-}"
if [ -z "$SCENARIO" ]; then
    SCENARIO_DIR="experiments/usecase_system_tuning/scenarios/deepseek-v3/tp2_pp8_ep64"
    for scenario in "$SCENARIO_DIR"/node*_bw*.yaml; do
        name=$(basename "$scenario" .yaml)
        run_name="system-tuning-deepseek-v3-tp2_pp8_ep64-${name}"
        echo "Submitting: $name"
        sbatch "$SCRIPT_DIR/simulate_deepseek.sh" "$scenario" "$run_name"
    done
    exit 0
fi

RUN_NAME="${2:-}"
if [ -n "$RUN_NAME" ]; then
    export WANDB_RUN_NAME="$RUN_NAME"
fi
uv run simulon simulate "$SCENARIO" --energy -v
