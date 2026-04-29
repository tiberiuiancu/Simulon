#!/bin/bash
#SBATCH --job-name=simulon-simulate
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=experiments/validate_compute/jobs/simulate_%j.out
#SBATCH --error=experiments/validate_compute/jobs/simulate_%j.err
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/../.."

source .venv/bin/activate
mkdir -p experiments/validate_compute/results
simulon simulate experiments/validate_compute/sim_training.yaml --chrome experiments/validate_compute/results/sim_trace.json
echo "=== Simulation done ==="
