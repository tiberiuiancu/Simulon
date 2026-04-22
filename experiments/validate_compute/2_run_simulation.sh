#!/bin/bash
#SBATCH --job-name=simulon-simulate
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=jobs/simulate_%j.out
#SBATCH --error=jobs/simulate_%j.err
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
source ../../.venv/bin/activate
mkdir -p results
simulon simulate sim_training.yaml --chrome results/sim_trace.json
echo "=== Simulation done ==="
