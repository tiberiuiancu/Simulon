#!/bin/bash
#SBATCH --job-name=simulon-profile-validate-compute-h100
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/profile_h100_%j.out
#SBATCH --error=jobs/logs/profile_h100_%j.err

set -euo pipefail
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

SCRIPT_DIR="experiments/validation/validate_compute"
cd "$SCRIPT_DIR"

uv sync --extra profiling
source ../../../.venv/bin/activate

mkdir -p jobs/logs

simulon profile gpu \
    --name H100 \
    --model validation-model \
    --tp 1 \
    --ep 1 \
    --batch-size 1 \
    --seq-len 8192 \
    --output templates/gpu/h100.yaml

echo "=== Done. Profile appended to templates/gpu/h100.yaml ==="
