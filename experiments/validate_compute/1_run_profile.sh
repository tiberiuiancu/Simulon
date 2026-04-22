#!/bin/bash
#SBATCH --job-name=simulon-profile-validate-compute-h100
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/profile_%j.out
#SBATCH --error=jobs/profile_%j.err

set -euo pipefail
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

cd "$(dirname "$(realpath "$0")")"

source ../../.venv/bin/activate

mkdir -p jobs

simulon profile gpu \
    --name H100 \
    --model validation-model \
    --tp 1 \
    --ep 1 \
    --batch-size 1 \
    --seq-len 4096 \
    --output ../../templates/gpu/h100.yaml

echo "=== Done. Profile appended to ../../templates/gpu/h100.yaml ==="
