#!/bin/bash
#SBATCH --job-name=simulon-profile-gpt-oss-1b-h100
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/profile_h100_%j.out
#SBATCH --error=jobs/logs/profile_h100_%j.err

set -euo pipefail
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

uv sync --extra profiling
source .venv/bin/activate

mkdir -p jobs/logs

simulon profile gpu \
    --name H100 \
    --model gpt-oss-1b-moe \
    --tp 1 \
    --ep 1 \
    --batch-size 1 \
    --seq-len 8192 \
    --output templates/gpu/h100.yaml

echo "=== Done. Profile appended to templates/gpu/h100.yaml ==="
