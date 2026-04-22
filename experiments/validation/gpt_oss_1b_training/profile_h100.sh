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
cd "$SCRIPT_DIR"

source "$REPO_ROOT/.venv/bin/activate"

mkdir -p results

python profile_h100.py

echo "=== Done. Profile saved to results/h100_profile.yaml ==="
