#!/bin/bash
#SBATCH --job-name=simulon-setup
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/setup_%j.out
#SBATCH --error=jobs/setup_%j.err
set -euo pipefail
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0
source ../../.venv/bin/activate
cd "$(dirname "$(realpath "$0")")"
uv pip install pip datasets transformers sentencepiece wget pybind11 torch
uv pip install --no-build-isolation transformer_engine[pytorch]
uv pip install -e megatron-lm
uv run simulon install apex --skip-cuda-version-check
echo "=== Setup done ==="
