#!/bin/bash
#SBATCH --job-name=simulon-validate-compute-synthetic
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=experiments/validate_compute/jobs/megatron_synthetic_%j.out
#SBATCH --error=experiments/validate_compute/jobs/megatron_synthetic_%j.err

set -euo pipefail
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

cd "$(dirname "$(realpath "$0")")"
source ../../.venv/bin/activate

export PYTHONPATH="$(pwd)/megatron-lm:${PYTHONPATH:-}"

mkdir -p results

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

uv run python run_megatron.py --mode synthetic

echo "=== Done. Check results/ for chrome_trace_synthetic.json and megatron_synthetic.log ==="
