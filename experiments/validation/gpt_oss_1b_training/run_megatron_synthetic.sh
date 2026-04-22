#!/bin/bash
#SBATCH --job-name=simulon-gpt-oss-1b-synthetic
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/megatron_synthetic_%j.out
#SBATCH --error=jobs/logs/megatron_synthetic_%j.err

set -euo pipefail
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

source .venv/bin/activate
SCRIPT_DIR="experiments/validation/gpt_oss_1b_training"
SCRIPT_DIR_REAL=$(realpath $SCRIPT_DIR)
cd "$SCRIPT_DIR_REAL"

uv pip install datasets transformers sentencepiece wget
uv pip install -e megatron-lm
export PYTHONPATH="$SCRIPT_DIR_REAL/megatron-lm:${PYTHONPATH:-}"

mkdir -p results

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

uv run python run_megatron.py --mode synthetic

echo "=== Done. Check results/ for chrome_trace_synthetic.json and megatron_synthetic.log ==="
