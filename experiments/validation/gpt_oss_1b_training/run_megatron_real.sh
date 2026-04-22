#!/bin/bash
#SBATCH --job-name=simulon-gpt-oss-1b-real
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/megatron_real_%j.out
#SBATCH --error=jobs/logs/megatron_real_%j.err

set -euo pipefail
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

SCRIPT_DIR="experiments/validation/gpt_oss_1b_training"
cd "$SCRIPT_DIR"

source ../../.venv/bin/activate
python -m pip install datasets transformers sentencepiece wget
export PYTHONPATH="$SCRIPT_DIR/megatron-lm:${PYTHONPATH:-}"

mkdir -p results

python run_megatron.py --mode real

echo "=== Done. Check results/ for chrome_trace_real.json and megatron_real.log ==="
