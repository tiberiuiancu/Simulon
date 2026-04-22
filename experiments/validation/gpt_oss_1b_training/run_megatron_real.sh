#!/bin/bash
#SBATCH --job-name=simulon-gpt-oss-1b-real
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/megatron_real_%j.out
#SBATCH --error=jobs/logs/megatron_real_%j.err

set -euo pipefail
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR"

source "$REPO_ROOT/.venv/bin/activate"
python -m pip install datasets transformers sentencepiece
export PYTHONPATH="$SCRIPT_DIR/megatron-lm:${PYTHONPATH:-}"

mkdir -p results

python run_megatron.py --mode real

echo "=== Done. Check results/ for chrome_trace_real.json and megatron_real.log ==="
