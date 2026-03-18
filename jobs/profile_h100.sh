#!/bin/bash
#SBATCH --job-name=simulon-profile-h100
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=jobs/logs/profile_h100_%j.out
#SBATCH --error=jobs/logs/profile_h100_%j.err

set -euo pipefail
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

# --- Environment setup ---
uv sync --extra profiling
uv pip install pip
source .venv/bin/activate
#uv run simulon install apex --skip-cuda-version-check
#uv run simulon install deepgemm

mkdir -p jobs/logs

# --- Sweep parameters ---
TP="1,2,4,8"
EP="1,2,4,8,16,32,64"  # ep > num_experts and ep > 1 for dense models are skipped automatically
BATCH="1,2,4,8,16,32,64,128"
SEQ="256,512,1024,2048,4096,8192"
GPU_NAME="H100"
OUTPUT="templates/gpu/h100.yaml"

MODELS=(
    llama-7b
    llama-3-70b
    deepseek-v3
    qwen3-30b-a3b
    qwen3-235b-a22b
)

# --- Profile each model ---
for MODEL in "${MODELS[@]}"; do
    echo "=== Profiling model: $MODEL ==="
    simulon profile gpu \
        --name "$GPU_NAME" \
        --model "$MODEL" \
        --tp "$TP" \
        --ep "$EP" \
        --batch-size "$BATCH" \
        --seq-len "$SEQ" \
        --output "$OUTPUT"
done

echo "=== Done. Profile saved to $OUTPUT ==="
