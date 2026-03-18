#!/bin/bash
#SBATCH --job-name=simulon-profile-h100
#SBATCH --nodelist=gpu_h100
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/profile_h100_%j.out
#SBATCH --error=jobs/logs/profile_h100_%j.err

set -euo pipefail

# TODO: load your CUDA module here, e.g.:
#   module load cuda/12.x
# Then remove the exit below.
echo "ERROR: No module loaded. Edit this script to load the correct CUDA module." >&2
exit 1

# --- Environment setup ---
pip install -e ".[profiling]"
simulon install apex
simulon install deepgemm

mkdir -p jobs/logs

# --- Sweep parameters ---
TP="1,2,4,8"
EP="1"
BATCH="1,2,4,8,16,32,64,128"
SEQ="256,512,1024,2048,4096,8192"
GPU_NAME="H100-SXM5-80GB"
OUTPUT="templates/gpu/h100-sxm5-80gb.yaml"

DENSE_MODELS=(
    llama-7b
    llama-13b
    llama-70b
    deepseek-7b
)

# --- Profile each model ---
for MODEL in "${DENSE_MODELS[@]}"; do
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
