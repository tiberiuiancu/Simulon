#!/bin/bash
#SBATCH --job-name=llama-fake-single-iter
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/llama_fake_%j.out
#SBATCH --error=jobs/llama_fake_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Llama 3 dense model single-iteration smoke test with --fake-process-group
#
# Parallelism: TP=2, PP=2 (DP derived = 8 / (2*2) = 2)
# World size: 8
# -----------------------------------------------------------------------------

module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

cd "$(dirname "$(realpath "$0")")/.."
source .venv/bin/activate

MEGATRON_PATH="experiments/validate_compute/megatron-lm"
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p jobs results

# Fake distributed env: we are rank 0 of an 8-GPU job
export WORLD_SIZE=8
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Memory profiling output
MEMORY_SNAPSHOT="results/llama_fake_iter1_memory.pickle"

PARALLELISM_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --context-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

BATCH_ARGS=(
    --micro-batch-size 2
    --global-batch-size 15360
    --train-iters 10
    --exit-interval 10
)

DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 128256
    --split 100,0,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 0
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --max-position-embeddings 4096
    --seq-length 4096
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

OPTIMIZER_ARGS=(
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 0.00015
    --min-lr 0.00001
    --lr-decay-style cosine
    --lr-warmup-iters 2
)

FAKE_ARGS=(
    --fake-process-group
    --no-check-for-nan-in-loss-and-grad
)

MEMORY_ARGS=(
    --record-memory-history
    --memory-snapshot-path "${MEMORY_SNAPSHOT}"
)

MISC_ARGS=(
    --bf16
    --log-interval 1
    --log-throughput
    --log-params-norm
    --log-num-zeros-in-grad
    --logging-level 20
    --distributed-timeout-minutes 60
    --transformer-impl transformer_engine
    --manual-gc
    --manual-gc-interval 1
    --eval-iters 0
    --eval-interval 100000
)

python "${MEGATRON_PATH}/pretrain_gpt.py" \
    "${PARALLELISM_ARGS[@]}" \
    "${BATCH_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${FAKE_ARGS[@]}" \
    "${MEMORY_ARGS[@]}" \
    "${MISC_ARGS[@]}"

echo "=== Done ==="
echo "Memory snapshot: ${MEMORY_SNAPSHOT}"
