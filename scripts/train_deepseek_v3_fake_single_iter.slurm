#!/bin/bash
#SBATCH --job-name=dsv3-fake-single-iter
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/dsv3_fake_%j.out
#SBATCH --error=jobs/dsv3_fake_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# DeepSeek-V3 (671B) single-iteration smoke test with --fake-process-group
#
# What this does:
#   - Simulates a 2048-GPU distributed run on a SINGLE GPU
#   --fake-process-group skips all NCCL collectives (no real multi-GPU needed)
#   - Runs exactly 1 training iteration (--train-iters 1 --exit-interval 1)
#   - Uses mock data so no dataset preparation is required
#   - Dumps a PyTorch memory snapshot so you can inspect memory usage
#
# Parallelism config (from examples/deepseek_v3_training.yaml):
#   TP=1, PP=16, EP=64, DP=2  ->  1*16*64*2 = 2048 GPUs total
#
# How to run:
#   sbatch scripts/train_deepseek_v3_fake_single_iter.slurm
# -----------------------------------------------------------------------------

module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

cd "$(dirname "$(realpath "$0")")/.."
source .venv/bin/activate

MEGATRON_PATH="experiments/validate_compute/megatron-lm"
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p jobs results

# Fake distributed env: we are rank 0 of a 2048-GPU job
export WORLD_SIZE=2048
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Memory profiling output
MEMORY_SNAPSHOT="results/dsv3_fake_iter1_memory.pickle"

PARALLELISM_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 16
    --expert-model-parallel-size 64
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

BATCH_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --train-iters 1
    --exit-interval 1
)

DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 32000
    --split 100,0,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 0
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 64
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --max-position-embeddings 4096
    --seq-length 4096
    --position-embedding-type rope
    --rotary-base 10000
    --make-vocab-size-divisible-by 3232
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --multi-latent-attention
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --qk-layernorm
)

MOE_ARGS=(
    --num-experts 256
    --moe-layer-freq "([0]*3+[1]*58)"
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-token-dispatcher-type alltoall
    --moe-router-pre-softmax
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-4
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --moe-permute-fusion
)

MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
)

MTP_ARGS=(
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

OPTIMIZER_ARGS=(
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 3.9e-6
    --min-lr 3.9e-7
    --lr-decay-style cosine
    --lr-warmup-samples 1
)

FAKE_ARGS=(
    --fake-process-group
    --no-check-for-nan-in-loss-and-grad
    --moe-router-force-load-balancing
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
    --disable-bias-linear
    --transformer-impl transformer_engine
    --manual-gc
    --manual-gc-interval 1
    --recompute-granularity selective
    --recompute-modules moe_act mlp layernorm mla_up_proj
    --eval-iters 0
)

python "${MEGATRON_PATH}/pretrain_gpt.py" \
    "${PARALLELISM_ARGS[@]}" \
    "${BATCH_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${MLA_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${FAKE_ARGS[@]}" \
    "${MEMORY_ARGS[@]}" \
    "${MISC_ARGS[@]}"

echo "=== Done ==="
echo "Memory snapshot: ${MEMORY_SNAPSHOT}"
echo "To visualize memory:  python -c \"import pickle, torch; d=pickle.load(open('${MEMORY_SNAPSHOT}','rb')); torch.cuda.memory._record_memory_history(d)\""
