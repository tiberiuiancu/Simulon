#!/bin/bash
# Confirm the config fault before spending another 3 h re-capturing the grid.
#
# THE FAULT. experiments/trace_1p7b_realpg.sbatch and close_rc_and_pp.sbatch never passed
# --data-parallel-sharding-strategy, and this Megatron DEFAULTS it to 'optim_grads_params'
# (arguments.py:2799) -- parameters sharded across DP and all-gathered per microbatch. The
# archived val40 campaign, and CANON_CFG which the simulator itself uses, set 'no_shard'.
# Confirmed from the two DistributedDataParallelConfig banners:
#     archived  data_parallel_sharding_strategy='no_shard'     check_for_nan_in_grad=False
#     mine      data_parallel_sharding_strategy='optim_grads_params'  check_for_nan_in_grad=True
#
# The symptom is a 1.4-5x slowdown that scales with the microbatch count:
#     cell            my run    archived   ratio   nmb
#     tp4pp1-mbs1      205 s      41.4 s    5.0x   256
#     tp2pp1-mbs1      100 s      21.2 s    4.7x   128
#     tp1pp2-mbs1     44.5 s      14.7 s    3.0x   128
#     tp1pp1-mbs2     21.2 s       9.5 s    2.2x    32
#     tp2pp1-mbs8     15.3 s      11.1 s    1.4x    16
# It is NOT the profiler: an untraced val40 (26.0 s) matched its own traced UNPROFILED
# iterations (25.4-26.0 s) exactly.
#
# WHAT THIS DECIDES. Runs tp2pp1-mbs1 twice back to back, same node, same clocks, same
# session -- once with the flags the archived campaign used, once with the (faulty) current
# BODY. Only the sharding strategy and the nan check differ. No tracing: this is purely
# about iteration time, so it is cheap.
#
# PRE-REGISTERED (write down before running):
#   * FIXED arm lands near the archived 21.2 s (say 19-24 s). Then the fault is fully
#     explained, every real-PG trace in job 1225995 is invalid, and host_cost_us=19.0 --
#     derived as archived_iteration / mismatched_traced_ops -- must be re-derived from
#     config-matched traces.
#   * FIXED arm still ~100 s. Then sharding is NOT the cause, something else in the arg
#     list is, and re-capturing would waste 3 h. Diff the two argument dumps again before
#     doing anything else.
#   * BROKEN arm should reproduce ~100 s. If it does not, the slowdown is environmental
#     (node, clocks, contention) rather than configuration, and BOTH conclusions above are
#     void -- that is why the control is run in the same job rather than compared to a log.
#
# COST: ~20 min, 1 node. 6 iterations per arm; iterations 4-6 are the measurement.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=00:40:00
#SBATCH --job-name=verify_no_shard
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --chdir=/e/scratch/e-sta-openeurollm/vanosch1
#SBATCH --output=/e/scratch/e-sta-openeurollm/vanosch1/logs/verify_no_shard-%j.log

set -uo pipefail
STAGE=/e/scratch/e-sta-openeurollm/vanosch1/simulon_stage
[ -f "$STAGE/simulon/cli/__init__.py" ] || { echo "FATAL: run stage_repo.sh first" >&2; exit 1; }
cd "$STAGE" || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
MEGATRON=/opt/simulon/vendor/Megatron-LM-traced

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=20085
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export OMP_NUM_THREADS=8

echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "node     : $(hostname)"
echo "started  : $(date)"

BODY="--seq-length 4096 --max-position-embeddings 4096 \
  --num-layers 28 --hidden-size 2048 --ffn-hidden-size 6144 \
  --num-attention-heads 16 --num-query-groups 8 --group-query-attention \
  --kv-channels 128 --swiglu --normalization RMSNorm --qk-layernorm \
  --position-embedding-type rope --use-flash-attn --bf16 \
  --attention-softmax-in-fp32 --attention-dropout 0.0 --hidden-dropout 0.0 \
  --disable-bias-linear --no-overlap-p2p-communication \
  --tokenizer-type NullTokenizer --vocab-size 262144 --padded-vocab-size 262272 \
  --mock-data --dataloader-type cyclic \
  --optimizer adam --lr 1e-8 --min-lr 0.0 --lr-decay-style WSD \
  --lr-decay-iters 1 --lr-wsd-decay-iters 1 --lr-wsd-decay-style linear \
  --weight-decay 0.05 --clip-grad 1.0 --adam-beta2 0.95 \
  --use-distributed-optimizer --distributed-timeout-minutes 30 \
  --eval-interval 999999 --eval-iters 0 --save-interval 1000000 \
  --log-interval 1 --log-throughput \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
  --micro-batch-size 1 --global-batch-size 256 --train-iters 6"

# The two flags the archived campaign had and the faulty BODY did not.
FIXED="--data-parallel-sharding-strategy no_shard --no-check-for-nan-in-loss-and-grad"

arm() {
    local label="$1"; shift
    local extra="$*"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] ARM: $label   extra='$extra'"
    srun --ntasks=1 --gres=gpu:4 \
    apptainer exec --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$STAGE:/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon:$MEGATRON \
            CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONDONTWRITEBYTECODE=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            torchrun --nproc-per-node 4 --nnodes 1 --max-restarts 0 \
              --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
              $MEGATRON/pretrain_gpt.py $BODY $extra" 2>&1 \
      | grep -E "data_parallel_sharding_strategy=|elapsed time per iteration" \
      | sed -e "s/^.*data_parallel_sharding_strategy=/  STRATEGY: /" \
            -e "s/,.*$//" -e "s/^.*iteration *\([0-9]*\)\/.*(ms): \([0-9.]*\).*$/  iter \1: \2 ms/"
}

arm "FIXED  (no_shard, as the archived campaign ran)" $FIXED
arm "BROKEN (current BODY: optim_grads_params default)"

echo "=============================================================="
echo "READ IT AS: compare iterations 4-6 of each arm."
echo "  archived val40 for this cell (tp2pp1-mbs1) = 21210 ms"
echo "  FIXED ~= 21 s  AND  BROKEN ~= 100 s  -> fault confirmed, re-capture the grid with"
echo "     experiments/trace_1p7b_realpg.sbatch (already corrected) into realpg_traces_v2,"
echo "     then re-derive host_cost_us from config-matched traces."
echo "  FIXED still ~100 s -> sharding is NOT the cause; do NOT re-capture, diff the arg"
echo "     dumps again."
echo "finished : $(date)"
