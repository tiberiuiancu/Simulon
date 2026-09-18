#!/bin/bash
# Why is tp2pp1-mbs1 91 s under our harness and 21 s under the campaign's?
#
# STATE OF THE INVESTIGATION. Two hypotheses have already been REFUTED by measurement:
#   1. --data-parallel-sharding-strategy (vendored Megatron defaults it to
#      'optim_grads_params', production to 'no_shard'). Tested head-to-head in job 1301886:
#      no_shard 90961/88676/92670 ms vs optim_grads_params 93302/90749/93194 ms. Real, but
#      worth ~1.5%, not 4.3x.
#   2. CPU starvation from a missing --cpus-per-task. sacct says the step was allocated all
#      288 CPUs.
#
# WHAT IS ESTABLISHED. The GPU work is IDENTICAL: the vendored real-PG trace of this cell
# reports total_kernel_device_ms = 9962 ms against production nsys 9890 ms (+0.7%). So the
# entire ~70 s difference is host-side -- roughly 84 us/op against 19.7. This is a
# framework-dispatch difference, not a compute one.
#
# WHAT IS STILL CONFOUNDED. Four variables separate the 21 s path from the 91 s path:
#   (a) Megatron checkout   vendored Megatron-LM-traced (2026-06-03, our tracer fork)
#                           vs production submodules/Megatron-LM (2026-03-06)
#   (b) OMP_NUM_THREADS     ours 8, the nsys/campaign harness $SLURM_CPUS_PER_TASK = 288
#   (c) launcher            our srun+torchrun vs the oellm-autoexp launcher
#   (d) filesystem          our /e/scratch stage vs the campaign's /e/home checkout
# Both trace registries (fake-PG AND real-PG) were captured through OUR harness, so NEITHER
# can be assumed clean until this is settled -- host_ops is exactly the quantity a host-side
# slowdown corrupts, and it is what host_cost_us is derived from.
#
# DESIGN. Four arms, same cell, same node, same container, same session. (d) is removed for
# every arm by running BOTH checkouts from /e/scratch. Each arm changes ONE thing from the
# arm above it, so whichever arm moves identifies the cause.
#
#   A  vendored   OMP=8     -- reproduce the 91 s, else the effect is not stable
#   B  PRODUCTION OMP=8     -- isolates the CHECKOUT
#   C  vendored   OMP=288   -- isolates OMP_NUM_THREADS
#   D  PRODUCTION OMP=288   -- positive control; must land near the archived 21.2 s
#
# PRE-REGISTERED READING:
#   * A~91 and D~21          -> the effect is real and inside (a)+(b); read B and C.
#   * B~21, C~91             -> the CHECKOUT is the cause. The tracer fork cannot be used
#                               for calibration; port the tracer onto production Megatron.
#   * C~21, B~91             -> OMP_NUM_THREADS is the cause. Trivial fix, and every trace
#                               we hold is invalid but cheaply re-capturable.
#   * B~21 AND C~21          -> two sufficient causes; fix both, and distrust any conclusion
#                               drawn from a run that had either.
#   * A NOT ~91              -> the slowdown is not reproducible; suspect node-to-node
#                               variation and re-examine whether 91 s was ever representative.
#   * D NOT ~21              -> the positive control FAILED. Then the campaign's 21.2 s is
#                               not reproducible from this harness at all, and the archived
#                               numbers -- not our traces -- are what need re-examining.
#     That last branch is the important one: it is the only outcome that would invalidate
#     the ground truth rather than the traces.
#
# COST: <=1 h. 6 iterations per arm; iterations 4-6 are the measurement.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=01:15:00
#SBATCH --job-name=isolate_slowdown
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --chdir=/e/scratch/e-sta-openeurollm/vanosch1
#SBATCH --output=/e/scratch/e-sta-openeurollm/vanosch1/logs/isolate_slowdown-%j.log

set -uo pipefail
STAGE=/e/scratch/e-sta-openeurollm/vanosch1/simulon_stage
PRODMEG=/e/scratch/e-sta-openeurollm/vanosch1/megatron_prod
[ -f "$STAGE/simulon/cli/__init__.py" ] || { echo "FATAL: run stage_repo.sh first" >&2; exit 1; }
[ -f "$PRODMEG/pretrain_gpt.py" ] || { echo "FATAL: production Megatron not staged" >&2; exit 1; }
cd "$STAGE" || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=20087
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3

echo "SM clock  : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "node      : $(hostname)"
echo "AllocCPUS : ${SLURM_CPUS_PER_TASK:-unset}"
echo "started   : $(date)"

# tp2pp1-mbs1. Deliberately NO --data-parallel-sharding-strategy: it is worth ~1.5% and
# leaving it at each checkout's own default keeps the arms comparable to what was captured.
CFG="--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
  --micro-batch-size 1 --global-batch-size 256 --train-iters 6 \
  --seq-length 4096 --max-position-embeddings 4096 \
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
  --log-interval 1 --log-throughput"

# arm <label> <megatron-dir-in-container> <omp>
arm() {
    local label="$1" meg="$2" omp="$3"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] ARM $label   megatron=$meg  OMP_NUM_THREADS=$omp"
    srun --ntasks=1 --gres=gpu:4 \
    apptainer exec --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$STAGE:/opt/simulon" --bind "$PRODMEG:/opt/megatron_prod" "$SIF" \
        bash -c "cd $meg && OMP_NUM_THREADS=$omp PYTHONPATH=$meg \
            CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONDONTWRITEBYTECODE=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            torchrun --nproc-per-node 4 --nnodes 1 --max-restarts 0 \
              --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
              $meg/pretrain_gpt.py $CFG" 2>&1 \
      | grep -E "data_parallel_sharding_strategy=|elapsed time per iteration|Error|Traceback" \
      | sed -e "s/^.*data_parallel_sharding_strategy=/  strategy: /" -e "s/,.*$//" \
            -e "s/^.*iteration *\([0-9]*\)\/.*(ms): \([0-9.]*\).*$/  iter \1: \2 ms/" \
      | head -12
}

arm "A vendored   OMP=8  " /opt/simulon/vendor/Megatron-LM-traced 8
arm "B PRODUCTION OMP=8  " /opt/megatron_prod                     8
arm "C vendored   OMP=288" /opt/simulon/vendor/Megatron-LM-traced 288
arm "D PRODUCTION OMP=288" /opt/megatron_prod                     288

echo "=============================================================="
echo "Compare iterations 4-6. Archived val40 for tp2pp1-mbs1 = 21210 ms."
echo "  D must land near 21 s -- it is the positive control. If it does not, the"
echo "  archived ground truth is what needs re-examining, not the traces."
echo "  Whichever of B / C moves away from A identifies the cause."
echo "finished  : $(date)"
