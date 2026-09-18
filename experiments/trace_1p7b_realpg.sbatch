#!/bin/bash
# Capture a Simulon trace during a REAL 4-GPU run, to measure what the fake process group
# is costing us -- rather than assuming.
#
# WHY. Comparing the fake-PG traces against nsys on the same configs (2026-08-03):
#     tp2pp1-mbs1 stage0   trace  9495   nsys  9890    -4.0%
#     tp4pp1-mbs2 stage0   trace  9898   nsys 10802    -8.4%
#     tp2pp2-mbs2 stage0   trace  6037   nsys  6686    -9.7%
#     tp2pp2-mbs2 stage1   trace  9841   nsys 12812   -23.2%   <-- the last stage
# Every stage is measured low, and the LAST pipeline stage -- the one owning the
# 262144-vocab lm_head and vocab-parallel cross-entropy -- is measured very low. A real-PG
# trace has no such gap by construction.
#
# WHAT THIS JOB DECIDES. It runs tp2pp2-mbs2 (the cell with the worst known bias AND an
# existing nsys profile AND a physical val40 number) under a real process group, so all
# three can be compared directly. Then the SAME comparison for tp1pp1-mbs1, which has no
# collectives at all and should therefore agree with the fake-PG trace -- if it does not,
# the difference is the real-PG capture machinery, not the process group.
#
# PRE-REGISTERED (write down before looking):
#   * tp2pp2 stage1 real-PG kernel time should land near the nsys 12812 ms, NOT the
#     fake-PG 9841 ms. If it comes back ~9841 the fake PG is exonerated and the deficit is
#     something else (most likely power/thermal contention, which a 4-GPU run reproduces
#     but a solo tracing GPU does not).
#   * tp1pp1-mbs1 should agree with its fake-PG trace to within the ~9% GH200 session
#     spread. A larger difference indicts the capture path.
#
# COST: two short real runs, 4 GPUs, one node. No new physical validation runs are needed --
# the val40 ground truth for both cells is already on disk.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=01:30:00
#SBATCH --job-name=trace_1p7b_realpg
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --chdir=/e/scratch/e-sta-openeurollm/vanosch1
#SBATCH --output=/e/scratch/e-sta-openeurollm/vanosch1/logs/trace_1p7b_realpg-%j.log

set -uo pipefail
# Run off /e/scratch: the /e/home GPFS export goes stale mid-job on booster nodes and kills
# the whole step (see experiments/stage_repo.sh). Re-stage after any code change.
STAGE=/e/scratch/e-sta-openeurollm/vanosch1/simulon_stage
OUT=/e/scratch/e-sta-openeurollm/vanosch1/realpg_traces_v2
[ -f "$STAGE/simulon/cli/__init__.py" ] || { echo "FATAL: run stage_repo.sh first" >&2; exit 1; }
cd "$STAGE" || exit 1
mkdir -p "$OUT"

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
MEGATRON=/opt/simulon/vendor/Megatron-LM-traced

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=20081
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export OMP_NUM_THREADS=8

echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "started  : $(date)"

# CPU AFFINITY -- the fault that cost jobs 1225995 / 1231006 / 1301886. Without
# --cpus-per-task the job is still ALLOCATED all 288 CPUs (sacct says so at job AND step
# level, which is what misled the first diagnosis) but the `srun --ntasks=1` step can be
# BOUND to a narrow mask, so the four torchrun children share a fraction of the node. These
# runs are 79-83% host-side framework time, so that alone made every iteration 4-5x slower
# -- host-side only, kernels untouched, scaling with the microbatch count.
#
# Read the mask with sched_getaffinity, NOT with `nproc`: nproc deliberately honours
# OMP_NUM_THREADS, so with OMP_NUM_THREADS=8 it reports 8 no matter how many cores the step
# actually owns. That false positive aborted jobs 1303639 and 1445115 before they ran a
# single cell.
CORES=$(srun --ntasks=1 apptainer exec --bind /e/scratch/ "$SIF" \
        python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null | tail -1)
echo "affinity : ${CORES:-unknown} cores in the step's mask (expect 288)"
echo "nproc    : $(srun --ntasks=1 bash -c 'nproc') (OMP-limited; NOT an affinity reading)"
case "$CORES" in
    ''|*[!0-9]*) echo "WARN: could not read affinity; continuing" >&2 ;;
    *) if [ "$CORES" -lt 64 ]; then
           echo "FATAL: step is bound to $CORES cores. Traces captured this way have valid" >&2
           echo "       op COUNTS but wildly inflated host_ms and span timings." >&2
           exit 1
       fi ;;
esac

# run <label> <tp> <pp> <mbs>
run() {
    local label="$1" tp="$2" pp="$3" mbs="$4" out="$OUT/$1"; shift 4
    local extra="$*"
    echo "=============================================================="
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] $label present -> skip"; return 0; fi
    echo "[$(date +%H:%M:%S)] $label  tp=$tp pp=$pp mbs=$mbs $extra (REAL process group)"
    mkdir -p "$out"
    srun --ntasks=1 --gres=gpu:4 \
    apptainer exec --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$STAGE:/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=1 \
            PYTHONPATH=/opt/simulon:$MEGATRON \
            CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONDONTWRITEBYTECODE=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            torchrun --nproc-per-node 4 --nnodes 1 --max-restarts 0 \
              --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
              $MEGATRON/pretrain_gpt.py \
              --trace-real-process-group --trace-dir '$out' --trace-warmup-iters 6 \
              --train-iters 8 \
              --tensor-model-parallel-size $tp --pipeline-model-parallel-size $pp \
              --micro-batch-size $mbs --global-batch-size 256 \
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
              --data-parallel-sharding-strategy no_shard \
              --no-check-for-nan-in-loss-and-grad \
              --eval-interval 999999 --eval-iters 0 --save-interval 1000000 \
              --log-interval 1 --log-throughput $extra" 2>&1 | tail -30
    local n=$(ls "$out"/trace_rank_*.json 2>/dev/null | wc -l)
    echo "  -> $n rank traces in $out"
}

# WHICH CELLS. Real-PG capture exists ONLY to fix what the fake process group gets wrong.
# Measured against nsys, fake PG is low by 4.0% (tp2pp1-mbs1 s0), 8.4% (tp4pp1-mbs2 s0),
# 9.7% (tp2pp2-mbs2 s0) and 23.2% (tp2pp2-mbs2 s1 -- the LAST pipeline stage, which owns
# the 262k-vocab lm_head). So fake PG is trustworthy for TP, mbs and PP=1, and untrustworthy
# for PP>=2 and recompute. Capturing only the untrustworthy cells costs a third of the
# walltime for all of the value; the fake-PG registry (which is config-CLEAN, being built
# by `simulon trace` from workload.yaml) continues to serve the rest.
#
# Ordered cheapest-first so a walltime overrun costs the most expensive cell, not the whole
# job; already-captured cells are skipped. recompute is an extra flag on the -rc entry.
run tp1pp1-mbs1  1 1 1              # control: no collectives, must match fake PG closely
run tp1pp2-mbs1  1 2 1
run tp1pp4-mbs1  1 4 1
run tp1pp1-mbs1-rc 1 1 1 --recompute-activations
run tp2pp2-mbs2  2 2 2

# The rest of the grid. Restored to the full 11 cells: with the CPU-affinity fault fixed
# the runs are ~4x faster, so the whole grid now costs less than the 5-cell subset did, and
# a complete real-PG registry allows like-for-like comparison against fake-PG on every axis.
run tp1pp1-mbs2  1 1 2
run tp1pp1-mbs4  1 1 4
run tp2pp1-mbs8  2 1 8
run tp2pp1-mbs1  2 1 1
run tp4pp1-mbs2  4 1 2
run tp4pp1-mbs1  4 1 1

echo "=============================================================="
echo "=== real-PG vs fake-PG vs nsys ==="
OUT=$OUT python3 - <<'PYEOF'
import json, os, pathlib
# Real-PG numbers only. The fake-PG registry is NOT visible here: stage_repo.sh excludes
# trace payloads from the stage, and jobs must not read /e/home (stale-handle fault). Do
# the fake-vs-real comparison on the login node, where both registries exist.
real = pathlib.Path(os.environ['OUT'])
NSYS = {('tp2pp2-mbs2', 0): 6686, ('tp2pp2-mbs2', 2): 12812}
hdr = ('cell / rank', 'kernel ms', 'host_ops', 'nsys', 'vs nsys')
print(f'{hdr[0]:<24}{hdr[1]:>11}{hdr[2]:>10}{hdr[3]:>8}{hdr[4]:>10}')
for cell in sorted(x.name for x in real.iterdir() if x.is_dir()):
    for f in sorted((real / cell).glob('trace_rank_*.json')):
        r = int(f.stem.split('_')[-1])
        t = json.load(open(f))
        k = t.get('total_kernel_device_ms') or 0.0
        ops = t.get('total_host_ops') or 0
        ns = NSYS.get((cell, r))
        vs = f'{100*(k-ns)/ns:+.1f}%' if ns else ''
        print(f'{cell + " r" + str(r):<24}{k:>11.0f}{ops:>10}{(ns or 0):>8}{vs:>10}')
PYEOF

echo
echo "NEXT: rsync $OUT/ into templates/gpu/gh200_jupiter-1p7b-realpg/traces/ and re-score:"
echo "  SIMULON_1P7B_REGISTRY=gh200_jupiter-1p7b-realpg SIMULON_HOST_COST_US=18.2 \\"
echo "    PYTHONPATH=. python experiments/analyze_1p7b.py"
