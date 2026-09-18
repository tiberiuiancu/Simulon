#!/bin/bash
# Close the two remaining structural residuals: recompute (+10.4%) and PP at dp=2 (+18.6%).
# Each experiment captures BOTH the val40 physical measurement and the real-PG trace for the
# same cell in the same session, so the comparison is trace-matched.
#
# ============================================================================
# EXPERIMENT A -- is the recompute per-op cost real, or is n=1 a coincidence?
# ============================================================================
# host_cost_us transfers across TP and mbs because the op MIX is invariant there. Recompute
# changes the mix: on tp1pp1-mbs1 it adds 100352 ops (+19.4%) for only 307 ms of kernel time
# (+3.4%) -- framework-heavy, GPU-cheap work. Solving the two cells' ceilings jointly
# (19.16 vs 17.38 us/op) implies recompute ops cost ~8.2 us/op against ~19.2 for normal
# ops, which is mechanically plausible (ops replayed under no_grad skip autograd graph
# construction) but rests on ONE cell, so it is not shippable.
#
# tp2pp1-mbs1 is host-bound (ceiling 19.51 us/op) and at a different TP, so its rc variant
# is an independent second point.
#
# PRE-REGISTERED. tp2pp1-mbs1 has 1076652 ops. If rc adds the same +19.4%, that is ~208700
# extra ops, and the two models predict measurably different iteration times:
#     uniform 19.0 us/op for every op  ->  24.4 s/iteration
#     19.0 normal + 8.2 for recompute  ->  22.2 s/iteration
# The 10% separation is well outside the ~2% noise floor at TP=2.
#   * ~22.2 s  => two-class op cost CONFIRMED at n=2 and shippable as a measured quantity.
#   * ~24.4 s  => uniform cost is right and the tp1pp1-rc miss is something else entirely;
#                 do NOT introduce a second constant.
#   * neither  => the marginal cost is not transferable; recompute stays fenced off.
#
# ============================================================================
# EXPERIMENT B -- is the PP failure about pipeline depth, or about DP?
# ============================================================================
# Of the three PP cells, the two with dp=1 are ACCEPTABLE (tp1pp4 +7.9%, tp2pp2 -7.0%) and
# the ONLY one with dp=2 FAILS (tp1pp2 +18.6%). That is suggestive but confounded: those
# cells also differ in TP and depth.
#
# This runs tp1pp2 on TWO GPUs with gbs=128, which makes dp=1 while keeping the microbatch
# count at 128 -- identical per-rank program to the 4-GPU gbs=256 cell already measured.
# The ONLY thing that changes is the data-parallel dimension. gbs=128 deviates from the
# campaign's fixed gbs=256, which is fine here because both the measurement and the
# simulation use it; this is a paired comparison, not an absolute-accuracy point.
#
# PRE-REGISTERED.
#   * error falls to within about +-8% (the band the other dp=1 PP cells sit in)
#       => the residual is DP-coupled, not pipeline depth. Suspect the distributed-optimizer
#          RS+AG serialization already recorded at PP=1, now interacting with the per-p2p
#          torch.cuda.synchronize().
#   * error stays near +18%
#       => it is pipeline depth, and the device-sync model over-destroys pipelining. DP is
#          exonerated.
#
# COST: ~1 h. 2 val40 runs + 2 real-PG traces, one node.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --job-name=close_rc_and_pp
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --chdir=/e/scratch/e-sta-openeurollm/vanosch1
#SBATCH --output=/e/scratch/e-sta-openeurollm/vanosch1/logs/close_rc_and_pp-%j.log

set -uo pipefail
# Run off /e/scratch: the /e/home GPFS export goes stale mid-job on booster nodes and kills
# the whole step. Re-stage after any code change (experiments/stage_repo.sh).
STAGE=/e/scratch/e-sta-openeurollm/vanosch1/simulon_stage
OUT=/e/scratch/e-sta-openeurollm/vanosch1/realpg_traces_v2
PHYS=/e/scratch/e-sta-openeurollm/vanosch1/v40_out
[ -f "$STAGE/simulon/cli/__init__.py" ] || { echo "FATAL: run stage_repo.sh first" >&2; exit 1; }
cd "$STAGE" || exit 1
mkdir -p "$OUT" "$PHYS"

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
MEGATRON=/opt/simulon/vendor/Megatron-LM-traced

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=20083
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export OMP_NUM_THREADS=8

echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "started  : $(date)"

# The 1.7B body. Identical across every cell; only the parallel/mbs/rc knobs differ.
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
  --data-parallel-sharding-strategy no_shard \
  --no-check-for-nan-in-loss-and-grad \
  --eval-interval 999999 --eval-iters 0 --save-interval 1000000 \
  --log-interval 1 --log-throughput"

# val40 <physdir> <ngpu> <tp> <pp> <mbs> <gbs> [extra...]
# 40 iterations, unprofiled, ONE log file per run (never concatenate -- mixing jobs
# fabricates fake warmup regimes). Mean of iters 21..40 is the measurement.
val40() {
    local dir="$1" ngpu="$2" tp="$3" pp="$4" mbs="$5" gbs="$6"; shift 6
    local extra="$*" log="$PHYS/$dir/slurm-${SLURM_JOB_ID}.log"
    echo "=============================================================="
    if compgen -G "$PHYS/$dir/slurm-*.log" > /dev/null; then
        echo "[$(date +%H:%M:%S)] val40 $dir present -> skip"; return 0; fi
    echo "[$(date +%H:%M:%S)] val40 $dir  ngpu=$ngpu tp=$tp pp=$pp mbs=$mbs gbs=$gbs $extra"
    mkdir -p "$PHYS/$dir"
    srun --ntasks=1 --gres=gpu:$ngpu \
    apptainer exec --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$STAGE:/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon:$MEGATRON \
            CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONDONTWRITEBYTECODE=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            torchrun --nproc-per-node $ngpu --nnodes 1 --max-restarts 0 \
              --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
              $MEGATRON/pretrain_gpt.py --train-iters 40 \
              --tensor-model-parallel-size $tp --pipeline-model-parallel-size $pp \
              --micro-batch-size $mbs --global-batch-size $gbs \
              $BODY $extra" > "$log" 2>&1
    echo "  -> $(grep -c 'elapsed time per iteration' "$log") iterations logged"
    grep 'elapsed time per iteration' "$log" | tail -3
}

# trace <cell> <ngpu> <tp> <pp> <mbs> <gbs> [extra...]
trace() {
    local cell="$1" ngpu="$2" tp="$3" pp="$4" mbs="$5" gbs="$6"; shift 6
    local extra="$*" out="$OUT/$cell"
    echo "=============================================================="
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] trace $cell present -> skip"; return 0; fi
    echo "[$(date +%H:%M:%S)] trace $cell  ngpu=$ngpu tp=$tp pp=$pp mbs=$mbs gbs=$gbs $extra"
    mkdir -p "$out"
    srun --ntasks=1 --gres=gpu:$ngpu \
    apptainer exec --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$STAGE:/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=1 \
            PYTHONPATH=/opt/simulon:$MEGATRON \
            CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONDONTWRITEBYTECODE=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            torchrun --nproc-per-node $ngpu --nnodes 1 --max-restarts 0 \
              --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
              $MEGATRON/pretrain_gpt.py \
              --trace-real-process-group --trace-dir '$out' --trace-warmup-iters 6 \
              --train-iters 8 \
              --tensor-model-parallel-size $tp --pipeline-model-parallel-size $pp \
              --micro-batch-size $mbs --global-batch-size $gbs \
              $BODY $extra" 2>&1 | tail -20
    echo "  -> $(ls "$out"/trace_rank_*.json 2>/dev/null | wc -l) rank traces in $out"
}

# --- B first: it is the cheaper of the two, so a walltime overrun costs A, not both ---
val40 v40_qwen3_1p7b_1_pp2_tp1_mbs1_rcFalse_vppNone_spFalse_gbs128_ngpu2 2 1 2 1 128
trace tp1pp2-mbs1-dp1 2 1 2 1 128

# --- A: second host-bound recompute cell ---
val40 v40_qwen3_1p7b_1_pp1_tp2_mbs1_rcTrue_vppNone_spFalse_gbs256 4 2 1 1 256 --recompute-activations
trace tp2pp1-mbs1-rc 4 2 1 1 256 --recompute-activations

echo "=============================================================="
echo "=== measured (mean of iters 21..40, the val40 protocol) ==="
PHYS=$PHYS OUT=$OUT python3 - <<'PYEOF'
import glob, json, os, re, statistics, pathlib
for d in sorted(glob.glob(os.environ['PHYS'] + "/*/")):
    logs = sorted(glob.glob(d + "slurm-*.log"))
    if not logs:
        continue
    # newest complete log only -- never concatenate across jobs
    for f in reversed(logs):
        v = [float(x) for x in re.findall(
            r"elapsed time per iteration \(ms\): ([0-9.]+)", open(f, errors="ignore").read())]
        if len(v) >= 40:
            print(f"{pathlib.Path(d).name:<62}{statistics.mean(v[20:40]):>9.0f} ms")
            break
    else:
        print(f"{pathlib.Path(d).name:<62}{'INCOMPLETE':>9} ({len(v)} iters)")
print()
print("=== traces ===")
for c in ("tp1pp2-mbs1-dp1", "tp2pp1-mbs1-rc"):
    p = pathlib.Path(os.environ['OUT']) / c
    for f in sorted(p.glob("trace_rank_*.json")):
        t = json.load(open(f))
        print(f"{c + ' ' + f.stem.split('_')[-1]:<24}"
              f"kernel={t.get('total_kernel_device_ms') or 0:>9.0f} ms"
              f"  ops={t.get('total_host_ops') or 0:>10}")
PYEOF

echo
echo "NEXT (login node):"
echo "  rsync -a $PHYS/ /e/home/jusers/vanosch1/jupiter/oellm-autoexp/output/qwen3-1p7b/"
echo "  rsync -a $OUT/ templates/gpu/gh200_jupiter-1p7b-realpg/traces/"
echo "  # then backfill workload.yaml for the two new cells and re-score"
