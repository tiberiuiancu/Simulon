#!/bin/bash
# Two measurements the simulator currently DERIVES instead of measuring. Both need >1 GPU,
# both are minutes long, and each one either confirms a model or falsifies it.
#
# ---------------------------------------------------------------------------------------
# EXPERIMENT 1 -- the data-parallel collectives, measured for the first time.
#
# The tracer instruments P2P (p2p_communication.py) and the TP collectives
# (tensor_parallel/mappings.py, extensions/transformer_engine.py) and NOTHING in
# megatron/core/distributed/, so every trace ever captured contains zero DP traffic. At the
# DP=1 capture topology that is invisible -- the collectives are genuine no-ops -- and at
# DP=128 it is the dominant scale-out cost. Simulon therefore SYNTHESIZES them
# (trace_tracer._inject_dp_grad_sync: bytes from the parameter count, placement from the
# overlap flags), which took 512n PP2 M=8 from -18.8% to +3.8% -- good, but derived.
#
# This job captures them: `_record_dp_collective` (added to param_and_grad_buffer.py) now
# records the distributed optimizer's grad reduce-scatter and param all-gather, with bytes =
# the FULL bucket buffer, the same convention nccl-tests reports and the synthesis emits.
# DP=4 is the smallest world where they are real work.
#
# PRE-REGISTERED (Qwen3-32B, 33.890e9 params; TP4/PP2 -> 4.236e9 params per rank):
#   * TOTAL per rank per iteration: ReduceScatter 16,944,988,160 B (fp32 grads, 15.78 GiB)
#     and AllGather 8,472,494,080 B (bf16 params, 7.89 GiB). Match => _dp_grad_sync_bytes is
#     right and the synthesis can stay as the fallback for DP=1 traces. Mismatch => read the
#     ratio: 2x on the RS means grads are NOT fp32 here, a smaller factor means the buffer is
#     padded or the layer split is not uniform (the synthesis assumes N/(TP*PP) exactly).
#   * COUNT: > 1 event per collective means Megatron buckets them. The bucket size is what
#     sets overlap granularity, which the replay currently models as one monolithic async
#     collective -- if buckets are ~40 MB there are ~400 of them and the exposure model has
#     to change. This number is the point of the experiment as much as the total is.
#   * GROUP: group_ranks must be a DP group -- {0,4,8,12} for TP4/PP2 (rank = tp + 4*dp +
#     16*pp), i.e. one GPU on each of 4 nodes = topology 4n1g, NOT 4n4g. Confirms which
#     measured busbw curve the replay must index.
#   * PLACEMENT (cell A vs cell B): with the overlap flags ON the RS events should interleave
#     with the backward slots and the AG with the next forward; with them OFF both should sit
#     at/after the last backward. That is exactly what _inject_dp_grad_sync assumes; cell B is
#     the control that says whether the assumption is real or just plausible.
#
# ---------------------------------------------------------------------------------------
# EXPERIMENT 2 -- the interleaved schedule, with the chunk ids it was missing.
#
# 98 of the 127 measured Qwen3-32B configs -- every production run and everything above 16
# nodes -- are blocked on one thing: Simulon has no interleaved-1F1B DAG with explicit
# layouts. Building it needs the real slot order, and the tracer keyed interleaved slots by
# virtual_microbatch_id only, so a slot could not be mapped to the layers it ran
# (schedules.py:1299). `model_chunk_id` is now recorded on every interleaved slot; this
# captures the production 16-chunk layout with it.
#
# PRE-REGISTERED:
#   * Every non-step slot carries model_chunk_id; slot count = M * chunks_per_stage * 2 + 1
#     (M=64, 4 chunks -> 513), which check_trace_health now expects.
#   * Per-chunk kernel time tracks the chunk's LAYER COUNT in the layout
#     (Et*5|t*4|...|t*3L -> 5,4,4,4 layers on stage 0's chunks, embedding on the first,
#     lm_head+CE on the last). Job 1836105 showed this holds per STAGE to ~2%; if it holds
#     per CHUNK, chunk compute = layers_in_chunk x t_layer and the DAG synthesis needs no
#     per-layout traces at all -- one trace per (TP, model) covers every layout.
#     If it does NOT hold, short chunks are less efficient and that is a real effect the
#     PP*VPP<=16 rule in SCALING_GUIDE is probably measuring.
#   * Cell D adds DP=2 on top: the production shape (layout + chunks + real DP collectives)
#     in one trace, so experiments 1 and 2 can be checked together rather than separately.
#
# COST: 8 nodes, ~35 min. No training run. Cells are independent sruns; a failure in one
# cannot take the others down, and re-submitting skips whatever already landed.
#
# NOTE ON ENV: matched to jobs 1829600/1836105 (the existing registry) so compute stays
# comparable, which means NCCL_NET_GDR_LEVEL is NOT set to production's 0. These cells are
# read for SIZES, COUNTS, GROUPS and ORDER -- not for DP collective timing, which Simulon
# takes from the measured busbw curves.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=00:45:00
#SBATCH --job-name=trace_32b_dpsync
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --chdir=/e/scratch/e-sta-openeurollm/vanosch1
#SBATCH --output=/e/scratch/e-sta-openeurollm/vanosch1/logs/trace_32b_dpsync-%j.log

set -uo pipefail
# Run off /e/scratch: jobs must not touch /e/home (stale-handle fault, see stage_repo.sh).
# RE-RUN experiments/stage_repo.sh after any code change -- this job needs the tracer edits
# in vendor/Megatron-LM-traced (param_and_grad_buffer.py, schedules.py, tracer.py).
STAGE=/e/scratch/e-sta-openeurollm/vanosch1/simulon_stage
OUT=/e/scratch/e-sta-openeurollm/vanosch1/traces_32b_target
[ -f "$STAGE/simulon/cli/__init__.py" ] || { echo "FATAL: run stage_repo.sh first" >&2; exit 1; }
grep -q "_record_dp_collective" "$STAGE/vendor/Megatron-LM-traced/megatron/core/distributed/param_and_grad_buffer.py" \
  || { echo "FATAL: staged Megatron lacks the DP-collective hook -- re-run stage_repo.sh" >&2; exit 1; }
grep -q "model_chunk_id" "$STAGE/vendor/Megatron-LM-traced/megatron/core/instrumentation/tracer.py" \
  || { echo "FATAL: staged tracer lacks model_chunk_id -- re-run stage_repo.sh" >&2; exit 1; }
cd "$STAGE" || exit 1
mkdir -p "$OUT"

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif      # production container
MEGATRON=/opt/simulon/vendor/Megatron-LM-traced
LAYOUT='Et*5|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3L'   # config-1827154

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${NODES[0]}
PORT=20300

echo "container: $SIF"
echo "nodes    : ${#NODES[@]}  (${NODES[*]})"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "started  : $(date)"

CORES=$(srun -N1 --ntasks=1 apptainer exec --bind /e/scratch/ "$SIF" \
        python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null | tail -1)
echo "affinity : ${CORES:-unknown} cores in the step's mask (expect 288)"
case "$CORES" in
    ''|*[!0-9]*) echo "WARN: could not read affinity; continuing" >&2 ;;
    *) [ "$CORES" -lt 64 ] && { echo "FATAL: step bound to $CORES cores" >&2; exit 1; } ;;
esac

# Production-pinned model + training args (models/qwen3_dense_32b.yaml + config-1827154.yaml).
COMMON="--tensor-model-parallel-size 4 --sequence-parallel --context-parallel-size 1
  --micro-batch-size 1 --global-batch-size 64
  --seq-length 4096 --max-position-embeddings 4096
  --num-layers 64 --hidden-size 5120 --ffn-hidden-size 25600
  --num-attention-heads 64 --num-query-groups 8 --group-query-attention --kv-channels 128
  --swiglu --normalization RMSNorm --norm-epsilon 1e-5 --qk-layernorm
  --position-embedding-type rope --rotary-base 100000 --rotary-percent 1.0
  --untie-embeddings-and-output-weights --disable-bias-linear --init-method-std 0.02
  --use-flash-attn --attention-backend auto --attention-softmax-in-fp32
  --attention-dropout 0.0 --hidden-dropout 0.0 --bf16
  --transformer-impl transformer_engine
  --cross-entropy-loss-fusion --cross-entropy-fusion-impl te
  --use-distributed-optimizer
  --data-parallel-sharding-strategy no_shard --manual-gc
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8
  --clip-grad 1.0 --weight-decay 0.1 --lr 1e-8 --min-lr 0.0
  --lr-decay-style WSD --lr-decay-iters 1 --lr-wsd-decay-iters 1 --lr-wsd-decay-style linear
  --tokenizer-type NullTokenizer --vocab-size 262144 --padded-vocab-size 262144
  --mock-data --dataloader-type cyclic --seed 1234
  --train-iters 8 --trace-warmup-iters 6
  --eval-interval 999999 --eval-iters 0 --save-interval 1000000
  --log-interval 1 --log-throughput --distributed-timeout-minutes 30"
# Collapse to ONE line: expanded inside bash -c "...", where a newline is a command separator.
COMMON=$(echo $COMMON)

ENV="SIMULON_TRACE_KERNEL_TIME=1 PYTHONPATH=/opt/simulon:$MEGATRON PYTHONDONTWRITEBYTECODE=1
  CUDA_DEVICE_MAX_CONNECTIONS=1 UB_SKIPMC=1 NCCL_NVLS_ENABLE=0
  NCCL_SOCKET_IFNAME=ib0 NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_AVOID_RECORD_STREAMS=1
  TORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8 OMP_NUM_THREADS=8"
ENV=$(echo $ENV)

# write_workload <out> <pp> <nodes> <ogr:0|1> <layout|->
write_workload() {
    python3 - "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import sys, yaml
out, pp, nodes, ogr, layout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
cfg = {
    "num-gpus": nodes * 4, "tokenizer-type": "NullTokenizer", "mock-data": True,
    "vocab-size": 262144, "padded-vocab-size": 262144,
    "tensor-model-parallel-size": 4, "pipeline-model-parallel-size": pp,
    "context-parallel-size": 1, "sequence-parallel": True,
    "micro-batch-size": 1, "global-batch-size": 64,
    "num-layers": 64, "hidden-size": 5120, "ffn-hidden-size": 25600,
    "num-attention-heads": 64, "num-query-groups": 8, "group-query-attention": True,
    "kv-channels": 128, "seq-length": 4096, "max-position-embeddings": 4096,
    "swiglu": True, "normalization": "RMSNorm", "norm-epsilon": 1.0e-5, "qk-layernorm": True,
    "position-embedding-type": "rope", "rotary-base": 100000, "rotary-percent": 1.0,
    "untie-embeddings-and-output-weights": True, "add-bias-linear": False,
    "use-flash-attn": True, "attention-backend": "auto", "attention-softmax-in-fp32": True,
    "attention-dropout": 0.0, "hidden-dropout": 0.0, "init-method-std": 0.02,
    "bf16": True, "transformer-impl": "transformer_engine",
    "cross-entropy-loss-fusion": True, "cross-entropy-fusion-impl": "te",
    "use-distributed-optimizer": True,
    "overlap-grad-reduce": ogr, "overlap-param-gather": ogr,
    # Megatron forces overlap_p2p_comm=False for the NON-interleaved schedule; only a
    # layout (interleaved) cell keeps it. Simulon reads this to decide whether to model
    # the per-exchange device sync.
    "overlap-p2p-comm": layout != "-",
    "tp-comm-overlap": False,
    "data-parallel-sharding-strategy": "no_shard",
    "optimizer": "adam", "adam-beta1": 0.9, "adam-beta2": 0.95, "adam-eps": 1.0e-8,
    "clip-grad": 1.0, "weight-decay": 0.1, "seed": 1234,
    "recompute-activations": False,
}
if layout != "-":
    cfg["pipeline-model-parallel-layout"] = layout
with open(f"{out}/workload.yaml", "w") as f:
    yaml.dump({"framework": "megatron", "config": cfg}, f, default_flow_style=False, sort_keys=False)
PYEOF
}

# run <label> <pp> <nodes> <extra megatron args...>
run() {
    local label="$1" pp="$2" nodes="$3"; shift 3
    local extra="$*" out="$OUT/$label"
    local ogr=0 layout='-'
    [[ "$extra" == *"--overlap-grad-reduce"* ]] && ogr=1
    [[ "$extra" == *"--pipeline-model-parallel-layout"* ]] && layout="$LAYOUT"
    local dp=$(( nodes * 4 / (4 * pp) ))
    echo "=============================================================="
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] $label present -> skip"; return 0; fi
    echo "[$(date +%H:%M:%S)] $label  pp=$pp nodes=$nodes dp=$dp  extra: $extra"
    mkdir -p "$out"
    PORT=$((PORT + 1))
    local sub; sub=$(printf '%s,' "${NODES[@]:0:$nodes}"); sub=${sub%,}
    # shellcheck disable=SC2086
    srun -N "$nodes" --ntasks="$nodes" --ntasks-per-node=1 -w "$sub" \
         --gpus-per-node=4 --cpus-per-task=288 \
    apptainer exec --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$STAGE:/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && env $ENV \
            torchrun --nproc-per-node 4 --nnodes $nodes --node-rank \$SLURM_NODEID \
              --max-restarts 0 --master-addr $MASTER_ADDR --master-port $PORT \
              $MEGATRON/pretrain_gpt.py \
              --trace-real-process-group --trace-dir '$out' \
              --pipeline-model-parallel-size $pp $COMMON $extra" \
        > "$out/.megatron.log" 2>&1
    local rc=$? n
    n=$(ls "$out"/trace_rank_*.json 2>/dev/null | wc -l)
    echo "  -> rc=$rc  $n rank traces in $out   (log: $out/.megatron.log)"
    if [ "$n" -gt 0 ]; then
        write_workload "$out" "$pp" "$nodes" "$ogr" "$layout"
    else
        echo "  !! NO TRACE -- tail of .megatron.log:"; tail -15 "$out/.megatron.log" | sed 's/^/     /'
    fi
}

OGR="--overlap-grad-reduce --overlap-param-gather"

# Ordered by decisiveness, cheapest first, so a walltime overrun costs the least.
run tp4pp4-layout-chunks     4 4 "--pipeline-model-parallel-layout '$LAYOUT'"        # EXP 2 primary
run tp4pp2-dp4-ogr           2 8 $OGR                                               # EXP 1 primary
run tp4pp2-dp4-noogr         2 8                                                    # EXP 1 control
run tp4pp4-layout-dp2-ogr    4 8 "--pipeline-model-parallel-layout '$LAYOUT'" $OGR   # both, production shape

echo "=============================================================="
echo "=== EXPERIMENT 1: the DP collectives, if they were recorded ==="
OUT=$OUT python3 - <<'PYEOF'
import json, os, pathlib, collections
base = pathlib.Path(os.environ['OUT'])
# TP4/PP2: 4.236e9 params/rank -> RS fp32 16,944,988,160 B ; AG bf16 8,472,494,080 B
# TP4/PP4: 2.118e9 params/rank -> RS fp32  8,472,494,080 B ; AG bf16 4,236,247,040 B
EXPECT = {2: (16944988160, 8472494080), 4: (8472494080, 4236247040)}
for cell in ("tp4pp2-dp4-ogr", "tp4pp2-dp4-noogr", "tp4pp4-layout-dp2-ogr", "tp4pp4-layout-chunks"):
    d = base / cell
    f = d / "trace_rank_0.json"
    if not f.exists():
        print(f"\n{cell}: NO TRACE"); continue
    t = json.load(open(f))
    ev = t.get("events", [])
    dp_ev = [e for e in ev if e["type"] == "collective"
             and str(e["metadata"].get("name", "")).startswith(("dist_opt_", "ddp_grad_"))]
    pp = int(cell.split("pp")[1][0])
    exp_rs, exp_ag = EXPECT.get(pp, (0, 0))
    print(f"\n{cell}: {len(dp_ev)} DP collective events on rank 0")
    if not dp_ev:
        print("   NONE -- the hook did not fire (DP=1 cell, or the staged Megatron is stale)")
        continue
    by = collections.defaultdict(lambda: [0, 0, set()])
    for e in dp_ev:
        m = e["metadata"]; k = m["collective_type"]
        by[k][0] += 1; by[k][1] += int(m.get("bytes") or 0)
        by[k][2].add(tuple(m.get("group_ranks") or []))
    for k, (n, tot, groups) in sorted(by.items()):
        exp = exp_rs if k == "ReduceScatter" else (exp_ag if k == "AllGather" else 0)
        delta = f"{(tot/exp-1)*100:+.1f}% vs predicted {exp:,}" if exp else ""
        g = sorted(groups)[0] if groups else ()
        print(f"   {k:<14} n={n:<4} total={tot:>15,} B  ({tot/2**30:6.2f} GiB)  {delta}")
        print(f"   {'':14} per-event={tot//max(1,n):>12,} B   group_ranks[0]={list(g)[:8]}"
              f"{' ...' if len(g) > 8 else ''}  ({len(groups)} distinct groups)")
    # placement: which slot direction were they issued during?
    order, cur = [], None
    for e in ev:
        if e["type"] == "slot_begin":
            cur = e["metadata"].get("direction")
        elif e["type"] == "slot_end":
            cur = None
        elif e["type"] == "collective" and str(e["metadata"].get("name", "")).startswith(("dist_opt_", "ddp_grad_")):
            order.append((e["metadata"]["collective_type"], cur or "between-slots"))
    print("   placement:", dict(collections.Counter(order)))
PYEOF

echo
echo "=== EXPERIMENT 2: interleaved chunk structure ==="
OUT=$OUT python3 - <<'PYEOF'
import json, os, pathlib, collections
base = pathlib.Path(os.environ['OUT'])
for cell in ("tp4pp4-layout-chunks", "tp4pp4-layout-dp2-ogr"):
    d = base / cell
    if not (d / "trace_rank_0.json").exists():
        print(f"\n{cell}: NO TRACE"); continue
    print(f"\n{cell}")
    for f in sorted(d.glob("trace_rank_*.json"), key=lambda p: int(p.stem.split('_')[-1]))[:16:4]:
        r = int(f.stem.split('_')[-1]); t = json.load(open(f))
        slots = [e for e in t["events"] if e["type"] == "slot_begin"]
        withid = [e for e in slots if "model_chunk_id" in e["metadata"]]
        per = collections.defaultdict(lambda: [0, 0.0])
        for e in slots:
            m = e["metadata"]
            if m.get("direction") == "step":
                continue
            k = (m.get("model_chunk_id"), m.get("direction"))
            per[k][0] += 1
            per[k][1] += float(m.get("kernel_device_ms") or 0.0)
        print(f"  rank {r} (stage {r//4}): {len(slots)} slots, {len(withid)} with model_chunk_id")
        fwd = {c: v for (c, dr), v in per.items() if dr == "fwd"}
        for c in sorted(fwd, key=lambda x: (x is None, x)):
            n, ms = fwd[c]
            print(f"     chunk {str(c):<5} fwd n={n:<4} kernel {ms:9.1f} ms  ({ms/max(1,n):6.2f} ms/slot)")
PYEOF

echo
echo "NEXT (login node):"
echo "  rsync -a $OUT/ templates/gpu/gh200_jupiter-32b-target/traces/"
echo "  PYTHONPATH=. python experiments/check_trace_health.py templates/gpu/gh200_jupiter-32b-target/traces"
echo "finished : $(date)"
