#!/bin/bash
# Step 1 of the 256-1024-node plan: capture Qwen3-32B traces in the PRODUCTION-PINNED setting
# (TP4, SP=true, bf16, tp_comm_overlap, dist-optimizer, nemo_26.04) under a REAL process
# group, one trace per pipeline depth the target search actually varies (PP 2/4/8).
#
# WHY THESE AND NOT THE 16-NODE SP=false GRID. At 256-1024 nodes with GBS 2048/4096 the
# search variables are PP x VPP x layout x mbs under the M>=8 constraint (M = microbatches
# per DP rank; SCALING_GUIDE.md ranks it the largest lever, 5.2x). TP is pinned to the
# NVLink domain (4), SP=true is required by tp_comm_overlap, and every existing 32B trace is
# fake-PG / SP=false / unfused / simulon.sif -- none is in the production kernel mix. The
# 512-node evidence table (oellm_32b_dense/SCALING_GUIDE.md s9, 12 runs) is the held-out set
# these traces will be scored against; it was never simulated.
#
# WHY REAL PG. Fake-PG traces are uniformly optimistic and 23% low on the last pipeline
# stage (CAMPAIGN.md "How far the fake process group can be trusted"); the real-PG grid at
# 1.7B fixed that (stage1 -23.2% -> -5.5% vs nsys). Real PG also means the full TP x PP group
# runs, so PP8 needs 8 nodes. DP=1 everywhere: these traces carry COMPUTE only -- DP
# collectives and their overlap come from the NCCL tables and the replay, not from here.
#
# THE PROCESS GROUP SETTING THAT DIFFERS FROM EVERY EARLIER JOB: production runs
# overlap_p2p_comm=true (config-1827154.yaml), which sets batch_p2p_comm=False and removes the
# per-exchange torch.cuda.synchronize() that dag.host_sync_nodes models. Do NOT pass
# --no-overlap-p2p-communication here; the trace must carry the production P2P schedule, and
# Simulon's host-sync path must be OFF when replaying these traces.
#
# CELLS (each a separate srun on a subset of the 8-node allocation; a failure in one cannot
# take down the rest; already-captured cells are skipped on resubmit):
#   A  tp4pp4-bf16             4n  anchor: plain PP4, no VPP, production kernels
#   B  tp4pp4-bf16-noubo       4n  A with tp_comm_overlap OFF -- the only variant Simulon can
#                                  currently replay (TP collectives as NCCL from the tables)
#   C  tp4pp4-fp8              4n  A in fp8 hybrid/delayed -- kernel-side size of the 1.38x
#   D  tp4pp8-bf16             8n  deepest PP in the target grid
#   E  tp4pp4-bf16-prodlayout  4n  the production 16-chunk layout (PP4 x 4 chunks/stage)
#   F  tp4pp2-bf16             2n  LAST: memory-marginal at DP=1 (~85 GB est. of 102), an OOM
#                                  here must not cost anything else
# GBS=64 at mbs=1, DP=1 -> M=64 microbatches per trace. Simulon TRIMS microbatches downward
# (replication for pp>1 is fenced off), so M=64 covers every 256-1024-node cell at mbs=2 and
# all but PP8@256n/GBS4096 at mbs=1 (M=128, will be refused -- acceptable).
#
# PRE-REGISTERED, written before looking:
#   1. SP=true sanity. A PP4 stage at M=64 does the same layer-microbatches as a PP1 rank at
#      M=16, and the profiled real SP=true TP4/PP1/16n/GBS256 run measured 8062 ms of non-NCCL
#      kernel time (STATUS.md sE.0). Middle stages of A should land within ~+-10% of that.
#      If they come out ~-17% (the fake-PG failure), the SP=true defect is NOT fake-PG and
#      this whole direction stops.
#   2. Last stage of A (lm_head + vocab-parallel CE, 262144 vocab) should exceed the middle
#      stages by a visible margin -- that margin is what fake PG lost.
#   3. B vs A: tp_comm_overlap changes kernel time by a measurable amount (userbuffer GEMMs
#      run with comm in flight). Sign and size are the information; no expectation.
#   4. C vs A: fp8 kernel time / bf16 kernel time. End-to-end measured 1/1.38 = 0.72 at 512n;
#      the GEMM-only ratio should be lower (more speedup) since attention/norm are not fp8.
#   5. E vs A: per-stage TOTAL kernel time equal within noise (chunking is compute-neutral).
#      If E is slower, the PP*VPP=32 penalty in SCALING_GUIDE is partly kernel efficiency of
#      short chunks, not only schedule -- and step 2 must model chunk-size-dependent compute.
#   NOTE on E: the tracer keys interleaved slots by virtual_microbatch_id and does NOT
#   record model_chunk_id (schedules.py:1299), so E's per-chunk slots are not yet consumable
#   by Simulon -- only its per-rank totals are. Fix the tracer before relying on E for step 2.
#
# KNOWN FIDELITY GAP: the traced fork is Megatron-core 0.18.0; production runs 0.16.0
# (v0.15-openeurollm-12). Same container (TE 2.14 / cuDNN), so kernel choices should match,
# but this is an assumption -- record it against any residual.
#
# COST: 8 nodes x ~1 h. No training run.
#
# RESULT (job 1829600, 2026-09-16; 1829371 was lost to a quoting bug). All 6 cells captured,
# PP2 fit at DP=1. Registry: templates/gpu/gh200_jupiter-32b-target/traces.
#   wall ms (median iters 3-6)  A 11703  B 11771  C(fp8) 10414  D(pp8) 6828  E(layout) 10290  F(pp2) 21420
#   [1] PASS on B: middle stages 7240/7605 vs 8062 -> -10.2%/-5.7%. Real-PG SP=true is sane;
#       the fake-PG -17% is not reproduced.
#   [2] PASS on B: last stage +10.4% over middle, embedding stage -0.9%.
#   [3] tp_comm_overlap is WALL-NEUTRAL at DP=1 (-0.6%), but its userbuffer kernels are booked
#       as compute: A's kernel time is 1.6x B's with 23% spread across identical TP ranks.
#       => every UB cell (A,C,D,E,F) has CONTAMINATED kernel_device_ms; use B-style (noubo)
#       traces for compute, or add a UB-kernel exclusion to the tracer. B is the compute anchor.
#   [4] fp8 wall -11% here vs -28% (1/1.38) at 512n -- most of the production fp8 gain is
#       NOT kernel time (fp8_param_gather halves DP bytes; mbs=1 here vs 2). Kernel ratio unclean.
#   [5] layout cell 12% faster wall than plain PP4 at M=64: the interleaved bubble saving is only
#       ~3.5pp, the rest is overlap_p2p_comm=True (no per-exchange device sync) -- supports the
#       host-sync mechanism. Compute-neutrality NOT judged (both cells UB-contaminated).
#   check_trace_health: WARNs are expected (all TP ranks saved; padded_vocab 262144 is
#   production's). E FAILs (513 slots, no chunk id) -- the known tracer gap, see NOTE on E.
#   NEXT: re-capture C, D, E, F with tp_comm_overlap OFF (~25 min, 8 nodes) for clean compute.
#
# RESULT WAVE 2 (job 1836105, 2026-09-16). All 4 noubo cells captured. 10 cells in the registry.
#   wall ms  C' fp8 10133  E' layout 10223  D' pp8 6626  F' pp2 20990 -- every noubo cell is
#   1-3% FASTER than its UB twin: tp_comm_overlap buys nothing at DP=1, so noubo traces are the
#   faithful compute source for production and Simulon can model TP comm from the NCCL tables.
#   [4'] fp8 CLEAN kernel ratio 0.62-0.63 on transformer stages, 0.70 on the lm_head stage
#        (1.6x kernel speedup) against 0.86 wall: at mbs=1 the iteration is host/sync-bound
#        (host_ms 7-9 s per rank vs 7.5 s kernel), so shrinking kernels exposes host time.
#        The 512n 1.38x is therefore consistent with kernels alone once mbs=2 halves host/sample.
#   [5'] production layout IS compute-neutral: per-stage ratios 1.060/1.030/1.007/0.974 track the
#        LAYER COUNT per stage exactly (17/16/16/15 layers vs 16 -> 1.0625/1/1/0.9375). The
#        PP*VPP=32 penalty in SCALING_GUIDE is schedule, not kernel efficiency.
#   [6'] compute is linear in layers across PP: PP8 interior/PP4 interior 0.478-0.497 (expect
#        0.5), PP2 1.99/2.10 (expect 2.0). lm_head+CE costs ~1.66 transformer layers, embedding
#        ~0. => one per-layer time + one lm_head term reproduces every stage of every PP.
#   These three results are what step 2 (interleaved-1F1B DAG synthesis with layouts) needs:
#   chunk compute = layers_in_chunk * t_layer, no per-chunk traces required.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=01:30:00
#SBATCH --job-name=trace_32b_target
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --chdir=/e/scratch/e-sta-openeurollm/vanosch1
#SBATCH --output=/e/scratch/e-sta-openeurollm/vanosch1/logs/trace_32b_target-%j.log

set -uo pipefail
# Run off /e/scratch: jobs must not touch /e/home (stale-handle fault, see stage_repo.sh).
# Re-run experiments/stage_repo.sh after any code change you want this job to pick up.
STAGE=/e/scratch/e-sta-openeurollm/vanosch1/simulon_stage
OUT=/e/scratch/e-sta-openeurollm/vanosch1/traces_32b_target
[ -f "$STAGE/simulon/cli/__init__.py" ] || { echo "FATAL: run stage_repo.sh first" >&2; exit 1; }
cd "$STAGE" || exit 1
mkdir -p "$OUT"

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif      # production container
MEGATRON=/opt/simulon/vendor/Megatron-LM-traced
LAYOUT='Et*5|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3L'   # config-1827154

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${NODES[0]}
PORT=20200

echo "container: $SIF"
echo "nodes    : ${#NODES[@]}  (${NODES[*]})"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "started  : $(date)"

# CPU affinity: the fault behind jobs 1225995/1231006/1301886 -- a step bound to a narrow
# mask inflates host time 4-5x. Read the real mask, not `nproc` (which honours OMP_NUM_THREADS).
CORES=$(srun -N1 --ntasks=1 apptainer exec --bind /e/scratch/ "$SIF" \
        python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null | tail -1)
echo "affinity : ${CORES:-unknown} cores in the step's mask (expect 288)"
case "$CORES" in
    ''|*[!0-9]*) echo "WARN: could not read affinity; continuing" >&2 ;;
    *) [ "$CORES" -lt 64 ] && { echo "FATAL: step bound to $CORES cores" >&2; exit 1; } ;;
esac

# Production-pinned model + training args (models/qwen3_dense_32b.yaml + config-1827154.yaml).
# Fusions (rope, bias-dropout, masked-softmax, swiglu, grad-accum) are Megatron defaults and
# are left at default so the trace inherits exactly what production runs; capture_config in
# the trace records the resolved values. NullTokenizer at vocab 262144 == production's padded
# vocab, so the lm_head / CE shapes are identical.
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
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
  --data-parallel-sharding-strategy no_shard --manual-gc
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8
  --clip-grad 1.0 --weight-decay 0.1 --lr 1e-8 --min-lr 0.0
  --lr-decay-style WSD --lr-decay-iters 1 --lr-wsd-decay-iters 1 --lr-wsd-decay-style linear
  --tokenizer-type NullTokenizer --vocab-size 262144 --padded-vocab-size 262144
  --mock-data --dataloader-type cyclic --seed 1234
  --train-iters 8 --trace-warmup-iters 6
  --eval-interval 999999 --eval-iters 0 --save-interval 1000000
  --log-interval 1 --log-throughput --distributed-timeout-minutes 30"
# Collapse to ONE line: both strings are expanded inside a bash -c "..." string, where an
# embedded newline is a command separator. Job 1829371 lost every cell to exactly that --
# `env` ran alone, torchrun got one line of args, and each following --flag line ran as a
# command (rc=127). Word-splitting here is safe: neither string contains globs or quotes.
COMMON=$(echo $COMMON)

# Environment exactly as the production sbatch (job_20260915_122434_549.sbatch), minus
# NCCL_DEBUG. UB_SKIPMC=1 and CUDA_DEVICE_MAX_CONNECTIONS=1 are what tp_comm_overlap needs.
ENV="SIMULON_TRACE_KERNEL_TIME=1 PYTHONPATH=/opt/simulon:$MEGATRON PYTHONDONTWRITEBYTECODE=1
  CUDA_DEVICE_MAX_CONNECTIONS=1 UB_SKIPMC=1 NCCL_NVLS_ENABLE=0
  NCCL_SOCKET_IFNAME=ib0 NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_AVOID_RECORD_STREAMS=1
  TORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8 OMP_NUM_THREADS=8"
ENV=$(echo $ENV)

# write_workload <out> <pp> <fp8:0|1> <ubo:0|1> <layout|->
# Simulon needs workload.yaml next to the traces (dash-form keys) for mbs extrapolation and
# microbatch trimming; the torchrun path bypasses `simulon trace generate`, which would have
# written it. Backfilling this by hand is what silently no-op'd every mbs leg once before.
write_workload() {
    local out="$1" pp="$2" fp8="$3" ubo="$4" layout="$5"
    python3 - "$out" "$pp" "$fp8" "$ubo" "$layout" <<'PYEOF'
import sys, yaml
out, pp, fp8, ubo, layout = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1", sys.argv[5]
cfg = {
    "num-gpus": 4 * pp, "tokenizer-type": "NullTokenizer", "mock-data": True,
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
    "use-distributed-optimizer": True, "overlap-grad-reduce": True, "overlap-param-gather": True,
    # Megatron forces overlap_p2p_comm=False for the NON-interleaved schedule (training log:
    # "Setting args.overlap_p2p_comm ... to False since non-interleaved schedule does not
    # support overlapping p2p communication"), so only the layout (interleaved) cell keeps it.
    # Simulon reads this key to decide whether the per-exchange device sync is modelled.
    "overlap-p2p-comm": layout != "-", "tp-comm-overlap": ubo,
    "data-parallel-sharding-strategy": "no_shard",
    "optimizer": "adam", "adam-beta1": 0.9, "adam-beta2": 0.95, "adam-eps": 1.0e-8,
    "clip-grad": 1.0, "weight-decay": 0.1, "seed": 1234,
    "recompute-activations": False,
}
if fp8:
    cfg.update({"fp8-format": "hybrid", "fp8-recipe": "delayed",
                "fp8-amax-history-len": 1024, "fp8-amax-compute-algo": "max"})
if layout != "-":
    cfg["pipeline-model-parallel-layout"] = layout
with open(f"{out}/workload.yaml", "w") as f:
    yaml.dump({"framework": "megatron", "config": cfg}, f, default_flow_style=False, sort_keys=False)
PYEOF
}

# run <label> <pp> <extra megatron args...>
run() {
    local label="$1" pp="$2"; shift 2
    local extra="$*" out="$OUT/$label" nodes=$((pp))     # TP4 fills a node -> nodes = PP
    local fp8=0 ubo=1 layout='-'
    [[ "$extra" == *"--fp8-format"* ]] && fp8=1
    [[ "$extra" == *"--tp-comm-overlap"* ]] || ubo=0
    [[ "$extra" == *"--pipeline-model-parallel-layout"* ]] && layout="$LAYOUT"
    echo "=============================================================="
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] $label present -> skip"; return 0; fi
    echo "[$(date +%H:%M:%S)] $label  pp=$pp nodes=$nodes  extra: $extra"
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
        write_workload "$out" "$pp" "$fp8" "$ubo" "$layout"
    else
        echo "  !! NO TRACE -- tail of .megatron.log:"; tail -15 "$out/.megatron.log" | sed 's/^/     /'
    fi
}

UBO="--tp-comm-overlap"
FP8="--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"

run tp4pp4-bf16            4  $UBO                                           # A  anchor
run tp4pp4-bf16-noubo      4                                                 # B  overlap off
run tp4pp4-fp8             4  $UBO $FP8                                      # C  fp8
run tp4pp8-bf16            8  $UBO                                           # D  PP8
run tp4pp4-bf16-prodlayout 4  $UBO "--pipeline-model-parallel-layout '$LAYOUT'"  # E  16 chunks
run tp4pp2-bf16            2  $UBO                                           # F  memory-marginal

# WAVE 2 (after job 1829600 showed UB kernels are booked as compute): the same grid with
# tp_comm_overlap OFF. UB is wall-neutral (-0.6%), so these are the faithful COMPUTE traces for
# production; B is their PP4/bf16 anchor. Cells above are skipped once captured, so
# resubmitting this file runs only what is missing.
run tp4pp4-fp8-noubo             4  $FP8                                          # C'
run tp4pp4-bf16-prodlayout-noubo 4  "--pipeline-model-parallel-layout '$LAYOUT'"  # E'
run tp4pp8-bf16-noubo            8                                                # D'
run tp4pp2-bf16-noubo            2                                                # F'

echo "=============================================================="
echo "=== per-rank kernel time (NCCL excluded) and the pre-registered comparisons ==="
OUT=$OUT python3 - <<'PYEOF'
import json, os, pathlib
base = pathlib.Path(os.environ['OUT'])
cells = {}
print(f"{'cell / rank':<32}{'kernel ms':>11}{'host_ops':>10}{'host ms':>11}")
for d in sorted(p for p in base.iterdir() if p.is_dir()):
    ranks = {}
    for f in sorted(d.glob('trace_rank_*.json'), key=lambda p: int(p.stem.split('_')[-1])):
        r = int(f.stem.split('_')[-1]); t = json.load(open(f))
        k = t.get('total_kernel_device_ms') or 0.0
        ranks[r] = k
        print(f"{d.name + ' r' + str(r):<32}{k:>11.0f}{str(t.get('total_host_ops')):>10}"
              f"{(t.get('total_host_ms') or 0):>11.0f}")
    if not ranks:
        print(f"{d.name:<32}{'NO TRACE':>11}")
    cells[d.name] = ranks

def stage_means(ranks, pp):
    # TP4 -> ranks 4s..4s+3 are stage s; take the stage's mean over its 4 TP ranks
    return [sum(ranks.get(4*s+i, 0) for i in range(4)) / 4 for s in range(pp)]

def show(label, a, b, pp):
    if a in cells and b in cells and cells[a] and cells[b]:
        sa, sb = stage_means(cells[a], pp), stage_means(cells[b], pp)
        print(f"\n{label}")
        for s, (x, y) in enumerate(zip(sa, sb)):
            print(f"  stage {s}: {x:8.0f} vs {y:8.0f}   ratio {y/x if x else float('nan'):.3f}")

A = 'tp4pp4-bf16'
if A in cells and cells[A]:
    st = stage_means(cells[A], 4)
    print(f"\n[1] SP=true sanity -- middle stages of A vs profiled real SP=true 8062 ms (STATUS sE.0):"
          f"  {st[1]:.0f} / {st[2]:.0f}  ({(st[1]/8062-1)*100:+.1f}% / {(st[2]/8062-1)*100:+.1f}%)")
    print(f"[2] last stage vs middle (lm_head+CE margin): {st[3]:.0f} vs {st[2]:.0f}  ({(st[3]/st[2]-1)*100:+.1f}%)")
show('[3] B/A -- tp_comm_overlap OFF vs ON', A, 'tp4pp4-bf16-noubo', 4)
show('[4] C/A -- fp8 vs bf16 (end-to-end measured 0.72 at 512n)', A, 'tp4pp4-fp8', 4)
show('[5] E/A -- production 16-chunk layout vs plain PP4 (compute-neutral => ratio ~1.0)', A, 'tp4pp4-bf16-prodlayout', 4)

# Wave 2: the same questions on UB-free traces, anchored on B (clean compute).
B = 'tp4pp4-bf16-noubo'
show("[4'] C'/B -- fp8 vs bf16, CLEAN kernel time (wall was 0.89)", B, 'tp4pp4-fp8-noubo', 4)
show("[5'] E'/B -- production layout vs plain PP4, CLEAN (compute-neutral => ~1.0)", B, 'tp4pp4-bf16-prodlayout-noubo', 4)
for name, pp, expect in (('tp4pp8-bf16-noubo', 8, 0.5), ('tp4pp2-bf16-noubo', 2, 2.0)):
    if B in cells and name in cells and cells[name]:
        sb, sx = stage_means(cells[B], 4), stage_means(cells[name], pp)
        mid_b = (sb[1] + sb[2]) / 2
        mids = sx[1:-1] if pp > 2 else sx
        print(f"\n[6'] {name}: interior stages / B interior (layers ratio => expect ~{expect}):",
              " ".join(f"{x/mid_b:.3f}" for x in mids), f"| last/first-of-B {sx[-1]/sb[3]:.3f} {sx[0]/sb[0]:.3f}")
PYEOF

echo
echo "NEXT (login node):"
echo "  rsync -a $OUT/ templates/gpu/gh200_jupiter-32b-target/traces/"
echo "  PYTHONPATH=. python experiments/check_trace_health.py templates/gpu/gh200_jupiter-32b-target/traces"
echo "finished : $(date)"
