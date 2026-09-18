#!/bin/bash
# UNIFIED single-session capture of the whole Qwen3-1.7B trace set — span AND kernel-timing,
# back-to-back on ONE GPU with pinned clocks.
#
# WHY. The existing 1.7B traces were captured across >=4 separate jobs (6 span traces still
# carry the stale padded_vocab=262144 from a cancelled run, and span vs kernel-timing came
# from different sessions entirely). Cross-session compute spread on GH200 is ~9% — the SAME
# SIZE as the effects being measured. Every span-vs-kernel decomposition (overhead = span -
# kernel) therefore mixes a real signal with a thermal/clock artifact of equal magnitude.
# The 32B campaign solved this with a single back-to-back session (TRACE_REGISTRY_UNIFIED);
# 1.7B never got one. This is that job.
#
# It also re-captures every KT trace with the FIXED per-slot attribution (timestamp-based,
# tracer.py:_kernel_ms_by_slot). The old KT traces book all kernel time on forward slots
# (bwd = exactly 0), which zeroes backward compute in the replay — they must not be reused.
#
# RESUMABLE: each cell is skipped if already present, so if the walltime is hit, just
# resubmit and it continues. Cells are ordered MOST-IMPORTANT-FIRST so a partial run still
# yields the cells the refounding depends on.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=trace_1p7b_tpfix
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_1p7b_tpfix-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/thomas_1p7b_workloads
SPAN_OUT=templates/gpu/gh200_jupiter-1p7b-unified/traces
KT_OUT=templates/gpu/gh200_jupiter-1p7b-unified-kt/traces

echo "container: $SIF"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "started  : $(date)"

# gen <cell> <outbase> <kernel_time 0|1>
gen() {
    local cell="$1" outbase="$2" kt="$3" out="$2/$1"
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] $cell (kt=$kt) present -> skip"; return 0; fi
    echo "[$(date +%H:%M:%S)] $cell (kt=$kt)"
    mkdir -p "$out"
    APPTAINERENV_SIMULON_TRACE_KERNEL_TIME="$kt" \
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=$kt PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cell.yaml' --gpu gh200_jupiter -o '$out'"
    local rc=$?
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "  ok (exit=$rc)"
    else
        echo "  !! NO TRACE (exit=$rc) -- continuing"
    fi
}

# Most-important-first: the refounding anchors and the mbs/TP axes, then PP, then the rest.
CELLS="tp2pp1-mbs1 tp4pp1-mbs1 tp2pp1-mbs8 tp2pp2-mbs2"

# Span and kernel-timing for each cell are captured ADJACENTLY, so the pair that gets
# subtracted (overhead = span - kernel) shares thermal/clock state as tightly as possible.
for cell in $CELLS; do
    gen "$cell" "$SPAN_OUT" 0
    gen "$cell" "$KT_OUT"   1
done

echo "[$(date +%H:%M:%S)] done"
echo "=== inventory ==="
for base in "$SPAN_OUT" "$KT_OUT"; do
    echo "--- $base"
    for cell in $CELLS; do
        [ -f "$base/$cell/trace_rank_0.json" ] && echo "    $cell OK" || echo "    $cell MISSING"
    done
done
echo "NEXT: PYTHONPATH=. python experiments/check_trace_health.py $SPAN_OUT $KT_OUT"
