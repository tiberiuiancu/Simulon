#!/bin/bash
# Trace the tp4pp1-mbs2 cell — the OUT-OF-SAMPLE test of the 4-rank skew constant.
#
# WHY THIS CELL. k(4)=0.911 ms in templates/node/jupiter-gh200-4g.yaml currently rests on
# exactly ONE observation (tp4pp1-mbs1), so it is calibrated but unfalsifiable. tp4pp1-mbs2
# keeps the communicator size at 4 while HALVING the collective count (28928 -> 14464,
# because 128 microbatches instead of 256). If the cost really is per-collective at 0.911 ms,
# the skew contribution must halve with it.
#
# Captures span and kernel-timing adjacently in one session, matching the unified registry
# they will be scored against (cross-session compute spread on GH200 is ~9%, the same size
# as the effect).
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --job-name=trace_1p7b_tp4mbs2
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_1p7b_tp4mbs2-%j.log

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
    # MUST run under `srun`. Invoking apptainer directly from the batch script — which is how
    # every earlier trace job did it — killed the entire cgroup at ~50 s with a silent,
    # byte-identical 1697-byte log and no traceback (jobs 1149575, 1150654). The identical
    # workload completes in 2m40s under srun with GPU peaking at 19.8 GB and host RAM barely
    # moving (760 -> 731 GB available), so it was never a resource problem. `--gres=gpu:1` on
    # the srun step is what makes the difference; the batch allocation hands out all 4 GPUs.
    APPTAINERENV_SIMULON_TRACE_KERNEL_TIME="$kt" \
    srun --ntasks=1 --gres=gpu:1 \
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=$kt PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python -u -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cell.yaml' --gpu gh200_jupiter -o '$out'" 2>&1
    local rc=$?
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "  ok (exit=$rc)"
    else
        echo "  !! NO TRACE (exit=$rc) -- continuing"
    fi
}

# Most-important-first: the refounding anchors and the mbs/TP axes, then PP, then the rest.
CELLS="tp4pp1-mbs2"

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
