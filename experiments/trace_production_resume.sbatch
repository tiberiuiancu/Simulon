#!/bin/bash
# RESUME the production-matched re-trace (job 1009860 was cancelled at 3/10 span + 2/8 kernel).
# Same toolchain as trace_production_session.sbatch: nemo_26.04.sif + fusions inherited from the
# FIXED _TRACE_DEFAULTS (no forced --no-*-fusion). This only regenerates configs still MISSING on
# disk, and skips any that already have trace_rank_0.json, so it is safe to re-run.
#
# Already OK (left untouched): span tp4pp2-mbs1, tp4pp4-mbs1, tp4pp4-mbs2 ;
#                              kernel tp4pp4-mbs1, tp4pp4-mbs2.
# NOT attempted -- genuinely infeasible on one fake-PG GPU (exhausted last session, ~155 MiB free):
#                              tp2pp4-mbs2, tp4pp1-mbs2.
#
# Sanity target after this lands: tp4pp2-mbs1 span compute ~7609 (factorial cell C), not old 7999.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=trace_prod_resume
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_production_resume-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif   # production container
W=experiments/sweep_qwen3_32b_workloads/unified
SPAN_OUT=templates/gpu/gh200_jupiter-prod/traces
KERN_OUT=templates/gpu/gh200_jupiter-prod-kerneltime/traces

echo "container: $SIF"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"

# $1 = config, $2 = "span" | "kernel"
gen() {
    local cfg="$1" mode="$2" out
    if [ "$mode" = "kernel" ]; then out="$KERN_OUT/$cfg"; else out="$SPAN_OUT/$cfg"; fi
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] $cfg [$mode] already present -> skip"
        return 0
    fi
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] $cfg [$mode] -> $out"
    mkdir -p "$out"
    if [ "$mode" = "kernel" ]; then
        APPTAINERENV_SIMULON_TRACE_KERNEL_TIME=1 \
        apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
            --bind "$(pwd):/opt/simulon" "$SIF" \
            bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=1 PYTHONPATH=/opt/simulon \
                CUDA_DEVICE_MAX_CONNECTIONS=1 \
                python -c 'from simulon.cli import app; app()' trace generate \
                '$W/$cfg.yaml' --gpu gh200_jupiter -o '$out'"
    else
        apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
            --bind "$(pwd):/opt/simulon" "$SIF" \
            bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon \
                CUDA_DEVICE_MAX_CONNECTIONS=1 \
                python -c 'from simulon.cli import app; app()' trace generate \
                '$W/$cfg.yaml' --gpu gh200_jupiter -o '$out'"
    fi
    echo "[$(date +%H:%M:%S)] $cfg [$mode] exit=$?"
    if [ -f "$out/trace_rank_0.json" ]; then
        python3 -c "
import json
d=json.load(open('$out/trace_rank_0.json'))
print('  ok  kernel_ms=%s  events=%d'%(d.get('total_kernel_device_ms'),len(d.get('events',[]))))"
    else
        echo "  !! NO TRACE (likely OOM under profiler) -- continuing"
    fi
}

# ---- span (needed for every corrected config + the rc references) ----
gen tp4pp2-mbs2   span
gen tp4pp1-mbs1   span
gen tp2pp4-mbs1   span
gen tp4pp2-mbs1-rc span
gen tp4pp4-mbs1-rc span

# ---- kernel (references first: tp4pp2-mbs1 is the pp2 overhead anchor) ----
gen tp4pp2-mbs1   kernel
gen tp4pp2-mbs2   kernel
gen tp4pp1-mbs1   kernel
gen tp2pp4-mbs1   kernel

echo "[$(date +%H:%M:%S)] resume done"
echo "=== final inventory ==="
for base in "$SPAN_OUT" "$KERN_OUT"; do
    echo "--- $base"
    for d in "$base"/*/; do
        c=$(basename "$d")
        [ -f "$d/trace_rank_0.json" ] && echo "    $c OK" || echo "    $c MISSING"
    done
done
