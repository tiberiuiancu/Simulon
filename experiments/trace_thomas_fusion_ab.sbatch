#!/bin/bash
# THOMAS re-trace of Tiberiu's top-5 workload-tuning configs, as a FUSED vs UNFUSED A/B,
# to measure the fusion bias his main-branch sweep carried (his traces forced
# masked-softmax + bias-swiglu fusion OFF via the old _TRACE_DEFAULTS; his workloads never
# set those flags, so our fix flips them ON).
#
# WHY AN A/B, AND ITS LIMITS. His traces are H100 (Snellius); we can only trace on Jupiter
# GH200. A GH200 is a Hopper H100-class die (both templates 989 TFLOPS), but ours is
# powercapped to 680W and has faster HBM3e, and this uses nemo_26.04.sif (aarch64) not his
# simulon-nemo.sif — so a single GH200 number is NOT directly his H100 MFU. What IS clean
# is the fused/unfused RATIO measured on the SAME GH200 + SAME container: that isolates the
# pure fusion effect, which transfers to H100 approximately. Applying that ratio to his
# existing H100 (unfused) traces gives a defensible fusion-corrected MFU to compare vs 54.4%.
# Separately, thomas-<cfg>-unfused vs his H100 trace gauges how good the GH200->H100 proxy is.
# The SP=true bias is NOT addressed here — it needs real H100 hardware.
#
# 5 configs x {fused, unfused} = 10 single-GPU fake-PG traces. Outputs:
#   templates/gpu/h100/traces/thomas-<cfg>-fused     (current tracer, fusions ON)
#   templates/gpu/h100/traces/thomas-<cfg>-unfused    (fusions forced OFF = reproduces his)
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=trace_thomas_ab
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_thomas_ab-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/usecase_workload_tuning/scenarios_thomas
OUT=templates/gpu/h100/traces

echo "container: $SIF"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"

gen() {
    local cfg="$1" variant="$2"
    local wl="$W/${cfg}_${variant}/workload.yaml"
    local out="$OUT/thomas-${cfg}-${variant}"
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] ${cfg}-${variant} already present -> skip"; return 0
    fi
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] ${cfg}-${variant}  <- $wl"
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$wl' --gpu h100 -o '$out'"
    echo "[$(date +%H:%M:%S)] ${cfg}-${variant} exit=$?"
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "  OK  $(ls "$out" | tr '\n' ' ')"
    else
        echo "  !! NO TRACE (likely fake-PG OOM: no DP to shard the optimizer) -- continuing"
    fi
}

for cfg in tp4_pp2_mbs2_vpp8 tp4_pp2_mbs2_vpp16 tp4_pp2_mbs2_vpp32 tp8_pp1_mbs4 tp4_pp2_mbs2_vpp1; do
    gen "$cfg" fused
    gen "$cfg" unfused
done

echo "[$(date +%H:%M:%S)] all done"
echo "=== inventory ==="
for d in "$OUT"/thomas-*; do
    [ -f "$d/trace_rank_0.json" ] && echo "  $(basename "$d") OK" || echo "  $(basename "$d") MISSING"
done
