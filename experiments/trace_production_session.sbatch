#!/bin/bash
# PRODUCTION-MATCHED re-trace: regenerate the whole isolation set with the SAME toolchain
# as the physical runs, retiring the two confounds found by the 2x2 factorial (job 1009816).
#
# WHAT CHANGED vs the -unified / -kerneltime sets:
#   1. CONTAINER: nemo_26.04.sif (TE 2.14.0, cuDNN 9.20.0) -- the container the physical
#      runs actually use -- instead of simulon.sif (TE 2.12.0, cuDNN 9.19.0). torch and
#      CUDA are identical between them, so TE/cuDNN was the entire delta, and TE owns the
#      fused attention/linear kernels that dominate this model.
#   2. FUSIONS: simulon/cli/trace.py no longer forces --no-masked-softmax-fusion /
#      --no-bias-swiglu-fusion onto every trace, so these inherit Megatron's defaults
#      (fusions ON) and finally match production (masked_softmax_fusion=true,
#      bias_swiglu_fusion=true in config-989395.yaml). Verified: the plain workloads now
#      emit NO --no-*-fusion flags.
# Measured effect on tp4pp2-mbs1 (hardware 9274 ms): +6.6% -> +2.6%.
#
# WHY SPAN AND KERNEL ARE CAPTURED ADJACENTLY. The GEMM-efficiency correction uses
# overhead(cfg) = span_compute(cfg) - kernel_compute(cfg). If the two came from different
# sessions the overhead would absorb thermal/clock drift, so each config's span trace and
# its kernel-timing trace are taken back-to-back on the same GPU, clocks pinned.
#
# SPAN traces  -> templates/gpu/gh200_jupiter-prod/traces/<cfg>            (all 10)
# KERNEL traces-> templates/gpu/gh200_jupiter-prod-kerneltime/traces/<cfg> (the 8 the
#   correction actually consumes; the two rc configs are tp4/mbs1 references that are
#   never themselves corrected, so they get span only).
# NOTE tp4pp1-mbs2 and tp2pp4-mbs2 OOM'd under the profiler last time (low shard, ~155 MiB
# free); if they OOM again they simply stay uncorrected -- the job continues.
#
# AFTER THIS LANDS: point TRACE_REGISTRY_UNIFIED / TRACE_REGISTRY_KERNELTIME at the -prod
# dirs and re-run experiments/analyze_val40.py. Sanity check: tp4pp2-mbs1 span compute
# should land near 7609 (factorial cell C), not the old 7999.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --job-name=trace_production
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_production-%j.log

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
    local cfg="$1" mode="$2" out env_pfx=""
    if [ "$mode" = "kernel" ]; then out="$KERN_OUT/$cfg"; else out="$SPAN_OUT/$cfg"; fi
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
import json,sys
d=json.load(open('$out/trace_rank_0.json'))
print('  ok  kernel_ms=%s  events=%d'%(d.get('total_kernel_device_ms'),len(d.get('events',[]))))"
    else
        echo "  !! NO TRACE (likely OOM) -- continuing"
    fi
}

# span + kernel back-to-back per config
for cfg in tp4pp4-mbs1 tp4pp4-mbs2 tp4pp2-mbs1 tp4pp2-mbs2 tp4pp1-mbs1 tp4pp1-mbs2 \
           tp2pp4-mbs1 tp2pp4-mbs2; do
    gen "$cfg" span
    gen "$cfg" kernel
done

# recompute configs: span only (they are tp4/mbs1 references, never corrected themselves)
gen tp4pp2-mbs1-rc span
gen tp4pp4-mbs1-rc span

echo "[$(date +%H:%M:%S)] all done"
