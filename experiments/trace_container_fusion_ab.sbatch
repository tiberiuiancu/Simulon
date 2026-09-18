#!/bin/bash
# 2x2 FACTORIAL: does the sim's persistent +5-8% compute over-prediction come from
# toolchain/config confounds rather than from fake-PG itself?
#
# TWO CONFOUNDS FOUND (2026-07-21), both biasing the SAME direction as the observed error:
#
#  1. FUSIONS. simulon/cli/trace.py _TRACE_DEFAULTS forces onto EVERY trace:
#         --no-masked-softmax-fusion   --no-bias-swiglu-fusion
#     but the physical run (config-989395.yaml) has masked_softmax_fusion=true AND
#     bias_swiglu_fusion=true. Unfused = slower kernels = traces overstate compute.
#     Both are hot paths here (attention softmax; SwiGLU MLP with ffn_hidden=25600).
#
#  2. CONTAINER. Traces are generated in simulon.sif (TE 2.12.0, cuDNN 9.19.0); the
#     physical runs use nemo_26.04.sif (TE 2.14.0, cuDNN 9.20.0). torch and CUDA are
#     IDENTICAL, so TE/cuDNN is the whole delta -- and TE implements the fused
#     attention/linear kernels that dominate this model's compute.
#
# DESIGN. Config is tp4pp2-mbs1 throughout (hardware = 9274 ms, mean iters 21-40;
# current sim = 9886 ms = +6.6%, compute 7999). Existing simulon.sif+fusions-OFF trace is
# the 4th cell, already on disk, so three new traces complete the square:
#
#                     fusions OFF            fusions ON
#   simulon.sif   (baseline, have it)   A  -> isolates FUSION effect
#   nemo_26.04    B -> CONTAINER effect  C  -> fully matched to production
#
# READ THE RESULT: compare each cell's compute_ms against the baseline 7999 and the
# hardware-implied compute. If C removes most of the +6.6%, the residual was a toolchain
# artifact, NOT a fake-PG limit -> re-trace everything in nemo_26.04.sif with the fused
# workload, and fix _TRACE_DEFAULTS. If C barely moves, the toolchain is exonerated.
#
# All three run back-to-back on one GPU with clocks pinned, so thermal drift is common-mode.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=trace_ctr_fusion_ab
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_ctr_fusion_ab-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIM_SIF=/e/project1/e-sta-openeurollm/vanosch1/containers/simulon.sif
NEMO_SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/sweep_qwen3_32b_workloads/unified
OUTBASE=templates/gpu/gh200_jupiter-ctrfusion/traces

echo "SM clock: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"

gen() {
    local label="$1" sif="$2" wl="$3" out="$OUTBASE/$1"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] $label : sif=$(basename $sif) workload=$wl"
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$sif" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$wl.yaml' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] $label exit=$?"
    [ -f "$out/trace_rank_0.json" ] && echo "  OK: $(ls $out | tr '\n' ' ')" || echo "  !! NO TRACE"
}

gen A_simulon_fused  "$SIM_SIF"  tp4pp2-mbs1-fused   # fusion effect (vs existing baseline)
gen B_nemo_unfused   "$NEMO_SIF" tp4pp2-mbs1         # container effect
gen C_nemo_fused     "$NEMO_SIF" tp4pp2-mbs1-fused   # both -> production-matched

echo "[$(date +%H:%M:%S)] all done"
