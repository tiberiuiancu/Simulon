#!/bin/bash
# Generate NATIVE mbs=2 traces for TP4/PP2 and TP4/PP4 at SP=false.
#
# WHY THIS EXISTS
# ---------------
# Every one of the 14 entries in TRACE_REGISTRY is mbs=1, even though the registry key
# is (tp, pp, sp, mbs). All mbs>1 results therefore come from _extrapolate_trace_for_mbs,
# which scales every timestamp by exactly new_mbs/trace_mbs -- so per-token compute is
# linear in mbs BY CONSTRUCTION and the total compute cannot change. That is structurally
# incapable of reproducing the measured effect:
#
#     TP4/PP2:  mbs1 10799 ms -> mbs2 7566 ms  (-30%, 315.8 -> 450.8 TFLOP/s)
#     TP4/PP4:  mbs1  8773 ms -> mbs2 7860 ms  (-10%)
#   simulator:  -5% and -0.07% respectively.
#
# The mbs effect is a compute-KERNEL effect (larger GEMMs, fewer fixed per-microbatch
# overheads). Trace generation executes real kernels on a real GH200 -- only the process
# group is faked -- so a native mbs=2 trace measures it directly. No fitted fudge factor.
#
# SELF-VALIDATING: measured ground truth for both points already exists and is solid
# (PP2 mbs2 replicates at 7566 / 7611, <1% apart, with a clear plateau). If the simulator
# moves 9961 -> ~7600 for PP2 and 11042 -> ~7900 for PP4, the mbs model is fixed.
#
# RISK: the fake process group cannot shard optimizer state, so one GPU holds all of it.
# mbs=2 doubles the activation term on top of that. TP4/PP2 mbs=1 traced fine, so there
# is headroom, but if this OOMs the fallback is to trace at PP=4 only (model state 14.8
# vs 23.3 GB) or to enable recompute during tracing.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=trace_mbs2_spfalse
#SBATCH --output=experiments/trace_mbs2_spfalse-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/vanosch1/containers/simulon.sif
W=experiments/sweep_qwen3_32b_workloads

gen() {
    local wl="$1" out="$2" label="$3"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] tracing $label -> $out"
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$wl' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] exit=$?"
    ls -la "$out"
    [ -f "$out/.OOM" ] && echo "!! OOM MARKER — retry at PP=4 only, or trace with recompute"
}

gen "$W/trace_tp4pp2spfalse_mbs2_gbs256.yaml" \
    "templates/gpu/gh200_jupiter-tp4pp2spfalse-mbs2/traces/gbs256" "TP4/PP2 mbs=2"

gen "$W/trace_tp4pp4spfalse_mbs2_gbs256.yaml" \
    "templates/gpu/gh200_jupiter-tp4pp4spfalse-mbs2/traces/gbs256" "TP4/PP4 mbs=2"

echo "[$(date +%H:%M:%S)] done"
