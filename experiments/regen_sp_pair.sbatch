#!/bin/bash
# Regenerate the TP4/PP1 SP=true / SP=false trace pair BACK-TO-BACK in one
# allocation and one container instance.
#
# Why: the existing pair was generated 2h41m apart and 12 days earlier than the
# other traces, and forward-pass trace time varies ~15% run-to-run between builds.
# That confound inflates the apparent SP=true compute "saving" (17.5% at TP4/PP1 vs
# only 1.8% for the pair generated 5 min apart), which is a large part of the
# SP=true accuracy gap. Matched conditions remove it.
#
# Writes to *-rerun dirs so the existing traces are preserved for comparison.
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:50:00
#SBATCH --job-name=regen_sp_pair
#SBATCH --output=experiments/regen_sp_pair-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/vanosch1/containers/simulon.sif

run_one () {
    local wl="$1" out="$2" label="$3"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] tracing $label"
    echo "  workload: $wl"
    echo "  output  : $out"
    echo "=============================================================="
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$wl' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] exit=$? for $label"
    ls -la "$out" 2>/dev/null
}

# Same container invocation, same node, seconds apart — the only difference is the
# sequence-parallel flag in the workload.
run_one templates/gpu/gh200_jupiter/traces/5a8d3f1964fbbea1/workload.yaml \
        templates/gpu/gh200_jupiter-tp4pp1sptrue-rerun/traces/gbs256 "SP=true"

run_one "templates/gpu/gh200_jupiter-sp=false/traces/5a8d3f1964fbbea1/workload.yaml" \
        templates/gpu/gh200_jupiter-tp4pp1spfalse-rerun/traces/gbs256 "SP=false"

echo "[$(date +%H:%M:%S)] done"
