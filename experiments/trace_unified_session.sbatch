#!/bin/bash
# UNIFIED single-session trace regeneration — all 10 SP=false traces, one GPU, one job.
#
# WHY ONE SESSION
# ---------------
# The existing PP2 and PP4 trace sets were captured in different sessions and disagree
# by a constant ~6.4% on TOTAL COMPUTE — a quantity that is provably PP-invariant
# (per-GPU work = (64/pp) layers x n_mb, n_mb = 16*pp/mbs => 1024/mbs layer-microbatches
# regardless of pp). That cross-session clock/thermal drift puts a ~6% floor under any
# validation. Capturing everything back-to-back on the same GPU removes it.
#
# WHAT IS NEW BESIDES THE REGENERATION
#   * tp4pp1 mbs1/mbs2  — PP=1 has no pipeline at all, so its mbs1->mbs2 ratio measures
#                         pure kernel efficiency + DP comm; arbitrates the PP2 anomaly.
#   * tp4pp2/tp4pp4 -rc — NATIVE recompute traces. These run the real recomputed
#                         backward kernels, replacing the fitted
#                         SELECTIVE_RECOMPUTE_COMPUTE_OVERHEAD=0.0504 constant with
#                         measured kernels (same medicine that fixed mbs).
#   * tp2pp4 mbs1/mbs2  — the TP axis' trace-side, regenerated in-session.
#
# All outputs land under templates/gpu/gh200_jupiter-unified/traces/<cfg>/ and are wired
# into the sweep via TRACE_REGISTRY_UNIFIED (sweep_qwen3_32b.py), which prefers them
# over the legacy per-session dirs whenever they exist.
#
# Order: interleaved so the two members of each comparison pair are captured
# adjacently (thermal drift within the session hits both sides equally).
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=trace_unified
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_unified-%j.log
# --comment: site prolog pins the SM clock (verified job 1006682). Removes the boost
# ramp-up variance in the wall-clock spans and cuts trace-to-trace noise ~a few %.

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/vanosch1/containers/simulon.sif
W=experiments/sweep_qwen3_32b_workloads/unified
OUTBASE=templates/gpu/gh200_jupiter-unified/traces

gen() {
    local cfg="$1"
    local out="$OUTBASE/$cfg"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] tracing $cfg -> $out"
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cfg.yaml' --gpu gh200_jupiter -o '$out'"
    local rc=$?
    echo "[$(date +%H:%M:%S)] $cfg exit=$rc"
    [ -f "$out/.OOM" ] && echo "!! OOM MARKER for $cfg"
    ls "$out"
}

# Comparison pairs adjacent:
gen tp4pp2-mbs1        # ┐ PP axis @ mbs1
gen tp4pp4-mbs1        # ┘
gen tp4pp2-mbs2        # ┐ PP axis @ mbs2 / mbs axis @ PP2+PP4
gen tp4pp4-mbs2        # ┘
gen tp4pp1-mbs1        # ┐ mbs axis @ PP1 (pipeline-free)
gen tp4pp1-mbs2        # ┘
gen tp4pp2-mbs1-rc     # ┐ recompute natives (vs their -mbs1 partners above)
gen tp4pp4-mbs1-rc     # ┘
gen tp2pp4-mbs1        # ┐ TP axis (vs tp4pp4-mbs1)
gen tp2pp4-mbs2        # ┘

echo "[$(date +%H:%M:%S)] all done"
