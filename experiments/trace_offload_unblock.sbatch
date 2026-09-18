#!/bin/bash
# UNBLOCK the two configs that fake-PG has never been able to trace: tp2pp4-mbs2 and
# tp4pp1-mbs2. Both OOM under --fake-process-group (0 rank traces in every attempt, incl.
# the unified and kernel-timing sessions) because a single tracing GPU has no data-parallel
# group to shard the distributed optimizer across, so Adam's fp32 moments (~8 B/param,
# ~33 GB at TP=2) must live on that one GPU. Measured failure: ~155 MiB free of 95 GiB.
#
# FIX: trace with --optimizer-cpu-offload: keeps Adam AND the distributed optimizer (so DP collectives stay correct) while moving states to host memory.
# WHY THIS IS SAFE: the optimizer runs inside a dedicated 'step' slot, and that slot is
# only 0.1-0.2% of traced compute -- 11.9 / 13.5 / 14.0 ms out of ~7900 ms on
# tp4pp4-mbs1 / tp4pp2-mbs1 / tp2pp4-mbs1 respectively. The fwd/bwd kernels, which are all
# the simulator's compute model consumes, are identical regardless of optimizer choice.
# So the fidelity cost is bounded well below every margin we validate against (5-10%).
#
# WHY IT MATTERS: tp2pp4-mbs2 is the missing piece for BOTH the TP axis (currently resting
# on a single measured point, tp2pp4-mbs1) and the MBS axis at TP=2. Without a native
# trace the sweep silently falls back to the tp2pp4-mbs1 trace and extrapolates, which
# holds compute constant and misses exactly the GEMM efficiency this session fixed.
#
# Production-matched toolchain: nemo_26.04.sif + fusions ON (see trace_production_session).
# span + kernel captured adjacently per config, clocks pinned, so overhead = span - kernel
# is drift-free.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=trace_sgd_unblock
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_sgd_unblock-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/sweep_qwen3_32b_workloads/unified
SPAN_OUT=templates/gpu/gh200_jupiter-prod/traces
KERN_OUT=templates/gpu/gh200_jupiter-prod-kerneltime/traces

echo "SM clock: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"

gen() {
    local wl="$1" name="$2" mode="$3" out
    if [ "$mode" = "kernel" ]; then out="$KERN_OUT/$name"; else out="$SPAN_OUT/$name"; fi
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] $name [$mode] <- $wl"
    mkdir -p "$out"
    local pfx=""
    [ "$mode" = "kernel" ] && pfx="SIMULON_TRACE_KERNEL_TIME=1"
    APPTAINERENV_SIMULON_TRACE_KERNEL_TIME=$([ "$mode" = kernel ] && echo 1 || echo 0) \
    APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && $pfx PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$wl.yaml' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] $name [$mode] exit=$?"
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "  *** SUCCESS -- SGD unblocked a config fake-PG could never trace ***"
        ls "$out" | tr '\n' ' '; echo
    else
        echo "  !! still no trace (SGD did not free enough) -- continuing"
    fi
}

gen tp2pp4-mbs2-offload tp2pp4-mbs2 span     # the high-value one: TP axis + MBS@TP2
gen tp2pp4-mbs2-offload tp2pp4-mbs2 kernel
# tp4pp1-mbs2 DROPPED: its hardware run OOM'd for real (job 989337), so MBS@pp1 can
# never be validated regardless of the trace -- tracing it would gain nothing.

echo "[$(date +%H:%M:%S)] done"
