#!/bin/bash
# KERNEL-TIMING trace session — the non-hardcoded fix for the mbs AND TP GEMM-efficiency
# axes. Re-traces the isolation set with SIMULON_TRACE_KERNEL_TIME=1 so each slot carries
# metadata.kernel_device_ms (pure GPU kernel time, fake-PG CPU-launch idle removed). The
# sweep (_apply_mbs_efficiency in sweep_qwen3_32b.py) then builds the compute correction
# from MEASURED trace quantities — overhead(cfg)=span_compute-kernel_compute, held at its
# mbs=1/ref value — with NO fitted latency constant.
#
# WHAT THIS ENABLES (once traces land, wired via TRACE_REGISTRY_KERNELTIME):
#   * mbs axis at PP1/PP2/PP4  (tp4pp{1,2,4} mbs1+mbs2) — extends the tp4pp4 result
#     (already +8.0% / -11.1% scaling, GOOD) to pp1 and pp2, retiring the hardcoded
#     amortization fallback entirely.
#   * TP axis  (tp2pp4-mbs1 vs tp4pp4-mbs1) — kernel_compute(tp2) is the missing piece
#     to correct the last failing axis (TP4->2 measured -16.4% vs span -4.1%).
#
# ONE session on one GPU: thermal/clock drift hits every trace equally (the cross-session
# ~9% compute spread is why we capture back-to-back). Clocks pinned via --comment.
# NOTE: with the profiler on, the SPAN timestamps are inflated by profiler CPU overhead —
# only the *_kernel_device_ms fields are trusted in a kernel-timing run (the correction
# uses kernel_device_ms; the span it needs comes from the non-profiler unified traces).
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=trace_kerneltime_sess
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_kerneltime_sess-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/vanosch1/containers/simulon.sif
W=experiments/sweep_qwen3_32b_workloads/unified
OUTBASE=templates/gpu/gh200_jupiter-kerneltime/traces

echo "SM clock: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"

gen() {
    local cfg="$1" out="$OUTBASE/$1"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] tracing $cfg (kernel-time ON) -> $out"
    mkdir -p "$out"
    APPTAINERENV_SIMULON_TRACE_KERNEL_TIME=1 \
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=1 PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cfg.yaml' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] $cfg exit=$?"
    python3 - "$out/trace_rank_0.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
tk = d.get("total_kernel_device_ms")
n = sum(1 for e in d.get("events", []) if e["type"] == "slot_begin"
        and "kernel_device_ms" in e.get("metadata", {}))
print(f"  total_kernel_device_ms={tk}  slots_with_kernel_ms={n}")
PY
}

# Interleaved so each comparison pair is captured adjacently:
gen tp4pp4-mbs1     # ┐ mbs @ pp4  +  TP-axis denominator
gen tp4pp4-mbs2     # ┘
gen tp4pp2-mbs1     # ┐ mbs @ pp2
gen tp4pp2-mbs2     # ┘
gen tp4pp1-mbs1     # ┐ mbs @ pp1 (pipeline-free)
gen tp4pp1-mbs2     # ┘
gen tp2pp4-mbs1     # ┐ TP axis numerator (vs tp4pp4-mbs1)
gen tp2pp4-mbs2     # ┘

echo "[$(date +%H:%M:%S)] all done"
