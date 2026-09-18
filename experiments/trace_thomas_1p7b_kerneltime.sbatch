#!/bin/bash
# KERNEL-TIMING traces for the Qwen3-1.7B mbs axis — the foundation experiment for
# "trace a base, extrapolate the sweep". SIMULON_TRACE_KERNEL_TIME=1 makes each slot carry
# metadata.kernel_device_ms (pure GPU kernel time, fake-PG CPU-launch idle removed).
#
# WHY. The native scorecard (experiments/analyze_1p7b.py) showed the mbs=1 ANCHORS
# over-predict (tp1pp1 +12.6%, tp4pp1 +14.3%) and single-base mbs-extrapolation FAILS
# (+17..+71%). Root cause: the sim uses the fake-PG wall-clock SPAN as per-op time; that
# span carries a CPU-launch idle gap that (a) accumulates with microbatch count and (b)
# does NOT shrink when the GEMM gets more efficient. kernel_device_ms is the clean signal.
#
# The 32B correction (_apply_gemm_efficiency) holds the launch overhead at the tp4/mbs1
# reference, ASSUMING that config's span matches hardware (true at 32B, +5-7%). At 1.7B the
# mbs=1 anchors ARE the inflated points, so that assumption breaks. These traces let us
# measure overhead(cfg)=span_compute-kernel_compute per cell and test whether overhead PER
# LAUNCH is config-invariant — the basis for a reference-free refounding (no fitted const).
#
# NOTE (from the 32B KT run): with the profiler ON the SPAN timestamps are inflated, so ONLY
# kernel_device_ms is trusted here; the span term comes from the non-profiler traces already
# in templates/gpu/gh200_jupiter-1p7b/traces. Same nemo_26.04.sif as the physical + span
# traces so the toolchain (TE 2.14, fusions ON) matches. Clocks pinned via --comment.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=trace_thomas_1p7b_kt
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_thomas_1p7b_kt-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/thomas_1p7b_workloads
OUT=templates/gpu/gh200_jupiter-1p7b-kerneltime/traces

echo "container: $SIF"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"

gen() {
    local cell="$1" out="$OUT/$1"
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] $cell present -> skip"; return 0; fi
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] $cell (kernel-time ON)"
    mkdir -p "$out"
    APPTAINERENV_SIMULON_TRACE_KERNEL_TIME=1 \
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=1 PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cell.yaml' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] $cell exit=$?"
    if [ -f "$out/trace_rank_0.json" ]; then
        python3 - "$out/trace_rank_0.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
tk = d.get("total_kernel_device_ms")
n = sum(1 for e in d.get("events", []) if e.get("type") == "slot_begin"
        and "kernel_device_ms" in (e.get("metadata") or {}))
print(f"  total_kernel_device_ms={tk}  slots_with_kernel_ms={n}")
PY
    else
        echo "  !! NO TRACE (likely OOM under profiler) -- continuing"
    fi
}

# mbs axis at tp1pp1 (the anchor + the two extrapolation targets), plus the tp2pp1 mbs8 leg.
# Ordered so each anchor/target pair is captured adjacently (thermal/clock drift hits equally).
for cell in tp1pp1-mbs1 tp1pp1-mbs2 tp1pp1-mbs4 tp2pp1-mbs1 tp2pp1-mbs8; do
    gen "$cell"
done
echo "[$(date +%H:%M:%S)] done"
echo "=== inventory ==="
for d in "$OUT"/*/; do [ -f "$d/trace_rank_0.json" ] && echo "  $(basename "$d") OK" || echo "  $(basename "$d") MISSING"; done
