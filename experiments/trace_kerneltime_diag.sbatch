#!/bin/bash
# DECISIVE experiment: does PER-KERNEL GPU time show the mbs2 efficiency the wall-clock
# span misses? Re-trace TP4/PP4 mbs1 + mbs2 with SIMULON_TRACE_KERNEL_TIME=1, which makes
# the tracer (vendored Megatron instrumentation/tracer.py) additionally run torch.profiler
# and attribute each kernel's DEVICE time to its slot — pure kernel time, idle gaps
# removed. Clocks pinned via the site --comment mechanism as well.
#
# The new trace fields are ADDITIVE (wall-clock timestamps untouched):
#   * total_kernel_device_ms  — sum of all kernel device time for the iteration
#   * total_span_ms           — wall-clock span (for the ratio comparison)
#   * per slot: metadata.kernel_device_ms
# NOTE: with the profiler on, the SPAN timestamps are further inflated by profiler CPU
# overhead — only the *_kernel_device_ms fields are meaningful in this run.
#
# READ THE RESULT (analysis script after the job):
#   efficiency(mbs) = total_kernel_device_ms / tokens.  If kernel-time mbs1->mbs2 shows
#   ~-15% per token (matching hardware) while the span shows ~-2%, then idle gaps were
#   hiding the efficiency and per-kernel timing is the fix -> wire kernel_device_ms into
#   the sim. If kernel-time is ALSO flat (~-2%), the efficiency genuinely isn't there
#   under fake-PG -> switch the mbs/TP axis to the hardware-anchored amortization law.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=trace_kerneltime
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_kerneltime-%j.log

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
    # SIMULON_TRACE_KERNEL_TIME reaches the Python process inside the container.
    APPTAINERENV_SIMULON_TRACE_KERNEL_TIME=1 \
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=1 PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cfg.yaml' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] $cfg exit=$?"
    # quick peek: are the new fields present?
    python3 - "$out/trace_rank_0.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
tk = d.get("total_kernel_device_ms"); ts = d.get("total_span_ms")
nslots = sum(1 for e in d.get("events", []) if e["type"] == "slot_begin"
             and "kernel_device_ms" in e.get("metadata", {}))
print(f"  total_kernel_device_ms={tk}  total_span_ms={ts}  slots_with_kernel_ms={nslots}")
PY
}

gen tp4pp4-mbs1
gen tp4pp4-mbs2

echo "[$(date +%H:%M:%S)] done"
