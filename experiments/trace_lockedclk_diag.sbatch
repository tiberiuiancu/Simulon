#!/bin/bash
# DIAGNOSTIC re-trace #2: lock the GPU clock via the Slurm site mechanism (prolog runs
# as root and honors the --comment), which my in-job `nvidia-smi -lgc` could not do
# (job 1006317: LOCK DENIED, needs elevated perms).
#
# WHAT WE LEARNED FROM #1: trace compute = CUDA-event span between slot boundaries, which
# includes GPU-stream idle gaps. The tracing GPU idled at ~400 MHz 32% of the session and
# hit max boost (1980) only 27% (temp 58C — not thermal). So the span is inflated by idle
# time AND by kernels running during boost ramp-up.
#
# WHAT LOCKING TESTS: pinning the SM clock to 1980 removes the boost ramp-up penalty
# (each kernel runs at full clock from the first cycle, like a pegged 64-GPU run). It does
# NOT remove idle wall-clock time. So:
#   * trace MFU jumps toward 50% / 59%  => ramp-up/boost-state was the dominant cause;
#     locking clocks during tracing is the fix (cheap, just this SBATCH comment).
#   * trace MFU barely moves            => the CPU-bound idle gaps dominate; the real fix
#     is per-kernel GPU-time instrumentation in the tracer (bigger change).
# The clock log tells us definitively whether the lock took effect.
#
# Comm timings are unaffected regardless (collective durations come from the measured
# NcclProfile, never from trace wall-clock).
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=trace_lockedclk
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_lockedclk-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/vanosch1/containers/simulon.sif
W=experiments/sweep_qwen3_32b_workloads/unified
OUTBASE=templates/gpu/gh200_jupiter-lockedclk/traces
CLKLOG="experiments/trace_lockedclk-${SLURM_JOB_ID}.clocks.csv"

echo "=== clocks at job start (did the --comment lock take effect?) ==="
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv || true

# background clock/temp/power logger for GPU 0 (the tracing GPU)
echo "timestamp,clocks.sm[MHz],temperature.gpu[C],power.draw[W],utilization.gpu[%]" > "$CLKLOG"
( while true; do
    nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu \
        --format=csv,noheader,nounits -i 0 >> "$CLKLOG" 2>/dev/null
    sleep 1
  done ) &
LOGGER=$!

gen() {
    local cfg="$1" out="$OUTBASE/$1"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] tracing $cfg -> $out"
    echo "[$(date +%H:%M:%S)] SM clock just before: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cfg.yaml' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] $cfg exit=$?  SM clock after: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
    [ -f "$out/.OOM" ] && echo "!! OOM MARKER for $cfg"
}

gen tp4pp4-mbs1
gen tp4pp4-mbs2

kill "$LOGGER" 2>/dev/null

echo; echo "=== SM clock summary during the session ==="
awk -F, 'NR>1{c=$2+0; if(c>0){s+=c;n++; if(c>mx)mx=c; if(mn==0||c<mn)mn=c}} END{if(n)printf "  SM clock  min=%d  mean=%d  max=%d MHz\n",mn,s/n,mx}' "$CLKLOG"
awk -F, 'NR>1{c=$2+0; if(c>1800)hi++; if(c>0)n++} END{if(n)printf "  time at >1800 MHz: %d%%  (was 27%% unlocked)\n",hi*100/n}' "$CLKLOG"
awk -F, 'NR>1{t=$3+0; if(t>0){s+=t;n++; if(t>mx)mx=t}} END{if(n)printf "  temp mean=%d max=%d C\n",s/n,mx}' "$CLKLOG"
echo "[$(date +%H:%M:%S)] done — clock log: $CLKLOG"
