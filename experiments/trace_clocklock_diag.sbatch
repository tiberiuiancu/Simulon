#!/bin/bash
# DIAGNOSTIC re-trace: is the trace's compute over-estimate a GPU clock/thermal effect?
#
# Roofline (VALIDATION_16N.md §RESULT) showed the traces record ~47% MFU flat across mbs,
# while real training does 47% (mbs1) -> 59% (mbs2). Trace compute = wall-clock gaps
# between collectives on ONE fake-PG GPU (trace_tracer.py:1129), which is NOT pegged the
# way a real 64-GPU run is. Hypothesis: the tracing GPU runs at a lower / throttling clock
# than sustained training, so its kernel wall-clock is inflated and the mbs2 GEMM
# efficiency gain is masked.
#
# THIS JOB tests the hypothesis AND the fix at once, for TP4/PP4 mbs1 and mbs2:
#   1. logs SM clock + temperature + power every 1s throughout (the mechanism evidence);
#   2. WARMS the GPU with a burn kernel before each capture;
#   3. ATTEMPTS to lock the SM clock to max (nvidia-smi -lgc). If the cluster denies it
#      (needs elevated perms) the run continues UNLOCKED — the clock log still shows what
#      the GPU did, which is the diagnostic we need either way.
#
# Compare afterwards: run the roofline (experiments/analyze_val40.py path or the ad-hoc
# roofline script) on templates/gpu/gh200_jupiter-clocklock/traces/*. If trace MFU jumps
# toward 50%/59%, clock/thermal was the cause and clock-locked tracing is the fix. The
# clock log tells us the mechanism regardless.
#
# Comm timings are unaffected by any of this: collective durations are recomputed from the
# measured NccclProfile (collective_populate.py), never from trace wall-clock.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=trace_clocklock
#SBATCH --output=experiments/trace_clocklock-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/vanosch1/containers/simulon.sif
W=experiments/sweep_qwen3_32b_workloads/unified
OUTBASE=templates/gpu/gh200_jupiter-clocklock/traces
CLKLOG="experiments/trace_clocklock-${SLURM_JOB_ID}.clocks.csv"

echo "=== GPU inventory / max clocks ==="
nvidia-smi --query-gpu=index,name,clocks.max.sm,clocks.max.memory --format=csv || true

MAXSM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
echo "max SM clock = ${MAXSM} MHz"

echo "=== attempting to lock SM clock to max (may be denied without elevated perms) ==="
if nvidia-smi -i 0 -lgc "${MAXSM},${MAXSM}" 2>&1; then
    echo "LOCK: succeeded"
    LOCKED=1
else
    echo "LOCK: DENIED — proceeding unlocked; the clock log below is the evidence"
    LOCKED=0
fi

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
    echo "[$(date +%H:%M:%S)] tracing $cfg (clock-locked=$LOCKED) -> $out"
    echo "[$(date +%H:%M:%S)] clock just before $cfg: $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader -i 0)"
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cfg.yaml' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] $cfg exit=$?  clock after: $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader -i 0)"
    [ -f "$out/.OOM" ] && echo "!! OOM MARKER for $cfg"
}

gen tp4pp4-mbs1
gen tp4pp4-mbs2

kill "$LOGGER" 2>/dev/null
[ "$LOCKED" = "1" ] && { echo "resetting clocks"; nvidia-smi -i 0 -rgc || true; }

echo; echo "=== SM clock summary during the session (col 2 of $CLKLOG) ==="
awk -F, 'NR>1{c=$2+0; if(c>0){s+=c;n++; if(c>mx)mx=c; if(mn==0||c<mn)mn=c}} END{if(n)printf "  SM clock  min=%d  mean=%d  max=%d MHz  (max hardware=%s)\n",mn,s/n,mx,"'"$MAXSM"'"}' "$CLKLOG"
awk -F, 'NR>1{t=$3+0; if(t>0){s+=t;n++; if(t>mx)mx=t}} END{if(n)printf "  temp      mean=%d  max=%d C\n",s/n,mx}' "$CLKLOG"
awk -F, 'NR>1{p=$4+0; if(p>0){s+=p;n++; if(p>mx)mx=p}} END{if(n)printf "  power     mean=%d  max=%d W\n",s/n,mx}' "$CLKLOG"
echo "[$(date +%H:%M:%S)] done — clock log: $CLKLOG"
