#!/bin/bash
# Bisect the silent death of the host-cost trace job (1222963, 1223085).
#
# WHAT WE KNOW
#   * Both runs died ~31-36 s into "Tracing rank 0 ..." with a <2 KB log, no traceback, and
#     without gen() reaching its own "!! NO TRACE" echo -- the batch shell died too.
#   * Not resources: the sampler on 1223085 showed host RAM flat at 736 GB available and
#     GPU under 11 MiB for the whole run. (Caveat: with --gres=gpu:1 the batch-side
#     `nvidia-smi -i 0` may not be the GPU the step was given, so treat the GPU column as
#     weak evidence.)
#   * Matching the known-good header (--gres=gpu:1 --mem=200G, job 1150846) did NOT fix it.
#
# THE TWO CANDIDATES, AND WHY THE EVIDENCE CONFLICTS
#   (1) The kernel-timing path. Job 1150972 died silently under srun with kt=1, and two KT
#       traces OOM'd under torch.profiler in job 1008192 -- so kt=1 has a known failure mode.
#   (2) The tracer change (per-slot host_ms / host_ops / launch_count / kernel_count).
#   Against (1): tp1pp1-mbs1 is the SMALLEST cell and it traced fine with kt=1 before -- it
#   is sitting in templates/gpu/gh200_jupiter-1p7b-unified-kt/traces right now.
#   Against (2): the new code runs at END of iteration inside a try/except, long after the
#   ~31 s mark, and the identical byte-signature predates it (jobs 1149575/1150654).
#   Neither story explains everything, so measure instead of arguing.
#
# THE BISECT: same cell, same everything, kt=0 then kt=1.
#   A passes, B fails -> the profiler path, and the tracer change is exonerated
#   A fails too      -> not the profiler; suspect the tracer import or the environment
#   both pass        -> the failure is cell-specific or intermittent; re-run the full job
#
# CRITICAL: the container's stderr is redirected to a FILE on /e, not left to the job log.
# Every previous investigation lost the traceback because the cgroup died before stdout was
# flushed. A file on shared storage survives that.
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=00:25:00
#SBATCH --job-name=diag_hostcost_bisect
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/diag_hostcost_bisect-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/thomas_1p7b_workloads
CELL=tp1pp1-mbs1
LOGDIR=/e/scratch/e-sta-openeurollm/vanosch1/diag_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

echo "container : $SIF"
echo "SM clock  : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "stderr to : $LOGDIR"
echo "started   : $(date)"

run_variant() {
    local kt="$1" out="$LOGDIR/out_kt$1"
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] VARIANT kt=$kt  -> $LOGDIR/megatron_kt$kt.log"
    mkdir -p "$out"
    # PYTHONFAULTHANDLER dumps a C-level traceback on a fatal signal, which is the one
    # thing that would survive a SIGSEGV/SIGABRT and has never been captured here.
    APPTAINERENV_SIMULON_TRACE_KERNEL_TIME="$kt" \
    srun --ntasks=1 --gres=gpu:1 \
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=$kt PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONFAULTHANDLER=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python -u -c 'from simulon.cli import app; app()' trace generate \
            '$W/$CELL.yaml' --gpu gh200_jupiter -o '$out'" \
        > "$LOGDIR/megatron_kt$kt.log" 2>&1
    local rc=$?
    sync
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "  kt=$kt PASS (exit=$rc)"
        python3 - "$out/trace_rank_0.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))
print("    span_ms      ", round(t.get("total_span_ms") or 0, 1))
print("    kernel_ms    ", round(t.get("total_kernel_device_ms") or 0, 1))
print("    host_ms      ", round(t.get("total_host_ms") or 0, 1), "  <-- the new field")
print("    host_ops     ", t.get("total_host_ops"))
print("    launch_count ", t.get("total_launch_count"))
PY
    else
        echo "  kt=$kt FAIL (exit=$rc)"
        # simulon/cli/trace.py now STREAMS Megatron's output here as it runs, so this
        # survives a kill. Previously it used capture_output=True and every byte was lost
        # whenever the child was killed rather than exiting -- which is why four jobs in a
        # row produced a mute log.
        echo "  --- tail of $out/.megatron.log (Megatron itself) ---"
        tail -40 "$out/.megatron.log" 2>/dev/null | sed 's/^/    /' || echo "    (absent)"
        echo "  --- tail of the simulon CLI's own output ---"
        tail -15 "$LOGDIR/megatron_kt$kt.log" | sed 's/^/    /'
    fi
    sync
}

run_variant 0
run_variant 1

echo "=============================================================="
echo "[$(date +%H:%M:%S)] done. Full logs kept in $LOGDIR"
echo "  kt=0 PASS + kt=1 FAIL -> profiler path; tracer change exonerated"
echo "  both FAIL             -> not the profiler; suspect tracer import / environment"
echo "  both PASS             -> intermittent or cell-specific; re-run the full job"
