#!/bin/bash
# R-HOST: regenerate the 1.7B kernel-timing registry so it carries HOST time.
#
# WHY THIS IS THE FIRST JOB OF THE CAMPAIGN. The physical val40 numbers for these cells
# already exist (experiments/results_1p7b.txt), so once the traces carry host_ms the whole
# host-cost model can be scored WITHOUT any new physical training run. This is the cheap,
# fast loop: ~11 single-GPU traces, then `analyze_1p7b.py` re-scores against measurements
# already in hand.
#
# WHAT CHANGED IN THE TRACER (vendor/.../instrumentation/tracer.py): each slot now also
# records
#   host_ms      union of the slot's cpu_op intervals = how long it kept the host thread
#                busy in framework code (python / dispatcher / autograd)
#   host_ops     aten-op count      launch_count  CUDA launch calls      kernel_count
# This is the term the simulator was missing. nsys profiling of four cells showed 36-41% of
# every iteration is GPU idle on the pace-setting rank, and that 79-83% of that idle is host
# time with no CUDA call in flight (experiments/gap_attribution.py). It was previously
# mis-modelled as a per-collective constant (NcclProfile.launch_latency_ms), which is why
# that constant drifted 22-26% across microbatch size.
#
# The container binds the repo at /opt/simulon, so the edited vendored tracer is picked up
# with no image rebuild.
#
# PRE-REGISTERED EXPECTATIONS (state before looking at results):
#   * total_host_ms should be of order 30-50% of the traced iteration span. Much below 10%
#     means cpu_op records are not being captured and the whole term is unmeasured -- treat
#     the traces as failed, do not "fix" it by scaling.
#   * host_ms should be roughly FLAT across mbs at fixed TP (same ops, bigger tensors) while
#     kernel_device_ms grows. If host_ms tracks kernel time instead, the union is picking up
#     GPU-side ranges and the attribution is wrong.
#   * host_ms should be roughly flat across TP as well (same program, sharded tensors).
#
# NOTHING IN THIS JOB MAY TOUCH /e/home. Jobs 1222963, 1223085 and 1224406 all died ~30 s
# in for the same reason, visible only once simulon/cli/trace.py stopped swallowing
# Megatron's output:
#     OSError: [Errno 116] Stale file handle: '/opt/simulon/vendor/.../quantization'
#     slurmstepd: couldn't chdir to `.../Simulon': Stale file handle: going to /tmp instead
#     FATAL: container creation failed ... stale file handle
# The compute node's handle on the /e/home GPFS export (`exa_home`) goes stale mid-job; the
# ~30 s time-to-death is just how long the megatron/torch import chain runs before it walks
# a directory that has gone. So --chdir and --output both live on /e/scratch, and the code
# is read from a staged snapshot there (experiments/stage_repo.sh).
# This was never the profiler, never memory, and never the tracer change -- and it says
# nothing about kt=0 vs kt=1, since the kt=1 variant never started (exit 255).
# Trace generation is single-GPU anyway (fake process group), so gpu:1 is correct.
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=288
#SBATCH --mem=200G
#SBATCH --time=00:55:00
#SBATCH --job-name=trace_1p7b_hostcost
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --chdir=/e/scratch/e-sta-openeurollm/vanosch1
#SBATCH --output=/e/scratch/e-sta-openeurollm/vanosch1/logs/trace_1p7b_hostcost-%j.log

set -uo pipefail
# RUN ENTIRELY FROM /e/scratch. Jobs 1222963/1223085/1224406 all died ~30 s in because the
# compute node's handle on the /e/home GPFS export went stale mid-import ("OSError: [Errno
# 116] Stale file handle", and slurmstepd could not even chdir into the repo). Nothing here
# may touch /e/home. Refresh the stage with `bash experiments/stage_repo.sh` after any code
# change -- it is a snapshot, not a live view.
STAGE=/e/scratch/e-sta-openeurollm/vanosch1/simulon_stage
OUTBASE=/e/scratch/e-sta-openeurollm/vanosch1/hostcost_traces
if [ ! -f "$STAGE/simulon/cli/__init__.py" ]; then
    echo "FATAL: no stage at $STAGE -- run 'bash experiments/stage_repo.sh' first" >&2
    exit 1
fi
cd "$STAGE" || exit 1
mkdir -p "$OUTBASE"

# If it dies silently again, leave evidence instead of another mute log.
SAMP=/e/scratch/e-sta-openeurollm/vanosch1/hostcost_samples_${SLURM_JOB_ID}.csv
( echo "t,host_avail_GB,gpu_used_MiB"
  while true; do
      printf "%s,%s,%s\n" "$(date +%H:%M:%S)" \
          "$(free -g | awk '/^Mem:/{print $7}')" \
          "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)"
      sleep 5
  done ) > "$SAMP" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/thomas_1p7b_workloads
# New registry, so the existing -unified-kt one stays intact as a fallback and an A/B.
KT_OUT="$OUTBASE"

echo "container: $SIF"
# Reported 1830 MHz on job 1222963 despite --comment=NVIDIA_GPU_CLOCKS=1980, so the clock
# lock is not taking on this partition. Not fatal here -- host_ms is what these traces are
# for, and it is CPU-side -- but it does mean kernel_device_ms across sessions carries the
# usual ~9% GH200 spread, so score cells against traces captured in the SAME session.
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "host mem : $(free -g | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
echo "samples  : $SAMP"
echo "started  : $(date)"

gen() {
    local cell="$1" out="$KT_OUT/$1"
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] $cell present -> skip"; return 0; fi
    echo "[$(date +%H:%M:%S)] $cell"
    mkdir -p "$out"
    # Run under `srun --ntasks=1 --gres=gpu:1` so the step gets its own GPU cgroup.
    # NOTE: srun was previously credited with fixing the silent trace deaths. It was not --
    # those were the /e/home stale-handle fault above, and srun-vs-direct was confounded
    # with it. Keep srun (it is correct for a single-GPU step), but do not treat it as the
    # cure for a mute job log.
    APPTAINERENV_SIMULON_TRACE_KERNEL_TIME=1 \
    srun --ntasks=1 --gres=gpu:1 \
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$STAGE:/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=1 PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONDONTWRITEBYTECODE=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python -u -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cell.yaml' --gpu gh200_jupiter -o '$out'" 2>&1
    local rc=$?
    [ -f "$out/trace_rank_0.json" ] && echo "  ok (exit=$rc)" || echo "  !! NO TRACE (exit=$rc)"
    # If the cgroup is being killed, this line is the last thing written -- its absence is
    # itself the diagnosis, so flush it rather than lose it to buffering.
    sync
}

# The 11 cells the 1.7B scorecard scores. tp4pp1-mbs2's kernel-timing capture has failed
# repeatedly under torch.profiler (jobs 1150972, 1008192); it is last so a failure there
# does not cost the rest.
CELLS="tp1pp1-mbs1 tp1pp1-mbs2 tp1pp1-mbs4 tp1pp1-mbs1-rc \
       tp2pp1-mbs1 tp2pp1-mbs8 tp1pp2-mbs1 tp1pp4-mbs1 \
       tp2pp2-mbs2 tp4pp1-mbs1 tp4pp1-mbs2"

for cell in $CELLS; do gen "$cell"; done

echo "[$(date +%H:%M:%S)] done"
echo "=== peak resource use ==="
awk -F, 'NR>1 && $2!="" {if(min==""||$2<min)min=$2; if($3>mx)mx=$3} END{
    print "  host available RAM bottomed at " min " GB"; print "  GPU used peaked at " mx " MiB"}' "$SAMP"
echo "=== inventory + host-time sanity ==="
OUTBASE=$OUTBASE python3 - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ["OUTBASE"])
print(f"{'cell':<20}{'rank0 span ms':>14}{'kernel ms':>12}{'host ms':>10}{'host%':>8}{'host_ops':>11}")
for d in sorted(base.iterdir()):
    f = d / "trace_rank_0.json"
    if not f.exists():
        print(f"{d.name:<20}  MISSING"); continue
    t = json.load(open(f))
    span = t.get("total_span_ms") or 0.0
    k = t.get("total_kernel_device_ms") or 0.0
    h = t.get("total_host_ms") or 0.0
    ops = t.get("total_host_ops") or 0
    pct = 100 * h / span if span else 0
    flag = "  <-- host time not captured" if pct < 10 else ""
    print(f"{d.name:<20}{span:>14.0f}{k:>12.0f}{h:>10.0f}{pct:>7.1f}%{ops:>11}{flag}")
PY
echo
echo "NEXT (on the login node), copy the traces into the registry:"
echo "  rsync -a $OUTBASE/ templates/gpu/gh200_jupiter-1p7b-host/traces/"
echo "THEN: SIMULON_1P7B_REGISTRY=gh200_jupiter-1p7b-host PYTHONPATH=. python experiments/analyze_1p7b.py"
