#!/bin/bash
# DIAGNOSTIC re-run of the tp4pp1-mbs2 trace. Two attempts (1149575, 1150654) both died at
# ~50 s with a BYTE-IDENTICAL 1697-byte log ending at "Tracing rank 0 ...", no traceback,
# and — the real clue — WITHOUT the gen() helper printing its own "!! NO TRACE" line. The
# batch shell is dying too, which a normal non-zero exit would not cause.
#
# Ruled out already:
#   GPU memory  — the physical tp4/mbs2 run peaks at 0.232 of capacity (mbs1: 0.182)
#   host RAM    — 878 GB requested, sacct MaxRSS ~0 (though sampling may simply have missed it)
#   disk/quota  — /e/home 10% used
#   walltime    — the working tp4pp1-mbs1 trace took ~3 min; this dies at 50 s
#   the script  — diffs clean against trace_1p7b_tpfix.sbatch, which traced tp4pp1-mbs1 fine
#
# So this job instruments rather than guesses:
#   1. runs the trace under `srun`, so slurm accounts the step separately and will report
#      OUT_OF_MEMORY explicitly instead of a bare FAILED
#   2. samples host RAM and GPU memory every 2 s into a side file, so a cgroup kill shows as
#      a visible ramp rather than silence
#   3. requests memory explicitly instead of inheriting the default
#   4. traps EXIT so the shell says something even if the child is killed
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=00:25:00
#SBATCH --job-name=tp4mbs2_diag
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_1p7b_tp4mbs2_diag-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/thomas_1p7b_workloads
OUT=templates/gpu/gh200_jupiter-1p7b-unified/traces/tp4pp1-mbs2
SAMP=experiments/trace_1p7b_tp4mbs2_diag-${SLURM_JOB_ID}.mem.csv

trap 'echo "[$(date +%H:%M:%S)] batch shell EXITING (rc=$?) — if this is the last line, the cgroup was killed"' EXIT

echo "container: $SIF"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "host mem : $(free -g | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
echo "sampling to $SAMP"

# Background sampler: host available RAM + GPU used MiB, every 2 s.
( echo "t,host_avail_GB,gpu_used_MiB"
  while true; do
      printf "%s,%s,%s\n" "$(date +%H:%M:%S)" \
          "$(free -g | awk '/^Mem:/{print $7}')" \
          "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)"
      sleep 2
  done ) > "$SAMP" 2>&1 &
SAMPLER=$!

mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] launching trace generate (span, kt=0) under srun"

srun --ntasks=1 --gres=gpu:1 \
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=0 PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONFAULTHANDLER=1 \
            python -u -X faulthandler -c 'from simulon.cli import app; app()' trace generate \
            '$W/tp4pp1-mbs2.yaml' --gpu gh200_jupiter -o '$OUT'" 2>&1
RC=$?

kill $SAMPLER 2>/dev/null
echo "[$(date +%H:%M:%S)] srun returned rc=$RC"
[ -f "$OUT/trace_rank_0.json" ] && echo "TRACE OK" || echo "NO TRACE"
echo "=== peak memory seen by the sampler ==="
awk -F, 'NR>1 && $2!="" {if(min==""||$2<min)min=$2; if($3>mx)mx=$3} END{
    print "  host available RAM bottomed at " min " GB"; print "  GPU used peaked at " mx " MiB"}' "$SAMP"
echo "=== last 10 samples ==="
tail -10 "$SAMP"
echo "NEXT: sacct -j $SLURM_JOB_ID --format=JobID%16,State,ExitCode,MaxRSS  # OUT_OF_MEMORY?"
