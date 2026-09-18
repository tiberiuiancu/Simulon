#!/bin/bash
# R7 -- the HELD-OUT test. Re-capture the 32B/16-node traces with the current tracer, so the
# host model can be scored on data that was never used to build it.
#
# WHY THIS IS THE ONLY HONEST TEST LEFT. Everything the host model rests on came from the
# 1.7B single-node grid: host_cost_us = 19.0 was derived from 1.7B host-bound cells, and the
# PP / recompute diagnoses came from 1.7B nsys profiles. Scoring against that same grid
# cannot separate "the physics is right" from "it was fitted to this data". The 32B set was
# never touched during any of it and differs on TWO axes at once -- 18x the model size and
# 16 nodes instead of 1, so inter-node NCCL and real DP scale-out are both exercised.
#
# WHY IT NEEDS A RE-CAPTURE AT ALL. The physical val40 measurements are intact (8 of 9 cells,
# 40 iterations each, under /e/project1/.../oellm-autoexp/output/qwen3-32b). The existing
# traces are not usable: gh200_jupiter-prod-kerneltime carries kernel_device_ms but
# host_ops=None -- it predates the op-counting tracer -- so it can only score the OLD span
# model, not the two-resource host model. No 16-node run is needed: these are fake-PG
# single-GPU captures (the original, job 1012359, took 29 min on 1 GPU).
#
# WHAT IS CLEAN AND WHAT IS NOT. Fake PG is measured unreliable at PP>=2 (last pipeline stage
# 23% low at 1.7B). Of the cells with physical numbers, exactly ONE is PP=1:
#     tp4pp1-mbs1   physical 8954 ms   <-- THE CLEAN HELD-OUT POINT
# The other seven are PP>=2 and their verdicts confound "does the host model generalize?"
# with "how wrong is fake PG at PP>=2?". They are captured anyway and MUST be reported
# separately, labelled as fake-PG-biased. tp4pp1-mbs2 and tp2pp4-mbs2 are omitted: they are
# genuinely infeasible on one fake-PG GPU (OOM, exhausted in job 1009860) and tp4pp1-mbs2 has
# no physical run either.
#
# PRE-REGISTERED (before looking). Judged on tp4pp1-mbs1 only.
#   * |delta| <= 10%  -> the host model GENERALIZES across an 18x model-size jump and 1->16
#     nodes. That is the strongest evidence available that host_cost_us is a machine constant
#     rather than a 1.7B fit.
#   * over-predicts by >10% -> host_cost_us does not transfer; most likely the op MIX differs
#     at 32B (the same failure mode already established for recompute), so the per-op average
#     is not portable across model shapes.
#   * under-predicts by >10% -> compute or inter-node comm is under-modelled at 16 nodes;
#     suspect the DP scale-out path, which has never been exercised by the 1.7B grid at all.
#   * NOTE the noise floor: every 32B cell is n=1, and the 1.7B repeat runs show a 5.0%
#     run-to-run spread at TP>=2. So anything inside +-5% is indistinguishable from scatter
#     and must NOT be reported as a precise agreement.
#
# COST: 1 node, 1 GPU, <= 1.5 h. No 32B training run.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=01:30:00
#SBATCH --job-name=trace_32b_hostops
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --chdir=/e/scratch/e-sta-openeurollm/vanosch1
#SBATCH --output=/e/scratch/e-sta-openeurollm/vanosch1/logs/trace_32b_hostops-%j.log

set -uo pipefail
# Run off /e/scratch: the /e/home GPFS export goes stale mid-job on booster nodes.
STAGE=/e/scratch/e-sta-openeurollm/vanosch1/simulon_stage
OUT=/e/scratch/e-sta-openeurollm/vanosch1/traces_32b_hostops
[ -f "$STAGE/simulon/cli/__init__.py" ] || { echo "FATAL: run stage_repo.sh first" >&2; exit 1; }
cd "$STAGE" || exit 1
mkdir -p "$OUT"

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif   # production container
W=/opt/simulon/experiments/sweep_qwen3_32b_workloads/unified

echo "container: $SIF"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "started  : $(date)"

# Affinity: read the real mask, NOT `nproc` (which honours OMP_NUM_THREADS and reported 8
# on a 288-core step, falsely aborting jobs 1303639 and 1445115).
CORES=$(apptainer exec --bind /e/scratch/ "$SIF" \
        python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null | tail -1)
echo "affinity : ${CORES:-unknown} cores (expect 288)"

gen() {
    local cfg="$1" out="$OUT/$1"
    echo "=============================================================="
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] $cfg present -> skip"; return 0; fi
    echo "[$(date +%H:%M:%S)] $cfg"
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$STAGE:/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=1 PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONDONTWRITEBYTECODE=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cfg.yaml' --gpu gh200_jupiter -o '$out'" 2>&1 | tail -12
    if [ -f "$out/trace_rank_0.json" ]; then
        python3 -c "
import json; d=json.load(open('$out/trace_rank_0.json'))
c=d.get('capture_config') or {}
print('  ok  kernel_ms=%s  host_ops=%s  capture_config=%d keys'%(
    round(d.get('total_kernel_device_ms') or 0), d.get('total_host_ops'), len(c)))
print('      HOST_OPS MISSING -- this trace cannot drive the host model') if not d.get('total_host_ops') else None"
    else
        echo "  !! NO TRACE (likely OOM under the profiler) -- continuing"
    fi
}

# THE clean held-out point first, so a walltime overrun cannot cost it.
gen tp4pp1-mbs1
# PP>=2: captured for context, fake-PG-biased, report separately.
gen tp4pp2-mbs1
gen tp4pp2-mbs2
gen tp4pp4-mbs1
gen tp4pp4-mbs2
gen tp2pp4-mbs1
gen tp4pp2-mbs1-rc
gen tp4pp4-mbs1-rc

echo "=============================================================="
echo "=== inventory (host_ops is the thing that matters) ==="
OUT=$OUT python3 - <<'PYEOF'
import json, os, pathlib
base = pathlib.Path(os.environ['OUT'])
print(f"{'cell':<18}{'kernel_ms':>10}{'host_ops':>11}{'ranks':>7}  clean?")
for d in sorted(p for p in base.iterdir() if p.is_dir()):
    fs = sorted(d.glob('trace_rank_*.json'))
    if not fs:
        print(f"{d.name:<18}{'NO TRACE':>10}"); continue
    t = json.load(open(fs[0]))
    pp = int(d.name.split('pp')[1][0])
    print(f"{d.name:<18}{round(t.get('total_kernel_device_ms') or 0):>10}"
          f"{str(t.get('total_host_ops')):>11}{len(fs):>7}  "
          f"{'YES (PP=1, held-out point)' if pp == 1 else 'no (fake-PG biased at PP>=2)'}")
PYEOF

echo
echo "NEXT (login node):"
echo "  rsync -a $OUT/ templates/gpu/gh200_jupiter-prod-hostops/traces/"
echo "  # then score tp4pp1-mbs1 against its physical 8954 ms at host_cost_us=19.0"
echo "finished : $(date)"
