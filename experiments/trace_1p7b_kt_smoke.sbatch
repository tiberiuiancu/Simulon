#!/bin/bash
# SMOKE TEST for the rewritten kernel-time attribution — run this BEFORE the full capture.
#
# WHY. _kernel_ms_by_slot (tracer.py) was rewritten to attribute kernels by timestamp from
# the kineto chrome trace. It has never been executed. Its failure mode is silent-ish: a
# category-name mismatch across PyTorch/kineto versions yields ZERO attributed kernels, and
# the traces still "succeed". Discovering that after a 1-hour 20-trace capture wastes the
# hour. One cell costs ~4 minutes and answers the only question that matters:
#
#     does bwd now carry non-zero kernel time?
#
# PASS  -> run trace_1p7b_unified.sbatch for the full set.
# FAIL  -> the chrome-trace schema differs; fix the category matching first (the job log
#          will carry a "SIMULON WARNING: kernel-time attribution ..." line).
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=trace_1p7b_kt_smoke
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_1p7b_kt_smoke-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
CELL=tp1pp1-mbs1                       # smallest cell: fastest, and it is the anchor
OUT=templates/gpu/gh200_jupiter-1p7b-ktsmoke/traces/$CELL

rm -rf "$OUT"; mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] smoke-tracing $CELL with the rewritten attribution"

APPTAINERENV_SIMULON_TRACE_KERNEL_TIME=1 \
apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
    --bind "$(pwd):/opt/simulon" "$SIF" \
    bash -c "cd /opt/simulon && SIMULON_TRACE_KERNEL_TIME=1 PYTHONPATH=/opt/simulon \
        CUDA_DEVICE_MAX_CONNECTIONS=1 \
        python -c 'from simulon.cli import app; app()' trace generate \
        'experiments/thomas_1p7b_workloads/$CELL.yaml' --gpu gh200_jupiter -o '$OUT'"
echo "[$(date +%H:%M:%S)] exit=$?"

python3 - "$OUT/trace_rank_0.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"VERDICT: FAIL — no readable trace ({e})"); sys.exit(1)
by = {}
for e in d.get("events", []):
    m = e.get("metadata") or {}
    if e.get("type") == "slot_begin" and "kernel_device_ms" in m:
        by[m.get("direction")] = by.get(m.get("direction"), 0.0) + m["kernel_device_ms"]
fwd, bwd = by.get("fwd", 0.0), by.get("bwd", 0.0)
tot, un = d.get("total_kernel_device_ms"), d.get("unattributed_kernel_device_ms")
print(f"  fwd={fwd:.0f}ms  bwd={bwd:.0f}ms  total={tot}  unattributed={un}")
PHYS_MS = 9921.0  # measured tp1pp1-mbs1 iteration (val40, iters 21-40)
if not by:
    print("VERDICT: FAIL — no kernel_device_ms at all (category mismatch; see WARNING above)")
elif bwd <= 0:
    print("VERDICT: FAIL — bwd still ZERO. Timestamp attribution did not capture backward;")
    print("         check 'unattributed' above: if large, the slot annotations do not bound")
    print("         the backward kernels and per-slot attribution needs a different anchor.")
elif tot and tot > PHYS_MS:
    # Pure device time cannot exceed the wall time of the iteration it ran in. Hitting this
    # means categories are being double-counted (e.g. a GPU-side annotation range summed
    # alongside the kernels it spans), which silently inflates every kernel total.
    print(f"VERDICT: FAIL — kernel total {tot:.0f}ms EXCEEDS the {PHYS_MS:.0f}ms physical")
    print("         iteration: impossible on one GPU => double-counted categories.")
else:
    r = bwd / fwd if fwd else 0
    print(f"VERDICT: PASS — bwd carries {bwd:.0f}ms (bwd/fwd={r:.2f}), total {tot:.0f}ms "
          f"< {PHYS_MS:.0f}ms physical.")
    print("         Proceed to: sbatch experiments/trace_1p7b_unified.sbatch")
PY
