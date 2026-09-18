#!/bin/bash
# Stage the repo onto /e/scratch so cluster jobs never touch /e/home.
#
# WHY. Jobs 1222963, 1223085 and 1224406 all died ~30 s in with the SAME cause, which only
# became visible once simulon/cli/trace.py stopped swallowing Megatron's output:
#
#   OSError: [Errno 116] Stale file handle:
#     '/opt/simulon/vendor/Megatron-LM-traced/megatron/core/inference/quantization'
#   slurmstepd: couldn't chdir to `/e/home/jusers/vanosch1/jupiter/Simulon': Stale file handle
#   FATAL: container creation failed ... stale file handle
#
# The compute node's handle on the /e/home (GPFS `exa_home`) export goes stale mid-job. The
# ~30 s time-to-death is just how long the megatron/torch import chain takes before it walks
# a directory that has gone stale. It is NOT the profiler, NOT memory, NOT the tracer change,
# and it says nothing about kt=0 vs kt=1 -- the kt=1 variant never even started (exit 255).
#
# It correlates with editing the repo from the login node between submissions, which is
# exactly the workflow here, so the robust answer is to decouple: copy the tree to
# /e/scratch (a different filesystem, which the nsys jobs read reliably) and run from there.
# ~361 MB excluding .git and existing traces, so a few seconds.
#
# RE-RUN THIS AFTER EVERY CODE CHANGE you want the cluster to pick up -- the stage is a
# snapshot, not a live view. That is the trade for not being at the mercy of the export.
#
#     bash experiments/stage_repo.sh
set -euo pipefail

SRC=/e/project1/e-sta-openeurollm/vanosch1/Simulon
STAGE=/e/scratch/e-sta-openeurollm/vanosch1/simulon_stage

mkdir -p "$STAGE"
rsync -a --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'templates/gpu/*/traces' \
    --exclude '*.nsys-rep' --exclude '*.sqlite' \
    "$SRC"/ "$STAGE"/

# The GPU/node templates ARE needed; only the bulky trace payloads were excluded above.
echo "staged -> $STAGE"
echo "  $(du -sh "$STAGE" | cut -f1),  $(find "$STAGE" -name '*.py' | wc -l) python files"
echo "  tracer host-time fields present: $(grep -c 'host_ms' \
    "$STAGE/vendor/Megatron-LM-traced/megatron/core/instrumentation/tracer.py")"
