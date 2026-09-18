#!/bin/bash
# THOMAS Qwen3-1.7B methodology validation — trace all 11 isolation cells (single GPU,
# fake-PG). Pairs with the physical grid (config experiments/thomas/qwen3_1p7b_val40.yaml,
# jobs 1021172-1021182). Small model => every cell traces fast and fits on one GPU, incl.
# the mbs=4/8 and TP=1 cells that OOM at 32B.
#
# Output -> templates/gpu/gh200_jupiter-1p7b/traces/<cell> (registry for the analysis).
# nemo_26.04.sif + fusions ON (fixed _TRACE_DEFAULTS), matched to the physical runs.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=trace_thomas_1p7b
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_thomas_1p7b-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/thomas_1p7b_workloads
OUT=templates/gpu/gh200_jupiter-1p7b/traces

echo "container: $SIF"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"

gen() {
    local cell="$1" out="$OUT/$1"
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] $cell present -> skip"; return 0; fi
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] $cell"
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$W/$cell.yaml' --gpu gh200_jupiter -o '$out'"
    echo "[$(date +%H:%M:%S)] $cell exit=$?"
    [ -f "$out/trace_rank_0.json" ] && echo "  OK  $(ls "$out" | tr '\n' ' ')" || echo "  !! NO TRACE -- continuing"
}

# NOTE: tp1pp1-mbs8 OOMs physically at tp1 (262k-vocab logits ~34GB) -> mbs8 validated at
# tp2pp1 instead (tp2pp1-mbs8). tp1pp2-mbs1-vpp2 DROPPED: Megatron forbids interleaved VPP
# at PP=2 without p2p overlap (protocol keeps overlap off); VPP already validated at 32B.
for cell in tp1pp1-mbs1 tp1pp1-mbs2 tp1pp1-mbs4 \
            tp2pp1-mbs1 tp2pp1-mbs8 tp4pp1-mbs1 tp1pp2-mbs1 tp1pp4-mbs1 \
            tp1pp1-mbs1-rc tp2pp2-mbs2; do
    gen "$cell"
done
echo "[$(date +%H:%M:%S)] done"
echo "=== inventory ==="
for d in "$OUT"/*/; do [ -f "$d/trace_rank_0.json" ] && echo "  $(basename "$d") OK" || echo "  $(basename "$d") MISSING"; done
