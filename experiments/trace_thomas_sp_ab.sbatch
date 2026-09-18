#!/bin/bash
# THOMAS SP=true vs SP=false A/B for Tiberiu's two validate_e2e Qwen3-32B configs (GBS=64).
# Goal: measure the SP=true optimism his Snellius traces carry. His validate_e2e sims came
# out -1.4% / -2.4% vs H100 physical WITH SP=true + fusions-off (two offsetting biases).
# Here we hold fusions OFF (= his state) and toggle ONLY sequence-parallel, tracing both on
# the SAME Jupiter GH200. The SP=true - SP=false SIM delta isolates how much the SP=true
# fake-PG trace speeds up the prediction; the 680W powercap cancels in the ratio, so the
# GH200!=H100 gap does not matter here. On real hardware SP is ~time-neutral (a memory
# optimisation), so a large SP=true speedup in sim IS the optimism.
#
# GBS=64 => these are SMALL, fast traces (unlike the GBS=1024 sweep). 4 single-GPU traces.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=trace_thomas_sp
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/trace_thomas_sp-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
W=experiments/validate_e2e/scenarios_thomas_sp
OUT=templates/gpu/h100/traces

echo "container: $SIF"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"

gen() {
    local cfg="$1" tag="$2"
    local wl="$W/${cfg}_${tag}/workload.yaml"
    local out="$OUT/thomas-sp-${cfg}-${tag}"
    if [ -f "$out/trace_rank_0.json" ]; then
        echo "[$(date +%H:%M:%S)] ${cfg}-${tag} present -> skip"; return 0; fi
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] ${cfg}-${tag}  <- $wl"
    mkdir -p "$out"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
            python -c 'from simulon.cli import app; app()' trace generate \
            '$wl' --gpu h100 -o '$out'"
    echo "[$(date +%H:%M:%S)] ${cfg}-${tag} exit=$?"
    [ -f "$out/trace_rank_0.json" ] && echo "  OK" || echo "  !! NO TRACE (OOM?) -- continuing"
}

for cfg in qwen3-32b-tp4-pp2-mbs2-vpp8 qwen3-32b-tp2-pp4-mbs1-vpp1; do
    gen "$cfg" sptrue
    gen "$cfg" spfalse
done
echo "[$(date +%H:%M:%S)] done"
