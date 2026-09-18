#!/bin/bash
# Generate the TP2/PP2 SP=false trace — the only remaining gap in the SP=false grid.
#
# Its measured counterpart already exists (qwen3_32b_16_pp2_tp2_mbs1_spFalse_gbs256,
# 9,745 ms), so this self-validates the moment it lands.
#
# NOTE: TP2 low-PP trace generation has OOMed before under the fake process group
# (which cannot shard the optimizer state, so one GPU holds the whole thing). If this
# fails, the fallback is pp_synth derivation from tp2pp4 (--derive-low-pp), which is
# flagged unvalidated. Check for .OOM / .error.log markers in the output dir.
#
# Container: simulon.sif, matching how the other four SP=false traces were generated.
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=trace_tp2pp2_spfalse
#SBATCH --output=experiments/trace_tp2pp2_spfalse-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/vanosch1/containers/simulon.sif
WL=experiments/sweep_qwen3_32b_workloads/trace_tp2pp2spfalse_gbs256.yaml
OUT=templates/gpu/gh200_jupiter-tp2pp2spfalse/traces/gbs256

echo "[$(date +%H:%M:%S)] tracing TP2/PP2 SP=false -> $OUT"
mkdir -p "$OUT"
apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
    --bind "$(pwd):/opt/simulon" "$SIF" \
    bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon CUDA_DEVICE_MAX_CONNECTIONS=1 \
        python -c 'from simulon.cli import app; app()' trace generate \
        '$WL' --gpu gh200_jupiter -o '$OUT'"
echo "[$(date +%H:%M:%S)] exit=$?"
ls -la "$OUT"
[ -f "$OUT/.OOM" ] && echo "!! OOM MARKER PRESENT — fall back to pp_synth (--derive-low-pp)"
echo "[$(date +%H:%M:%S)] done"
