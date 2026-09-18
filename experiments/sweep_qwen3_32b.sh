#!/bin/bash
# Simulon sweep — Qwen3-32B parallelism configs on Jupiter GH200
#
# Run locally (from the repo root):
#   bash experiments/sweep_qwen3_32b.sh
#   bash experiments/sweep_qwen3_32b.sh --tp 4 --pp 1 2 --nodes 16 32
#   bash experiments/sweep_qwen3_32b.sh --out results/my_sweep.json
#
# Submit to SLURM (from the repo root):
#   sbatch experiments/sweep_qwen3_32b.sh
#   sbatch experiments/sweep_qwen3_32b.sh --tp 4 --pp 1 2 --nodes 16 32
#
# ── SLURM headers ─────────────────────────────────────────────────────────────
#SBATCH --job-name=simulon-sweep-qwen3-32b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=booster
#SBATCH --output=experiments/sweep_qwen3_32b_%j.log
#SBATCH --error=experiments/sweep_qwen3_32b_%j.err
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Repo root — SLURM_SUBMIT_DIR is set by sbatch; fall back for local runs.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

# Default output file with timestamp so repeated runs don't overwrite each other.
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUT="results/sweep_qwen3_32b_${TIMESTAMP}.json"
mkdir -p results

echo "── Simulon sweep: Qwen3-32B on Jupiter ──────────────────────────────────────"
echo "Repo:   ${REPO_ROOT}"
echo "Output: ${DEFAULT_OUT} (override with --out <path>)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM:  job ${SLURM_JOB_ID} on $(hostname)"
fi
echo ""

# Pass all script arguments through; if --out is not supplied, inject the default.
ARGS=("$@")
HAS_OUT=false
for arg in "${ARGS[@]}"; do
    [[ "${arg}" == "--out" ]] && HAS_OUT=true
done
if [[ "${HAS_OUT}" == "false" ]]; then
    ARGS+=("--out" "${DEFAULT_OUT}")
fi

source /e/scratch/e-sta-openeurollm/vanosch1/simulon/bin/activate
python experiments/sweep_qwen3_32b.py "${ARGS[@]}"

echo ""
echo "Done. Results written to ${DEFAULT_OUT}"
