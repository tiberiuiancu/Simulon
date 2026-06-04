#!/bin/bash
# Run nccl-tests benchmarks for AllReduce, AllGather, ReduceScatter, AllToAll
# for a single cluster config determined by the number of nodes at submit time.
#
# Submit with the exact node count you need:
#   sbatch --nodes=1 experiments/validate_simccl/run_nccl_jupiter.sh   # 1n4g
#   sbatch --nodes=2 experiments/validate_simccl/run_nccl_jupiter.sh   # 2n4g
#   sbatch --nodes=4 experiments/validate_simccl/run_nccl_jupiter.sh   # 4n4g
#
# Output JSON: results/nccl_<collective>_<Nn>4g_jupiter.json
#
# ── SLURM headers (--nodes is overridden on the command line above) ────────
#SBATCH --job-name=nccl-validation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --partition=booster
#SBATCH --output=experiments/validate_simccl/results/nccl_slurm_%j.log
#SBATCH --error=experiments/validate_simccl/results/nccl_slurm_%j.err
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
# SLURM copies the script to its spool dir, so BASH_SOURCE[0] won't point to
# the repo. Use SLURM_SUBMIT_DIR (set by sbatch) with a fallback for local runs.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPT_DIR="${REPO_ROOT}/experiments/validate_simccl"
NCCL_TESTS_DIR="${SCRIPT_DIR}/nccl-tests"
RESULT_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULT_DIR}"

GPUS_PER_NODE=4
NUM_NODES="${SLURM_NNODES}"
NUM_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))
CONFIG_LABEL="${NUM_NODES}n${GPUS_PER_NODE}g"

echo "Running nccl-tests: ${CONFIG_LABEL} (${NUM_NODES} nodes × ${GPUS_PER_NODE} GPUs = ${NUM_GPUS} GPUs total)"

# ── Modules ────────────────────────────────────────────────────────────────
ml Stages/2026 nvidia-compilers/25.9-CUDA-13 OpenMPI/5.0.8

# ── Build nccl-tests (MPI-enabled, once) ───────────────────────────────────
if [[ ! -f "${NCCL_TESTS_DIR}/build/all_reduce_perf" ]]; then
    echo "Building nccl-tests with MPI under ${NCCL_TESTS_DIR}"
    CUDA_HOME_DETECTED="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}"
    MPI_HOME_DETECTED="${MPI_HOME:-$(dirname "$(dirname "$(which mpirun)")")}"
    make -C "${NCCL_TESTS_DIR}" -j MPI=1 \
        CUDA_HOME="${CUDA_HOME_DETECTED}" \
        MPI_HOME="${MPI_HOME_DETECTED}"
fi

# ── Benchmark parameters ───────────────────────────────────────────────────
MIN_BYTES="8M"
MAX_BYTES="8192M"
STEP_FACTOR=2
ITERS=20
WARMUP=5

declare -A BINARY=(
    [AllReduce]=all_reduce_perf
    [AllGather]=all_gather_perf
    [ReduceScatter]=reduce_scatter_perf
    [AllToAll]=alltoall_perf
)

# ── Run all collectives for this config ────────────────────────────────────
for COLLECTIVE in AllReduce AllGather ReduceScatter AllToAll; do
    BIN="${NCCL_TESTS_DIR}/build/${BINARY[$COLLECTIVE]}"
    CNAME_LOWER=$(echo "$COLLECTIVE" | tr '[:upper:]' '[:lower:]')
    OUT="${RESULT_DIR}/nccl_${CNAME_LOWER}_${CONFIG_LABEL}_jupiter.json"

    echo "=== ${COLLECTIVE} ${CONFIG_LABEL} ==="
    srun --nodes="${NUM_NODES}" --ntasks="${NUM_GPUS}" --ntasks-per-node="${GPUS_PER_NODE}" --gpus-per-node="${GPUS_PER_NODE}" \
        "${BIN}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
                 -n "${ITERS}" -w "${WARMUP}" -g 1 \
                 -J "${OUT}"
    echo "    -> ${OUT}"
done

echo "Done. Results in ${RESULT_DIR}/"