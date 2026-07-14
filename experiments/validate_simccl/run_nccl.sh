#!/bin/bash
# Run nccl-tests benchmarks for AllReduce, AllGather, ReduceScatter, AllToAll
# for a single cluster config determined by the number of nodes at submit time.
#
# Submit with the exact node count you need:
#   sbatch --nodes=1 experiments/validate_simccl/run_nccl.sh   # 1n4g
#   sbatch --nodes=2 experiments/validate_simccl/run_nccl.sh   # 2n4g
#   sbatch --nodes=4 experiments/validate_simccl/run_nccl.sh   # 4n4g
#
# Output JSON: results/nccl_<collective>_<Nn>4g.json
#
# ── SLURM headers (--nodes is overridden on the command line above) ────────
#SBATCH --job-name=nccl-validation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --switches=1
#SBATCH --exclusive
#SBATCH --partition=gpu_h100
#SBATCH --output=nccl_slurm_%j.log
#SBATCH --error=nccl_slurm_%j.err
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"
SCRIPT_DIR="$REPO_ROOT/experiments/validate_simccl"
NCCL_TESTS_DIR="${SCRIPT_DIR}/nccl-tests"
RESULT_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULT_DIR}"

GPUS_PER_NODE=4
NUM_NODES="${SLURM_NNODES}"
NUM_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))
CONFIG_LABEL="${NUM_NODES}n${GPUS_PER_NODE}g"

echo "Running nccl-tests: ${CONFIG_LABEL} (${NUM_NODES} nodes × ${GPUS_PER_NODE} GPUs = ${NUM_GPUS} GPUs total)"

# ── Modules ────────────────────────────────────────────────────────────────
MPI_MODULE=OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
NCCL_MODULE=NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0
MODULE_HOME=/sw/arch/RHEL9/EB_production/2025/software
export NCCL_HOME=$MODULE_HOME/$NCCL_MODULE
export MPI_HOME=$MODULE_HOME/$MPI_MODULE
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 $NCCL_MODULE $MPI_MODULE

# ── Build nccl-tests (MPI-enabled, once) ───────────────────────────────────
if [[ ! -f "${NCCL_TESTS_DIR}/build/all_reduce_perf_mpi" ]]; then
    echo "Building nccl-tests with MPI under ${NCCL_TESTS_DIR}"
    make -C "${NCCL_TESTS_DIR}" -j MPI=1 NAME_SUFFIX=_mpi \
        CUDA_HOME="${CUDA_HOME}" \
        NCCL_HOME="${NCCL_HOME}" \
        MPI_HOME="${MPI_HOME}"
fi

export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_ALGO=Ring

# ── Benchmark parameters ───────────────────────────────────────────────────
MIN_BYTES="8M"
MAX_BYTES="8192M"
STEP_FACTOR=2
ITERS=20
WARMUP=5

declare -A BINARY=(
    [AllReduce]=all_reduce_perf_mpi
    [AllGather]=all_gather_perf_mpi
    [ReduceScatter]=reduce_scatter_perf_mpi
    [AllToAll]=alltoall_perf_mpi
)

# ── Run all collectives for this config ────────────────────────────────────
# Run each collective 3 times with -a 1 (avg), -a 2 (min time = max bw),
# -a 3 (max time = min bw) to produce error bands.
for COLLECTIVE in AllReduce AllGather ReduceScatter AllToAll; do
    BIN="${NCCL_TESTS_DIR}/build/${BINARY[$COLLECTIVE]}"
    CNAME_LOWER=$(echo "$COLLECTIVE" | tr '[:upper:]' '[:lower:]')
    BASE="${RESULT_DIR}/nccl_${CNAME_LOWER}_${CONFIG_LABEL}_snellius"

    echo "=== ${COLLECTIVE} ${CONFIG_LABEL} ==="
    for AVG_MODE in 1 2 3; do
        case $AVG_MODE in
            1) SUFFIX="" ;;
            2) SUFFIX="_maxbw" ;;
            3) SUFFIX="_minbw" ;;
        esac
        OUT="${BASE}${SUFFIX}.json"
        srun --mpi=pmix --nodes="${NUM_NODES}" --ntasks="${NUM_GPUS}" --ntasks-per-node="${GPUS_PER_NODE}" --gpus-per-node="${GPUS_PER_NODE}" \
            "${BIN}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${STEP_FACTOR}" \
                     -n "${ITERS}" -w "${WARMUP}" -g 1 -a "${AVG_MODE}" \
                     -J "${OUT}"
    done
    echo "    -> ${BASE}{{,_maxbw,_minbw}}.json"
done

echo "Done. Results in ${RESULT_DIR}/"
