#!/bin/bash
# Run nccl-tests benchmarks for AllReduce, AllGather, ReduceScatter
# across three cluster configs: 1×4 GPUs, 2×4 GPUs, 4×4 GPUs.
#
# Output JSON files land in experiments/validation/simccl/results/ and are
# named nccl_<collective>_<Nn><G>g.json to match sim_ccl.py output names.
#
#SBATCH --job-name=nccl-validation
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_h100
#SBATCH --output=experiments/validation/simccl/results/nccl_slurm_%j.log
#SBATCH --error=experiments/validation/simccl/results/nccl_slurm_%j.err
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Modules ────────────────────────────────────────────────────────────────
module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_TESTS_DIR="${SCRIPT_DIR}/nccl-tests"
RESULT_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULT_DIR}"

# ── Build nccl-tests (MPI-enabled) ─────────────────────────────────────────
if [[ ! -f "${NCCL_TESTS_DIR}/build/all_reduce_perf_mpi" ]]; then
    echo "Building nccl-tests with MPI ..."
    make -C "${NCCL_TESTS_DIR}" -j MPI=1 NAME_SUFFIX=_mpi \
        CUDA_HOME="${CUDA_HOME}" \
        NCCL_HOME="${NCCL_HOME}" \
        MPI_HOME="${MPI_HOME}"
fi

# ── Benchmark parameters ───────────────────────────────────────────────────
MIN_BYTES="8M"
MAX_BYTES="8192M"
STEP_FACTOR=2
ITERS=20
WARMUP=5

# collective_name → binary base name
declare -A BINARY=(
    [AllReduce]=all_reduce_perf_mpi
    [AllGather]=all_gather_perf_mpi
    [ReduceScatter]=reduce_scatter_perf_mpi
    [AllToAll]=alltoall_perf_mpi
)

# ── Run sweeps ─────────────────────────────────────────────────────────────
for COLLECTIVE in AllReduce AllGather ReduceScatter AllToAll; do
    BIN="${NCCL_TESTS_DIR}/build/${BINARY[$COLLECTIVE]}"
    CNAME_LOWER=$(echo "$COLLECTIVE" | tr '[:upper:]' '[:lower:]')
    COMMON_ARGS="-b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} -n ${ITERS} -w ${WARMUP} -g 1"

    # -- 1 node, 4 GPUs (intra-node NVSwitch only) --------------------------
    echo "=== ${COLLECTIVE} 1n4g ==="
    srun --nodes=1 --ntasks=4 --ntasks-per-node=4 \
        "${BIN}" ${COMMON_ARGS} \
        -J "${RESULT_DIR}/nccl_${CNAME_LOWER}_1n4g.json"

    # -- 2 nodes, 8 GPUs ----------------------------------------------------
    echo "=== ${COLLECTIVE} 2n4g ==="
    srun --nodes=2 --ntasks=8 --ntasks-per-node=4 \
        "${BIN}" ${COMMON_ARGS} \
        -J "${RESULT_DIR}/nccl_${CNAME_LOWER}_2n4g.json"

    # -- 4 nodes, 16 GPUs ---------------------------------------------------
    echo "=== ${COLLECTIVE} 4n4g ==="
    srun --nodes=4 --ntasks=16 --ntasks-per-node=4 \
        "${BIN}" ${COMMON_ARGS} \
        -J "${RESULT_DIR}/nccl_${CNAME_LOWER}_4n4g.json"
done

echo "Done. Results in ${RESULT_DIR}/"
