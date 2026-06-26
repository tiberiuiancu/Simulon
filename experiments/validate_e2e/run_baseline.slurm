#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=1:00:00
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --switches=1
#SBATCH --job-name=validate-e2e-baseline
#SBATCH --output=baseline-%x-%j.out
#SBATCH --error=baseline-%x-%j.err

set -euo pipefail

WORKLOAD_PATH="${1:-}"
if [ -z "$WORKLOAD_PATH" ]; then
    echo "Usage: sbatch $0 <path/to/workload.yaml or scenario.yaml>"
    exit 1
fi

if [ -z "${APPTAINER_CONTAINER:-}" ] && [ -z "${SINGULARITY_CONTAINER:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        REPO_ROOT_HOST="$SLURM_SUBMIT_DIR"
    else
        REPO_ROOT_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
    cd "$REPO_ROOT_HOST"
    if [[ "$(basename "$WORKLOAD_PATH")" == *"3nic"* ]] || [[ "$(basename "$WORKLOAD_PATH")" == *"3NIC"* ]]; then
        export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4"
    fi
    apptainer exec --nv \
        --bind "$REPO_ROOT_HOST/experiments:/opt/simulon/experiments" \
        --bind "$REPO_ROOT_HOST/examples:/opt/simulon/examples" \
        --bind "$REPO_ROOT_HOST/vendor:/opt/simulon/vendor" \
        --bind "$REPO_ROOT_HOST/simulon:/opt/simulon/simulon" \
        --bind "$REPO_ROOT_HOST/templates:/opt/simulon/templates" \
        --bind "$REPO_ROOT_HOST/output:/opt/simulon/output" \
        --bind "$REPO_ROOT_HOST/.tracking.env:/opt/simulon/.tracking.env" \
        "$REPO_ROOT_HOST/simulon-nemo.sif" \
        bash -c "cd /opt/simulon && python3 experiments/validate_e2e/run_baseline.py \"$WORKLOAD_PATH\""
    exit $?
fi

cd /opt/simulon
python3 experiments/validate_e2e/run_baseline.py "$WORKLOAD_PATH"
