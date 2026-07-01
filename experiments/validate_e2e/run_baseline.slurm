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

INPUT_PATH="${1:-}"
if [ -z "$INPUT_PATH" ]; then
    echo "Usage: sbatch $0 <relative/path/to/scenario.yaml or workload.yaml>"
    exit 1
fi


if [ -z "${APPTAINER_CONTAINER:-}" ] && [ -z "${SINGULARITY_CONTAINER:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        REPO_ROOT_HOST="$SLURM_SUBMIT_DIR"
    else
        REPO_ROOT_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
    cd "$REPO_ROOT_HOST"

    MASTER_PORT="${MASTER_PORT:-6000}"
    if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
        export MASTER_ADDR
    fi
    export MASTER_PORT

    if [[ "$(basename "$(dirname "$INPUT_PATH")")" == *"3nic"* ]]; then
        export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4"
    fi

    srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 \
        apptainer exec --nv \
        --env "CUDA_DEVICE_MAX_CONNECTIONS=1" \
        --env "MASTER_ADDR=$MASTER_ADDR" \
        --env "MASTER_PORT=$MASTER_PORT" \
        --env "NCCL_IB_HCA=${NCCL_IB_HCA:-}" \
        --env "WANDB_ENTITY=${WANDB_ENTITY:-}" \
        --env "WANDB_PROJECT=${WANDB_PROJECT:-}" \
        --bind "$REPO_ROOT_HOST/experiments:/opt/simulon/experiments" \
        --bind "$REPO_ROOT_HOST/examples:/opt/simulon/examples" \
        --bind "$REPO_ROOT_HOST/vendor:/opt/simulon/vendor" \
        --bind "$REPO_ROOT_HOST/simulon:/opt/simulon/simulon" \
        --bind "$REPO_ROOT_HOST/templates:/opt/simulon/templates" \
        --bind "$REPO_ROOT_HOST/output:/opt/simulon/output" \
        --bind "$REPO_ROOT_HOST/.tracking.env:/opt/simulon/.tracking.env" \
        "$REPO_ROOT_HOST/simulon-nemo.sif" \
        bash -c "cd /opt/simulon && python3 experiments/validate_e2e/run_baseline.py \"$INPUT_PATH\""

    exit $?
fi

cd /opt/simulon
python3 experiments/validate_e2e/run_baseline.py "$INPUT_PATH"
