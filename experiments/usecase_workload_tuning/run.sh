#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=1:45:00
#SBATCH --gpus=1
#SBATCH --job-name=qwen32b-workload-tuning
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [ -z "${APPTAINER_CONTAINER:-}" ] && [ -z "${SINGULARITY_CONTAINER:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        REPO_ROOT_HOST="$SLURM_SUBMIT_DIR"
    else
        SCRIPT_PATH_HOST="$(realpath "${BASH_SOURCE[0]}")"
        REPO_ROOT_HOST="$(cd "$(dirname "$SCRIPT_PATH_HOST")/../.." && pwd)"
    fi
    echo "[run.sh host] SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
    echo "[run.sh host] REPO_ROOT_HOST=$REPO_ROOT_HOST"
    echo "[run.sh host] Host Megatron commit: $(cd "$REPO_ROOT_HOST/vendor/Megatron-LM-traced" && git log --oneline -1 2>/dev/null || echo 'not a git repo')"
    SCRIPT_PATH_CONTAINER="$REPO_ROOT_HOST/experiments/usecase_workload_tuning/run.sh"
    SCRIPT_PATH_CONTAINER="/opt/simulon${SCRIPT_PATH_CONTAINER#$REPO_ROOT_HOST}"
    cd "$REPO_ROOT_HOST"
    apptainer run --nv \
        --bind "$REPO_ROOT_HOST/experiments:/opt/simulon/experiments" \
        --bind "$REPO_ROOT_HOST/examples:/opt/simulon/examples" \
        --bind "$REPO_ROOT_HOST/vendor:/opt/simulon/vendor" \
        --bind "$REPO_ROOT_HOST/simulon:/opt/simulon/simulon" \
        --bind "$REPO_ROOT_HOST/templates:/opt/simulon/templates" \
        --bind "$REPO_ROOT_HOST/output:/opt/simulon/output" \
        --bind "$REPO_ROOT_HOST/.tracking.env:/opt/simulon/.tracking.env" \
        "$REPO_ROOT_HOST/simulon-nemo.sif" \
        bash "$SCRIPT_PATH_CONTAINER" "$@"
    exit $?
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "[run.sh] Megatron submodule commit: $(cd "$REPO_ROOT/vendor/Megatron-LM-traced" && git log --oneline -1)"
echo "[run.sh] pretrain_gpt.py OOM handler present: $(grep -c 'except Exception as exc' "$REPO_ROOT/vendor/Megatron-LM-traced/pretrain_gpt.py")"
echo "[run.sh] /opt/simulon/vendor is symlink? $(test -L /opt/simulon/vendor && echo yes || echo no)"
echo "[run.sh] /opt/simulon/vendor real path: $(readlink -f /opt/simulon/vendor)"

python3 "$SCRIPT_DIR/grid_search.py" --clean-invalid-markers ${MAX_RUNS:+--max-runs "$MAX_RUNS"}
