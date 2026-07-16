#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --job-name=validate-bridge
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/experiments/validate_bridge/run.sh" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/experiments/validate_bridge"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$SCRIPT_DIR/logs"
PIDS=()

cleanup() {
    echo ""
    echo "Interrupted — killing background jobs..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

MODELS=(llama3-70b gptoss-120b qwen3-30b)
echo "Selected models: ${MODELS[*]}"

set +e
for model in "${MODELS[@]}"; do
    echo "  TRACE: $model"
    scenario="$SCRIPT_DIR/$model/scenario.yaml"
    bash scripts/apptainer-trace.sh "$scenario" \
        --gpu h100 --memory-snapshot oom.pickle
    trace_rc=$?
    if [ $trace_rc -ne 0 ]; then
        echo "    Warning: trace generation exited with code $trace_rc (continuing)"
    fi
done
set -e

TOTAL=0
for model in "${MODELS[@]}"; do
    if [ -f "$SCRIPT_DIR/$model/scenario.yaml" ]; then
        TOTAL=$((TOTAL + 1))
    fi
done

echo "Starting $TOTAL simulations in parallel (logs: $SCRIPT_DIR/logs/) ..."

COMPLETED=0
for model in "${MODELS[@]}"; do
    scenario="$SCRIPT_DIR/$model/scenario.yaml"
    if [ ! -f "$scenario" ]; then
        echo "  Warning: no scenario.yaml found for $model, skipping"
        continue
    fi
    echo "  SIMULATE: $model"
    log_file="$SCRIPT_DIR/logs/${model}.log"
    WANDB_RUN_NAME="validate-bridge-${model}" uv run simulon simulate "$scenario" --skip-if-tracked --energy --chrome-compact --chrome "output/validate-bridge-${model}" > "$log_file" 2>&1 &
    PIDS+=($!)
done

while [ $COMPLETED -lt $TOTAL ]; do
    wait -n 2>/dev/null || true
    COMPLETED=$((COMPLETED + 1))
    echo "Progress: $COMPLETED/$TOTAL simulations completed"
done

echo "=== Plotting ==="
uv run python "$SCRIPT_DIR/plot.py"
