#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

MODELS=()
for dir in "$SCRIPT_DIR"/*/; do
    if [ -d "$dir" ]; then
        model="$(basename "$dir")"
        MODELS+=("$model")
    fi
done

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No model sub-folders found in $SCRIPT_DIR"
    exit 1
fi

echo "Detected models: ${MODELS[*]}"

set +e
for model in "${MODELS[@]}"; do
    echo "  TRACE: $model"
    bash scripts/apptainer-trace.sh "$SCRIPT_DIR/$model/workload.yaml" \
        --gpu h100
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
    WANDB_RUN_NAME="validate-e2e-${model}" uv run simulon simulate "$scenario" --energy --chrome-compact --chrome "output/validate-e2e-${model}" > "$log_file" 2>&1 &
    PIDS+=($!)
done

while [ $COMPLETED -lt $TOTAL ]; do
    wait -n 2>/dev/null || true
    COMPLETED=$((COMPLETED + 1))
    echo "Progress: $COMPLETED/$TOTAL simulations completed"
done

echo "=== Plotting ==="
uv run python "$SCRIPT_DIR/plot.py"
