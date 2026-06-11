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

MODELS=(deepseekv3 gptoss-120b llama3-70b)

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
    for size in 4 8; do
        ((TOTAL++))
    done
done

echo "Starting $TOTAL simulations in parallel (logs: $SCRIPT_DIR/logs/) ..."

COMPLETED=0
for model in "${MODELS[@]}"; do
    for size in 4 8; do
        scenario="$SCRIPT_DIR/$model/scenario${size}.yaml"
        name="${model}-node${size}"
        echo "  SIMULATE: $name"
        log_file="$SCRIPT_DIR/logs/${name}.log"
        WANDB_RUN_NAME="node-size-${name}" uv run simulon simulate "$scenario" --chrome-compact --chrome "output/trace-node-size-${name}" > "$log_file" 2>&1 &
        PIDS+=($!)
    done
done

while [ $COMPLETED -lt $TOTAL ]; do
    wait -n 2>/dev/null || true
    ((COMPLETED++))
    echo "Progress: $COMPLETED/$TOTAL simulations completed"
done

echo "=== Plotting ==="
python "$SCRIPT_DIR/plot.py"
