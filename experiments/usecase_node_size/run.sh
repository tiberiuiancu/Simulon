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

for model in "${MODELS[@]}"; do
    echo "  TRACE: $model"
    bash scripts/apptainer-trace.sh "$SCRIPT_DIR/$model/workload.yaml" \
        --gpu h100
done

TOTAL=0
for model in "${MODELS[@]}"; do
    for size in 4 8; do
        ((TOTAL++))
    done
done

COMPLETED=0
for model in "${MODELS[@]}"; do
    for size in 4 8; do
        scenario="$SCRIPT_DIR/$model/scenario${size}.yaml"
        name="${model}-node${size}"
        log_file="$SCRIPT_DIR/logs/${name}.log"
        WAND_RUN_NAME="node-size-${name}" uv run simulon simulate "$scenario" --chrome-compact "--chrome output/trace-node-size-${name}" > "$log_file" 2>&1 &
        PIDS+=($!)
    done
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
    ((COMPLETED++))
    echo "Progress: $COMPLETED/$TOTAL simulations completed"
done

echo "=== Plotting ==="
python "$SCRIPT_DIR/plot.py"
