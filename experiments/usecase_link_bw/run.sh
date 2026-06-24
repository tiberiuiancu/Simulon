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
BWS=(10 100 200 400 800)
OVERLAP_MODELS=(deepseekv3)
OVERLAP_BWS=(10 100 200 400 800)

set +e
for model in "${MODELS[@]}"; do
    echo "  TRACE: $model"
    bash scripts/apptainer-trace.sh "$SCRIPT_DIR/../usecase_node_size/$model/workload.yaml" \
        --gpu h100
    trace_rc=$?
    if [ $trace_rc -ne 0 ]; then
        echo "    Warning: trace generation exited with code $trace_rc (continuing)"
    fi
done
set -e

TOTAL=0
for model in "${MODELS[@]}"; do
    for bw in "${BWS[@]}"; do
        TOTAL=$((TOTAL + 1))
    done
done
for model in "${OVERLAP_MODELS[@]}"; do
    for bw in "${OVERLAP_BWS[@]}"; do
        TOTAL=$((TOTAL + 1))
    done
done

echo "Starting $TOTAL simulations in parallel (logs: $SCRIPT_DIR/logs/) ..."

COMPLETED=0
for model in "${MODELS[@]}"; do
    for bw in "${BWS[@]}"; do
        scenario="$SCRIPT_DIR/$model/scenario${bw}.yaml"
        name="${model}-bw${bw}"
        echo "  SIMULATE: $name"
        log_file="$SCRIPT_DIR/logs/${name}.log"
        WANDB_RUN_NAME="link-bw-${name}" uv run simulon simulate "$scenario" --energy --chrome "output/link-nw-${name}.json" --chrome-compact > "$log_file" 2>&1 &
        PIDS+=($!)
    done
done
for model in "${OVERLAP_MODELS[@]}"; do
    for bw in "${OVERLAP_BWS[@]}"; do
        scenario="$SCRIPT_DIR/$model/scenario${bw}_overlap.yaml"
        name="${model}-bw${bw}-overlap"
        echo "  SIMULATE: $name"
        log_file="$SCRIPT_DIR/logs/${name}.log"
        WANDB_RUN_NAME="link-bw-${name}" uv run simulon simulate "$scenario" --energy --chrome "output/link-nw-${name}.json" --chrome-compact > "$log_file" 2>&1 &
        PIDS+=($!)
    done
done

while [ $COMPLETED -lt $TOTAL ]; do
    wait -n 2>/dev/null || true
    COMPLETED=$((COMPLETED + 1))
    echo "Progress: $COMPLETED/$TOTAL simulations completed"
done

echo "=== Plotting ==="
uv run python "$SCRIPT_DIR/plot.py" --output output/link_bw.png
