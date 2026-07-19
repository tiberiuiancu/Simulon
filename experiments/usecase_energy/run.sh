#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$SCRIPT_DIR/logs"

MODELS=(deepseekv3 gptoss-120b llama3-70b)
NODE_SIZE=8

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

TOTAL=${#MODELS[@]}
echo "Starting $TOTAL energy simulations (logs: $SCRIPT_DIR/logs/) ..."

COMPLETED=0
for model in "${MODELS[@]}"; do
    scenario="$SCRIPT_DIR/../usecase_node_size/$model/scenario${NODE_SIZE}.yaml"
    name="${model}-node${NODE_SIZE}"
    echo "  SIMULATE: $name"
    log_file="$SCRIPT_DIR/logs/${name}.log"
    WANDB_RUN_NAME="node-size-${name}" uv run simulon simulate "$scenario" --skip-if-tracked --energy --chrome-compact --chrome "output/trace-node-size-${name}" > "$log_file" 2>&1 &
    PIDS+=($!)
done

while [ $COMPLETED -lt $TOTAL ]; do
    wait -n 2>/dev/null || true
    COMPLETED=$((COMPLETED + 1))
    echo "Progress: $COMPLETED/$TOTAL simulations completed"
done

echo "=== Plotting ==="
uv run python "$SCRIPT_DIR/plot.py"
