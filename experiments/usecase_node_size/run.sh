#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODELS=(deepseekv3 gptoss-120b llama3-70b)

for model in "${MODELS[@]}"; do
    echo "  TRACE: $model"
    bash scripts/apptainer-trace.sh "$SCRIPT_DIR/$model/workload.yaml" \
        --gpu h100
done

for model in "${MODELS[@]}"; do
    for size in 4 8; do
        scenario="$SCRIPT_DIR/$model/scenario${size}.yaml"
        name="${model}-node${size}"
        echo "  SIMULATE: $name"
        WAND_RUN_NAME="node-size-${name}" uv run simulon simulate "$scenario" --chrome-compact "--chrome output/trace-node-size-${name}" &
    done
done

wait $(jobs -p)

echo "=== Plotting ==="
python "$SCRIPT_DIR/plot.py"

