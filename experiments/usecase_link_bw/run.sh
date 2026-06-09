#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODELS=(deepseekv3 gptoss-120b llama3-70b)
BWS=(10 100 200 400 800)

for model in "${MODELS[@]}"; do
    echo "  TRACE: $model"
    bash scripts/apptainer-trace.sh "$SCRIPT_DIR/../usecase_node_size/$model/workload.yaml" \
        --gpu h100
done

for model in "${MODELS[@]}"; do
    for bw in "${BWS[@]}"; do
        scenario="$SCRIPT_DIR/$model/scenario${bw}.yaml"
        name="${model}-bw${bw}"
        echo "  SIMULATE: $name"
        uv run simulon simulate "$scenario" &
    done
done

wait $(jobs -p)

echo "=== Plotting ==="
python "$SCRIPT_DIR/plot.py"
