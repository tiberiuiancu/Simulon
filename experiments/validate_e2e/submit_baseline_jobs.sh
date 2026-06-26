#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXCLUDE=(qwen3-32b)

for dir in "$SCRIPT_DIR"/*/; do
    model="$(basename "$dir")"
    if printf '%s\n' "${EXCLUDE[@]}" | grep -qx "$model"; then
        echo "Skipping excluded model: $model"
        continue
    fi
    workload="$dir/workload.yaml"
    if [ ! -f "$workload" ]; then
        echo "Warning: no workload.yaml for $model, skipping"
        continue
    fi
    echo "Submitting baseline job for $model"
    bash -c "sbatch '$SCRIPT_DIR/run_baseline.slurm' '$workload'"
done
