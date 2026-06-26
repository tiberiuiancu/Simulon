#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EXCLUDE=(qwen3-32b)

for dir in "$SCRIPT_DIR"/*/; do
    model="$(basename "$dir")"
    if printf '%s\n' "${EXCLUDE[@]}" | grep -qx "$model"; then
        echo "Skipping excluded model: $model"
        continue
    fi
    scenario="$dir/scenario.yaml"
    if [ ! -f "$scenario" ]; then
        echo "Warning: no scenario.yaml for $model, skipping"
        continue
    fi
    scenario_rel="${scenario#$REPO_ROOT/}"
    echo "Submitting baseline job for $model ($scenario_rel)"
    bash -c "sbatch '$SCRIPT_DIR/run_baseline.slurm' '$scenario_rel'"
done
