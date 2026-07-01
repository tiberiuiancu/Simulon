#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

EXCLUDE=(__pycache__ configs)

TOTAL_BASELINES=0
for dir in "$SCRIPT_DIR/configs"/*/; do
    model="$(basename "$dir")"
    if printf '%s\n' "${EXCLUDE[@]}" | grep -qx "$model"; then
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
    TOTAL_BASELINES=$((TOTAL_BASELINES + 1))
done

echo "Submitting simulation sweep job"
bash -c "sbatch '$SCRIPT_DIR/run_sim.slurm'"

echo "Submitted $TOTAL_BASELINES baseline job(s) + 1 simulation sweep job."
