#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

EXCLUDE=(__pycache__ configs qwen3-32b)
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

if [ -f "$REPO_ROOT/.tracking.env" ]; then
    set -a
    . "$REPO_ROOT/.tracking.env"
    set +a
fi

check_wandb_exists() {
    local model="$1"
    local prefix="validate-e2e-baseline-${model}-"
    uv run python3 -c "
import os, sys, wandb
api = wandb.Api()
entity = os.environ.get('WANDB_ENTITY')
project = os.environ.get('WANDB_PROJECT', 'simulon')
path = f'{entity}/{project}' if entity else project
runs = api.runs(path, filters={'state': 'finished', 'display_name': {'\$regex': r'^${prefix}'}})
for r in runs:
    rest = r.display_name[len('${prefix}'):]
    if rest and rest != 'local':
        try:
            int(rest)
            sys.exit(0)
        except ValueError:
            pass
sys.exit(1)
" 2>/dev/null
}

TOTAL_BASELINES=0
TOTAL_SIMS=0
for dir in "$SCRIPT_DIR/configs"/*/; do
    model="$(basename "$dir")"
    if printf '%s\n' "${EXCLUDE[@]}" | grep -qx "$model"; then
        continue
    fi
    scenario="${dir%/}/scenario.yaml"
    if [ ! -f "$scenario" ]; then
        echo "Warning: no scenario.yaml for $model, skipping"
        continue
    fi
    scenario_rel="${scenario#$REPO_ROOT/}"

    if check_wandb_exists "$model"; then
        echo "Skipping baseline for $model: already tracked in W&B"
    else
        echo "Submitting baseline job for $model ($scenario_rel)"
        if [ "$DRY_RUN" = false ]; then
            bash -c "sbatch '$SCRIPT_DIR/run_baseline.slurm' '$scenario_rel'"
        fi
        TOTAL_BASELINES=$((TOTAL_BASELINES + 1))
    fi

    echo "Submitting sim job for $model ($scenario_rel)"
    if [ "$DRY_RUN" = false ]; then
        bash -c "sbatch '$SCRIPT_DIR/run_sim.slurm' '$scenario_rel'"
    fi
    TOTAL_SIMS=$((TOTAL_SIMS + 1))
done

echo "Submitted $TOTAL_BASELINES baseline job(s) + $TOTAL_SIMS simulation job(s)."
