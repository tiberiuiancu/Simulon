#!/bin/bash
# Convenience wrapper to submit all jobs
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
sbatch 3_run_setup.sh
sbatch 1_run_profile.sh
sbatch 2_run_simulation.sh
sbatch 4_run_megatron_synthetic.sh
sbatch 5_run_megatron_real.sh
echo "=== All jobs submitted ==="
