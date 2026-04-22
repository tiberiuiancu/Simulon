#!/bin/bash
# Submit all jobs in order with SLURM dependencies
set -euo pipefail
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/../.."
SETUP_JOB=$(sbatch --parsable experiments/validate_compute/3_run_setup.sh)
echo "Submitted setup: $SETUP_JOB"
PROFILE_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB experiments/validate_compute/1_run_profile.sh)
echo "Submitted profile: $PROFILE_JOB"
SIM_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB experiments/validate_compute/2_run_simulation.sh)
echo "Submitted simulation: $SIM_JOB"
SYNTH_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB experiments/validate_compute/4_run_megatron_synthetic.sh)
echo "Submitted synthetic: $SYNTH_JOB"
REAL_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB experiments/validate_compute/5_run_megatron_real.sh)
echo "Submitted real: $REAL_JOB"
echo "=== All jobs submitted ==="
