#!/bin/bash
# Submit all jobs in order with SLURM dependencies
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
SETUP_JOB=$(sbatch --parsable 3_run_setup.sh)
echo "Submitted setup: $SETUP_JOB"
PROFILE_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB 1_run_profile.sh)
echo "Submitted profile: $PROFILE_JOB"
SIM_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB 2_run_simulation.sh)
echo "Submitted simulation: $SIM_JOB"
SYNTH_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB 4_run_megatron_synthetic.sh)
echo "Submitted synthetic: $SYNTH_JOB"
REAL_JOB=$(sbatch --parsable --dependency=afterok:$SETUP_JOB 5_run_megatron_real.sh)
echo "Submitted real: $REAL_JOB"
echo "=== All jobs submitted ==="
