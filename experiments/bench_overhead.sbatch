#!/bin/bash
# Measure the two missing fixed per-operation costs (see experiments/bench_overhead.py).
# Cluster+toolchain properties: run ONCE per machine, not per model. ~10 minutes.
#
# Runs, in one session on one node so nothing drifts between them:
#   world=1  -> kernel launch cost      (compute-side fixed cost behind the fitted `f`)
#   world=2  -> collective exposure @2 ranks   (predicted 531 us)
#   world=4  -> collective exposure @4 ranks   (predicted 900 us)
#
# Variant C in each distributed run reproduces the nccl-tests regime and is checked against
# the measured Jupiter reference (178.8 us @2, 105.3 us @4). If that control disagrees, the
# harness is wrong and no other number from this job should be used.
#
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:25:00
#SBATCH --job-name=bench_overhead
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --output=experiments/bench_overhead-%j.log

set -uo pipefail
cd /e/home/jusers/vanosch1/jupiter/Simulon || exit 1

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
echo "container: $SIF"
echo "SM clock : $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"

run() {
    local n="$1"
    echo
    echo "############### world_size=$n ###############"
    apptainer run --nv --bind /e/scratch/ --bind /e/project1/ \
        --bind "$(pwd):/opt/simulon" "$SIF" \
        bash -c "cd /opt/simulon && PYTHONPATH=/opt/simulon \
            CUDA_DEVICE_MAX_CONNECTIONS=1 \
            torchrun --nproc_per_node=$n --nnodes=1 \
              --master_addr=127.0.0.1 --master_port=29531 \
              experiments/bench_overhead.py"
    echo "exit=$?"
}

run 1
run 2
run 4
echo
echo "[$(date +%H:%M:%S)] done"
