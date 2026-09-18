#!/bin/bash
#SBATCH --account=e-sta-openeurollm
#SBATCH --nodes=1
#SBATCH --partition=booster
#SBATCH --time=00:35:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --output=/e/home/jusers/vanosch1/jupiter/Simulon/experiments/nsys_tp2-%j.log
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH --job-name=nsys_tp2_profile

export HF_HOME=/e/scratch/e-sta-openeurollm/vanosch1/hf
export WANDB_MODE=offline
export MACHINE_NAME=JUPITER
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MASTER_ADDR=$(nslookup $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) | grep "Address: " | sed "s/Address: //" | tail) 
export MASTER_PORT=20074
export NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4
export NUM_GPUS_PER_NODE=4
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
export ARCH=$(uname -m)
export SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
export NCCL_SOCKET_IFNAME=ib0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_TIMEOUT=18
export NCCL_IB_DISABLE=0
export NCCL_IB_RETRY_CNT=20
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export GLOO_SOCKET_FAMILY=AF_INET
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
cd /e/home/jusers/vanosch1/jupiter/oellm-autoexp || exit 1
mkdir -p /e/scratch/e-sta-openeurollm/vanosch1/nsys
command -v nsys >/dev/null 2>&1 || echo 'NOTE: nsys not on host PATH; it is invoked INSIDE the container'

echo "Launching v40_qwen3_1p7b_1_pp1_tp2_mbs1_rcFalse_vppNone_spFalse_gbs256_4 on JUPITER"
srun bash -c 'export LOCAL_ADDR=$(nslookup $(hostname) | grep "Address: " | sed "s/Address: //" | tail) ;\
 apptainer exec --env PYTHONPATH=.:submodules/Megatron-LM \
 --env RUN_DIR=./submodules/Megatron-LM \
 --env PROJECT_DIR=. \
 --env HF_HOME=/e/scratch/e-sta-openeurollm/vanosch1/hf \
 --env TOKENIZERS_PARALLELISM=false \
 --env TRANSFORMERS_CACHE=/e/scratch/e-sta-openeurollm/vanosch1/hf/transformers \
 --env HUGGINGFACE_HUB_CACHE=/e/scratch/e-sta-openeurollm/vanosch1/hf/hub \
 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 --env PYTHONWARNINGS=ignore::UserWarning \
 --env NCCL_NET_GDR_LEVEL=0 \
 --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
 --writable-tmpfs --nv --bind /e \
 --bind /dev/shm \
 /e/project1/e-sta-openeurollm/container/nemo_26.04.sif \
 nsys profile -t cuda,nvtx --delay=550 --duration=50 --trace-fork-before-exec=true --cuda-memory-usage=false --force-overwrite true -o /e/scratch/e-sta-openeurollm/vanosch1/nsys/tp2pp1_mbs1_%q{SLURM_JOB_ID} torchrun \
  --nproc-per-node 4\
  --nnodes 1\
  --max-restarts 0\
  --tee 3\
  --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT\
  --rdzv-backend static\
  --node-rank $SLURM_NODEID\
  --master-addr $MASTER_ADDR\
  --master-port $MASTER_PORT\
  --local-addr $LOCAL_ADDR\
   ./submodules/Megatron-LM/pretrain_gpt.py \
   --adam-beta2 0.95\
   --disable-bias-linear\
   --attention-dropout 0.0\
   --attention-softmax-in-fp32\
   --bf16\
   --no-check-for-nan-in-loss-and-grad\
   --data-cache-path .cache\
   --dataloader-type cyclic\
   --distributed-timeout-minutes 30\
   --eval-interval 999999\
   --eval-iters 0\
   --ffn-hidden-size 6144\
   --global-batch-size 256\
   --group-query-attention\
   --hidden-dropout 0.0\
   --hidden-size 2048\
   --kv-channels 128\
   --log-interval 1\
   --log-progress\
   --log-throughput\
   --log-timers-to-tensorboard\
   --lr 1e-08\
   --lr-decay-iters 1\
   --lr-decay-style WSD\
   --lr-wsd-decay-iters 1\
   --lr-wsd-decay-style linear\
   --max-position-embeddings 4096\
   --micro-batch-size 1\
   --mock-data\
   --moe-aux-loss-coeff 0.0\
   --moe-router-load-balancing-type aux_loss\
   --normalization RMSNorm\
   --num-attention-heads 16\
   --num-layers 28\
   --num-query-groups 8\
   --no-overlap-p2p-communication\
   --padded-vocab-size 262272\
   --position-embedding-type rope\
   --profile-ranks 0\
   --profile-step-start 5\
   --qk-layernorm\
   --rerun-mode disabled\
   --save-interval 2000\
   --seq-length 4096\
   --swiglu\
   --tensor-model-parallel-size 2\
   --tensorboard-dir ./output/qwen3-1p7b/v40_qwen3_1p7b_1_pp1_tp2_mbs1_rcFalse_vppNone_spFalse_gbs256/tensorboard\
   --tensorboard-queue-size 5\
   --tokenizer-type NullTokenizer\
   --train-iters 40\
   --use-distributed-optimizer\
   --use-flash-attn\
   --vocab-size 262144\
   --wandb-exp-name v40_qwen3_1p7b_1_pp1_tp2_mbs1_rcFalse_vppNone_spFalse_gbs256\
   --wandb-project sanity-checks\
   --wandb-save-dir ./output/qwen3-1p7b/v40_qwen3_1p7b_1_pp1_tp2_mbs1_rcFalse_vppNone_spFalse_gbs256/wandb\
   --weight-decay 0.05'

# ---------------------------------------------------------------------------
# Export to SQLite and decompose into COMPUTE / NCCL / IDLE.
# `nsys stats --report cuda_gpu_kern_sum` is NOT sufficient: it sums kernel durations, so it
# cannot yield idle time and it double-counts concurrent kernels on different streams. The
# SQLite export gives raw kernel intervals, which analyze_nsys.py unions per device to get
# true GPU busy time — and idle is the quantity that decides this question.
# ---------------------------------------------------------------------------
SCRATCH=/e/scratch/e-sta-openeurollm/vanosch1/nsys
REP=$(ls -t "$SCRATCH"/tp2pp1_mbs1_*.nsys-rep 2>/dev/null | head -1)
echo
echo "=== nsys report: $REP ==="
if [ -z "$REP" ]; then
  echo "NO REPORT PRODUCED — the capture window likely fell outside the run."
  echo "Check the iteration timestamps in this log and adjust --delay."
  exit 1
fi
ls -lh "$REP"

SIF=/e/project1/e-sta-openeurollm/container/nemo_26.04.sif
SQL="${REP%.nsys-rep}.sqlite"
echo "[$(date +%H:%M:%S)] exporting sqlite -> $SQL"
apptainer exec --bind /e "$SIF" bash -lc \
  "nsys export --type sqlite --force-overwrite true -o '$SQL' '$REP'" 2>&1 | tail -3
ls -lh "$SQL" 2>/dev/null || { echo "export failed"; exit 1; }

echo
python3 /e/home/jusers/vanosch1/jupiter/Simulon/experiments/analyze_nsys.py "$SQL"
echo "[$(date +%H:%M:%S)] done"
