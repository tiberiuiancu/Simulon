import torch
from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed

# Sequence length constants
SEQUENCE_LENGTH_16K: int = 16384
SEQUENCE_LENGTH_64K: int = 65536
SEQUENCE_LENGTH_128K: int = 131072


def llama3_70b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3 70B.

    Recommended parallelism: TP=4, PP=4, VPP=5, CP=2, SP=True with CommOverlap.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-70B").to_megatron_provider(
        load_weights=False
    )

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 8192

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    cfg.mixed_precision.fp8_recipe = "tensorwise"
    cfg.mixed_precision.fp8 = "hybrid"
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Communication overlap for 70B
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True, tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
    )

    # Mixed precision - explicitly use bf16_mixed
    cfg.mixed_precision = bf16_mixed()

    return cfg
