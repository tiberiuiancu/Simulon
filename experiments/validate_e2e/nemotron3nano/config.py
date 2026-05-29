# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.core.activations import squared_relu


def nemotron_3_nano_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron 3 Nano (30B-A3B MoE).

    This is a MoE (Mixture of Experts) model with the following default parallelism:
    - TP=4, PP=1, EP=8, SP=True
    - DeepEP enabled for MoE token dispatch

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3 Nano.
    """
    cfg = _pretrain_common()

    # Model Configuration (MoE)
    cfg.model = MambaModelProvider(
        # Architecture (Nemotron 3 Nano 30B-A3B)
        hybrid_layer_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        num_layers=52,
        hidden_size=2688,
        mamba_num_heads=64,
        kv_channels=128,
        mamba_state_dim=128,
        ffn_hidden_size=1856,
        num_attention_heads=32,
        mamba_head_dim=64,
        seq_length=8192,
        num_query_groups=2,
        # MoE
        num_moe_experts=128,
        moe_ffn_hidden_size=1856,
        moe_shared_expert_intermediate_size=3712,
        moe_router_topk=6,
        moe_router_topk_scaling_factor=2.5,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        # NemotronH base
        mamba_num_groups=8,
        make_vocab_size_divisible_by=128,
        activation_func=squared_relu,
        masked_softmax_fusion=True,
        apply_query_key_layer_scaling=False,
        persist_layer_norm=True,
        attention_softmax_in_fp32=False,
        first_last_layers_bf16=True,
        is_hybrid_model=True,
        moe_aux_loss_coeff=0.0001,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        moe_shared_expert_overlap=True,
        # Parallelism
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
    )

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    # Dataset Configuration
    cfg.dataset.seq_length = 8192
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False

    # Parallelism Settings (MoE-specific)
    cfg.model.pipeline_model_parallel_layout = None

    # MoE Token Dispatcher Settings
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training Configuration
    cfg.train.train_iters = 39735
    cfg.train.global_batch_size = 3072
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Transformer Engine (TE)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel Selections
    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory Saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # =========================================================================
    # FP8 & MXFP8 (Mixed Precision Settings)
    # =========================================================================
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # FP8 settings (disabled by default, uncomment to enable)
    cfg.mixed_precision.fp8_recipe = "tensorwise"
    cfg.mixed_precision.fp8 = "hybrid"
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = True

    # Optimizer Precision Settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Optimizer hyperparameters
    cfg.optimizer.lr = 1.6e-3
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.min_lr = 1.6e-5
    cfg.scheduler.lr_warmup_iters = 333

    # Communication Overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_bootstrap_backend="nccl", tp_comm_overlap=True)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = False

    # DDP Configuration
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    cfg.model.init_method_std = 0.0173
    cfg.model.apply_rope_fusion = False
    cfg.model.use_fused_weighted_squared_relu = True

    return cfg
