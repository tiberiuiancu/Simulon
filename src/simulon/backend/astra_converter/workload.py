"""Converter for workload configurations to ASTRA-Sim trace format."""

from dataclasses import dataclass

from simulon.config.dc import DatacenterConfig
from simulon.config.workload import LLMSpec, MegatronWorkload


@dataclass
class LayerTrace:
    """Trace for a single layer in the workload."""

    layer_id: str
    dependency: int  # -1 for first layer, otherwise index of parent layer
    fwd_compute_time_ns: int
    fwd_comm_type: str  # "ALLGATHER", "REDUCESCATTER", "ALLREDUCE", "NONE"
    fwd_comm_size_bytes: int
    ig_compute_time_ns: int
    ig_comm_type: str
    ig_comm_size_bytes: int
    wg_compute_time_ns: int
    wg_comm_type: str
    wg_comm_size_bytes: int
    wg_update_time_ns: int


@dataclass
class WorkloadTrace:
    """Complete workload trace for ASTRA-Sim."""

    parallelism_policy: str  # "HYBRID_TRANSFORMER"
    model_parallel_npu_group: int  # TP degree
    expert_parallel_npu_group: int  # EP degree
    pipeline_model_parallelism: int  # PP degree
    ga: int  # Gradient accumulation steps
    vpp: int  # Virtual pipeline parallelism
    all_gpus: int
    num_layers: int
    layers: list[LayerTrace]


class WorkloadConverter:
    """Converts MegatronWorkload to WorkloadTrace for ASTRA-Sim."""

    def convert(
        self, workload: MegatronWorkload, datacenter: DatacenterConfig
    ) -> WorkloadTrace:
        """Convert Megatron workload configuration to ASTRA-Sim trace.

        Args:
            workload: The Megatron workload configuration
            datacenter: The datacenter configuration (for GPU specs)

        Returns:
            WorkloadTrace suitable for passing to ASTRA-Sim
        """
        # Extract parallelism configuration
        tp = workload.parallelism.tp
        pp = workload.parallelism.pp
        vpp = workload.parallelism.vpp

        # Get model spec
        model_spec = workload.model
        if isinstance(model_spec, str):
            raise ValueError("Model must be inline spec, not a reference")

        ep = workload.parallelism.ep if model_spec.moe else 1

        # Calculate data parallelism and gradient accumulation
        total_gpus = workload.training.num_gpus
        dp = workload.parallelism.dp if workload.parallelism.dp is not None else (
            total_gpus // (tp * pp * ep)
        )

        micro_batch_size = workload.training.micro_batch_size
        global_batch_size = workload.training.global_batch_size
        ga = global_batch_size // (dp * micro_batch_size)

        # Model parameters
        hidden_size = model_spec.hidden_size
        num_attention_heads = model_spec.num_heads
        seq_length = workload.training.sequence_length
        num_layers = model_spec.num_layers

        # Data type size (assuming bf16/fp16 for activations, fp32 for weights)
        dtype_bytes = 2  # bf16/fp16

        # Generate layers
        layers: list[LayerTrace] = []

        for layer_idx in range(num_layers):
            # Generate attention sublayer
            attn_layer = self._create_attention_layer(
                layer_idx * 2,  # Even indices for attention
                layer_idx,
                workload,
                datacenter,
                hidden_size,
                num_attention_heads,
                seq_length,
                micro_batch_size,
                tp,
                dp,
                dtype_bytes,
            )
            layers.append(attn_layer)

            # Generate MLP sublayer
            mlp_layer = self._create_mlp_layer(
                layer_idx * 2 + 1,  # Odd indices for MLP
                layer_idx,
                workload,
                datacenter,
                hidden_size,
                seq_length,
                micro_batch_size,
                tp,
                dp,
                dtype_bytes,
            )
            layers.append(mlp_layer)

        # Determine parallelism policy
        if ep > 1:
            parallelism_policy = "HYBRID_TRANSFORMER_FP8_MoE"
        else:
            parallelism_policy = "HYBRID_TRANSFORMER"

        return WorkloadTrace(
            parallelism_policy=parallelism_policy,
            model_parallel_npu_group=tp,
            expert_parallel_npu_group=ep,
            pipeline_model_parallelism=pp,
            ga=ga,
            vpp=vpp,
            all_gpus=total_gpus,
            num_layers=len(layers),
            layers=layers,
        )

    def _create_attention_layer(
        self,
        global_layer_id: int,
        transformer_layer_idx: int,
        workload: MegatronWorkload,
        datacenter: DatacenterConfig,
        hidden_size: int,
        num_attention_heads: int,
        seq_length: int,
        micro_batch_size: int,
        tp: int,
        dp: int,
        dtype_bytes: int,
    ) -> LayerTrace:
        """Create attention layer trace."""
        # Dependency: first layer has -1, others depend on previous sublayer
        dependency = -1 if global_layer_id == 0 else global_layer_id - 1

        # Forward pass
        # QKV projection: (batch, seq, hidden) @ (hidden, 3*hidden) = compute
        # Then attention: FlashAttention if enabled
        flash_attention = workload.training.flash_attention
        fwd_compute_time_ns = self._estimate_attention_compute(
            flash_attention, datacenter, micro_batch_size, seq_length, hidden_size, num_attention_heads
        )

        # Communication: TP ALLGATHER on input (if TP > 1)
        if tp > 1:
            fwd_comm_type = "ALLGATHER_TP"
            # Input activations: batch * seq * hidden
            fwd_comm_size_bytes = micro_batch_size * seq_length * hidden_size * dtype_bytes
        else:
            fwd_comm_type = "NONE"
            fwd_comm_size_bytes = 0

        # Input gradient (backward on input)
        ig_compute_time_ns = fwd_compute_time_ns  # Roughly same as forward

        if tp > 1:
            ig_comm_type = "REDUCESCATTER_TP"
            ig_comm_size_bytes = fwd_comm_size_bytes
        else:
            ig_comm_type = "NONE"
            ig_comm_size_bytes = 0

        # Weight gradient (backward on weights)
        wg_compute_time_ns = fwd_compute_time_ns  # Roughly same as forward

        # DP ALLREDUCE on weight gradients
        if dp > 1:
            wg_comm_type = "ALLREDUCE_DP"
            # Weight size: 3 * hidden * hidden (QKV) + hidden * hidden (output projection)
            num_params = 4 * hidden_size * hidden_size
            wg_comm_size_bytes = num_params * 4  # fp32 for gradients
        else:
            wg_comm_type = "NONE"
            wg_comm_size_bytes = 0

        # Weight update time (optimizer step)
        wg_update_time_ns = int(wg_compute_time_ns * 0.1)  # ~10% of compute

        return LayerTrace(
            layer_id=f"layer_{transformer_layer_idx}_attention",
            dependency=dependency,
            fwd_compute_time_ns=fwd_compute_time_ns,
            fwd_comm_type=fwd_comm_type,
            fwd_comm_size_bytes=fwd_comm_size_bytes,
            ig_compute_time_ns=ig_compute_time_ns,
            ig_comm_type=ig_comm_type,
            ig_comm_size_bytes=ig_comm_size_bytes,
            wg_compute_time_ns=wg_compute_time_ns,
            wg_comm_type=wg_comm_type,
            wg_comm_size_bytes=wg_comm_size_bytes,
            wg_update_time_ns=wg_update_time_ns,
        )

    def _create_mlp_layer(
        self,
        global_layer_id: int,
        transformer_layer_idx: int,
        workload: MegatronWorkload,
        datacenter: DatacenterConfig,
        hidden_size: int,
        seq_length: int,
        micro_batch_size: int,
        tp: int,
        dp: int,
        dtype_bytes: int,
    ) -> LayerTrace:
        """Create MLP layer trace."""
        dependency = global_layer_id - 1

        # Forward pass
        # Two matmuls: (batch, seq, hidden) @ (hidden, 4*hidden) and vice versa
        # Use SwiGLU if enabled (3 matmuls instead of 2)
        model_spec = workload.model
        if isinstance(model_spec, str):
            raise ValueError("Model must be inline spec")
        swiglu = model_spec.swiglu

        fwd_compute_time_ns = self._estimate_mlp_compute(
            swiglu, datacenter, micro_batch_size, seq_length, hidden_size
        )

        # Communication: TP ALLGATHER on input
        if tp > 1:
            fwd_comm_type = "ALLGATHER_TP"
            fwd_comm_size_bytes = micro_batch_size * seq_length * hidden_size * dtype_bytes
        else:
            fwd_comm_type = "NONE"
            fwd_comm_size_bytes = 0

        # Input gradient
        ig_compute_time_ns = fwd_compute_time_ns

        if tp > 1:
            ig_comm_type = "REDUCESCATTER_TP"
            ig_comm_size_bytes = fwd_comm_size_bytes
        else:
            ig_comm_type = "NONE"
            ig_comm_size_bytes = 0

        # Weight gradient
        wg_compute_time_ns = fwd_compute_time_ns

        if dp > 1:
            wg_comm_type = "ALLREDUCE_DP"
            # Weight size: hidden * 4*hidden (up proj) + 4*hidden * hidden (down proj)
            # If SwiGLU: add another hidden * 4*hidden
            ffn_hidden = hidden_size * 4
            num_params = 2 * hidden_size * ffn_hidden
            if workload.model.swiglu:
                num_params += hidden_size * ffn_hidden
            wg_comm_size_bytes = num_params * 4  # fp32 for gradients
        else:
            wg_comm_type = "NONE"
            wg_comm_size_bytes = 0

        wg_update_time_ns = int(wg_compute_time_ns * 0.1)

        return LayerTrace(
            layer_id=f"layer_{transformer_layer_idx}_mlp",
            dependency=dependency,
            fwd_compute_time_ns=fwd_compute_time_ns,
            fwd_comm_type=fwd_comm_type,
            fwd_comm_size_bytes=fwd_comm_size_bytes,
            ig_compute_time_ns=ig_compute_time_ns,
            ig_comm_type=ig_comm_type,
            ig_comm_size_bytes=ig_comm_size_bytes,
            wg_compute_time_ns=wg_compute_time_ns,
            wg_comm_type=wg_comm_type,
            wg_comm_size_bytes=wg_comm_size_bytes,
            wg_update_time_ns=wg_update_time_ns,
        )

    def _estimate_attention_compute(
        self,
        flash_attention: bool,
        datacenter: DatacenterConfig,
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        num_heads: int,
    ) -> int:
        """Estimate attention compute time in nanoseconds."""
        # Try to find kernel benchmarks
        gpu_spec = datacenter.node.gpu
        if isinstance(gpu_spec, str):
            kernel_runs = {}
        else:
            kernel_runs = {kr.kernel: kr for kr in gpu_spec.kernel_runs}

        if flash_attention and "flash_attention" in kernel_runs:
            # Use flash attention benchmark
            kernel_info = kernel_runs["flash_attention"]
            # Average the times
            base_time_ms = sum(kernel_info.times_ms) / len(kernel_info.times_ms)
            flops_multiplier = gpu_spec.flops_multiplier

            # Scale by batch size and sequence length
            # FlashAttention time scales roughly as O(batch * seq^2)
            ref_batch = 8
            ref_seq = 2048
            scale_factor = (batch_size / ref_batch) * (seq_length / ref_seq) ** 2

            time_ms = base_time_ms * scale_factor * flops_multiplier
        elif "matmul" in kernel_runs:
            # Use matmul benchmark for QKV projection + output projection
            kernel_info = kernel_runs["matmul"]
            base_time_ms = sum(kernel_info.times_ms) / len(kernel_info.times_ms)
            flops_multiplier = gpu_spec.flops_multiplier

            # 4 matmuls: Q, K, V projections + output projection
            # Plus attention compute
            time_ms = base_time_ms * 4 * flops_multiplier
        else:
            # Fallback: estimate based on FLOPs
            # QKV projection: 3 * (batch * seq * hidden * hidden) FLOPs
            # Attention: 2 * (batch * num_heads * seq^2 * (hidden/num_heads)) FLOPs
            # Output projection: (batch * seq * hidden * hidden) FLOPs
            flops = (
                3 * batch_size * seq_length * hidden_size * hidden_size
                + 2 * batch_size * num_heads * seq_length * seq_length * (hidden_size // num_heads)
                + batch_size * seq_length * hidden_size * hidden_size
            )

            # Assume ~300 TFLOPS for H100 (bf16)
            tflops = 300e12
            time_ms = (flops / tflops) * 1000

        return int(time_ms * 1e6)  # Convert ms to ns

    def _estimate_mlp_compute(
        self,
        swiglu: bool,
        datacenter: DatacenterConfig,
        batch_size: int,
        seq_length: int,
        hidden_size: int,
    ) -> int:
        """Estimate MLP compute time in nanoseconds."""
        gpu_spec = datacenter.node.gpu
        if isinstance(gpu_spec, str):
            kernel_runs = {}
        else:
            kernel_runs = {kr.kernel: kr for kr in gpu_spec.kernel_runs}

        ffn_hidden = hidden_size * 4

        if "matmul" in kernel_runs:
            kernel_info = kernel_runs["matmul"]
            base_time_ms = sum(kernel_info.times_ms) / len(kernel_info.times_ms)
            flops_multiplier = gpu_spec.flops_multiplier

            # 2 matmuls (up + down) or 3 if SwiGLU (gate + up + down)
            num_matmuls = 3 if swiglu else 2
            time_ms = base_time_ms * num_matmuls * flops_multiplier
        else:
            # Fallback: estimate based on FLOPs
            flops = 2 * batch_size * seq_length * hidden_size * ffn_hidden
            if swiglu:
                flops += batch_size * seq_length * hidden_size * ffn_hidden

            tflops = 300e12
            time_ms = (flops / tflops) * 1000

        return int(time_ms * 1e6)  # Convert ms to ns
