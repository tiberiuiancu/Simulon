"""Megatron-LM workload trace generator.

Generates a WorkloadTrace from a MegatronWorkload config + optional GPU profile.
No AICB imports — fully self-contained.
"""

import statistics
from typing import Optional

from simulon.config.dc import GPUSpec
from simulon.config.workload import LLMSpec, MegatronWorkload

from .trace import (
    CommOp,
    CommOpType,
    ComputeOp,
    ParallelGroup,
    TraceOp,
    WorkloadTrace,
)

# ---------------------------------------------------------------------------
# Kernel timing lookup
# ---------------------------------------------------------------------------

_NUMERIC_KEYS = frozenset(
    ["hidden_size", "ffn_hidden_size", "num_heads", "seq_len", "batch_size", "vocab_size", "tp"]
)


def _lookup_kernel_time(profile: GPUSpec, kernel: str, params: dict) -> Optional[float]:
    """Return mean runtime in microseconds for the closest matching KernelRun entry.

    Exact match on `dtype`; nearest-neighbor by Euclidean distance on numeric params.
    Returns None if no entry exists for that kernel name.
    """
    candidates = [kr for kr in profile.kernel_runs if kr.kernel == kernel]
    if not candidates:
        return None

    dtype = params.get("dtype")
    if dtype:
        typed = [kr for kr in candidates if kr.params.get("dtype") == dtype]
        if typed:
            candidates = typed

    if len(candidates) == 1:
        return statistics.mean(candidates[0].times_ms) * 1000.0  # ms → µs

    numeric_keys = [k for k in params if k in _NUMERIC_KEYS and k != "dtype"]

    def _dist(kr):
        d = 0.0
        for k in numeric_keys:
            val = params[k]
            pval = kr.params.get(k)
            if pval is not None and val != 0:
                d += ((val - pval) / max(abs(val), abs(pval))) ** 2
        return d

    best = min(candidates, key=_dist)
    return statistics.mean(best.times_ms) * 1000.0  # ms → µs


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_megatron_trace(
    workload: MegatronWorkload,
    model_spec: LLMSpec,
    gpu_profile: Optional[GPUSpec] = None,
) -> WorkloadTrace:
    """Generate a WorkloadTrace for one training iteration.

    Models the perspective of the first PP stage (which is also the last when pp=1).
    For pp>1, ISend/IRecv pipeline ops are included; embedding/logit are present only
    when the GPU is both first and last stage (pp=1).
    """
    # --- Parallelism ---
    tp = workload.parallelism.tp
    pp = workload.parallelism.pp
    ep = workload.parallelism.ep
    sp = workload.parallelism.sp
    dist_opt = workload.parallelism.distributed_optimizer

    num_gpus = workload.training.num_gpus
    dp = num_gpus // (tp * pp * ep)
    global_batch = workload.training.global_batch_size
    micro_batch = workload.training.micro_batch_size
    seq_len = workload.training.sequence_length
    num_microbatches = global_batch // (micro_batch * dp)

    # --- Model dims ---
    hidden = model_spec.hidden_size
    num_layers = model_spec.num_layers
    num_heads = model_spec.num_heads
    vocab_size = model_spec.vocab_size
    swiglu = model_spec.swiglu

    if model_spec.ffn_hidden_size:
        ffn_hidden = model_spec.ffn_hidden_size
    elif swiglu:
        ffn_hidden = round(8 / 3 * hidden / 64) * 64
    else:
        ffn_hidden = 4 * hidden

    layers_per_stage = num_layers // pp

    # --- Parameter counts ---
    attn_per_layer = 4 * hidden * hidden // tp
    mlp_factor = 3 if swiglu else 2
    mlp_per_layer = mlp_factor * hidden * ffn_hidden // tp
    embedding_params = vocab_size * hidden // tp
    layernorm_params = layers_per_stage * hidden * 2
    params_per_stage = (attn_per_layer + mlp_per_layer) * layers_per_stage
    total_params = embedding_params + pp * params_per_stage

    # --- Comm sizes (bytes, bf16 = 2 bytes/element) ---
    B = 2
    tp_comm = B * seq_len * micro_batch * hidden
    pp_act = B * hidden * seq_len * micro_batch

    ops: list[TraceOp] = []

    # --- Helpers ---
    def comm(comm_type, group, group_size, size_bytes, stage, **kw):
        return CommOp(
            comm_type=comm_type,
            group=group,
            group_size=group_size,
            size_bytes=size_bytes,
            stage=stage,
            **kw,
        )

    def compute(kernel, input_shapes, stage, params=None):
        t = None
        if gpu_profile is not None and params is not None:
            t = _lookup_kernel_time(gpu_profile, kernel, params)
        return ComputeOp(kernel=kernel, input_shapes=input_shapes, compute_time_us=t, stage=stage)

    def _prof(extra=None):
        """Build profile params dict with common keys."""
        base = {
            "hidden_size": hidden,
            "seq_len": seq_len,
            "batch_size": micro_batch,
            "dtype": "bf16",
        }
        if extra:
            base.update(extra)
        return base

    # -----------------------------------------------------------------------
    # Init ops
    # -----------------------------------------------------------------------
    for _ in range(4):
        ops.append(comm(CommOpType.all_reduce, ParallelGroup.dp, dp, 4, "init"))
    ops.append(comm(CommOpType.all_gather, ParallelGroup.dp, dp, 4, "init"))
    ops.append(comm(CommOpType.broadcast, ParallelGroup.tp, tp, 24, "init"))
    if pp > 1:
        # Last pp stage syncs embedding weights across TP group after init
        ops.append(
            comm(CommOpType.all_reduce, ParallelGroup.tp, tp, B * embedding_params, "init")
        )

    # -----------------------------------------------------------------------
    # Per-microbatch forward
    # -----------------------------------------------------------------------
    def fwd_mb() -> list[TraceOp]:
        mb: list[TraceOp] = []
        head_dim = hidden // num_heads

        # Attention mask metadata broadcast (once per microbatch when tp > 1)
        if tp > 1:
            mb.append(comm(CommOpType.broadcast, ParallelGroup.tp, tp, 40, "forward"))
            mb.append(
                comm(
                    CommOpType.broadcast,
                    ParallelGroup.tp,
                    tp,
                    8 * (tp + seq_len * micro_batch),
                    "forward",
                )
            )

        for _ in range(layers_per_stage):
            # Pre-attention layernorm
            mb.append(
                compute(
                    "layernorm",
                    [(seq_len * micro_batch, hidden)],
                    "forward",
                    _prof(),
                )
            )
            # QKV projection
            mb.append(
                compute(
                    "attn_qkv",
                    [(seq_len * micro_batch, hidden)],
                    "forward",
                    _prof(),
                )
            )
            # SP: AllGather for attention ColumnLinear (qkv) — MegatronColumnLinear.forward()
            # emits AllGather(tp) when sequence_parallel_enabled. AICB: MockedMegatron.py:173-181.
            if sp:
                mb.append(
                    comm(CommOpType.all_gather, ParallelGroup.tp, tp, tp_comm, "forward", additional="overlap")
                )
            # Flash attention
            mb.append(
                compute(
                    "attn_flash",
                    [(micro_batch, num_heads // tp, seq_len, head_dim)],
                    "forward",
                    _prof({"num_heads": num_heads}),
                )
            )
            # Output projection
            mb.append(
                compute(
                    "attn_proj",
                    [(seq_len * micro_batch, hidden // tp)],
                    "forward",
                    _prof(),
                )
            )
            # RowLinear comm (attention)
            if sp:
                mb.append(comm(CommOpType.reduce_scatter, ParallelGroup.tp, tp, tp_comm, "forward"))
            else:
                mb.append(comm(CommOpType.all_reduce, ParallelGroup.tp, tp, tp_comm, "forward"))

            # Pre-MLP layernorm
            mb.append(
                compute(
                    "layernorm",
                    [(seq_len * micro_batch, hidden)],
                    "forward",
                    _prof(),
                )
            )
            # SP: AllGather for MLP ColumnLinear (dense_h_to_4h) — same pattern as attn qkv.
            # AICB: MegatronColumnLinear.forward() always emits AllGather when sp=True.
            # Previous code only had one AllGather (attn), missing this one for MLP.
            if sp:
                mb.append(comm(CommOpType.all_gather, ParallelGroup.tp, tp, tp_comm, "forward"))
            # MLP linear1
            mb.append(
                compute(
                    "mlp_linear1",
                    [(seq_len * micro_batch, hidden)],
                    "forward",
                    _prof({"ffn_hidden_size": ffn_hidden}),
                )
            )
            # MLP activation
            mb.append(
                compute(
                    "mlp_act",
                    [(seq_len * micro_batch, ffn_hidden // tp)],
                    "forward",
                    {"ffn_hidden_size": ffn_hidden, "seq_len": seq_len, "batch_size": micro_batch, "dtype": "bf16"},
                )
            )
            # MLP linear2
            mb.append(
                compute(
                    "mlp_linear2",
                    [(seq_len * micro_batch, ffn_hidden // tp)],
                    "forward",
                    _prof({"ffn_hidden_size": ffn_hidden}),
                )
            )
            # RowLinear comm (MLP)
            if sp:
                mb.append(comm(CommOpType.reduce_scatter, ParallelGroup.tp, tp, tp_comm, "forward"))
            else:
                mb.append(comm(CommOpType.all_reduce, ParallelGroup.tp, tp, tp_comm, "forward"))

        # First pp stage (always — we model rank 0): embedding lookup
        mb.append(
            compute(
                "embedding",
                [(micro_batch * seq_len, hidden // tp)],
                "forward",
                _prof(),
            )
        )

        # Last pp stage (only when pp == 1 — same GPU is first and last): logit + loss
        if pp == 1:
            mb.append(
                compute(
                    "logit",
                    [(seq_len * micro_batch, vocab_size // tp)],
                    "forward",
                    _prof({"vocab_size": vocab_size}),
                )
            )
            # Cross-entropy loss reduction
            for _ in range(3):
                mb.append(
                    comm(CommOpType.all_reduce, ParallelGroup.tp, tp, seq_len * micro_batch * 4, "forward")
                )
            # Average loss across DP
            mb.append(comm(CommOpType.all_reduce, ParallelGroup.dp, dp, 4, "forward"))

        # Pipeline activation transfer
        if pp > 1:
            mb.append(
                comm(CommOpType.isend, ParallelGroup.pp, pp, pp_act, "forward", additional="send_next")
            )
            mb.append(
                comm(CommOpType.irecv, ParallelGroup.pp, pp, pp_act, "forward", additional="recv_prev")
            )

        return mb

    # -----------------------------------------------------------------------
    # Per-microbatch backward
    # -----------------------------------------------------------------------
    # AICB (utils/utils.py extract_averages): backward compute time = forward
    # compute time (1:1 ratio). No separate backward kernels are benchmarked;
    # the same profiled forward times are reused for input-grad computation.
    # We therefore call compute() with the same kernel names as forward —
    # _lookup_kernel_time returns the forward-measured time for all of them.
    def bwd_mb() -> list[TraceOp]:
        mb: list[TraceOp] = []

        for _ in range(layers_per_stage):
            # Input-grad compute: reversed kernel order matches backward data flow.
            # Timing reuses forward kernel entries (AICB 1:1 ratio).
            mb.append(compute("mlp_linear2", [(seq_len * micro_batch, ffn_hidden // tp)], "backward", _prof({"ffn_hidden_size": ffn_hidden})))
            mb.append(compute("mlp_act",     [(seq_len * micro_batch, ffn_hidden // tp)], "backward", {"ffn_hidden_size": ffn_hidden, "seq_len": seq_len, "batch_size": micro_batch, "dtype": "bf16"}))
            mb.append(compute("mlp_linear1", [(seq_len * micro_batch, hidden)],           "backward", _prof({"ffn_hidden_size": ffn_hidden})))
            mb.append(compute("layernorm",   [(seq_len * micro_batch, hidden)],           "backward", _prof()))
            mb.append(compute("attn_proj",   [(seq_len * micro_batch, hidden // tp)],     "backward", _prof()))
            mb.append(compute("attn_flash",  [(micro_batch, num_heads // tp, seq_len, hidden // num_heads)], "backward", _prof({"num_heads": num_heads})))
            mb.append(compute("attn_qkv",    [(seq_len * micro_batch, hidden)],           "backward", _prof()))
            mb.append(compute("layernorm",   [(seq_len * micro_batch, hidden)],           "backward", _prof()))

            if sp:
                # Per AICB MockedMegatron.py, per transformer layer backward with SP:
                #   MegatronAttention.backward():
                #     qkv ColumnLinear.backward()       → AllGather(tp) + ReduceScatter(tp)
                #     attn_proj RowLinear.backward()    → AllGather(tp)
                #   MegatronMlp.backward():
                #     dense_h_to_4h ColumnLinear.backward() → AllGather(tp) + ReduceScatter(tp)
                #     dense_4h_to_h RowLinear.backward()    → AllGather(tp)
                # Total per layer: 4× AllGather + 2× ReduceScatter.
                # Previous code had 2 AG + 1 RS (off by 2 AG + 1 RS per layer).
                mb.append(comm(CommOpType.all_gather, ParallelGroup.tp, tp, tp_comm, "backward"))     # attn qkv col bwd
                mb.append(comm(CommOpType.reduce_scatter, ParallelGroup.tp, tp, tp_comm, "backward")) # attn qkv col bwd
                mb.append(comm(CommOpType.all_gather, ParallelGroup.tp, tp, tp_comm, "backward"))     # attn proj row bwd
                mb.append(comm(CommOpType.all_gather, ParallelGroup.tp, tp, tp_comm, "backward"))     # mlp linear1 col bwd
                mb.append(comm(CommOpType.reduce_scatter, ParallelGroup.tp, tp, tp_comm, "backward")) # mlp linear1 col bwd
                mb.append(comm(CommOpType.all_gather, ParallelGroup.tp, tp, tp_comm, "backward"))     # mlp linear2 row bwd
            else:
                # Without SP: ColumnLinear.backward() emits AllReduce(tp); RowLinear.backward()
                # emits nothing. One AllReduce per sublayer × 2 sublayers = 2 per layer. Unchanged.
                mb.append(comm(CommOpType.all_reduce, ParallelGroup.tp, tp, tp_comm, "backward"))
                mb.append(comm(CommOpType.all_reduce, ParallelGroup.tp, tp, tp_comm, "backward"))

        # Pipeline gradient transfer
        if pp > 1:
            mb.append(
                comm(CommOpType.isend, ParallelGroup.pp, pp, pp_act, "backward", additional="send_prev")
            )
            mb.append(
                comm(CommOpType.irecv, ParallelGroup.pp, pp, pp_act, "backward", additional="recv_next")
            )

        return mb

    # -----------------------------------------------------------------------
    # Assemble forward + backward for all microbatches
    # -----------------------------------------------------------------------
    for _ in range(num_microbatches):
        ops.extend(fwd_mb())
    for _ in range(num_microbatches):
        ops.extend(bwd_mb())

    # -----------------------------------------------------------------------
    # Optimizer step
    # -----------------------------------------------------------------------
    if dist_opt:
        # Distributed optimizer: ReduceScatter fp32 grads, AllGather updated params
        ops.append(
            comm(CommOpType.reduce_scatter, ParallelGroup.dp, dp, 4 * total_params // pp, "step")
        )
        ops.append(
            comm(CommOpType.all_gather, ParallelGroup.dp, dp, 2 * total_params // pp, "step")
        )
    else:
        ops.append(
            comm(CommOpType.all_reduce, ParallelGroup.dp, dp, 4 * total_params // pp, "step")
        )
    # LayerNorm grad sync across TP
    ops.append(
        comm(CommOpType.all_reduce, ParallelGroup.tp, tp, 2 * layernorm_params, "step")
    )
    # NaN/Inf check
    ops.append(comm(CommOpType.all_reduce, ParallelGroup.tp, tp, 4, "step"))

    return WorkloadTrace(
        framework="megatron",
        model_name=model_spec.name or "unknown",
        num_gpus=num_gpus,
        tp=tp,
        pp=pp,
        dp=dp,
        ep=ep,
        iterations=workload.training.iterations,
        ops=ops,
    )
