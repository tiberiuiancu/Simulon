"""GPU kernel benchmarking for simulon GPU profile generation.

Benchmarks each transformer kernel in isolation using CUDA event timing.
Produces a list of KernelRun entries suitable for GPUSpec.kernel_runs.

Optional dependencies (falls back gracefully if absent):
  - flash_attn : flash_attn_varlen_func
  - apex       : FastLayerNormFN
"""

from __future__ import annotations

from typing import Optional

from simulon.config.common import DType
from simulon.config.dc import KernelRun

# ---------------------------------------------------------------------------
# CUDA event timing helper
# ---------------------------------------------------------------------------


def _cuda_time(fn, warmup: int = 3, repeats: int = 10) -> list[float]:
    """Run *fn* and return per-iteration elapsed times in milliseconds."""
    import torch

    device = torch.device("cuda")
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    return times


# ---------------------------------------------------------------------------
# Individual kernel benchmarks
# ---------------------------------------------------------------------------


def _bench_embedding(
    vocab_size: int,
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    tp: int,
    dtype,
) -> list[float]:
    import torch

    local_vocab = vocab_size // tp
    weight = torch.randn(local_vocab, hidden_size, dtype=dtype, device="cuda")
    pos_weight = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
    ids = torch.randint(0, local_vocab, (batch_size, seq_len), device="cuda")

    def fn():
        out = torch.embedding(weight, ids)
        out = out + pos_weight.unsqueeze(0)
        return out

    return _cuda_time(fn)


def _bench_layernorm(
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    dtype,
) -> list[float]:
    import torch
    import torch.nn.functional as F

    x = torch.randn(batch_size * seq_len, hidden_size, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    bias = torch.zeros(hidden_size, dtype=dtype, device="cuda")

    try:
        from apex.normalization.fused_layer_norm import FastLayerNormFN  # type: ignore

        def fn():
            return FastLayerNormFN.apply(x, weight, bias, 1e-5)

    except ImportError:

        def fn():  # type: ignore[misc]
            return F.layer_norm(x, (hidden_size,), weight, bias)

    return _cuda_time(fn)


def _bench_attn_qkv(
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    tp: int,
    dtype,
) -> list[float]:
    import torch

    x = torch.randn(batch_size * seq_len, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(3 * hidden_size // tp, hidden_size, dtype=dtype, device="cuda")

    def fn():
        return torch.matmul(x, w.t())

    return _cuda_time(fn)


def _bench_attn_flash(
    hidden_size: int,
    num_heads: int,
    seq_len: int,
    batch_size: int,
    tp: int,
    dtype,
) -> list[float]:
    import torch
    import torch.nn.functional as F

    local_heads = num_heads // tp
    head_dim = hidden_size // num_heads
    q = torch.randn(batch_size, local_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, local_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, local_heads, seq_len, head_dim, dtype=dtype, device="cuda")

    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore

        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda"
        )
        q_flat = q.transpose(1, 2).reshape(batch_size * seq_len, local_heads, head_dim)
        k_flat = k.transpose(1, 2).reshape(batch_size * seq_len, local_heads, head_dim)
        v_flat = v.transpose(1, 2).reshape(batch_size * seq_len, local_heads, head_dim)

        def fn():
            return flash_attn_varlen_func(
                q_flat, k_flat, v_flat, cu_seqlens, cu_seqlens, seq_len, seq_len
            )

    except ImportError:

        def fn():  # type: ignore[misc]
            return F.scaled_dot_product_attention(q, k, v)

    return _cuda_time(fn)


def _bench_attn_proj(
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    tp: int,
    dtype,
) -> list[float]:
    import torch

    x = torch.randn(batch_size * seq_len, hidden_size // tp, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, hidden_size // tp, dtype=dtype, device="cuda")

    def fn():
        return torch.matmul(x, w.t())

    return _cuda_time(fn)


def _bench_mlp_linear1(
    hidden_size: int,
    ffn_hidden_size: int,
    seq_len: int,
    batch_size: int,
    tp: int,
    dtype,
) -> list[float]:
    import torch

    x = torch.randn(batch_size * seq_len, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(ffn_hidden_size // tp, hidden_size, dtype=dtype, device="cuda")

    def fn():
        return torch.matmul(x, w.t())

    return _cuda_time(fn)


def _bench_mlp_act(
    ffn_hidden_size: int,
    seq_len: int,
    batch_size: int,
    tp: int,
    swiglu: bool,
    dtype,
) -> list[float]:
    import torch
    import torch.nn.functional as F

    if swiglu:
        x = torch.randn(batch_size * seq_len, 2 * ffn_hidden_size // tp, dtype=dtype, device="cuda")

        def fn():
            gate, up = x.chunk(2, dim=-1)
            return F.silu(gate) * up

    else:
        x = torch.randn(batch_size * seq_len, ffn_hidden_size // tp, dtype=dtype, device="cuda")

        def fn():  # type: ignore[misc]
            return F.gelu(x)

    return _cuda_time(fn)


def _bench_mlp_linear2(
    hidden_size: int,
    ffn_hidden_size: int,
    seq_len: int,
    batch_size: int,
    tp: int,
    dtype,
) -> list[float]:
    import torch

    x = torch.randn(batch_size * seq_len, ffn_hidden_size // tp, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, ffn_hidden_size // tp, dtype=dtype, device="cuda")

    def fn():
        return torch.matmul(x, w.t())

    return _cuda_time(fn)


def _bench_moe_route(
    hidden_size: int,
    num_experts: int,
    seq_len: int,
    batch_size: int,
    dtype,
) -> list[float]:
    import torch

    x = torch.randn(batch_size * seq_len, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(num_experts, hidden_size, dtype=dtype, device="cuda")

    def fn():
        return torch.matmul(x, w.t())

    return _cuda_time(fn)


def _bench_moe_expert(
    hidden_size: int,
    ffn_hidden_size: int,
    num_experts: int,
    ep: int,
    top_k: int,
    seq_len: int,
    batch_size: int,
    dtype,
) -> list[float]:
    import torch

    num_local_experts = max(1, num_experts // ep)
    # tokens dispatched to this rank: seq_len * batch_size * top_k / ep (approx)
    local_tokens = max(1, batch_size * seq_len * top_k // ep)

    # AICB uses grouped_gemm (CUTLASS, https://github.com/fanshiqing/grouped_gemm@v1.0)
    # for Megatron MoE and deep_gemm (DeepSeek FP8, SM90+) for DeepSeek variants.
    # Try CUTLASS grouped GEMM first; fall back to sequential torch.matmul otherwise.
    try:
        from grouped_gemm import ops as gg_ops  # type: ignore

        # grouped_gemm expects: x (total_tokens, hidden), w (num_experts, out, in)
        x = torch.randn(num_local_experts * local_tokens, hidden_size, dtype=dtype, device="cuda")
        w1 = torch.randn(num_local_experts, ffn_hidden_size, hidden_size, dtype=dtype, device="cuda")
        w2 = torch.randn(num_local_experts, hidden_size, ffn_hidden_size, dtype=dtype, device="cuda")
        # batch_sizes: how many tokens each expert receives
        batch_sizes = torch.full((num_local_experts,), local_tokens, dtype=torch.long, device="cuda")

        def fn():
            h = gg_ops.gmm(x, w1, batch_sizes, trans_b=True)
            return gg_ops.gmm(h, w2, batch_sizes, trans_b=True)

    except (ImportError, AttributeError):
        # Fallback: sequential matmul per expert. This is pessimistic vs. real
        # grouped GEMM (which fuses all experts) but produces the right shape.
        x = torch.randn(num_local_experts * local_tokens, hidden_size, dtype=dtype, device="cuda")
        w1 = torch.randn(num_local_experts, ffn_hidden_size, hidden_size, dtype=dtype, device="cuda")
        w2 = torch.randn(num_local_experts, hidden_size, ffn_hidden_size, dtype=dtype, device="cuda")

        def fn():  # type: ignore[misc]
            out = torch.zeros_like(x)
            for i in range(num_local_experts):
                t = x[i * local_tokens:(i + 1) * local_tokens]
                h = torch.matmul(t, w1[i].t())
                out[i * local_tokens:(i + 1) * local_tokens] = torch.matmul(h, w2[i].t())
            return out

    return _cuda_time(fn)


def _bench_logit(
    hidden_size: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    tp: int,
    dtype,
) -> list[float]:
    import torch

    x = torch.randn(batch_size * seq_len, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(vocab_size // tp, hidden_size, dtype=dtype, device="cuda")

    def fn():
        return torch.matmul(x, w.t())

    return _cuda_time(fn)


# ---------------------------------------------------------------------------
# _DTYPE_MAP
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    DType.fp32: None,  # resolved lazily
    DType.fp16: None,
    DType.bf16: None,
    DType.fp8: None,
}


def _torch_dtype(dtype: DType):
    import torch

    return {
        DType.fp32: torch.float32,
        DType.fp16: torch.float16,
        DType.bf16: torch.bfloat16,
        DType.fp8: torch.float8_e4m3fn,
    }[dtype]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def benchmark_kernels(
    hidden_size: int,
    num_heads: int,
    ffn_hidden_size: int,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    tp: int = 1,
    dtype: DType = DType.bf16,
    epoch_num: int = 10,
    swiglu: bool = False,
    num_experts: int = 0,
    ep: int = 1,
    top_k: int = 1,
    existing_runs: Optional[list[dict]] = None,
) -> list[KernelRun]:
    """Benchmark all transformer kernels and return KernelRun measurements.

    Must be called on a machine with a CUDA GPU. Tensors are pre-allocated with
    TP-sharded shapes matching the intended deployment configuration.

    Args:
        hidden_size: Transformer hidden dimension.
        num_heads: Total number of attention heads (pre-TP-sharding).
        ffn_hidden_size: FFN intermediate dimension (pre-TP-sharding).
        seq_len: Sequence length in tokens.
        batch_size: Micro-batch size.
        vocab_size: Vocabulary size.
        tp: Tensor Parallelism degree. Weights are sharded by this factor.
        dtype: Compute precision.
        epoch_num: Number of timed iterations per kernel.
        swiglu: Use SwiGLU activation (affects mlp_act tensor shape).
        num_experts: Total number of MoE experts. If > 0, moe_route and moe_expert are benchmarked.
        ep: Expert Parallelism degree (experts are sharded across ep ranks).
        top_k: Top-k routing (tokens dispatched to k experts each).

    Returns:
        List of KernelRun objects with raw per-iteration times in milliseconds.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for kernel benchmarking")

    tdt = _torch_dtype(dtype)
    dtype_str = dtype.value
    params_base = {
        "hidden_size": hidden_size,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "tp": tp,
    }

    # Build a fast lookup: (kernel, frozenset(params)) -> timing count
    _sufficient: set[tuple] = set()
    for run in (existing_runs or []):
        key = (run["kernel"], frozenset(run["params"].items()))
        if len(run["times_ms"]) >= epoch_num:
            _sufficient.add(key)

    results: list[KernelRun] = []

    def _run(kernel, fn, extra_params=None):
        p = {**params_base, **(extra_params or {})}
        if (kernel, frozenset(p.items())) in _sufficient:
            return
        times = fn()
        results.append(KernelRun(kernel=kernel, params=p, times_ms=times[:epoch_num]))

    _run(
        "embedding",
        lambda: _bench_embedding(vocab_size, hidden_size, seq_len, batch_size, tp, tdt),
    )
    _run("layernorm", lambda: _bench_layernorm(hidden_size, seq_len, batch_size, tdt))
    _run("attn_qkv", lambda: _bench_attn_qkv(hidden_size, seq_len, batch_size, tp, tdt))
    _run(
        "attn_flash",
        lambda: _bench_attn_flash(hidden_size, num_heads, seq_len, batch_size, tp, tdt),
        {"num_heads": num_heads},
    )
    _run("attn_proj", lambda: _bench_attn_proj(hidden_size, seq_len, batch_size, tp, tdt))
    _run(
        "mlp_linear1",
        lambda: _bench_mlp_linear1(hidden_size, ffn_hidden_size, seq_len, batch_size, tp, tdt),
        {"ffn_hidden_size": ffn_hidden_size},
    )
    _run(
        "mlp_act",
        lambda: _bench_mlp_act(ffn_hidden_size, seq_len, batch_size, tp, swiglu, tdt),
        {"ffn_hidden_size": ffn_hidden_size, "swiglu": swiglu},
    )
    _run(
        "mlp_linear2",
        lambda: _bench_mlp_linear2(hidden_size, ffn_hidden_size, seq_len, batch_size, tp, tdt),
        {"ffn_hidden_size": ffn_hidden_size},
    )
    _run(
        "logit",
        lambda: _bench_logit(hidden_size, vocab_size, seq_len, batch_size, tp, tdt),
        {"vocab_size": vocab_size},
    )
    if num_experts > 0:
        _run(
            "moe_route",
            lambda: _bench_moe_route(hidden_size, num_experts, seq_len, batch_size, tdt),
            {"num_experts": num_experts},
        )
        _run(
            "moe_expert",
            lambda: _bench_moe_expert(
                hidden_size, ffn_hidden_size, num_experts, ep, top_k, seq_len, batch_size, tdt
            ),
            {"num_experts": num_experts, "ep": ep, "top_k": top_k, "ffn_hidden_size": ffn_hidden_size},
        )

    return results
