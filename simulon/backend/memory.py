"""Per-GPU memory feasibility model for Megatron training.

Simulon predicts *throughput* from traces but has no notion of whether a config
fits in GPU memory — so it will happily rank configs that OOM on hardware.  This
module estimates peak per-GPU memory and flags infeasible configs.

Everything structural is DERIVED (no fitted constants):

  model state   Megatron mixed-precision + distributed-optimizer accounting.
                Per parameter held on a GPU: 2 B bf16 weights + 2 B bf16 grads
                (replicated) + (4 B fp32 master + 4 B Adam m + 4 B Adam v) sharded
                across the data-parallel group.  => p·(4 + 12/dp) bytes,
                p = N_params / (TP·PP).

  activations   Korthikanti et al., "Reducing Activation Recomputation in Large
                Transformer Models" (2022), generalized to the actual architecture
                (SwiGLU ffn, GQA) and to FlashAttention.  FlashAttention does NOT
                materialize the s×s score matrix, so the dominant 5·a·s/h term of
                the paper's formula is dropped — this is what makes the estimate
                match reality at seq=4096.  Sequence parallelism divides every
                activation by TP.  Peak is on pipeline stage 0, which under 1F1B
                holds min(PP, n_microbatches) in-flight microbatches × (L/PP) layers
                — i.e. ≈ L layer-activations, nearly PP-independent (which is why
                TP2/PP1 OOMs while TP2/PP2 fits: same activations, 2× the model
                state).

The only non-derived quantity is `overhead_gb` — CUDA context + NCCL/cuBLAS work
buffers + allocator fragmentation + transient backward grad buffer.  It is not
cleanly derivable a priori; the default is grounded against the measured fit/OOM
boundary (see calibrate_overhead / experiments), not invented.

Validated against the 16-node Qwen3-32B fit/OOM runs (2026-06-18): the derived
model-state + activation cleanly separates the measured FIT configs (≤65 GB) from
the OOM configs (≥99 GB), with GH200's 96 GB in the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bytes per element.
_BF16 = 2
_FP32 = 4
# Distributed-optimizer per-parameter bytes: replicated bf16 weight+grad, plus the
# fp32 master/m/v sharded over the DP group.
_REPLICATED_BYTES = _BF16 + _BF16          # 4
_SHARDED_BYTES = _FP32 + _FP32 + _FP32     # 12


@dataclass(frozen=True)
class ModelDims:
    """Architecture dimensions needed for the memory estimate."""

    num_layers: int
    hidden: int
    ffn_hidden: int
    seq: int
    num_heads: int
    num_query_groups: int
    kv_channels: int
    vocab: int
    mlp_gated: bool = True          # SwiGLU → 3 weight matrices (gate, up, down)
    tie_embeddings: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> "ModelDims":
        def g(*keys, default=None):
            for k in keys:
                if k in cfg and cfg[k] is not None:
                    return cfg[k]
            return default

        heads = int(g("num-attention-heads", "num_attention_heads"))
        hidden = int(g("hidden-size", "hidden_size"))
        return cls(
            num_layers=int(g("num-layers", "num_layers")),
            hidden=hidden,
            ffn_hidden=int(g("ffn-hidden-size", "ffn_hidden_size")),
            seq=int(g("seq-length", "seq_length")),
            num_heads=heads,
            num_query_groups=int(g("num-query-groups", "num_query_groups", default=heads)),
            kv_channels=int(g("kv-channels", "kv_channels", default=hidden // heads)),
            vocab=int(g("padded-vocab-size", "padded_vocab_size", "vocab-size",
                        "vocab_size", default=0)),
            tie_embeddings=bool(g("tie-embeddings", "tie_word_embeddings", default=False)),
        )


def param_count(d: ModelDims) -> int:
    """Total parameters, derived from the architecture (GQA attention + SwiGLU MLP)."""
    q = d.num_heads * d.kv_channels
    kv = d.num_query_groups * d.kv_channels
    attn = d.hidden * (q + 2 * kv) + q * d.hidden          # fused QKV + output proj
    mlp = (3 if d.mlp_gated else 2) * d.hidden * d.ffn_hidden
    per_layer = attn + mlp
    embed = d.vocab * d.hidden * (1 if d.tie_embeddings else 2)
    return per_layer * d.num_layers + embed


def model_state_bytes(n_params: int, tp: int, pp: int, dp: int) -> float:
    """Per-GPU weights + grads + (DP-sharded) optimizer state."""
    p = n_params / (tp * pp)
    return p * (_REPLICATED_BYTES + _SHARDED_BYTES / dp)


def _activation_per_layer_bytes(d: ModelDims, tp: int, mbs: int, recompute: str | None) -> float:
    """Stored activation bytes for one transformer layer on one microbatch.

    FlashAttention assumed (no s×s score matrix). Sequence-parallel → all /TP.
    recompute: None (store all), "selective" (recompute core attention → drop the
    attention-projection activations), "full" (store only the layer input).
    """
    s, b, h = d.seq, mbs, d.hidden
    attn_proj = s * b * (d.num_heads * d.kv_channels)     # attention-output proj input
    if recompute == "full":
        return _BF16 * (s * b * h) / tp                  # only the layer input
    # bf16-stored activations: norm/QKV inputs, attn-out input, MLP in/hidden/down-in
    linear = (
        s * b * h                       # input to attention (norm output)
        + attn_proj                     # attention output projection input
        + s * b * h                     # input to MLP
        + 2 * s * b * d.ffn_hidden      # SwiGLU gate & up outputs
        + s * b * d.ffn_hidden          # down projection input
    )
    if recompute == "selective":
        linear -= attn_proj             # core attention recomputed
    return _BF16 * linear / tp


def activation_bytes(d: ModelDims, tp: int, pp: int, mbs: int, recompute: str | None,
                     n_microbatches: int, vpp_v: int = 1) -> float:
    """Per-GPU peak activation memory (pipeline stage 0 under 1F1B)."""
    a_layer = _activation_per_layer_bytes(d, tp, mbs, recompute)
    in_flight = min(pp, n_microbatches) if pp > 1 else 1
    # Interleaved 1F1B (VPP) keeps more microbatches in flight during warmup; the
    # activation grows by ≈ 1 + (pp-1)/(pp·v).  Approximate, flagged for v>1.
    if vpp_v > 1 and pp > 1:
        in_flight *= 1 + (pp - 1) / (pp * vpp_v) * (vpp_v)  # ~ v× the warmup depth
    return a_layer * (d.num_layers / pp) * in_flight


@dataclass(frozen=True)
class MemoryEstimate:
    model_state_gb: float
    activation_gb: float
    overhead_gb: float
    capacity_gb: float

    @property
    def peak_gb(self) -> float:
        return self.model_state_gb + self.activation_gb + self.overhead_gb

    @property
    def fits(self) -> bool:
        return self.peak_gb <= self.capacity_gb


# CUDA context + NCCL/cuBLAS work buffers + fragmentation + transient backward grad
# buffer.  Not derivable a priori; this default is consistent with the measured
# fit/OOM boundary (largest no-recompute FIT ≈ 65 GB, smallest OOM ≈ 99 GB on a
# 96 GB GH200) and a typical ~3-5 GB runtime footprint.
#
# CAVEAT (the one non-derived scalar in this module): pinned to the fit/OOM *boundary*,
# NOT to a measured peak — a good feasibility *classifier* (fits vs OOMs) but not a
# GB-accurate predictor.  To re-ground: log real peak (torch.cuda.max_memory_allocated)
# from a few runs and solve overhead = measured_peak − model_state − activations, then
# replace this constant.  Kept at 4.0 deliberately until that data exists.
DEFAULT_OVERHEAD_GB = 4.0


def estimate_memory(cfg: dict, num_gpus: int, capacity_gb: float,
                    overhead_gb: float = DEFAULT_OVERHEAD_GB) -> MemoryEstimate:
    """Estimate peak per-GPU memory (GB) for a Megatron config."""
    d = ModelDims.from_config(cfg)
    tp = int(cfg.get("tensor-model-parallel-size", cfg.get("tensor_model_parallel_size", 1)))
    pp = int(cfg.get("pipeline-model-parallel-size", cfg.get("pipeline_model_parallel_size", 1)))
    ep = int(cfg.get("expert-model-parallel-size", cfg.get("expert_model_parallel_size", 1)))
    mbs = int(cfg.get("micro-batch-size", cfg.get("micro_batch_size", 1)))
    gbs = int(cfg.get("global-batch-size", cfg.get("global_batch_size", 0)))
    dp = max(1, num_gpus // (tp * pp * ep))
    n_mb = max(1, gbs // (mbs * dp)) if gbs else pp

    nlvps = cfg.get("num-layers-per-virtual-pipeline-stage",
                    cfg.get("num_layers_per_virtual_pipeline_stage"))
    vpp_v = (d_num_layers(cfg) // (pp * int(nlvps))) if nlvps else 1

    rc = _recompute_mode(cfg)
    n = param_count(d)
    ms = model_state_bytes(n, tp, pp, dp) / 1e9
    ac = activation_bytes(d, tp, pp, mbs, rc, n_mb, vpp_v) / 1e9
    return MemoryEstimate(ms, ac, overhead_gb, capacity_gb)


def d_num_layers(cfg: dict) -> int:
    return int(cfg.get("num-layers", cfg.get("num_layers", 1)))


def _recompute_mode(cfg: dict) -> str | None:
    gran = cfg.get("recompute-granularity", cfg.get("recompute_granularity"))
    if gran == "full":
        return "full"
    if cfg.get("recompute-activations", cfg.get("recompute_activations")) or gran == "selective":
        return "selective"
    return None
