from __future__ import annotations

import statistics
import warnings
from typing import Any, Optional

from simulon.config.dc import GPUSpec

# ---------------------------------------------------------------------------
# Per-kernel canonical match parameters
# ---------------------------------------------------------------------------

# Defines which parameters are relevant for matching each kernel.
# Callers may pass a superset; only keys listed here are used.
# Unknown kernels fall back to using all provided params (no filtering).
KERNEL_MATCH_KEYS: dict[str, frozenset[str]] = {
    "embedding":   frozenset({"vocab_size", "hidden_size", "seq_len", "batch_size", "tp", "dtype"}),
    "layernorm":   frozenset({"hidden_size", "seq_len", "batch_size", "dtype"}),
    "attn_qkv":    frozenset({"hidden_size", "seq_len", "batch_size", "tp", "dtype"}),
    "attn_flash":  frozenset({"hidden_size", "num_heads", "seq_len", "batch_size", "tp", "dtype"}),
    "attn_proj":   frozenset({"hidden_size", "seq_len", "batch_size", "tp", "dtype"}),
    "mlp_linear1": frozenset({"hidden_size", "ffn_hidden_size", "seq_len", "batch_size", "tp", "dtype"}),
    "mlp_act":     frozenset({"ffn_hidden_size", "seq_len", "batch_size", "tp", "dtype"}),
    "mlp_linear2": frozenset({"hidden_size", "ffn_hidden_size", "seq_len", "batch_size", "tp", "dtype"}),
    "logit":       frozenset({"hidden_size", "vocab_size", "seq_len", "batch_size", "tp", "dtype"}),
    "loss_ce":     frozenset({"vocab_size", "seq_len", "batch_size", "tp", "dtype"}),
    "moe_route":   frozenset({"hidden_size", "num_experts", "seq_len", "batch_size", "dtype"}),
    "moe_expert":  frozenset({"hidden_size", "ffn_hidden_size", "num_experts", "ep", "top_k",
                               "seq_len", "batch_size", "dtype"}),
    "adamw":       frozenset({"num_params", "dtype"}),
}

# Preferred parameter relaxation order for extrapolation.
# When no exact or partial match is found, each parameter is tried in priority
# order: we search for a run that matches all OTHER params exactly, then scale
# the missing one linearly.  dtype is never relaxed and is excluded here.
# Parameters that increase memory/compute: scale by req/ref.
# Sharding parameters (tp, ep) decrease work per GPU: scale by ref/req.
KERNEL_EXTRAPOLATION_ORDER: dict[str, list[str]] = {
    "embedding":   ["batch_size", "vocab_size", "seq_len", "tp"],
    "layernorm":   ["batch_size", "hidden_size", "seq_len"],
    "attn_qkv":    ["batch_size", "hidden_size", "tp", "seq_len"],
    "attn_flash":  ["batch_size", "hidden_size", "num_heads", "tp", "seq_len"],
    "attn_proj":   ["batch_size", "hidden_size", "tp", "seq_len"],
    "mlp_linear1": ["batch_size", "hidden_size", "ffn_hidden_size", "tp", "seq_len"],
    "mlp_act":     ["batch_size", "ffn_hidden_size", "tp", "seq_len"],
    "mlp_linear2": ["batch_size", "hidden_size", "ffn_hidden_size", "tp", "seq_len"],
    "logit":       ["batch_size", "hidden_size", "vocab_size", "tp", "seq_len"],
    "loss_ce":     ["batch_size", "vocab_size", "tp", "seq_len"],
    "moe_route":   ["batch_size", "hidden_size", "num_experts", "seq_len"],
    "moe_expert":  ["batch_size", "hidden_size", "ffn_hidden_size", "ep", "num_experts", "top_k", "seq_len"],
    "adamw":       ["num_params"],
}

# Sharding parameters: higher value = less work per GPU → scale time by ref/req.
_SHARDING_PARAMS = frozenset({"tp", "ep"})

# Cache: (kernel, frozen_params, id(gpu_spec)) → (Optional[float], bool)
# bool = True when the result was obtained via extrapolation.
_cache: dict[tuple, tuple[Optional[float], bool]] = {}


def _scale_ratio(param: str, req_val: Any, ref_val: Any) -> float:
    """Return the linear scaling factor when extrapolating from ref_val to req_val."""
    if ref_val == 0:
        return 1.0
    if param in _SHARDING_PARAMS:
        # Higher sharding degree = less work; scale inversely.
        return ref_val / req_val
    return req_val / ref_val


def is_kernel_oom(
    kernel: str,
    match_params: dict[str, Any],
    gpu_spec: GPUSpec,
) -> bool:
    """Return True if the given kernel+params matches a known OOM entry in gpu_spec.

    Uses the same canonical-key filtering as lookup_kernel_time.  An OOM entry
    matches when every key in the filtered query params is present in the entry's
    params with an equal value.  In practice OOM entries are generated with exactly
    the canonical key set, so this is an equality check; the subset direction
    handles any hand-edited or legacy entries that carry extra keys.
    """
    filtered = _filter_params(kernel, match_params)
    for run in gpu_spec.oom_kernel_runs:
        if run.kernel != kernel:
            continue
        if all(k in run.params and run.params[k] == v for k, v in filtered.items()):
            return True
    return False


def lookup_kernel_time(
    kernel: str,
    match_params: dict[str, Any],
    gpu_spec: GPUSpec,
    warn: bool = True,
) -> tuple[Optional[float], bool]:
    """Find the median runtime (ms) for a kernel with the given parameters.

    Returns ``(time_ms, is_extrapolated)`` where ``is_extrapolated`` is True
    when the result came from structured single-parameter extrapolation rather
    than an exact or partial match.

    Results are cached per (kernel, match_params, gpu_spec identity).

    Matching strategy (tried in order):

    1. **Filter params** — only keys listed in ``KERNEL_MATCH_KEYS[kernel]`` are
       used for matching.  Unknown kernels use all provided params unchanged.
    2. **Exact match** — all filtered params present in ``run.params`` with equal
       values.
    3. **Partial match** — only keys present in *both* dicts must agree.
    4. **Structured extrapolation** — for each parameter in
       ``KERNEL_EXTRAPOLATION_ORDER[kernel]`` (priority order), find a run that
       exactly matches all other filtered params.  Scale the median time linearly
       by the ratio of the mismatched parameter, respecting sharding direction.
       A ``UserWarning`` is emitted (once per kernel per session) unless
       *warn* is ``False``.

    Returns ``(None, False)`` if no usable run is found.
    """
    cache_key = (kernel, frozenset(match_params.items()), id(gpu_spec))
    if cache_key in _cache:
        return _cache[cache_key]

    result = _lookup_kernel_time_impl(kernel, match_params, gpu_spec, warn=warn)
    _cache[cache_key] = result
    return result


def _filter_params(kernel: str, match_params: dict[str, Any]) -> dict[str, Any]:
    """Return only the params relevant to this kernel (drop None values)."""
    canonical = KERNEL_MATCH_KEYS.get(kernel)
    if canonical is None:
        return {k: v for k, v in match_params.items() if v is not None}
    return {k: v for k, v in match_params.items() if k in canonical and v is not None}


def _lookup_kernel_time_impl(
    kernel: str,
    match_params: dict[str, Any],
    gpu_spec: GPUSpec,
    warn: bool = True,
) -> tuple[Optional[float], bool]:
    filtered = _filter_params(kernel, match_params)

    exact: list[float] = []
    partial: list[float] = []

    for run in gpu_spec.kernel_runs:
        if run.kernel != kernel:
            continue

        # Exact: every key in filtered exists in run.params with equal value.
        if all(k in run.params and run.params[k] == v for k, v in filtered.items()):
            exact.extend(run.times_ms)
            continue

        # Partial: only check keys present in both.
        overlap = {k: v for k, v in filtered.items() if k in run.params}
        if overlap and all(run.params[k] == v for k, v in overlap.items()):
            partial.extend(run.times_ms)

    if exact:
        return statistics.median(exact), False
    if partial:
        return statistics.median(partial), False

    # --- Structured extrapolation ---
    extrap_order = KERNEL_EXTRAPOLATION_ORDER.get(kernel, [])
    for relax_param in extrap_order:
        if relax_param not in filtered:
            continue
        req_val = filtered[relax_param]
        if req_val is None:
            continue

        # Match all params except relax_param exactly.
        fixed = {k: v for k, v in filtered.items() if k != relax_param}

        candidates: list[tuple[float, Any]] = []
        for run in gpu_spec.kernel_runs:
            if run.kernel != kernel:
                continue
            if relax_param not in run.params:
                continue
            if all(k in run.params and run.params[k] == v for k, v in fixed.items()):
                candidates.append((statistics.median(run.times_ms), run.params[relax_param]))

        if not candidates:
            continue

        ref_time, ref_val = candidates[0]
        scale = _scale_ratio(relax_param, req_val, ref_val)

        if warn:
            warnings.warn(
                f"kernel '{kernel}': no exact/partial match for {relax_param}={req_val}; "
                f"scaling from {relax_param}={ref_val} by {scale:.2f}x",
                UserWarning,
                stacklevel=3,
            )

        return ref_time * scale, True

    return None, False
