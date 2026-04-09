from __future__ import annotations

import statistics
import warnings
from typing import Any, Optional

from simulon.config.dc import GPUSpec

# Params treated as architecture identity for proportional-scaling fallback.
# Variable params (batch_size, seq_len) are excluded so that a run at a
# different token count can be used as a scaling reference.
# num_params is included so adamw scaling uses num_params ratio rather than batch*seq.
_SCALE_ARCH_KEYS = frozenset(
    {"hidden_size", "num_heads", "ffn_hidden_size", "num_experts", "ep", "top_k", "tp", "dtype", "num_params"}
)

# Cache: (kernel, frozen_params, id(gpu_spec)) → Optional[float]
# Keyed on id(gpu_spec) so different GPUSpec objects don't share entries.
# The gpu_spec object is kept alive by the caller throughout the simulation,
# so using its id() as a key is safe.
_cache: dict[tuple, Optional[float]] = {}


def lookup_kernel_time(
    kernel: str,
    match_params: dict[str, Any],
    gpu_spec: GPUSpec,
    warn: bool = True,
) -> Optional[float]:
    """Find the median runtime (ms) for a kernel with the given parameters.

    Results are cached per (kernel, match_params, gpu_spec identity) so that
    repeated lookups for the same kernel across many DAG nodes are O(1).
    The proportional-scaling warning is emitted at most once per kernel.

    Matching strategy (tried in order, returns on first hit):

    1. **Exact match** — all ``match_params`` keys are present in ``run.params``
       with equal values.
    2. **Partial match** — only keys present in *both* dicts must agree.
    3. **Proportional scaling** — find a run whose architecture params
       (``hidden_size``, ``num_heads``, ``ffn_hidden_size``, ``num_experts``,
       ``ep``, ``top_k``, ``tp``, ``dtype``) match, then scale the median time
       by ``(req_batch * req_seq) / (ref_batch * ref_seq)``.  Emits a
       ``UserWarning`` (once per kernel) unless *warn* is ``False``.

    Returns ``None`` if no usable run is found.
    """
    cache_key = (kernel, frozenset(match_params.items()), id(gpu_spec))
    if cache_key in _cache:
        return _cache[cache_key]

    result = _lookup_kernel_time_impl(kernel, match_params, gpu_spec, warn=warn)
    _cache[cache_key] = result
    return result


def _lookup_kernel_time_impl(
    kernel: str,
    match_params: dict[str, Any],
    gpu_spec: GPUSpec,
    warn: bool = True,
) -> Optional[float]:
    exact: list[float] = []
    partial: list[float] = []

    for run in gpu_spec.kernel_runs:
        if run.kernel != kernel:
            continue

        # Exact: every key in match_params exists in run.params with equal value
        if all(k in run.params and run.params[k] == v for k, v in match_params.items()):
            exact.extend(run.times_ms)
            continue

        # Partial: only check keys present in both
        overlap = {k: v for k, v in match_params.items() if k in run.params}
        if overlap and all(run.params[k] == v for k, v in overlap.items()):
            partial.extend(run.times_ms)

    times = exact or partial
    if times:
        return statistics.median(times)

    # --- Proportional scaling fallback ---
    # For kernels parameterised by num_params (e.g. adamw) scale by that ratio;
    # for token-count kernels scale by batch_size * seq_len.
    req_num_params = match_params.get("num_params")

    # num_params is the scaling dimension for adamw, so exclude it from the arch
    # identity match (analogous to how batch_size / seq_len are excluded for
    # token-count kernels).
    arch_params = {
        k: v for k, v in match_params.items()
        if k in _SCALE_ARCH_KEYS and k != "num_params"
    }
    if not arch_params:
        return None

    scale_candidates: list[tuple[float, float, float]] = []
    for run in gpu_spec.kernel_runs:
        if run.kernel != kernel:
            continue
        if all(k in run.params and run.params[k] == v for k, v in arch_params.items()):
            if req_num_params is not None:
                ref_dim = run.params.get("num_params", 1)
                scale_candidates.append((statistics.median(run.times_ms), req_num_params, ref_dim))
            else:
                req_dim = match_params.get("batch_size", 1) * match_params.get("seq_len", 1)
                ref_dim = run.params.get("batch_size", 1) * run.params.get("seq_len", 1)
                scale_candidates.append((statistics.median(run.times_ms), req_dim, ref_dim))

    if not scale_candidates:
        return None

    ref_time, req_dim, ref_dim = scale_candidates[0]
    scale = req_dim / ref_dim if ref_dim else 1.0

    if warn:
        if req_num_params is not None:
            warnings.warn(
                f"kernel '{kernel}': no exact match for num_params={req_num_params}; "
                f"scaling from num_params={int(ref_dim)} by factor {scale:.2f}x",
                UserWarning,
                stacklevel=3,
            )
        else:
            req_batch = match_params.get("batch_size", 1)
            req_seq = match_params.get("seq_len", 1)
            warnings.warn(
                f"kernel '{kernel}': no exact match for batch_size={req_batch} seq_len={req_seq}; "
                f"scaling from tokens={int(ref_dim)} by factor {scale:.2f}x",
                UserWarning,
                stacklevel=3,
            )

    return ref_time * scale
