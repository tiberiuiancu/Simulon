from __future__ import annotations

import statistics
import warnings
from typing import Any, Optional

from simulon.config.dc import GPUSpec

# Params treated as architecture identity for proportional-scaling fallback.
# Variable params (batch_size, seq_len) are excluded so that a run at a
# different token count can be used as a scaling reference.
_SCALE_ARCH_KEYS = frozenset(
    {"hidden_size", "num_heads", "ffn_hidden_size", "num_experts", "ep", "top_k", "tp", "dtype"}
)


def lookup_kernel_time(
    kernel: str,
    match_params: dict[str, Any],
    gpu_spec: GPUSpec,
    warn: bool = True,
) -> Optional[float]:
    """Find the median runtime (ms) for a kernel with the given parameters.

    Matching strategy (tried in order, returns on first hit):

    1. **Exact match** — all ``match_params`` keys are present in ``run.params``
       with equal values.
    2. **Partial match** — only keys present in *both* dicts must agree.
    3. **Proportional scaling** — find a run whose architecture params
       (``hidden_size``, ``num_heads``, ``ffn_hidden_size``, ``num_experts``,
       ``ep``, ``top_k``, ``tp``, ``dtype``) match, then scale the median time
       by ``(req_batch * req_seq) / (ref_batch * ref_seq)``.  Emits a
       ``UserWarning`` unless *warn* is ``False``.

    Returns ``None`` if no usable run is found.
    """
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
    arch_params = {k: v for k, v in match_params.items() if k in _SCALE_ARCH_KEYS}
    if not arch_params:
        return None

    req_batch = match_params.get("batch_size", 1)
    req_seq = match_params.get("seq_len", 1)

    scale_candidates: list[tuple[float, int, int]] = []
    for run in gpu_spec.kernel_runs:
        if run.kernel != kernel:
            continue
        if all(k in run.params and run.params[k] == v for k, v in arch_params.items()):
            ref_batch = run.params.get("batch_size", 1)
            ref_seq = run.params.get("seq_len", 1)
            scale_candidates.append((statistics.median(run.times_ms), ref_batch, ref_seq))

    if not scale_candidates:
        return None

    ref_time, ref_batch, ref_seq = scale_candidates[0]
    scale = (req_batch * req_seq) / (ref_batch * ref_seq)

    if warn:
        warnings.warn(
            f"kernel '{kernel}': no exact match for batch_size={req_batch} seq_len={req_seq}; "
            f"scaling from batch_size={ref_batch} seq_len={ref_seq} by factor {scale:.2f}x",
            UserWarning,
            stacklevel=2,
        )

    return ref_time * scale
