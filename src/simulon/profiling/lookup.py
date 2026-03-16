from __future__ import annotations

import statistics
from typing import Any, Optional

from simulon.config.dc import GPUSpec


def lookup_kernel_time(
    kernel: str,
    match_params: dict[str, Any],
    gpu_spec: GPUSpec,
) -> Optional[float]:
    """Find the median runtime (ms) for a kernel with the given parameters.

    Matching strategy:
    1. Exact match: all match_params keys are present in run.params and equal.
    2. Partial match: only keys present in both dicts must agree.
    Returns the median of times_ms for the first matching run, or None.
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
    if not times:
        return None
    return statistics.median(times)
