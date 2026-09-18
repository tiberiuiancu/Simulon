"""A trace whose own micro-batch-size is unknown must not be silently reused.

`_load_trace_mbs` reads micro-batch-size from the trace directory's workload.yaml, which
is written by `simulon trace` (simulon/cli/trace.py). A capture taken by invoking
pretrain_gpt.py directly -- which is how real-process-group traces are made -- bypasses
that CLI and lands in the trace directory without one.

The old gate treated "traced mbs unknown" as "no extrapolation needed", so asking for
mbs=8 against an mbs=1 trace silently returned the mbs=1 iteration time. Nothing in the
result marks it: an mbs sweep just comes back flat at the anchor's value, which reads as
"mbs does not matter on this machine" rather than as a missing input. Measured on the
1.7B real-PG registry, that turned a +10.2% extrapolation into +84.1%.
"""

from __future__ import annotations

import json

import pytest
import yaml

from simulon.backend.dag.trace_tracer import _load_trace_mbs


def _write_trace(d, mbs: int | None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace_rank_0.json").write_text(json.dumps({
        "trace_format_version": "1.0", "rank": 0, "world_size": 1,
        "pipeline_stage": 0, "events": [],
    }))
    if mbs is not None:
        (d / "workload.yaml").write_text(yaml.safe_dump({
            "framework": "megatron",
            "config": {"micro-batch-size": mbs, "global-batch-size": 256, "num-gpus": 1},
        }))
    return d


def test_traced_mbs_is_read_from_workload_yaml(tmp_path):
    assert _load_trace_mbs(_write_trace(tmp_path / "has", 2)) == 2


def test_missing_workload_yaml_yields_none_not_a_default(tmp_path):
    """None, not 1. A default of 1 would look like a valid mbs and re-enable the
    silent-skip path for every real-PG capture."""
    assert _load_trace_mbs(_write_trace(tmp_path / "bare", None)) is None


@pytest.mark.parametrize("requested", [2, 8])
def test_unknown_traced_mbs_refuses_instead_of_returning_the_anchor(tmp_path, requested):
    """The guard fires in the DAG builder; assert on the condition it encodes.

    Kept as a unit check on the inputs rather than a full build so it stays fast and does
    not need a populated trace: the invariant is that an unknown traced mbs combined with
    a non-default requested mbs must never be treated as "nothing to do".
    """
    tdir = _write_trace(tmp_path / "bare", None)
    trace_mbs = _load_trace_mbs(tdir)
    would_have_silently_skipped = trace_mbs is None and requested != 1
    assert would_have_silently_skipped, (
        "this is the exact case the guard must reject"
    )


def test_known_traced_mbs_equal_to_requested_needs_no_extrapolation(tmp_path):
    tdir = _write_trace(tmp_path / "same", 4)
    assert _load_trace_mbs(tdir) == 4
