"""Host-time attribution in the Megatron tracer, driven by a synthetic kineto trace.

The tracer runs inside the container on the cluster, where a silent attribution bug costs a
whole job and only surfaces hours later in analysis (that is exactly how the fwd=all /
bwd=0 kernel-timing defect happened). These tests exercise _kernel_ms_by_slot without a GPU
by feeding it a hand-built chrome trace.

What is being protected: per-slot `host_ms` is the UNION of that slot's cpu_op intervals.
cpu_op records nest -- aten::linear contains aten::addmm -- so summing durations would
count the same wall-clock host time several times over and inflate the host term, which is
the term the whole two-resource replay rests on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TRACER = (Path(__file__).resolve().parents[1] / "vendor" / "Megatron-LM-traced"
           / "megatron" / "core" / "instrumentation" / "tracer.py")


def _load_tracer_module():
    spec = importlib.util.spec_from_file_location("_simulon_tracer_under_test", _TRACER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tracer_mod():
    return _load_tracer_module()


class _FakeProf:
    """Stands in for torch.profiler; writes the chrome trace the tracer parses."""

    def __init__(self, events):
        self._events = events

    def export_chrome_trace(self, path):
        Path(path).write_text(json.dumps({"traceEvents": self._events}))


def _ev(cat, name, ts, dur, corr=None):
    e = {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}
    if corr is not None:
        e["args"] = {"correlation": corr}
    return e


def _run(tracer_mod, events, scope_to_slot):
    obj = object.__new__(tracer_mod.CudaEventTracer)
    obj._prof = _FakeProf(events)
    obj._scope_to_slot = scope_to_slot
    return obj._kernel_ms_by_slot()


def test_host_ms_is_a_union_not_a_sum(tracer_mod):
    """Nested cpu_ops must collapse. Summing them would double-count host time.

    One 100 us outer op containing two 30 us inner ops: the host was busy for 100 us, not
    160 us.
    """
    events = [
        _ev("user_annotation", "slot0", 1000, 1000),
        _ev("cpu_op", "aten::linear", 1000, 100),
        _ev("cpu_op", "aten::addmm", 1010, 30),
        _ev("cpu_op", "aten::t", 1050, 30),
    ]
    _, _, _, host = _run(tracer_mod, events, {"slot0": 0})
    assert host[0]["host_ms"] == pytest.approx(0.100)  # 100 us, not 160
    assert host[0]["host_ops"] == 3


def test_disjoint_cpu_ops_add_up(tracer_mod):
    events = [
        _ev("user_annotation", "slot0", 0, 1000),
        _ev("cpu_op", "a", 0, 100),
        _ev("cpu_op", "b", 500, 200),
    ]
    _, _, _, host = _run(tracer_mod, events, {"slot0": 0})
    assert host[0]["host_ms"] == pytest.approx(0.300)


def test_counts_and_kernel_time_are_attributed_per_slot(tracer_mod):
    """Two slots, each with its own kernels and launches, attributed by launch site."""
    events = [
        _ev("user_annotation", "slotA", 0, 1000),
        _ev("user_annotation", "slotB", 2000, 1000),
        # slot A: one launch at t=100 -> kernel runs later, on the device clock
        _ev("cuda_runtime", "cudaLaunchKernel", 100, 5, corr=1),
        _ev("kernel", "gemm", 5000, 400, corr=1),
        _ev("cpu_op", "aten::mm", 100, 50),
        # slot B: two launches
        _ev("cuda_runtime", "cudaLaunchKernel", 2100, 5, corr=2),
        _ev("kernel", "gemm", 6000, 200, corr=2),
        _ev("cuda_runtime", "cudaLaunchKernel", 2200, 5, corr=3),
        _ev("kernel", "gemm", 6400, 100, corr=3),
        _ev("cpu_op", "aten::mm", 2100, 40),
        _ev("cpu_op", "aten::mm", 2200, 40),
    ]
    per_slot, total, unattr, host = _run(tracer_mod, events, {"slotA": 0, "slotB": 1})

    # Kernels are booked to the slot containing their LAUNCH, not their execution.
    assert per_slot[0] == pytest.approx(0.400)
    assert per_slot[1] == pytest.approx(0.300)
    assert total == pytest.approx(0.700)
    assert unattr == pytest.approx(0.0)

    assert host[0] == {"host_ms": pytest.approx(0.050), "host_ops": 1,
                       "launch_count": 1, "kernel_count": 1}
    assert host[1] == {"host_ms": pytest.approx(0.080), "host_ops": 2,
                       "launch_count": 2, "kernel_count": 2}


def test_work_outside_every_slot_is_not_attributed(tracer_mod):
    """A kernel launched outside any slot window must land in `unattributed`, not a slot.

    check_trace_health.py gates on that value; quietly folding it into a slot would hide a
    capture whose annotations do not cover the iteration.
    """
    events = [
        _ev("user_annotation", "slot0", 0, 100),
        _ev("cuda_runtime", "cudaLaunchKernel", 9000, 5, corr=1),
        _ev("kernel", "gemm", 9500, 300, corr=1),
        _ev("cpu_op", "aten::mm", 9000, 50),
    ]
    per_slot, total, unattr, host = _run(tracer_mod, events, {"slot0": 0})
    assert per_slot == {}
    assert total == pytest.approx(0.300)
    assert unattr == pytest.approx(0.300)
    assert host == {}


def test_no_kernel_timing_yields_no_host_block(tracer_mod):
    """No cpu_op / kernel records at all -> empty host dict, so downstream stays inert."""
    events = [_ev("user_annotation", "slot0", 0, 100)]
    per_slot, total, unattr, host = _run(tracer_mod, events, {"slot0": 0})
    assert per_slot == {}
    assert total is None
    assert host == {}


def test_nccl_kernels_are_excluded_from_compute(tracer_mod):
    """Under a REAL process group, NCCL kernels must not be booked as compute.

    NCCL spin-waits, so a rank that arrives early at a collective reports enormous kernel
    time that is pure waiting -- measured on 4x GH200 as 15.7 s on idle ranks against 3.7 s
    on the rank actually setting the pace. Counting that as compute would be a worse error
    than the fake-PG bias real-PG tracing exists to remove. Simulon models collectives from
    the measured NCCL bandwidth tables, so the trace only carries the compute side.
    """
    events = [
        _ev("user_annotation", "slot0", 0, 10000),
        _ev("cuda_runtime", "cudaLaunchKernel", 100, 5, corr=1),
        _ev("kernel", "sm90_xmma_gemm_bf16", 5000, 400, corr=1),
        _ev("cuda_runtime", "cudaLaunchKernel", 200, 5, corr=2),
        _ev("kernel", "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 5500, 9000, corr=2),
        _ev("cpu_op", "aten::mm", 100, 50),
    ]
    per_slot, total, unattr, host = _run(tracer_mod, events, {"slot0": 0})
    assert per_slot[0] == pytest.approx(0.400)  # the GEMM only
    assert total == pytest.approx(0.400)  # the 9 s of NCCL spin is gone
    assert host[0]["kernel_count"] == 1


def test_fake_pg_traces_are_unaffected_by_the_nccl_filter(tracer_mod):
    """Back-compat: a fake-PG capture has no NCCL kernels, so nothing changes."""
    events = [
        _ev("user_annotation", "slot0", 0, 10000),
        _ev("cuda_runtime", "cudaLaunchKernel", 100, 5, corr=1),
        _ev("kernel", "sm90_xmma_gemm_bf16", 5000, 400, corr=1),
    ]
    per_slot, total, _u, host = _run(tracer_mod, events, {"slot0": 0})
    assert per_slot[0] == pytest.approx(0.400)
    assert total == pytest.approx(0.400)
    assert host[0]["kernel_count"] == 1
