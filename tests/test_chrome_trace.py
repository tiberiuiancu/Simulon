"""Tests for chrome_trace.to_chrome_trace() — specifically the fused_kernels arg behaviour."""

from __future__ import annotations

from simulon.backend.dag.chrome_trace import _TID_COMPUTE, to_chrome_trace
from simulon.backend.dag.nodes import ComputeNode, ExecutionDAG


def _make_compute_node(node_id: int, fused_kernels: list[str], kernel: str = "mlp_linear1") -> ComputeNode:
    """Return a minimal, timing-populated ComputeNode."""
    return ComputeNode(
        node_id=node_id,
        gpu_rank=0,
        kernel=kernel,
        layer_id=0,
        microbatch_id=0,
        pipeline_stage=0,
        phase="fwd",
        duration_ms=1.0,
        start_ms=0.0,
        finish_ms=1.0,
        fused_kernels=fused_kernels,
    )


def _compute_events(dag: ExecutionDAG) -> list[dict]:
    """Extract all compute 'X' events from the trace."""
    trace = to_chrome_trace(dag, tp=1, pp=1, dp=1)
    return [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("tid") == _TID_COMPUTE]


class TestFusedKernelsArg:
    def test_empty_fused_kernels_omits_arg(self):
        """A ComputeNode with no fused kernels must not emit a fused_kernels key."""
        dag = ExecutionDAG(compute_nodes=[_make_compute_node(0, [])])
        events = _compute_events(dag)
        assert len(events) == 1
        assert "fused_kernels" not in events[0]["args"]

    def test_fused_kernels_arg_is_comma_joined(self):
        """The fused_kernels arg value must be the kernel names joined with ', '."""
        kernels = ["layernorm", "attn_qkv", "attn_flash"]
        dag = ExecutionDAG(compute_nodes=[_make_compute_node(0, kernels)])
        events = _compute_events(dag)
        assert events[0]["args"]["fused_kernels"] == "layernorm, attn_qkv, attn_flash"

    def test_single_fused_kernel_no_trailing_comma(self):
        """A single-element fused_kernels list must produce a plain string with no comma."""
        dag = ExecutionDAG(compute_nodes=[_make_compute_node(0, ["attn_proj"])])
        events = _compute_events(dag)
        assert events[0]["args"]["fused_kernels"] == "attn_proj"

    def test_fused_node_event_name_unchanged(self):
        """The event name must be the node's kernel field, not the joined fused names."""
        dag = ExecutionDAG(compute_nodes=[_make_compute_node(0, ["mlp_act", "mlp_linear2"], kernel="2 kernels")])
        events = _compute_events(dag)
        assert events[0]["name"] == "2 kernels"

    def test_mixed_nodes_only_fused_gets_arg(self):
        """When both fused and plain nodes exist, only the fused one carries the arg."""
        plain = _make_compute_node(0, [])
        fused = _make_compute_node(1, ["mlp_act", "mlp_linear2"])
        fused.start_ms = 2.0
        fused.finish_ms = 3.0
        dag = ExecutionDAG(compute_nodes=[plain, fused])
        events = _compute_events(dag)
        assert len(events) == 2
        plain_args = next(e["args"] for e in events if "fused_kernels" not in e["args"])
        fused_args = next(e["args"] for e in events if "fused_kernels" in e["args"])
        assert "fused_kernels" not in plain_args
        assert fused_args["fused_kernels"] == "mlp_act, mlp_linear2"
