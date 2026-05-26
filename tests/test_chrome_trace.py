"""Tests for chrome_trace.to_chrome_trace() — specifically the fused_kernels arg behaviour."""

from __future__ import annotations

from simulon.backend.dag.chrome_trace import _TID_COMPUTE, to_chrome_trace
from simulon.backend.dag.nodes import CommNode, ComputeNode, ExecutionDAG


def _make_compute_node(
    node_id: int, fused_kernels: list[str], kernel: str = "mlp_linear1"
) -> ComputeNode:
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


class TestOnlyProfiledFiltering:
    def test_only_profiled_filters_compute_comm_and_metadata(self):
        """When only_profiled=True, only ranks in dag.profiled_ranks emit events."""
        compute_profiled = ComputeNode(
            node_id=0,
            gpu_rank=0,
            kernel="mlp_linear1",
            layer_id=0,
            microbatch_id=0,
            pipeline_stage=0,
            phase="fwd",
            duration_ms=1.0,
            start_ms=0.0,
            finish_ms=1.0,
        )
        compute_unprofiled = ComputeNode(
            node_id=1,
            gpu_rank=1,
            kernel="mlp_linear2",
            layer_id=0,
            microbatch_id=0,
            pipeline_stage=0,
            phase="fwd",
            duration_ms=1.0,
            start_ms=2.0,
            finish_ms=3.0,
        )
        comm = CommNode(
            node_id=2,
            src_gpu=0,
            dst_gpu=1,
            bytes=1024,
            collective_type="AllGather",
            layer_id=0,
            phase="fwd",
            flow_id=0,
            duration_ms=0.5,
            start_ms=0.0,
            finish_ms=0.5,
        )
        dag = ExecutionDAG(
            compute_nodes=[compute_profiled, compute_unprofiled],
            comm_nodes=[comm],
            profiled_ranks={0},
        )

        trace = to_chrome_trace(dag, tp=1, pp=1, dp=1, only_profiled=True)
        events = trace["traceEvents"]

        meta_events = [e for e in events if e.get("ph") == "M"]
        pids_with_meta = {e["pid"] for e in meta_events}
        assert pids_with_meta == {1000}, f"Expected only pid 1000 metadata, got {pids_with_meta}"

        compute_x = [e for e in events if e.get("ph") == "X" and e.get("tid") == _TID_COMPUTE]
        assert len(compute_x) == 1
        assert compute_x[0]["pid"] == 1000

        comm_x = [e for e in events if e.get("ph") == "X" and e.get("tid") != _TID_COMPUTE]
        assert len(comm_x) == 0
