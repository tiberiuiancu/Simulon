import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from simulon.backend.dag.nodes import DAGEdge
from simulon.backend.dag.trace_parser import TraceFileParser
from simulon.backend.dag.trace_tracer import MegatronDagTracer, ParallelConfig, _remap_collectives
from simulon.backend.dag.tracer import DAGTracerConfig
from simulon.collective import NCCLDecomposer
from simulon.collective.decompose import decompose_collective
from simulon.config.dc import (
    ClusterSpec,
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    NodeSpec,
)
from simulon.config.workload import MegatronWorkload


def _write_trace(data: dict[str, Any], traces_dir: Path, rank: int) -> Path:
    path = traces_dir / f"trace_rank_{rank}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_trace(
    events: list[dict[str, Any]],
    rank: int = 0,
    world_size: int = 2,
    pipeline_stage: int = 0,
) -> dict[str, Any]:
    return {
        "trace_format_version": "1.0",
        "rank": rank,
        "world_size": world_size,
        "pipeline_stage": pipeline_stage,
        "events": events,
    }


def _make_datacenter(num_gpus: int = 2, traces_dir: str | None = None) -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test", traces_dir=traces_dir),
        cluster=ClusterSpec(num_nodes=1),
        node=NodeSpec(
            gpus_per_node=num_gpus,
            gpu=GPUSpec(name="H100", memory_capacity_gb=80.0),
        ),
    )


def _make_workload(
    tp: int = 1,
    pp: int = 1,
    num_gpus: int = 2,
) -> MegatronWorkload:
    return MegatronWorkload(
        framework="megatron",
        config={
            "tensor-model-parallel-size": tp,
            "pipeline-model-parallel-size": pp,
            "num-layers": 2,
            "hidden-size": 512,
            "num-attention-heads": 8,
            "ffn-hidden-size": 11008,
            "seq-length": 128,
            "micro-batch-size": 1,
            "global-batch-size": num_gpus,
            "num_gpus": num_gpus,
        },
    )


def _run_tracer(workload: MegatronWorkload, datacenter: DatacenterConfig):
    tracer = MegatronDagTracer(cfg=DAGTracerConfig(), ccl=NCCLDecomposer())
    return tracer.trace(workload, datacenter)


def test_trace_builds_dag_with_compute_and_comm_nodes():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {
            "type": "collective",
            "timestamp_ms": 1.0,
            "metadata": {
                "collective_type": "AllReduce",
                "bytes": 2048,
                "group_ranks": [0, 1],
            },
        },
        {"type": "slot_end", "timestamp_ms": 2.0, "metadata": {"slot": "fwd"}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        trace = _make_trace(events, rank=0, world_size=2, pipeline_stage=0)
        _write_trace(trace, traces_dir, rank=0)
        workload = _make_workload(tp=2, pp=1, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        assert len(dag.compute_nodes) > 0
        assert len(dag.comm_nodes) > 0


def test_compute_nodes_have_duration_ms():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {
            "type": "collective",
            "timestamp_ms": 2.0,
            "metadata": {
                "collective_type": "AllReduce",
                "bytes": 2048,
                "group_ranks": [0, 1],
            },
        },
        {"type": "slot_end", "timestamp_ms": 5.0, "metadata": {"slot": "fwd"}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        trace = _make_trace(events, rank=0, world_size=2, pipeline_stage=0)
        _write_trace(trace, traces_dir, rank=0)
        workload = _make_workload(tp=2, pp=1, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        for cn in dag.compute_nodes:
            assert cn.duration_ms is not None
            assert cn.duration_ms > 0


def test_comm_nodes_have_correct_bytes_and_type():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {
            "type": "collective",
            "timestamp_ms": 1.0,
            "metadata": {
                "collective_type": "AllReduce",
                "bytes": 2048,
                "group_ranks": [0, 1],
            },
        },
        {"type": "slot_end", "timestamp_ms": 2.0, "metadata": {"slot": "fwd"}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        trace = _make_trace(events, rank=0, world_size=2, pipeline_stage=0)
        _write_trace(trace, traces_dir, rank=0)
        workload = _make_workload(tp=2, pp=1, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        ar_nodes = [n for n in dag.comm_nodes if n.collective_type == "AllReduce"]
        assert len(ar_nodes) > 0
        for n in ar_nodes:
            assert n.bytes == 1024
            assert n.collective_type == "AllReduce"


def _has_path(edges: list[DAGEdge], src: int, dst: int) -> bool:
    adj: dict[int, list[int]] = {}
    for e in edges:
        adj.setdefault(e.src_node_id, []).append(e.dst_node_id)
    visited: set[int] = set()
    stack: list[int] = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node not in visited:
            visited.add(node)
            stack.extend(adj.get(node, []))
    return False


def test_dag_edges_wired_sequentially():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {
            "type": "collective",
            "timestamp_ms": 1.0,
            "metadata": {
                "collective_type": "AllReduce",
                "bytes": 2048,
                "group_ranks": [0, 1],
            },
        },
        {"type": "slot_end", "timestamp_ms": 2.0, "metadata": {"slot": "fwd"}},
        {"type": "slot_begin", "timestamp_ms": 3.0, "metadata": {"slot": "bwd"}},
        {"type": "slot_end", "timestamp_ms": 4.0, "metadata": {"slot": "bwd"}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        trace = _make_trace(events, rank=0, world_size=2, pipeline_stage=0)
        _write_trace(trace, traces_dir, rank=0)
        workload = _make_workload(tp=2, pp=1, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        for rank in {n.gpu_rank for n in dag.compute_nodes}:
            rank_cnodes = sorted(
                [n for n in dag.compute_nodes if n.gpu_rank == rank],
                key=lambda n: n.node_id,
            )
            for i in range(len(rank_cnodes) - 1):
                assert _has_path(
                    dag.edges,
                    rank_cnodes[i].node_id,
                    rank_cnodes[i + 1].node_id,
                ), f"No path between compute nodes on rank {rank}"


def test_trace_with_multiple_pp_stages():
    events0 = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {"type": "slot_end", "timestamp_ms": 1.0, "metadata": {"slot": "fwd"}},
    ]
    events1 = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {"type": "slot_end", "timestamp_ms": 1.0, "metadata": {"slot": "fwd"}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        _write_trace(
            _make_trace(events0, rank=0, world_size=2, pipeline_stage=0),
            traces_dir,
            rank=0,
        )
        _write_trace(
            _make_trace(events1, rank=1, world_size=2, pipeline_stage=1),
            traces_dir,
            rank=1,
        )
        workload = _make_workload(tp=1, pp=2, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        stages = {n.pipeline_stage for n in dag.compute_nodes}
        assert 0 in stages
        assert 1 in stages


def test_trace_skips_pp_send_events():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {
            "type": "collective",
            "timestamp_ms": 1.0,
            "metadata": {
                "collective_type": "PP_Send",
                "bytes": 1024,
                "group_ranks": [0, 1],
            },
        },
        {"type": "slot_end", "timestamp_ms": 2.0, "metadata": {"slot": "fwd"}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        trace = _make_trace(events, rank=0, world_size=2, pipeline_stage=0)
        _write_trace(trace, traces_dir, rank=0)
        workload = _make_workload(tp=1, pp=1, num_gpus=1)
        dc = _make_datacenter(num_gpus=1, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        pp_send_nodes = [n for n in dag.comm_nodes if n.collective_type == "PP_Send"]
        assert len(pp_send_nodes) == 0


def test_slot_markers_assign_microbatch_and_phase():
    events = [
        {
            "type": "collective",
            "timestamp_ms": 0.0,
            "metadata": {
                "collective_type": "AllReduce",
                "bytes": 2048,
                "group_ranks": [0, 1],
            },
        },
        {
            "type": "slot_begin",
            "timestamp_ms": 1.0,
            "metadata": {"microbatch_id": 3, "phase": "fwd"},
        },
        {
            "type": "collective",
            "timestamp_ms": 2.0,
            "metadata": {
                "collective_type": "AllReduce",
                "bytes": 2048,
                "group_ranks": [0, 1],
            },
        },
        {"type": "slot_end", "timestamp_ms": 3.0, "metadata": {}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        trace = _make_trace(events, rank=0, world_size=2, pipeline_stage=0)
        _write_trace(trace, traces_dir, rank=0)
        workload = _make_workload(tp=2, pp=1, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        for rank in {n.gpu_rank for n in dag.compute_nodes}:
            rank_cnodes = sorted(
                [n for n in dag.compute_nodes if n.gpu_rank == rank],
                key=lambda n: n.node_id,
            )
            assert rank_cnodes[0].microbatch_id == -1
            assert rank_cnodes[0].phase == ""
            for cn in rank_cnodes[1:]:
                assert cn.microbatch_id == 3
                assert cn.phase == "fwd"


def test_missing_middle_trace_reused():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"microbatch_id": 0, "phase": "fwd"}},
        {"type": "slot_end", "timestamp_ms": 1.0, "metadata": {}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        _write_trace(_make_trace(events, rank=0, world_size=4, pipeline_stage=0), traces_dir, rank=0)
        _write_trace(_make_trace(events, rank=0, world_size=4, pipeline_stage=2), traces_dir, rank=2)
        _write_trace(_make_trace(events, rank=0, world_size=4, pipeline_stage=3), traces_dir, rank=3)
        workload = _make_workload(tp=1, pp=4, num_gpus=4)
        dc = _make_datacenter(num_gpus=4, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        stages = {n.pipeline_stage for n in dag.compute_nodes}
        assert stages == {0, 1, 2, 3}, f"Expected all 4 stages, got {stages}"


def test_pp_send_has_activation_bytes():
    events0 = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"microbatch_id": 0, "phase": "fwd"}},
        {
            "type": "collective",
            "timestamp_ms": 0.5,
            "metadata": {
                "collective_type": "PP_Send",
                "bytes": 2048,
                "group_ranks": [0, 1],
                "microbatch_id": 0,
                "direction": "fwd",
            },
        },
        {"type": "slot_end", "timestamp_ms": 1.0, "metadata": {}},
    ]
    events1 = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"microbatch_id": 0, "phase": "fwd"}},
        {"type": "slot_end", "timestamp_ms": 1.0, "metadata": {}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        _write_trace(_make_trace(events0, rank=0, world_size=2, pipeline_stage=0), traces_dir, rank=0)
        _write_trace(_make_trace(events1, rank=0, world_size=2, pipeline_stage=1), traces_dir, rank=1)
        workload = MegatronWorkload(
            framework="megatron",
            config={
                "tensor-model-parallel-size": 1,
                "pipeline-model-parallel-size": 2,
                "num-layers": 2,
                "hidden-size": 512,
                "num-attention-heads": 8,
                "ffn-hidden-size": 11008,
                "seq-length": 128,
                "micro-batch-size": 1,
                "global-batch-size": 2,
                "num_gpus": 2,
            },
        )
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        pp_sends = [n for n in dag.comm_nodes if n.collective_type == "PP_Send"]
        assert len(pp_sends) > 0
        for n in pp_sends:
            assert n.bytes > 0, f"PP_Send must have non-zero bytes, got {n.bytes}"


def test_missing_first_trace_raises():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"microbatch_id": 0, "phase": "fwd"}},
        {"type": "slot_end", "timestamp_ms": 1.0, "metadata": {}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        _write_trace(_make_trace(events, rank=0, world_size=2, pipeline_stage=1), traces_dir, rank=1)
        workload = _make_workload(tp=1, pp=2, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        with pytest.raises(ValueError, match="First PP stage"):
            _run_tracer(workload, dc)


def test_missing_last_trace_raises():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"microbatch_id": 0, "phase": "fwd"}},
        {"type": "slot_end", "timestamp_ms": 1.0, "metadata": {}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        _write_trace(_make_trace(events, rank=0, world_size=2, pipeline_stage=0), traces_dir, rank=0)
        workload = _make_workload(tp=1, pp=2, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        with pytest.raises(ValueError, match="Last PP stage"):
            _run_tracer(workload, dc)


def test_remap_dp_allreduce_to_target_dp_group():
    config = ParallelConfig(tp=2, cp=1, pp=1, ep=1, dp=2, num_gpus=4)
    events = [{
        "type": "collective", "timestamp_ms": 0.0,
        "metadata": {
            "collective_type": "AllReduce", "bytes": 2048,
            "group_ranks": [0, 2],
        },
    }]
    trace_dict = {"trace_format_version": "1.0", "rank": 0, "world_size": 4, "pipeline_stage": 0, "events": events}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "trace_rank_0.json"
        p.write_text(json.dumps(trace_dict))
        tf = TraceFileParser.parse(str(p))
    remapped = _remap_collectives(tf, from_rank=0, to_rank=1, config=config)
    assert len(remapped.events) == 1
    new_group = remapped.events[0].metadata["group_ranks"]
    assert set(new_group) == {1, 3}, f"Expected DP group [1,3] for rank 1, got {new_group}"


def test_remap_tp_allreduce_to_target_tp_group():
    config = ParallelConfig(tp=4, cp=1, pp=1, ep=1, dp=1, num_gpus=4)
    events = [{
        "type": "collective", "timestamp_ms": 0.0,
        "metadata": {
            "collective_type": "AllReduce", "bytes": 2048,
            "group_ranks": [0, 1, 2, 3],
        },
    }]
    trace_dict = {"trace_format_version": "1.0", "rank": 0, "world_size": 4, "pipeline_stage": 0, "events": events}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "trace_rank_0.json"
        p.write_text(json.dumps(trace_dict))
        tf = TraceFileParser.parse(str(p))
    remapped = _remap_collectives(tf, from_rank=0, to_rank=0, config=config)
    assert remapped.events[0].metadata["group_ranks"] == [0, 1, 2, 3]


def test_derived_trace_from_same_stage():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"microbatch_id": 0, "phase": "fwd"}},
        {"type": "slot_end", "timestamp_ms": 1.0, "metadata": {}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        _write_trace(_make_trace(events, rank=0, world_size=4, pipeline_stage=0), traces_dir, rank=0)
        _write_trace(_make_trace(events, rank=3, world_size=4, pipeline_stage=3), traces_dir, rank=3)
        workload = MegatronWorkload(
            framework="megatron",
            config={"tensor-model-parallel-size": 1, "pipeline-model-parallel-size": 4, "num-layers": 2,
                    "hidden-size": 512, "num-attention-heads": 8, "ffn-hidden-size": 11008,
                    "seq-length": 128, "micro-batch-size": 1, "global-batch-size": 4, "num_gpus": 4},
        )
        dc = _make_datacenter(num_gpus=4, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)
        stages = {n.pipeline_stage for n in dag.compute_nodes}
        assert stages == {0, 1, 2, 3}


def test_collective_node_comes_before_compute_after_it():
    """
    For a non-PP collective, the correct DAG order is:
    compute(gap_before) -> collective -> compute(gap_after)
    The buggy order was: compute(gap_before) -> compute(gap_after) -> collective
    """
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {
            "type": "collective",
            "timestamp_ms": 1.0,
            "metadata": {
                "collective_type": "AllReduce",
                "bytes": 2048,
                "group_ranks": [0, 1],
            },
        },
        {"type": "slot_end", "timestamp_ms": 2.0, "metadata": {"slot": "fwd"}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        trace = _make_trace(events, rank=0, world_size=2, pipeline_stage=0)
        _write_trace(trace, traces_dir, rank=0)
        workload = _make_workload(tp=2, pp=1, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)

        rank0_compute = sorted(
            [n for n in dag.compute_nodes if n.gpu_rank == 0],
            key=lambda n: n.node_id,
        )
        assert len(rank0_compute) == 2, f"Expected 2 compute nodes on rank 0, got {len(rank0_compute)}"
        c_before = rank0_compute[0]
        c_after = rank0_compute[1]

        rank0_comm = [n for n in dag.comm_nodes if n.src_gpu == 0 or n.dst_gpu == 0]
        assert len(rank0_comm) > 0, "Expected at least one comm node on rank 0"
        collective_node = rank0_comm[0]

        assert _has_path(
            dag.edges, c_before.node_id, collective_node.node_id
        ), f"Expected edge from compute before ({c_before.node_id}) to collective ({collective_node.node_id})"
        assert _has_path(
            dag.edges, collective_node.node_id, c_after.node_id
        ), f"Expected edge from collective ({collective_node.node_id}) to compute after ({c_after.node_id})"


def test_unknown_collective_kept_unchanged():
    config = ParallelConfig(tp=2, cp=1, pp=1, ep=1, dp=2, num_gpus=4)
    events = [{
        "type": "collective", "timestamp_ms": 0.0,
        "metadata": {
            "collective_type": "SomeWeirdCollective", "bytes": 1024,
            "group_ranks": [0, 1, 2, 3],
        },
    }]
    trace_dict = {"trace_format_version": "1.0", "rank": 0, "world_size": 4, "pipeline_stage": 0, "events": events}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "trace_rank_0.json"
        p.write_text(json.dumps(trace_dict))
        tf = TraceFileParser.parse(str(p))
    remapped = _remap_collectives(tf, from_rank=0, to_rank=1, config=config)
    assert remapped.events[0].metadata["group_ranks"] == [0, 1, 2, 3]


def test_pp_send_preserved_in_remap():
    config = ParallelConfig(tp=2, cp=1, pp=1, ep=1, dp=2, num_gpus=4)
    events = [{
        "type": "collective", "timestamp_ms": 0.0,
        "metadata": {
            "collective_type": "PP_Send", "bytes": 1024,
            "group_ranks": [0, 2], "direction": "fwd",
        },
    }]
    trace_dict = {"trace_format_version": "1.0", "rank": 0, "world_size": 4, "pipeline_stage": 0, "events": events}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "trace_rank_0.json"
        p.write_text(json.dumps(trace_dict))
        tf = TraceFileParser.parse(str(p))
    remapped = _remap_collectives(tf, from_rank=0, to_rank=1, config=config)
    assert remapped.events[0].metadata["group_ranks"] == [1, 3]


def test_multi_rank_collective_dedup():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {
            "type": "collective",
            "timestamp_ms": 1.0,
            "metadata": {
                "collective_type": "AllReduce",
                "bytes": 2048,
                "group_ranks": [0, 1],
            },
        },
        {"type": "slot_end", "timestamp_ms": 2.0, "metadata": {"slot": "fwd"}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        _write_trace(_make_trace(events, rank=0, world_size=2, pipeline_stage=0), traces_dir, rank=0)
        _write_trace(_make_trace(events, rank=1, world_size=2, pipeline_stage=0), traces_dir, rank=1)
        workload = _make_workload(tp=2, pp=1, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)

        expected_result, _ = decompose_collective(
            collective_type="AllReduce",
            group_ranks=[0, 1],
            data_size=2048,
            num_channels=1,
            algorithm="ring",
            flow_id_start=0,
        )
        expected_n = len(expected_result.flows)

        ar_nodes = [n for n in dag.comm_nodes if n.collective_type == "AllReduce"]
        assert len(ar_nodes) == expected_n, (
            f"Expected {expected_n} AllReduce P2P nodes, got {len(ar_nodes)}"
        )


def test_collective_decomposition_edge_rewiring():
    events = [
        {"type": "slot_begin", "timestamp_ms": 0.0, "metadata": {"slot": "fwd"}},
        {
            "type": "collective",
            "timestamp_ms": 1.0,
            "metadata": {
                "collective_type": "AllReduce",
                "bytes": 2048,
                "group_ranks": [0, 1],
            },
        },
        {"type": "slot_end", "timestamp_ms": 2.0, "metadata": {"slot": "fwd"}},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        traces_dir = Path(tmp_dir)
        trace = _make_trace(events, rank=0, world_size=2, pipeline_stage=0)
        _write_trace(trace, traces_dir, rank=0)
        _write_trace(_make_trace(events, rank=1, world_size=2, pipeline_stage=0), traces_dir, rank=1)
        workload = _make_workload(tp=2, pp=1, num_gpus=2)
        dc = _make_datacenter(num_gpus=2, traces_dir=str(traces_dir))
        dag = _run_tracer(workload, dc)

        rank0_compute = sorted(
            [n for n in dag.compute_nodes if n.gpu_rank == 0],
            key=lambda n: n.node_id,
        )
        assert len(rank0_compute) == 2, f"Expected 2 compute nodes on rank 0, got {len(rank0_compute)}"
        c_before = rank0_compute[0]
        c_after = rank0_compute[1]

        rank0_comm = [n for n in dag.comm_nodes if n.src_gpu == 0 or n.dst_gpu == 0]
        assert len(rank0_comm) > 0, "Expected at least one CommNode touching rank 0"
        rank0_comm_sorted = sorted(rank0_comm, key=lambda n: n.node_id)
        first_p2p = rank0_comm_sorted[0]
        last_p2p = rank0_comm_sorted[-1]

        assert _has_path(
            dag.edges, c_before.node_id, first_p2p.node_id
        ), f"Expected path from compute before ({c_before.node_id}) to first P2P ({first_p2p.node_id})"
        assert _has_path(
            dag.edges, last_p2p.node_id, c_after.node_id
        ), f"Expected path from last P2P ({last_p2p.node_id}) to compute after ({c_after.node_id})"
