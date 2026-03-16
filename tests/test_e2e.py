"""End-to-end tests for the DAG tracer."""

import pytest

from simulon.backend.dag import DAGTracer, DAGTracerConfig, ExecutionDAG
from simulon.backend.astra_sim import AstraSimBackend
from simulon.config.common import DType
from simulon.config.dc import (
    ClusterSpec,
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    NICSpec,
    NetworkSpec,
    NodeSpec,
    ScaleOutSpec,
    ScaleUpSpec,
    SwitchSpec,
    TopologySpec,
    TopologyType,
)
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import (
    LLMSpec,
    MegatronParallelism,
    MegatronTraining,
    MegatronWorkload,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_datacenter() -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test_cluster"),
        cluster=ClusterSpec(num_nodes=2),
        node=NodeSpec(
            gpus_per_node=4,
            gpu=GPUSpec(name="H100", memory_capacity_gb=80.0),
        ),
        network=NetworkSpec(
            scale_up=ScaleUpSpec(
                switch=SwitchSpec(port_speed="2880Gbps", latency="0.000025ms"),
            ),
            scale_out=ScaleOutSpec(
                nic=NICSpec(speed="400Gbps", latency="0.005ms"),
                topology=TopologySpec(type=TopologyType.fat_tree, params={"k": 4}),
            ),
        ),
    )


def make_workload(
    tp: int = 1,
    pp: int = 1,
    num_gpus: int = 4,
    num_layers: int = 2,
    hidden_size: int = 512,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    seq_len: int = 128,
) -> MegatronWorkload:
    return MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name="test-model",
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=8,
            vocab_size=32000,
        ),
        parallelism=MegatronParallelism(tp=tp, pp=pp),
        training=MegatronTraining(
            num_gpus=num_gpus,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            sequence_length=seq_len,
            dtype=DType.bf16,
        ),
    )


@pytest.fixture
def simple_scenario():
    dc = make_datacenter()
    wl = make_workload(tp=1, pp=1, num_gpus=4)
    return ScenarioConfig(datacenter=dc, workload=wl)


# ---------------------------------------------------------------------------
# DAGTracer basic tests
# ---------------------------------------------------------------------------


def test_dag_tracer_returns_execution_dag(simple_scenario):
    """DAGTracer.trace() returns an ExecutionDAG instance."""
    tracer = DAGTracer()
    dag = tracer.trace(simple_scenario.workload, simple_scenario.datacenter)
    assert isinstance(dag, ExecutionDAG)


def test_dag_tracer_has_compute_nodes(simple_scenario):
    """DAG has compute nodes."""
    tracer = DAGTracer()
    dag = tracer.trace(simple_scenario.workload, simple_scenario.datacenter)
    assert len(dag.compute_nodes) > 0


def test_dag_tracer_tp1_no_comm_stubs(simple_scenario):
    """With tp=1, no AllGather/ReduceScatter comm nodes are generated."""
    tracer = DAGTracer()
    dag = tracer.trace(simple_scenario.workload, simple_scenario.datacenter)
    tp_collectives = [
        n for n in dag.comm_nodes
        if n.collective_type in ("AllGather", "ReduceScatter")
    ]
    assert tp_collectives == []


def test_dag_tracer_tp2_generates_comm_nodes():
    """With tp=2, AllGather and ReduceScatter comm nodes are generated."""
    wl = make_workload(tp=2, pp=1, num_gpus=4, num_layers=1)
    dc = make_datacenter()
    tracer = DAGTracer()
    dag = tracer.trace(wl, dc)

    ag_nodes = [n for n in dag.comm_nodes if n.collective_type == "AllGather"]
    rs_nodes = [n for n in dag.comm_nodes if n.collective_type == "ReduceScatter"]
    assert len(ag_nodes) > 0
    assert len(rs_nodes) > 0


def test_dag_tracer_pp2_generates_pp_sends():
    """With pp=2, PP_Send comm nodes are generated at stage boundaries."""
    wl = make_workload(tp=1, pp=2, num_gpus=4, num_layers=1, global_batch_size=4)
    dc = make_datacenter()
    tracer = DAGTracer()
    dag = tracer.trace(wl, dc)

    pp_sends = [n for n in dag.comm_nodes if n.collective_type == "PP_Send"]
    assert len(pp_sends) > 0


def test_dag_to_json_is_valid_json(simple_scenario):
    """ExecutionDAG.to_json() returns valid JSON."""
    import json
    tracer = DAGTracer()
    dag = tracer.trace(simple_scenario.workload, simple_scenario.datacenter)
    json_str = dag.to_json()
    data = json.loads(json_str)
    assert "compute_nodes" in data
    assert "comm_nodes" in data
    assert "edges" in data


def test_dag_to_dict_structure(simple_scenario):
    """ExecutionDAG.to_dict() has correct keys."""
    tracer = DAGTracer()
    dag = tracer.trace(simple_scenario.workload, simple_scenario.datacenter)
    d = dag.to_dict()
    assert isinstance(d["compute_nodes"], list)
    assert isinstance(d["comm_nodes"], list)
    assert isinstance(d["edges"], list)


def test_compute_node_fields(simple_scenario):
    """ComputeNodes have expected fields."""
    tracer = DAGTracer()
    dag = tracer.trace(simple_scenario.workload, simple_scenario.datacenter)
    cn = dag.compute_nodes[0]
    assert hasattr(cn, "node_id")
    assert hasattr(cn, "gpu_rank")
    assert hasattr(cn, "kernel")
    assert hasattr(cn, "layer_id")
    assert hasattr(cn, "microbatch_id")
    assert hasattr(cn, "pipeline_stage")
    assert hasattr(cn, "phase")


def test_flow_ids_nonnegative():
    """All flow_ids in comm nodes (from actual flows) are >= 0."""
    wl = make_workload(tp=2, pp=1, num_gpus=4, num_layers=1)
    dc = make_datacenter()
    tracer = DAGTracer()
    dag = tracer.trace(wl, dc)
    for n in dag.comm_nodes:
        assert n.flow_id >= 0 or n.flow_id == -1, f"Unexpected flow_id={n.flow_id}"
    # PP_Send nodes get real flow_ids
    actual = [n for n in dag.comm_nodes if n.collective_type != "PP_Send"]
    for n in actual:
        assert n.flow_id >= 0, f"comm node has flow_id={n.flow_id}"


# ---------------------------------------------------------------------------
# AstraSimBackend tests
# ---------------------------------------------------------------------------


def test_astra_sim_backend_run(simple_scenario):
    """AstraSimBackend.run() returns a dict with expected keys."""
    backend = AstraSimBackend()
    result = backend.run(simple_scenario)
    assert result["status"] == "success"
    assert "compute_nodes" in result
    assert "comm_nodes" in result
    assert "edges" in result
    assert "dag" in result


def test_astra_sim_backend_run_trace(simple_scenario):
    """AstraSimBackend.run_trace() returns an ExecutionDAG."""
    backend = AstraSimBackend()
    dag = backend.run_trace(simple_scenario)
    assert isinstance(dag, ExecutionDAG)


def test_astra_sim_backend_rejects_non_megatron():
    """AstraSimBackend raises ValueError for non-MegatronWorkload."""
    from simulon.config.workload import InferenceWorkload, InferenceParallelism, InferenceRun
    dc = make_datacenter()
    wl = InferenceWorkload(
        framework="inference",
        model=LLMSpec(name="test", hidden_size=512, num_layers=2, num_heads=8),
        parallelism=InferenceParallelism(),
        inference=InferenceRun(num_gpus=4, batch_size=1, seq_length=128),
    )
    sc = ScenarioConfig(datacenter=dc, workload=wl)
    backend = AstraSimBackend()
    with pytest.raises(ValueError, match="AstraSimBackend only supports MegatronWorkload"):
        backend.run_trace(sc)


def test_astra_sim_num_channels():
    """num_channels=2 produces more flows than num_channels=1."""
    wl = make_workload(tp=2, pp=1, num_gpus=4, num_layers=1)
    dc = make_datacenter()
    sc = ScenarioConfig(datacenter=dc, workload=wl)

    dag1 = AstraSimBackend(num_channels=1).run_trace(sc)
    dag2 = AstraSimBackend(num_channels=2).run_trace(sc)
    assert len(dag2.comm_nodes) > len(dag1.comm_nodes)
