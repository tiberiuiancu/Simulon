"""Integration tests for multi-workload simulation."""

import pytest

from simulon.backend.analytical import AnalyticalBackend
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
from simulon.config.scenario import ScenarioConfig, StartConfig, WorkloadInstance
from simulon.config.workload import (
    LLMSpec,
    MegatronParallelism,
    MegatronTraining,
    MegatronWorkload,
)


def _make_datacenter() -> DatacenterConfig:
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


def _make_workload(
    *,
    tp: int = 2,
    pp: int = 2,
    num_gpus: int = 4,
    num_layers: int = 2,
    hidden_size: int = 256,
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
            global_batch_size=4,
            micro_batch_size=1,
            sequence_length=128,
            dtype=DType.bf16,
        ),
    )


def _make_single_scenario(workload: MegatronWorkload) -> ScenarioConfig:
    return ScenarioConfig(datacenter=_make_datacenter(), workload=workload)


def test_multi_workload_concurrent():
    dc = _make_datacenter()
    wl_a = _make_workload()
    wl_b = _make_workload()

    scenario = ScenarioConfig(
        datacenter=dc,
        workloads=[
            WorkloadInstance(name="A", workload=wl_a, start=StartConfig(offset_ms=0.0)),
            WorkloadInstance(name="B", workload=wl_b, start=StartConfig(offset_ms=0.0)),
        ],
    )

    backend = AnalyticalBackend()
    output = backend.simulate(scenario, ignore_missing=True)

    assert output.dag is not None
    assert output.result is not None
    assert len(output.by_workload) == 2
    assert "A" in output.by_workload
    assert "B" in output.by_workload

    a_time = output.by_workload["A"].total_time_ms
    b_time = output.by_workload["B"].total_time_ms
    assert a_time > 0
    assert b_time > 0

    agg_time = output.result.total_time_ms
    assert abs(agg_time - max(a_time, b_time)) < 1.0

    dag_a = backend.run_trace(_make_single_scenario(wl_a))
    dag_b = backend.run_trace(_make_single_scenario(wl_b))
    individual_nodes = len(dag_a.compute_nodes) + len(dag_a.comm_nodes)
    individual_nodes += len(dag_b.compute_nodes) + len(dag_b.comm_nodes)
    merged_nodes = len(output.dag.compute_nodes) + len(output.dag.comm_nodes)
    assert merged_nodes == individual_nodes


def test_multi_workload_sequential():
    dc = _make_datacenter()
    wl_a = _make_workload()
    wl_b = _make_workload()

    scenario = ScenarioConfig(
        datacenter=dc,
        workloads=[
            WorkloadInstance(name="A", workload=wl_a, start=StartConfig(offset_ms=0.0)),
            WorkloadInstance(
                name="B", workload=wl_b, start=StartConfig(offset_ms=0.0, after_finish=["A"])
            ),
        ],
    )

    backend = AnalyticalBackend()
    output = backend.simulate(scenario, ignore_missing=True)

    assert len(output.by_workload) == 2
    a_time = output.by_workload["A"].total_time_ms
    b_time = output.by_workload["B"].total_time_ms
    assert a_time > 0
    assert b_time > 0

    b_offset = output.start_offsets.get(4, 0.0)
    assert abs(b_offset - a_time) < 1.0

    agg_time = output.result.total_time_ms
    assert abs(agg_time - (a_time + b_time)) < 1.0


def test_multi_workload_output_is_iterable():
    dc = _make_datacenter()
    wl = _make_workload()

    scenario = ScenarioConfig(
        datacenter=dc,
        workloads=[
            WorkloadInstance(name="A", workload=wl, start=StartConfig(offset_ms=0.0)),
            WorkloadInstance(name="B", workload=wl, start=StartConfig(offset_ms=0.0)),
        ],
    )

    backend = AnalyticalBackend()
    output = backend.simulate(scenario, ignore_missing=True)

    dag, result = output
    assert dag is output.dag
    assert result is output.result
    assert output[0] is dag
    assert output[1] is result
