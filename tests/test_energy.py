import pytest

from simulon.backend.dag.nodes import CommNode, ComputeNode, ExecutionDAG
from simulon.config.common import LinearPowerModel
from simulon.config.dc import DatacenterConfig, DatacenterMeta, GPUSpec, NodeSpec
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import MegatronWorkload
from simulon.energy import EnergyResult, compute_energy


def _dc_with_power_model() -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test", pue=1.1),
        num_nodes=2,
        node=NodeSpec(
            gpus_per_node=2,
            gpu=GPUSpec(
                name="test-gpu",
                power_model=LinearPowerModel(tdp_w=700.0, idle_power_w=67.0),
            ),
        ),
    )


def _scenario(dc: DatacenterConfig) -> ScenarioConfig:
    return ScenarioConfig(
        datacenter=dc,
        workload=MegatronWorkload(
            framework="megatron",
            config={},
        ),
    )


def _dag_with_nodes() -> ExecutionDAG:
    dag = ExecutionDAG()
    dag.compute_nodes.append(
        ComputeNode(
            node_id=0,
            gpu_rank=0,
            kernel="attn_flash",
            layer_id=0,
            microbatch_id=0,
            pipeline_stage=0,
            phase="fwd",
            start_ms=0.0,
            finish_ms=100.0,
        )
    )
    dag.comm_nodes.append(
        CommNode(
            node_id=1,
            src_gpu=0,
            dst_gpu=1,
            bytes=1024,
            collective_type="AllReduce",
            layer_id=0,
            phase="fwd",
            flow_id=0,
            start_ms=100.0,
            finish_ms=150.0,
        )
    )
    return dag


def test_measured_mode_with_codecarbon_data():
    dag = _dag_with_nodes()
    dag.energy_kwh = 0.5
    dag.co2eq_kg = 0.05

    dc = _dc_with_power_model()
    sc = _scenario(dc)

    result = compute_energy(dag, sc)
    assert isinstance(result, EnergyResult)
    assert result.source == "measured"
    assert result.total_wh == pytest.approx(500.0)
    assert result.co2eq_g == pytest.approx(50.0)
    assert result.hardware_subtotal_wh == pytest.approx(500.0)
    assert result.pue_overhead_wh == pytest.approx(0.0)
    assert result.breakdown[0].component == "measured_energy"
    assert result.breakdown[0].pct == 100.0


def test_fallback_mode_without_codecarbon_data():
    dag = _dag_with_nodes()

    dc = _dc_with_power_model()
    sc = _scenario(dc)

    result = compute_energy(dag, sc)
    assert isinstance(result, EnergyResult)
    assert result.source == "estimated"
    assert result.co2eq_g is None
    assert result.total_wh > 0
    assert result.hardware_subtotal_wh > 0
    assert result.pue_overhead_wh >= 0
    assert any(comp.component == "gpu" for comp in result.breakdown)


def test_no_power_model_returns_none():
    dag = _dag_with_nodes()
    dc = DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        num_nodes=2,
        node=NodeSpec(gpus_per_node=2, gpu=GPUSpec(name="test-gpu")),
    )
    sc = _scenario(dc)

    result = compute_energy(dag, sc)
    assert result is None
