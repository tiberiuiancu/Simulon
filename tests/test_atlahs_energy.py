"""Energy computation tests for ATLAHS backends.

Tests the compute_energy() function with non-replayed DAGs (start_ms=None)
where total_time_ms is passed explicitly — the typical ATLAHS backend scenario.
Also verifies backward compatibility with replayed DAGs using finish_ms.

Key scenarios:
  - Non-replayed DAG (start_ms=None) + explicit total_time_ms → utilisation from duration_ms
  - LinearPowerModel should NOT compute at idle power when total_time_ms is provided
  - ConstantPowerModel draws full TDP regardless of utilisation
  - Backward compatibility: compute_energy(dag, sc) without total_time_ms still works
  - EnergyResult output structure matches analytical backend expectations
"""
from __future__ import annotations

import pytest

from simulon.backend.dag.nodes import CommNode, ComputeNode, ExecutionDAG
from simulon.config.common import ConstantPowerModel, LinearPowerModel, DType
from simulon.config.dc import (
    ClusterSpec,
    CPUSpec,
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    NodeSpec,
)
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import (
    LLMSpec,
    MegatronParallelism,
    MegatronTraining,
    MegatronWorkload,
)
from simulon.energy import compute_energy, EnergyResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MS_PER_HOUR = 3_600_000


def _make_workload() -> MegatronWorkload:
    """Minimal MegatronWorkload that satisfies ScenarioConfig validation."""
    return MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name="test-model",
            hidden_size=1024,
            num_layers=2,
            num_heads=8,
            vocab_size=32000,
            ffn_hidden_size=4096,
        ),
        parallelism=MegatronParallelism(tp=1, pp=1, ep=1, dp=1),
        training=MegatronTraining(
            num_gpus=1,
            global_batch_size=1,
            micro_batch_size=1,
            sequence_length=512,
            dtype=DType.bf16,
        ),
    )


def make_datacenter(
    *,
    gpu_power_model=None,
    pue: float = 1.0,
    num_nodes: int = 1,
    gpus_per_node: int = 1,
) -> DatacenterConfig:
    """Build a minimal DatacenterConfig for energy tests."""
    gpu = GPUSpec(
        name="TestGPU",
        power_model=gpu_power_model,
    )
    return DatacenterConfig(
        datacenter=DatacenterMeta(pue=pue),
        cluster=ClusterSpec(num_nodes=num_nodes),
        node=NodeSpec(
            gpus_per_node=gpus_per_node,
            gpu=gpu,
        ),
    )


def make_scenario(dc: DatacenterConfig) -> ScenarioConfig:
    """Wrap a DatacenterConfig into a ScenarioConfig."""
    return ScenarioConfig(datacenter=dc, workload=_make_workload())


def _non_replayed_node(
    gpu_rank: int,
    duration_ms: float,
    node_id: int = 0,
) -> ComputeNode:
    """ComputeNode with start_ms=None — as produced by ATLAHS backends (no replay)."""
    return ComputeNode(
        node_id=node_id,
        gpu_rank=gpu_rank,
        kernel="matmul",
        layer_id=0,
        microbatch_id=0,
        pipeline_stage=0,
        phase="fwd",
        duration_ms=duration_ms,
        start_ms=None,
    )


def _replayed_node(
    gpu_rank: int,
    start_ms: float,
    finish_ms: float,
    node_id: int = 0,
) -> ComputeNode:
    """ComputeNode with replay timing — as produced by AnalyticalBackend replay."""
    return ComputeNode(
        node_id=node_id,
        gpu_rank=gpu_rank,
        kernel="matmul",
        layer_id=0,
        microbatch_id=0,
        pipeline_stage=0,
        phase="fwd",
        start_ms=start_ms,
        finish_ms=finish_ms,
    )


# ---------------------------------------------------------------------------
# Tests: total_time_ms parameter with non-replayed DAGs (ATLAHS scenario)
# ---------------------------------------------------------------------------


class TestTotalTimeMsParameter:
    """compute_energy(dag, sc, total_time_ms=N) with non-replayed DAGs."""

    def test_explicit_total_time_ms_used_as_wall_clock(self):
        """total_time_ms parameter is used as the wall-clock time, not derived from DAG."""
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=3_600_000.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=100.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        # run_duration_hours = 1000 / 3_600_000
        assert result.run_duration_hours == pytest.approx(1000.0 / _MS_PER_HOUR)
        # 3_600_000 W * 1 GPU * 1000 ms / 3_600_000 ms_per_hour = 1000 Wh
        assert result.total_wh == pytest.approx(1000.0)

    def test_utilisation_from_duration_ms_over_total_time_ms(self):
        """With start_ms=None, utilisation = sum(duration_ms) / num_gpus / total_time_ms."""
        dc = make_datacenter(
            gpu_power_model=LinearPowerModel(tdp_w=700.0, idle_power_w=200.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        # 500 ms compute out of 1000 ms total → utilisation = 0.5
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        # power = 200 + 0.5 * (700 - 200) = 450 W
        # energy = 450 W * 1 GPU * 1000/3600000 h
        gpu_comp = next(c for c in result.breakdown if c.component == "gpu")
        expected_wh = 450.0 * 1000.0 / _MS_PER_HOUR
        assert gpu_comp.wh == pytest.approx(expected_wh, rel=1e-9)

    def test_utilisation_positive_when_duration_ms_present(self):
        """With non-replayed DAG and total_time_ms, utilisation > 0 even for small compute."""
        dc = make_datacenter(
            gpu_power_model=LinearPowerModel(tdp_w=700.0, idle_power_w=200.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        # 1 ms compute out of 1000 ms → utilisation = 0.001
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=1.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        gpu_comp = next(c for c in result.breakdown if c.component == "gpu")
        # At utilisation=0.001, power = 200 + 0.001*500 = 200.5 W
        # At idle (utilisation=0), power = 200 W
        idle_wh = 200.0 * 1000.0 / _MS_PER_HOUR
        assert gpu_comp.wh > idle_wh, (
            "GPU power should be above idle when duration_ms > 0"
        )

    def test_multiple_ranks_utilisation_averaged(self):
        """Utilisation is averaged over ALL cluster GPUs (num_nodes × gpus_per_node)."""
        dc = make_datacenter(
            gpu_power_model=LinearPowerModel(tdp_w=700.0, idle_power_w=200.0),
            pue=1.0,
            num_nodes=1,
            gpus_per_node=2,
        )
        sc = make_scenario(dc)
        # Rank 0: 200 ms compute, Rank 1: 800 ms compute, total_time=1000
        # avg_active_ms = (200 + 800) / 2 = 500
        # utilisation = 500 / 1000 = 0.5
        dag = ExecutionDAG(compute_nodes=[
            _non_replayed_node(0, duration_ms=200.0, node_id=0),
            _non_replayed_node(1, duration_ms=800.0, node_id=1),
        ])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        # power = 200 + 0.5 * (700 - 200) = 450 W
        # 2 GPUs × 450 W × 1000/3600000 h
        gpu_comp = next(c for c in result.breakdown if c.component == "gpu")
        expected_wh = 450.0 * 2 * 1000.0 / _MS_PER_HOUR
        assert gpu_comp.wh == pytest.approx(expected_wh, rel=1e-9)


# ---------------------------------------------------------------------------
# Tests: LinearPowerModel with non-replayed DAG — must NOT compute at idle
# ---------------------------------------------------------------------------


class TestLinearPowerModelNonReplayed:
    """LinearPowerModel should NOT compute at idle power when total_time_ms is provided."""

    def test_linear_power_model_not_at_idle_with_total_time_ms(self):
        """LinearPowerModel power should exceed idle when utilisation > 0."""
        dc = make_datacenter(
            gpu_power_model=LinearPowerModel(tdp_w=700.0, idle_power_w=200.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        gpu_comp = next(c for c in result.breakdown if c.component == "gpu")
        idle_only_wh = 200.0 * 1000.0 / _MS_PER_HOUR
        assert gpu_comp.wh > idle_only_wh, (
            "Linear model with duration > 0 must exceed idle-only energy"
        )

    def test_linear_power_half_utilisation(self):
        """LinearPowerModel at 50% utilisation produces correct midpoint power."""
        dc = make_datacenter(
            gpu_power_model=LinearPowerModel(tdp_w=600.0, idle_power_w=100.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        gpu_comp = next(c for c in result.breakdown if c.component == "gpu")
        # power = 100 + 0.5 * (600 - 100) = 350 W
        # energy = 350 W * 1 GPU * 1000 / 3600000 h
        expected_wh = 350.0 * 1000.0 / _MS_PER_HOUR
        assert gpu_comp.wh == pytest.approx(expected_wh, rel=1e-9)


# ---------------------------------------------------------------------------
# Tests: ConstantPowerModel with non-replayed DAG
# ---------------------------------------------------------------------------


class TestConstantPowerModelNonReplayed:
    """ConstantPowerModel draws full TDP regardless of utilisation."""

    def test_constant_power_with_non_replayed_dag(self):
        """ConstantPowerModel with non-replayed DAG draws full TDP."""
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=500.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=100.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        gpu_comp = next(c for c in result.breakdown if c.component == "gpu")
        # 500 W * 1 GPU * 1000 ms / 3600000 ms_per_hour
        expected_wh = 500.0 * 1000.0 / _MS_PER_HOUR
        assert gpu_comp.wh == pytest.approx(expected_wh, rel=1e-9)

    def test_constant_power_utilisation_irrelevant(self):
        """ConstantPowerModel produces same energy regardless of duration_ms."""
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=500.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        dag_short = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=1.0)])
        dag_long = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=999.0)])
        result_short = compute_energy(dag_short, sc, total_time_ms=1000.0)
        result_long = compute_energy(dag_long, sc, total_time_ms=1000.0)
        assert result_short is not None
        assert result_long is not None
        # Both should be the same since ConstantPowerModel ignores utilisation
        assert result_short.total_wh == pytest.approx(result_long.total_wh)


# ---------------------------------------------------------------------------
# Tests: backward compatibility — no total_time_ms
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """compute_energy(dag, sc) without total_time_ms still works for replayed DAGs."""

    def test_no_total_time_ms_derives_from_finish_ms(self):
        """Without total_time_ms, wall time derived from max finish_ms in DAG."""
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        node = _replayed_node(0, start_ms=0.0, finish_ms=_MS_PER_HOUR)
        dag = ExecutionDAG(compute_nodes=[node])
        result = compute_energy(dag, sc)
        assert result is not None
        # 700 W * 1 GPU * 1 h = 700 Wh
        assert result.total_wh == pytest.approx(700.0)

    def test_no_total_time_ms_uses_max_finish_across_comm(self):
        """total_time_ms defaults to max finish_ms across compute + comm nodes."""
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=_MS_PER_HOUR),
            pue=1.0,
        )
        sc = make_scenario(dc)
        compute_node = _replayed_node(0, start_ms=0.0, finish_ms=500_000.0, node_id=0)
        comm_node = CommNode(
            node_id=1,
            src_gpu=0,
            dst_gpu=0,
            bytes=0,
            collective_type="AllReduce",
            layer_id=0,
            phase="fwd",
            flow_id=0,
            start_ms=400_000.0,
            finish_ms=float(_MS_PER_HOUR),
        )
        dag = ExecutionDAG(compute_nodes=[compute_node], comm_nodes=[comm_node])
        result = compute_energy(dag, sc)
        assert result is not None
        # total_time = _MS_PER_HOUR = 1 h
        # _MS_PER_HOUR W * 1 GPU * 1 h = _MS_PER_HOUR Wh
        assert result.total_wh == pytest.approx(float(_MS_PER_HOUR))

    def test_no_total_time_ms_with_multi_gpu(self):
        """Multi-GPU replayed DAG without total_time_ms works correctly."""
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=100.0),
            pue=1.0,
            num_nodes=2,
            gpus_per_node=4,
        )
        sc = make_scenario(dc)
        nodes = [
            _replayed_node(rank, start_ms=0.0, finish_ms=_MS_PER_HOUR, node_id=rank)
            for rank in range(8)
        ]
        dag = ExecutionDAG(compute_nodes=nodes)
        result = compute_energy(dag, sc)
        assert result is not None
        # 100 W * 8 GPUs * 1 h = 800 Wh
        gpu_comp = next(c for c in result.breakdown if c.component == "gpu")
        assert gpu_comp.wh == pytest.approx(800.0)

    def test_no_total_time_ms_returns_none_without_finish_times(self):
        """Without total_time_ms and no finish_ms in DAG, returns None."""
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
        )
        sc = make_scenario(dc)
        # Non-replayed dag with start_ms=None — no finish_ms available
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc)
        assert result is None, (
            "Should return None when total_time_ms is not provided "
            "and DAG has no finish_ms"
        )


# ---------------------------------------------------------------------------
# Tests: energy output structure matches EnergyResult spec
# ---------------------------------------------------------------------------


class TestEnergyOutputStructure:
    """EnergyResult output structure matches analytical backend expectations."""

    def test_returns_energy_result_type(self):
        """compute_energy returns an EnergyResult instance."""
        dc = make_datacenter(gpu_power_model=ConstantPowerModel(tdp_w=700.0))
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert isinstance(result, EnergyResult)

    def test_all_energy_result_fields_populated(self):
        """All EnergyResult fields are populated with non-zero values."""
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            pue=1.2,
        )
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        assert result.total_wh > 0
        assert result.hardware_subtotal_wh > 0
        assert result.pue_overhead_wh > 0
        assert result.avg_power_kw > 0
        assert result.run_duration_hours > 0
        assert len(result.breakdown) > 0

    def test_pue_math_correct(self):
        """total_wh = hardware_subtotal_wh * pue, overhead = subtotal * (pue - 1)."""
        pue = 1.5
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            pue=pue,
        )
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        assert result.total_wh == pytest.approx(result.hardware_subtotal_wh * pue)
        assert result.pue_overhead_wh == pytest.approx(
            result.hardware_subtotal_wh * (pue - 1), rel=1e-9
        )

    def test_breakdown_always_has_gpu(self):
        """Energy breakdown always includes a GPU component."""
        dc = make_datacenter(gpu_power_model=ConstantPowerModel(tdp_w=700.0))
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        gpu_comp = next((c for c in result.breakdown if c.component == "gpu"), None)
        assert gpu_comp is not None
        assert gpu_comp.wh > 0
        assert gpu_comp.pct == pytest.approx(100.0)

    def test_breakdown_percentages_sum_to_100(self):
        """All component pct values sum to 100 of the hardware subtotal."""
        cpu_spec = CPUSpec(
            name="TestCPU",
            sockets=2,
            power_model=ConstantPowerModel(tdp_w=300.0),
        )
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            pue=1.0,
        )
        dc.node.cpu = cpu_spec
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        total_pct = sum(c.pct for c in result.breakdown)
        assert total_pct == pytest.approx(100.0)

    def test_avg_power_kw_computed_correctly(self):
        """avg_power_kw = total_wh / run_duration_hours / 1000."""
        dc = make_datacenter(
            gpu_power_model=ConstantPowerModel(tdp_w=1_000_000.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=_MS_PER_HOUR)
        assert result is not None
        # total_wh = 1_000_000 W * 1 GPU * 1 h = 1_000_000 Wh
        # avg_power_kw = 1_000_000 / 1 / 1000 = 1000 kW
        assert result.avg_power_kw == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for ATLAHS energy computation."""

    def test_zero_total_time_ms_does_not_crash(self):
        """total_time_ms=0 results in utilisation=0 and zero energy, no division errors."""
        dc = make_datacenter(
            gpu_power_model=LinearPowerModel(tdp_w=700.0, idle_power_w=200.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=100.0)])
        result = compute_energy(dag, sc, total_time_ms=0.0)
        assert result is not None
        # utilisation = 0 (guarded by total_time_ms > 0), power = 200 W
        # _component_wh(200, 1, 0) = 0 Wh
        assert result.total_wh == pytest.approx(0.0)
        assert result.avg_power_kw == pytest.approx(0.0)
        assert result.run_duration_hours == pytest.approx(0.0)

    def test_empty_compute_nodes_with_total_time_ms(self):
        """Empty compute_nodes list gives utilisation=0 (idle power for LinearPowerModel)."""
        dc = make_datacenter(
            gpu_power_model=LinearPowerModel(tdp_w=700.0, idle_power_w=200.0),
            pue=1.0,
        )
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is not None
        # utilisation = 0, power = 200 W (idle)
        gpu_comp = next(c for c in result.breakdown if c.component == "gpu")
        expected_wh = 200.0 * 1000.0 / _MS_PER_HOUR
        assert gpu_comp.wh == pytest.approx(expected_wh, rel=1e-9)

    def test_no_power_model_returns_none(self):
        """compute_energy returns None when GPU has no power_model, even with total_time_ms."""
        dc = make_datacenter(gpu_power_model=None)
        sc = make_scenario(dc)
        dag = ExecutionDAG(compute_nodes=[_non_replayed_node(0, duration_ms=500.0)])
        result = compute_energy(dag, sc, total_time_ms=1000.0)
        assert result is None
