"""Unit tests for energy.py and cost.py.

Tests pin: Wh formula correctness, power model interpolation, utilisation
derivation from DAG node times, PUE multiplier, per-component breakdown and
percentages, no-power-model returns None, CAPEX scalar and Cost() range
propagation, OPEX per run, combined cost-per-run, and cost_per_run absence
when lifetime_years is not set.
"""
from __future__ import annotations

import math
import pytest

from simulon.backend.dag.nodes import ComputeNode, CommNode, ExecutionDAG
from simulon.config.common import ConstantPowerModel, LinearPowerModel, Cost
from simulon.config.dc import (
    ClusterSpec,
    CPUSpec,
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    NodeSpec,
    NICSpec,
    NetworkSpec,
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
from simulon.config.common import DType
from simulon.energy import _power_w, compute_energy, EnergyResult
from simulon.cost import compute_cost, CostResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MS_PER_HOUR = 3_600_000  # 1 hour in milliseconds


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


def _make_dc(
    *,
    gpu_power_model=None,
    pue: float = 1.0,
    num_nodes: int = 1,
    gpus_per_node: int = 1,
    gpus_per_nic: int = 1,
    gpu_cost=None,
    cpu_spec: CPUSpec | None = None,
    electricity_cost_per_kwh: float | None = None,
    datacenter_lifetime_years: float | None = None,
    idle_fraction: float = 0.0,
) -> DatacenterConfig:
    gpu = GPUSpec(
        name="TestGPU",
        power_model=gpu_power_model,
        cost=gpu_cost,
    )
    return DatacenterConfig(
        datacenter=DatacenterMeta(
            pue=pue,
            electricity_cost_per_kwh=electricity_cost_per_kwh,
            datacenter_lifetime_years=datacenter_lifetime_years,
            idle_fraction=idle_fraction,
        ),
        cluster=ClusterSpec(num_nodes=num_nodes),
        node=NodeSpec(
            gpus_per_node=gpus_per_node,
            gpus_per_nic=gpus_per_nic,
            gpu=gpu,
            cpu=cpu_spec,
        ),
    )


def _make_scenario(dc: DatacenterConfig) -> ScenarioConfig:
    return ScenarioConfig(datacenter=dc, workload=_make_workload())


def _single_compute_node(gpu_rank: int, start_ms: float, finish_ms: float) -> ComputeNode:
    return ComputeNode(
        node_id=0,
        gpu_rank=gpu_rank,
        kernel="matmul",
        layer_id=0,
        microbatch_id=0,
        pipeline_stage=0,
        phase="fwd",
        start_ms=start_ms,
        finish_ms=finish_ms,
    )


def _dag_with_finish(finish_ms: float, gpu_rank: int = 0) -> ExecutionDAG:
    """DAG with one compute node that spans [0, finish_ms] on the given rank."""
    node = _single_compute_node(gpu_rank, start_ms=0.0, finish_ms=finish_ms)
    return ExecutionDAG(compute_nodes=[node])


# ---------------------------------------------------------------------------
# _power_w unit tests
# ---------------------------------------------------------------------------


class TestPowerW:
    def test_constant_model_returns_tdp_regardless_of_util(self):
        """ConstantPowerModel always returns tdp_w for any utilisation."""
        model = ConstantPowerModel(tdp_w=700.0)
        assert _power_w(model, 0.0) == pytest.approx(700.0)
        assert _power_w(model, 0.5) == pytest.approx(700.0)
        assert _power_w(model, 1.0) == pytest.approx(700.0)

    def test_linear_model_at_zero_util_returns_idle_power(self):
        """LinearPowerModel at util=0 returns idle_power_w."""
        model = LinearPowerModel(tdp_w=700.0, idle_power_w=200.0)
        assert _power_w(model, 0.0) == pytest.approx(200.0)

    def test_linear_model_at_full_util_returns_tdp(self):
        """LinearPowerModel at util=1 returns tdp_w."""
        model = LinearPowerModel(tdp_w=700.0, idle_power_w=200.0)
        assert _power_w(model, 1.0) == pytest.approx(700.0)

    def test_linear_model_interpolates_at_half_util(self):
        """LinearPowerModel at util=0.5 returns midpoint between idle and tdp."""
        model = LinearPowerModel(tdp_w=700.0, idle_power_w=200.0)
        # 200 + 0.5 * (700 - 200) = 450
        assert _power_w(model, 0.5) == pytest.approx(450.0)

    def test_linear_model_at_quarter_util(self):
        """LinearPowerModel interpolates correctly at util=0.25."""
        model = LinearPowerModel(tdp_w=600.0, idle_power_w=100.0)
        # 100 + 0.25 * (600 - 100) = 225
        assert _power_w(model, 0.25) == pytest.approx(225.0)


# ---------------------------------------------------------------------------
# compute_energy — no power model → returns None
# ---------------------------------------------------------------------------


class TestComputeEnergyNoPowerModel:
    def test_returns_none_when_gpu_has_no_power_model(self):
        """compute_energy returns None when GPU spec has no power_model."""
        dc = _make_dc(gpu_power_model=None)
        scenario = _make_scenario(dc)
        dag = _dag_with_finish(1000.0)
        result = compute_energy(dag, scenario)
        assert result is None

    def test_logs_warning_when_gpu_has_no_power_model(self, caplog):
        """compute_energy emits a warning when GPU spec has no power_model."""
        import logging
        dc = _make_dc(gpu_power_model=None)
        scenario = _make_scenario(dc)
        dag = _dag_with_finish(1000.0)
        with caplog.at_level(logging.WARNING, logger="simulon.energy"):
            compute_energy(dag, scenario)
        assert any("no power_model" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# compute_energy — Wh calculation correctness
# ---------------------------------------------------------------------------


class TestComputeEnergyWh:
    def test_constant_gpu_one_hour_energy(self):
        """One GPU at 700 W for exactly 1 hour produces 700 Wh (PUE=1)."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            pue=1.0,
            num_nodes=1,
            gpus_per_node=1,
        )
        scenario = _make_scenario(dc)
        # DAG: single compute node spanning exactly 1 hour
        dag = _dag_with_finish(_MS_PER_HOUR)
        result = compute_energy(dag, scenario)
        assert result is not None
        # gpu energy = 700 W * 1 GPU * 1 h = 700 Wh
        assert result.hardware_subtotal_wh == pytest.approx(700.0)
        assert result.total_wh == pytest.approx(700.0)

    def test_pue_multiplies_hardware_subtotal(self):
        """total_wh equals hardware_subtotal_wh times PUE."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            pue=1.2,
        )
        scenario = _make_scenario(dc)
        dag = _dag_with_finish(_MS_PER_HOUR)
        result = compute_energy(dag, scenario)
        assert result is not None
        assert result.total_wh == pytest.approx(result.hardware_subtotal_wh * 1.2)
        assert result.pue_overhead_wh == pytest.approx(result.hardware_subtotal_wh * 0.2)

    def test_multi_gpu_energy_scales_with_count(self):
        """Energy scales linearly with total GPU count (gpus_per_node * num_nodes)."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=100.0),
            pue=1.0,
            num_nodes=2,
            gpus_per_node=4,
        )
        scenario = _make_scenario(dc)
        # DAG: one compute node per rank (8 ranks)
        nodes = [
            _single_compute_node(rank, 0.0, _MS_PER_HOUR)
            for rank in range(8)
        ]
        dag = ExecutionDAG(compute_nodes=nodes)
        result = compute_energy(dag, scenario)
        assert result is not None
        # 100 W * 8 GPUs * 1 h = 800 Wh
        gpu_component = next(c for c in result.breakdown if c.component == "gpu")
        assert gpu_component.wh == pytest.approx(800.0)

    def test_run_duration_hours(self):
        """run_duration_hours equals total_time_ms / 3_600_000."""
        total_time_ms = 7_200_000.0  # 2 hours
        dc = _make_dc(gpu_power_model=ConstantPowerModel(tdp_w=100.0))
        scenario = _make_scenario(dc)
        dag = _dag_with_finish(total_time_ms)
        result = compute_energy(dag, scenario)
        assert result is not None
        assert result.run_duration_hours == pytest.approx(2.0)

    def test_avg_power_kw(self):
        """avg_power_kw equals (total_wh / run_duration_hours) / 1000."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=1000.0),
            pue=1.0,
        )
        scenario = _make_scenario(dc)
        dag = _dag_with_finish(_MS_PER_HOUR)
        result = compute_energy(dag, scenario)
        assert result is not None
        # 1000 Wh / 1 h / 1000 = 1.0 kW
        assert result.avg_power_kw == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_energy — utilisation derivation
# ---------------------------------------------------------------------------


class TestComputeEnergyUtilisation:
    def test_linear_model_uses_derived_utilisation(self):
        """With a linear power model, GPU energy reflects the actual compute fraction."""
        # GPU active for 0.5 of the total iteration time → utilisation = 0.5
        # power = 200 + 0.5 * (700 - 200) = 450 W
        # energy = 450 W * 1 GPU * 1 h = 450 Wh
        idle_w = 200.0
        tdp_w = 700.0
        dc = _make_dc(
            gpu_power_model=LinearPowerModel(tdp_w=tdp_w, idle_power_w=idle_w),
            pue=1.0,
        )
        scenario = _make_scenario(dc)
        # total_time = 1 h, active = 0.5 h
        active_ms = _MS_PER_HOUR * 0.5
        compute_node = _single_compute_node(0, start_ms=0.0, finish_ms=active_ms)
        # comm node to extend total span to 1 h without adding active compute
        comm_node = CommNode(
            node_id=1,
            src_gpu=0,
            dst_gpu=0,
            bytes=0,
            collective_type="AllReduce",
            layer_id=0,
            phase="fwd",
            flow_id=0,
            start_ms=active_ms,
            finish_ms=float(_MS_PER_HOUR),
        )
        dag = ExecutionDAG(compute_nodes=[compute_node], comm_nodes=[comm_node])
        result = compute_energy(dag, scenario)
        assert result is not None
        expected_power_w = idle_w + 0.5 * (tdp_w - idle_w)  # 450 W
        expected_wh = expected_power_w * 1.0  # 1 hour
        gpu_component = next(c for c in result.breakdown if c.component == "gpu")
        assert gpu_component.wh == pytest.approx(expected_wh, rel=1e-6)

    def test_full_utilisation_constant_model_equals_partial_utilisation(self):
        """ConstantPowerModel draws full TDP regardless of actual utilisation fraction."""
        tdp_w = 500.0
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=tdp_w),
            pue=1.0,
        )
        scenario = _make_scenario(dc)
        # active only for 10% of the iteration
        compute_node = _single_compute_node(0, 0.0, _MS_PER_HOUR * 0.1)
        comm_node = CommNode(
            node_id=1,
            src_gpu=0,
            dst_gpu=0,
            bytes=0,
            collective_type="AllReduce",
            layer_id=0,
            phase="fwd",
            flow_id=0,
            start_ms=_MS_PER_HOUR * 0.1,
            finish_ms=float(_MS_PER_HOUR),
        )
        dag = ExecutionDAG(compute_nodes=[compute_node], comm_nodes=[comm_node])
        result = compute_energy(dag, scenario)
        assert result is not None
        # Constant model: always 500 W * 1 h = 500 Wh
        gpu_component = next(c for c in result.breakdown if c.component == "gpu")
        assert gpu_component.wh == pytest.approx(500.0)

    def test_utilisation_averaged_across_all_cluster_gpus(self):
        """GPU utilisation is averaged over ALL cluster GPUs (num_nodes × gpus_per_node).

        Rank 0 is active for 50% of the iteration, rank 1 is active for 100%.
        Cluster has 2 GPUs total, so avg util = (0.5 + 1.0) / 2 = 0.75.
        power = 200 + 0.75 * (700 - 200) = 575 W.
        2 GPUs × 575 W × 1 h = 1150 Wh.
        """
        dc = _make_dc(
            gpu_power_model=LinearPowerModel(tdp_w=700.0, idle_power_w=200.0),
            pue=1.0,
            num_nodes=1,
            gpus_per_node=2,
            gpus_per_nic=2,
        )
        scenario = _make_scenario(dc)
        # rank 0: active for half the hour (first 0.5 h)
        node_rank0 = _single_compute_node(0, 0.0, _MS_PER_HOUR * 0.5)
        # rank 1: active for the full hour
        node_rank1 = ComputeNode(
            node_id=1,
            gpu_rank=1,
            kernel="matmul",
            layer_id=0,
            microbatch_id=0,
            pipeline_stage=0,
            phase="fwd",
            start_ms=0.0,
            finish_ms=float(_MS_PER_HOUR),
        )
        dag = ExecutionDAG(compute_nodes=[node_rank0, node_rank1])
        result = compute_energy(dag, scenario)
        assert result is not None
        # total_time = 1 h (max finish_ms = _MS_PER_HOUR from rank 1)
        # active_ms_by_rank: {0: 0.5*MS_PER_HOUR, 1: MS_PER_HOUR}
        # avg_active_ms = (0.5 + 1.0) / 2 * MS_PER_HOUR = 0.75 * MS_PER_HOUR
        # utilisation = 0.75
        # power = 200 + 0.75 * (700 - 200) = 575 W
        # energy = 575 W × 2 GPUs × 1 h = 1150 Wh
        gpu_component = next(c for c in result.breakdown if c.component == "gpu")
        assert gpu_component.wh == pytest.approx(1150.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_energy — breakdown percentages
# ---------------------------------------------------------------------------


class TestComputeEnergyBreakdownPct:
    def test_single_component_pct_is_100(self):
        """With only a GPU component, its percentage of hardware subtotal is 100%."""
        dc = _make_dc(gpu_power_model=ConstantPowerModel(tdp_w=700.0), pue=1.0)
        scenario = _make_scenario(dc)
        dag = _dag_with_finish(_MS_PER_HOUR)
        result = compute_energy(dag, scenario)
        assert result is not None
        assert len(result.breakdown) == 1
        assert result.breakdown[0].pct == pytest.approx(100.0)

    def test_percentages_sum_to_100(self):
        """All component percentages sum to 100 of hardware subtotal."""
        # Build a scenario with GPU + CPU so there are two components
        cpu_spec = CPUSpec(
            name="TestCPU",
            sockets=2,
            power_model=ConstantPowerModel(tdp_w=300.0),
        )
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            pue=1.0,
            cpu_spec=cpu_spec,
        )
        scenario = _make_scenario(dc)
        dag = _dag_with_finish(_MS_PER_HOUR)
        result = compute_energy(dag, scenario)
        assert result is not None
        total_pct = sum(c.pct for c in result.breakdown)
        assert total_pct == pytest.approx(100.0)

    def test_pue_overhead_is_hardware_subtotal_times_pue_minus_one(self):
        """pue_overhead_wh equals hardware_subtotal_wh * (pue - 1)."""
        pue = 1.5
        dc = _make_dc(gpu_power_model=ConstantPowerModel(tdp_w=700.0), pue=pue)
        scenario = _make_scenario(dc)
        dag = _dag_with_finish(_MS_PER_HOUR)
        result = compute_energy(dag, scenario)
        assert result is not None
        assert result.pue_overhead_wh == pytest.approx(
            result.hardware_subtotal_wh * (pue - 1), rel=1e-9
        )


# ---------------------------------------------------------------------------
# compute_energy — total_time_ms from max finish_ms across all node types
# ---------------------------------------------------------------------------


class TestComputeEnergyTotalTime:
    def test_total_time_is_max_finish_ms_across_compute_and_comm_nodes(self):
        """total_time_ms uses the maximum finish_ms over both compute and comm nodes."""
        compute_node = _single_compute_node(0, 0.0, 500_000.0)
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
        dc = _make_dc(gpu_power_model=ConstantPowerModel(tdp_w=3_600_000.0), pue=1.0)
        scenario = _make_scenario(dc)
        result = compute_energy(dag, scenario)
        assert result is not None
        # total_time = 3600000 ms = 1 h; power = 3600000 W → energy = 3600000 Wh
        # but there is only 1 GPU; energy per GPU = 3600000 W * 1 h = 3600000 Wh
        assert result.run_duration_hours == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_cost — CAPEX tests
# ---------------------------------------------------------------------------


def _energy_stub(total_wh: float, run_duration_hours: float) -> EnergyResult:
    """Minimal EnergyResult for cost tests."""
    return EnergyResult(
        total_wh=total_wh,
        hardware_subtotal_wh=total_wh,
        pue_overhead_wh=0.0,
        avg_power_kw=0.0,
        run_duration_hours=run_duration_hours,
        breakdown=[],
    )


class TestComputeCostCapex:
    def test_scalar_gpu_cost_propagates_correctly(self):
        """Scalar GPU cost × GPU count equals gpu CapexComponent total."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=30_000.0,
            num_nodes=2,
            gpus_per_node=4,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(700.0, 1.0)
        result = compute_cost(scenario, energy)
        gpu_comp = next(c for c in result.capex.breakdown if c.component == "gpu")
        # 30_000 × 8 = 240_000
        assert gpu_comp.total == pytest.approx(240_000.0)
        assert gpu_comp.min is None
        assert gpu_comp.max is None

    def test_cost_object_range_propagates_to_capex_result(self):
        """Cost(value, min, max) propagates min/max to the CapexResult totals."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=Cost(value=30_000.0, min=25_000.0, max=35_000.0),
            num_nodes=1,
            gpus_per_node=1,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(700.0, 1.0)
        result = compute_cost(scenario, energy)
        assert result.capex.total == pytest.approx(30_000.0)
        assert result.capex.min == pytest.approx(25_000.0)
        assert result.capex.max == pytest.approx(35_000.0)

    def test_capex_range_absent_when_all_scalar(self):
        """When all cost fields are scalars, capex min and max are None."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=50_000.0,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(700.0, 1.0)
        result = compute_cost(scenario, energy)
        assert result.capex.min is None
        assert result.capex.max is None

    def test_capex_total_is_sum_of_all_components(self):
        """capex.total equals the sum of all component totals in the breakdown."""
        cpu_spec = CPUSpec(name="TestCPU", sockets=2, cost=4_000.0)
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=30_000.0,
            cpu_spec=cpu_spec,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(700.0, 1.0)
        result = compute_cost(scenario, energy)
        component_sum = sum(c.total for c in result.capex.breakdown)
        assert result.capex.total == pytest.approx(component_sum)

    def test_capex_percentages_sum_to_100(self):
        """All CapexComponent percentages sum to 100% of the capex total."""
        cpu_spec = CPUSpec(name="TestCPU", sockets=2, cost=4_000.0)
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=30_000.0,
            cpu_spec=cpu_spec,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(700.0, 1.0)
        result = compute_cost(scenario, energy)
        pct_sum = sum(c.pct for c in result.capex.breakdown)
        assert pct_sum == pytest.approx(100.0)

    def test_cpu_memory_cost_included_in_capex(self):
        """memory_cost_per_gb × memory_gb × num_nodes appears as cpu_memory component."""
        cpu_spec = CPUSpec(
            name="TestCPU",
            sockets=2,
            memory_gb=512.0,
            memory_cost_per_gb=5.0,
        )
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=30_000.0,
            num_nodes=2,
            cpu_spec=cpu_spec,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(700.0, 1.0)
        result = compute_cost(scenario, energy)
        mem_comp = next(
            (c for c in result.capex.breakdown if c.component == "cpu_memory"), None
        )
        assert mem_comp is not None
        # 5 $/GB × 512 GB × 2 nodes = 5120
        assert mem_comp.total == pytest.approx(5_120.0)

    def test_range_propagation_multi_component(self):
        """When one component has a range, total_min/max sum all components correctly."""
        # GPU: Cost(value=30000, min=25000, max=35000), count=1
        # CPU: scalar 4000, count=1 node × 2 sockets = 2 → 8000
        # Expected: total=38000, min=25000+8000=33000, max=35000+8000=43000
        cpu_spec = CPUSpec(name="TestCPU", sockets=2, cost=4_000.0)
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=Cost(value=30_000.0, min=25_000.0, max=35_000.0),
            num_nodes=1,
            gpus_per_node=1,
            cpu_spec=cpu_spec,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(700.0, 1.0)
        result = compute_cost(scenario, energy)
        assert result.capex.total == pytest.approx(38_000.0)
        assert result.capex.min == pytest.approx(33_000.0)
        assert result.capex.max == pytest.approx(43_000.0)


# ---------------------------------------------------------------------------
# compute_cost — OPEX tests
# ---------------------------------------------------------------------------


class TestComputeCostOpex:
    def test_opex_per_run_from_energy_and_electricity_cost(self):
        """opex_per_run equals total_wh / 1000 × electricity_cost_per_kwh."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            electricity_cost_per_kwh=0.10,
        )
        scenario = _make_scenario(dc)
        # 1000 Wh = 1 kWh → opex = 1 kWh × $0.10 = $0.10
        energy = _energy_stub(total_wh=1_000.0, run_duration_hours=1.0)
        result = compute_cost(scenario, energy)
        assert result.opex_per_run == pytest.approx(0.10)

    def test_opex_per_run_zero_when_no_electricity_cost(self):
        """opex_per_run is 0.0 when electricity_cost_per_kwh is not set."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            electricity_cost_per_kwh=None,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(total_wh=1_000.0, run_duration_hours=1.0)
        result = compute_cost(scenario, energy)
        assert result.opex_per_run == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_cost — combined cost_per_run tests
# ---------------------------------------------------------------------------


class TestComputeCostPerRun:
    def test_cost_per_run_absent_without_lifetime_years(self):
        """cost_per_run is None when datacenter_lifetime_years is not set."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=30_000.0,
            electricity_cost_per_kwh=0.10,
            datacenter_lifetime_years=None,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(total_wh=700.0, run_duration_hours=1.0)
        result = compute_cost(scenario, energy)
        assert result.cost_per_run is None

    def test_cost_per_run_present_with_lifetime_years(self):
        """cost_per_run is populated when datacenter_lifetime_years is set."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=30_000.0,
            electricity_cost_per_kwh=0.10,
            datacenter_lifetime_years=5.0,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(total_wh=700.0, run_duration_hours=1.0)
        result = compute_cost(scenario, energy)
        assert result.cost_per_run is not None

    def test_cost_per_run_values(self):
        """cost_per_run.total equals capex_component + opex_component."""
        # Setup:
        #   lifetime = 5 years, idle_fraction = 0.0
        #   run_duration = 1 h
        #   runs_per_lifetime = floor(5 * 8760 * 1.0 / 1.0) = 43800
        #   capex_total = 30000 (1 GPU × $30000)
        #   capex_per_run = 30000 / 43800
        #   total_wh = 700, electricity = $0.10/kWh
        #   opex_per_run = 700/1000 * 0.10 = 0.07
        gpu_cost_val = 30_000.0
        electricity = 0.10
        total_wh = 700.0
        run_hours = 1.0
        lifetime_years = 5.0
        idle_fraction = 0.0

        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=700.0),
            gpu_cost=gpu_cost_val,
            electricity_cost_per_kwh=electricity,
            datacenter_lifetime_years=lifetime_years,
            idle_fraction=idle_fraction,
        )
        scenario = _make_scenario(dc)
        energy = _energy_stub(total_wh=total_wh, run_duration_hours=run_hours)
        result = compute_cost(scenario, energy)

        runs_per_lifetime = math.floor(lifetime_years * 8760 * (1 - idle_fraction) / run_hours)
        expected_capex_per_run = gpu_cost_val / runs_per_lifetime
        expected_opex_per_run = total_wh / 1000 * electricity

        assert result.cost_per_run is not None
        assert result.cost_per_run.capex_component == pytest.approx(expected_capex_per_run)
        assert result.cost_per_run.opex_component == pytest.approx(expected_opex_per_run)
        assert result.cost_per_run.total == pytest.approx(
            expected_capex_per_run + expected_opex_per_run
        )

    def test_cost_per_run_idle_fraction_reduces_runs(self):
        """Higher idle_fraction reduces runs_per_lifetime, raising capex_per_run."""
        gpu_cost_val = 100_000.0
        run_hours = 1.0
        lifetime_years = 1.0

        def make_result(idle_fraction):
            dc = _make_dc(
                gpu_power_model=ConstantPowerModel(tdp_w=700.0),
                gpu_cost=gpu_cost_val,
                datacenter_lifetime_years=lifetime_years,
                idle_fraction=idle_fraction,
            )
            scenario = _make_scenario(dc)
            energy = _energy_stub(total_wh=0.0, run_duration_hours=run_hours)
            return compute_cost(scenario, energy)

        result_no_idle = make_result(0.0)
        result_half_idle = make_result(0.5)

        assert result_no_idle.cost_per_run is not None
        assert result_half_idle.cost_per_run is not None
        # Fewer runs when idle_fraction=0.5 → higher capex per run
        assert result_half_idle.cost_per_run.capex_component > result_no_idle.cost_per_run.capex_component


# ---------------------------------------------------------------------------
# Merged DAG / concurrent workload smoke tests
# ---------------------------------------------------------------------------


class TestMergedDagEnergyCost:
    """Smoke-test energy and cost on a merged DAG representing concurrent workloads."""

    def _make_merged_dag(self) -> ExecutionDAG:
        """Return a merged DAG with two concurrent workloads on an 8-GPU cluster.

        Workload A uses GPUs 0-3 and runs from 0 ms to 1000 ms.
        Workload B uses GPUs 4-7 and runs from 500 ms to 1500 ms (offset 500 ms).
        """
        # Workload A: 4 compute nodes on ranks 0-3, duration 1000 ms each
        nodes_a = [
            ComputeNode(
                node_id=i,
                gpu_rank=i,
                kernel="matmul",
                layer_id=0,
                microbatch_id=0,
                pipeline_stage=0,
                phase="fwd",
                start_ms=0.0,
                finish_ms=1000.0,
            )
            for i in range(4)
        ]
        # Workload B: 4 compute nodes on ranks 4-7, duration 1000 ms each, offset 500 ms
        nodes_b = [
            ComputeNode(
                node_id=4 + i,
                gpu_rank=4 + i,
                kernel="matmul",
                layer_id=0,
                microbatch_id=0,
                pipeline_stage=0,
                phase="fwd",
                start_ms=500.0,
                finish_ms=1500.0,
            )
            for i in range(4)
        ]
        return ExecutionDAG(compute_nodes=nodes_a + nodes_b)

    def test_energy_total_time_is_concurrent_makespan_not_sum(self):
        """max(finish_ms) on merged DAG equals concurrent makespan (1500 ms), not 2000 ms."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=100.0),
            pue=1.0,
            num_nodes=2,
            gpus_per_node=4,
        )
        scenario = _make_scenario(dc)
        dag = self._make_merged_dag()
        result = compute_energy(dag, scenario)
        assert result is not None
        # Makespan = max(1000, 1500) = 1500 ms
        assert result.run_duration_hours == pytest.approx(1500.0 / 3_600_000)
        # Total energy = 100 W * 8 GPUs * 1500 ms / 3.6M = 0.333... Wh
        assert result.total_wh == pytest.approx(100.0 * 8 * 1500.0 / 3_600_000)

    def test_energy_utilisation_on_merged_dag(self):
        """Utilisation = avg_active_ms / total_time_ms across all cluster GPUs."""
        dc = _make_dc(
            gpu_power_model=LinearPowerModel(tdp_w=400.0, idle_power_w=100.0),
            pue=1.0,
            num_nodes=2,
            gpus_per_node=4,
        )
        scenario = _make_scenario(dc)
        dag = self._make_merged_dag()
        result = compute_energy(dag, scenario)
        assert result is not None
        # avg_active_ms = (4*1000 + 4*1000) / 8 = 1000 ms
        # utilisation = 1000 / 1500 = 2/3
        # power = 100 + (2/3)*(400-100) = 300 W
        # energy = 300 W * 8 GPUs * 1500 ms / 3.6M = 1.0 Wh
        expected_power = 100.0 + (2.0 / 3.0) * (400.0 - 100.0)
        expected_wh = expected_power * 8 * 1500.0 / 3_600_000
        gpu_component = next(c for c in result.breakdown if c.component == "gpu")
        assert gpu_component.wh == pytest.approx(expected_wh, rel=1e-6)

    def test_energy_is_not_additive(self):
        """Energy for concurrent workloads is less than sum of isolated energies."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=100.0),
            pue=1.0,
            num_nodes=2,
            gpus_per_node=4,
        )
        scenario = _make_scenario(dc)
        dag = self._make_merged_dag()
        result = compute_energy(dag, scenario)
        assert result is not None
        # Isolated energy for A: 100 W * 8 GPUs * 1000 ms / 3.6M = 0.222... Wh
        # Isolated energy for B: 100 W * 8 GPUs * 1000 ms / 3.6M = 0.222... Wh
        # Sum = 0.444... Wh
        # Concurrent energy: 100 W * 8 GPUs * 1500 ms / 3.6M = 0.333... Wh
        assert result.total_wh < (100.0 * 8 * 1000.0 / 3_600_000) * 2
        assert result.total_wh == pytest.approx(100.0 * 8 * 1500.0 / 3_600_000)

    def test_capex_not_double_counted_on_merged_dag(self):
        """CAPEX reflects total cluster hardware, not per-workload sum."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=100.0),
            gpu_cost=10_000.0,
            num_nodes=2,
            gpus_per_node=4,
        )
        scenario = _make_scenario(dc)
        dag = self._make_merged_dag()
        energy = compute_energy(dag, scenario)
        assert energy is not None
        cost_result = compute_cost(scenario, energy)
        gpu_comp = next(c for c in cost_result.capex.breakdown if c.component == "gpu")
        # Total cluster = 2 nodes * 4 GPUs = 8 GPUs
        assert gpu_comp.total == pytest.approx(8 * 10_000.0)

    def test_cost_per_run_uses_merged_run_duration(self):
        """runs_per_lifetime uses the merged DAG makespan, not per-workload time."""
        dc = _make_dc(
            gpu_power_model=ConstantPowerModel(tdp_w=100.0),
            gpu_cost=10_000.0,
            num_nodes=2,
            gpus_per_node=4,
            datacenter_lifetime_years=1.0,
            idle_fraction=0.0,
            electricity_cost_per_kwh=0.10,
        )
        scenario = _make_scenario(dc)
        dag = self._make_merged_dag()
        energy = compute_energy(dag, scenario)
        assert energy is not None
        cost_result = compute_cost(scenario, energy)
        assert cost_result.cost_per_run is not None
        # run_duration_hours = 1500 / 3.6M
        # runs_per_lifetime = floor(1 * 8760 / (1500/3.6M))
        # = floor(8760 * 3.6M / 1500) = floor(21,024,000)
        expected_runs = math.floor(1.0 * 8760 * (1 - 0.0) / (1500.0 / 3_600_000))
        expected_capex_per_run = 8 * 10_000.0 / expected_runs
        assert cost_result.cost_per_run.capex_component == pytest.approx(expected_capex_per_run)
