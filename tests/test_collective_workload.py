"""Tests for the collective-only simulation mode.

Covers:
- CollectiveWorkload config parsing and validation
- build_collective_dag() DAG structure
- AnalyticalBackend.simulate() end-to-end with CollectiveWorkload
- extract_params() for CollectiveWorkload
"""

import pytest
from pydantic import ValidationError

from simulon.backend.analytical import AnalyticalBackend
from simulon.backend.dag.collective_tracer import build_collective_dag
from simulon.backend.dag.nodes import CommNode, ExecutionDAG
from simulon.backend.network import decompose_collectives_in_dag
from simulon.collective import NCCLDecomposer
from simulon.config.dc import (
    ClusterSpec,
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    NetworkSpec,
    NICSpec,
    NodeSpec,
    ScaleOutSpec,
    ScaleUpSpec,
    SwitchSpec,
    TopologySpec,
    TopologyType,
)
from simulon.config.scenario import NcclConfig, ScenarioConfig
from simulon.config.workload import CollectiveType, CollectiveWorkload
from simulon.tracking.params import extract_params

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_datacenter(num_nodes: int = 2, gpus_per_node: int = 2) -> DatacenterConfig:
    """Minimal DatacenterConfig with num_nodes * gpus_per_node ranks."""
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=num_nodes),
        node=NodeSpec(
            gpus_per_node=gpus_per_node, gpu=GPUSpec(from_="h100", memory_capacity_gb=80.0)
        ),
        network=NetworkSpec(
            scale_up=ScaleUpSpec(switch=SwitchSpec(port_speed="2880Gbps", latency="0.000025ms")),
            scale_out=ScaleOutSpec(
                nic=NICSpec(speed="400Gbps", latency="0.005ms"),
                topology=TopologySpec(type=TopologyType.fat_tree, params={"k": 4}),
            ),
        ),
    )


def make_collective_workload(
    collective_type: str = "AllReduce", message_size_bytes: int = 1024 * 1024
) -> CollectiveWorkload:
    return CollectiveWorkload(
        framework="collective",
        collective_type=CollectiveType(collective_type),
        message_size_bytes=message_size_bytes,
    )


def make_collective_scenario(
    collective_type: str = "AllReduce",
    message_size_bytes: int = 1024 * 1024,
    num_nodes: int = 2,
    gpus_per_node: int = 2,
) -> ScenarioConfig:
    from simulon.config.scenario import NcclConfig

    collective = NcclConfig(
        algorithm="auto",  # calbusbw drives BW from the nccl profile
        num_channels=1,
    )
    return ScenarioConfig(
        datacenter=make_datacenter(num_nodes=num_nodes, gpus_per_node=gpus_per_node),
        workload=make_collective_workload(collective_type, message_size_bytes),
        collective=collective,
    )


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------


class TestCollectiveWorkloadConfig:
    def test_valid_allreduce_parses(self):
        """CollectiveWorkload parses correctly for AllReduce."""
        wl = CollectiveWorkload(
            framework="collective",
            collective_type=CollectiveType.AllReduce,
            message_size_bytes=1_000_000,
        )
        assert wl.framework == "collective"
        assert wl.collective_type == CollectiveType.AllReduce
        assert wl.message_size_bytes == 1_000_000

    def test_all_collective_types_parse(self):
        """All four CollectiveType enum values are accepted."""
        for ct in ["AllReduce", "AllGather", "ReduceScatter", "AllToAll"]:
            wl = CollectiveWorkload(
                framework="collective", collective_type=CollectiveType(ct), message_size_bytes=512
            )
            assert wl.collective_type.value == ct

    def test_missing_collective_type_raises(self):
        """CollectiveWorkload rejects a payload missing collective_type."""
        with pytest.raises(ValidationError):
            CollectiveWorkload(framework="collective", message_size_bytes=1024)

    def test_missing_message_size_raises(self):
        """CollectiveWorkload rejects a payload missing message_size_bytes."""
        with pytest.raises(ValidationError):
            CollectiveWorkload(framework="collective", collective_type=CollectiveType.AllGather)

    def test_extra_field_rejected(self):
        """CollectiveWorkload rejects unknown extra fields (extra='forbid')."""
        with pytest.raises(ValidationError):
            CollectiveWorkload(
                framework="collective",
                collective_type=CollectiveType.AllReduce,
                message_size_bytes=1024,
                unknown_field="bad",
            )

    def test_discriminated_union_routes_correctly(self):
        """ScenarioConfig correctly dispatches framework='collective' to CollectiveWorkload."""
        sc = make_collective_scenario()
        assert isinstance(sc.workload, CollectiveWorkload)

    def test_yaml_round_trip(self):
        """CollectiveWorkload survives a YAML serialise → deserialise round trip."""
        import yaml

        sc = make_collective_scenario(collective_type="ReduceScatter", message_size_bytes=8192)
        dumped = yaml.dump(
            sc.model_dump(mode="json", by_alias=True, exclude_none=True), default_flow_style=False
        )
        restored = ScenarioConfig.model_validate(yaml.safe_load(dumped))
        assert isinstance(restored.workload, CollectiveWorkload)
        assert restored.workload.collective_type == CollectiveType.ReduceScatter
        assert restored.workload.message_size_bytes == 8192


# ---------------------------------------------------------------------------
# build_collective_dag tests
# ---------------------------------------------------------------------------


class TestBuildCollectiveDag:
    def test_returns_execution_dag(self):
        """build_collective_dag returns an ExecutionDAG instance."""
        wl = make_collective_workload("AllReduce", 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        assert isinstance(dag, ExecutionDAG)

    def test_no_compute_nodes(self):
        """Collective DAG contains zero compute nodes."""
        wl = make_collective_workload("AllReduce", 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        assert dag.compute_nodes == []

    def test_no_explicit_edges(self):
        """Collective DAG has no explicit DAGEdge objects (deps tracked via parent_flow_ids)."""
        wl = make_collective_workload("AllReduce", 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        assert dag.edges == []

    def test_allreduce_4ranks_1channel_flow_count(self):
        """Ring AllReduce with N=4, C=1 produces exactly 24 CommNodes."""
        wl = make_collective_workload("AllReduce", 4 * 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        decompose_collectives_in_dag(dag)
        assert len(dag.comm_nodes) == 24

    def test_allreduce_4ranks_2channels_doubles_flows(self):
        """Doubling num_channels doubles the CommNode count for AllReduce."""
        wl = make_collective_workload("AllReduce", 4 * 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        decompose_collectives_in_dag(dag)
        dag2 = build_collective_dag(wl, dc, algorithm="ring", num_channels=2, ccl=NCCLDecomposer())
        decompose_collectives_in_dag(dag2)
        assert len(dag2.comm_nodes) == 2 * len(dag.comm_nodes)

    def test_reduce_scatter_4ranks_1channel_flow_count(self):
        wl = make_collective_workload("ReduceScatter", 4 * 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        decompose_collectives_in_dag(dag)
        assert len(dag.comm_nodes) == 12

    def test_allgather_4ranks_1channel_flow_count(self):
        wl = make_collective_workload("AllGather", 4 * 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        decompose_collectives_in_dag(dag)
        assert len(dag.comm_nodes) == 12

    def test_comm_nodes_are_comm_node_instances(self):
        wl = make_collective_workload("AllReduce", 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        decompose_collectives_in_dag(dag)
        for node in dag.comm_nodes:
            assert isinstance(node, CommNode)

    def test_comm_nodes_have_correct_collective_type(self):
        for ct in ["AllReduce", "AllGather", "ReduceScatter"]:
            wl = make_collective_workload(ct, 4 * 1024)
            dc = make_datacenter(num_nodes=2, gpus_per_node=2)
            dag = build_collective_dag(
                wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer()
            )
            decompose_collectives_in_dag(dag)
            for node in dag.comm_nodes:
                assert node.collective_type == ct

    def test_comm_node_phase_is_collective(self):
        wl = make_collective_workload("AllReduce", 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        decompose_collectives_in_dag(dag)
        for node in dag.comm_nodes:
            assert node.phase == "collective"

    def test_comm_node_ids_are_unique(self):
        wl = make_collective_workload("AllReduce", 4 * 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        decompose_collectives_in_dag(dag)
        ids = [n.node_id for n in dag.comm_nodes]
        assert len(ids) == len(set(ids))

    def test_ranks_are_within_range(self):
        """src_gpu and dst_gpu on every CommNode are valid rank indices."""
        num_ranks = 4
        wl = make_collective_workload("AllReduce", 4 * 1024)
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        dag = build_collective_dag(wl, dc, algorithm="ring", num_channels=1, ccl=NCCLDecomposer())
        for node in dag.comm_nodes:
            assert 0 <= node.src_gpu < num_ranks
            assert 0 <= node.dst_gpu < num_ranks


# ---------------------------------------------------------------------------
# AnalyticalBackend end-to-end tests
# ---------------------------------------------------------------------------


class TestAnalyticalBackendCollective:
    def test_simulate_returns_dag_and_result(self):
        """AnalyticalBackend.simulate() returns a (ExecutionDAG, SimulationResult) tuple."""
        from simulon.backend.dag.replayer import SimulationResult

        sc = make_collective_scenario()
        backend = AnalyticalBackend()
        dag, result = backend.simulate(sc)
        assert isinstance(dag, ExecutionDAG)
        assert isinstance(result, SimulationResult)

    def test_simulate_total_time_positive(self):
        """Simulated collective completes in positive time."""
        sc = make_collective_scenario()
        _, result = AnalyticalBackend().simulate(sc)
        assert result.total_time_ms > 0

    def test_simulate_compute_ms_is_zero(self):
        """Collective-only simulation has zero compute time (no compute nodes)."""
        sc = make_collective_scenario()
        _, result = AnalyticalBackend().simulate(sc)
        assert result.compute_ms == 0.0

    def test_simulate_dag_has_no_compute_nodes(self):
        """DAG returned by simulate() contains no compute nodes."""
        sc = make_collective_scenario()
        dag, _ = AnalyticalBackend().simulate(sc)
        assert dag.compute_nodes == []

    def test_simulate_dag_has_comm_nodes(self):
        """DAG returned by simulate() contains CommNodes."""
        sc = make_collective_scenario()
        dag, _ = AnalyticalBackend().simulate(sc)
        assert len(dag.comm_nodes) > 0

    def test_simulate_allreduce_larger_message_takes_longer(self):
        """Larger message sizes produce higher total_time_ms."""
        sc_small = make_collective_scenario(message_size_bytes=1024)
        sc_large = make_collective_scenario(message_size_bytes=1024 * 1024 * 100)
        backend = AnalyticalBackend()
        _, r_small = backend.simulate(sc_small)
        _, r_large = backend.simulate(sc_large)
        assert r_large.total_time_ms > r_small.total_time_ms

    @pytest.mark.parametrize("ct", ["AllReduce", "AllGather", "ReduceScatter", "AllToAll"])
    def test_simulate_all_collective_types(self, ct):
        """AnalyticalBackend.simulate() works for all four CollectiveType values."""
        backend = AnalyticalBackend()
        sc = make_collective_scenario(collective_type=ct)
        dag, result = backend.simulate(sc)
        assert result.total_time_ms > 0, f"Expected positive time for {ct}"
        assert dag.compute_nodes == []

    def test_run_trace_returns_execution_dag(self):
        """AnalyticalBackend.run_trace() returns an ExecutionDAG for CollectiveWorkload."""
        sc = make_collective_scenario()
        dag = AnalyticalBackend().run_trace(sc)
        assert isinstance(dag, ExecutionDAG)

    def test_num_channels_affects_dag(self):
        """Increasing num_channels produces more CommNodes in the DAG."""
        dc = make_datacenter(num_nodes=2, gpus_per_node=2)
        wl = make_collective_workload("AllReduce", 4 * 1024)
        sc1 = ScenarioConfig(datacenter=dc, workload=wl, collective=NcclConfig(num_channels=1))
        sc2 = ScenarioConfig(datacenter=dc, workload=wl, collective=NcclConfig(num_channels=2))
        backend = AnalyticalBackend()
        dag1 = backend.run_trace(sc1)
        dag2 = backend.run_trace(sc2)
        assert len(dag2.comm_nodes) > len(dag1.comm_nodes)


# ---------------------------------------------------------------------------
# extract_params tests
# ---------------------------------------------------------------------------


class TestExtractParamsCollective:
    def test_framework_key_is_collective(self):
        """extract_params emits workload.framework = 'collective'."""
        sc = make_collective_scenario()
        params = extract_params(sc)
        assert params["workload.framework"] == "collective"

    def test_collective_type_key_present(self):
        """extract_params emits workload.collective_type matching the workload."""
        sc = make_collective_scenario(collective_type="ReduceScatter")
        params = extract_params(sc)
        assert params["workload.collective_type"] == "ReduceScatter"

    def test_message_size_bytes_key_present(self):
        """extract_params emits workload.message_size_bytes matching the workload."""
        sc = make_collective_scenario(message_size_bytes=65536)
        params = extract_params(sc)
        assert params["workload.message_size_bytes"] == 65536

    def test_collective_config_keys_present(self):
        """extract_params always includes the collective.* keys."""
        sc = make_collective_scenario()
        params = extract_params(sc)
        assert "collective.library" in params
        assert "collective.algorithm" in params
        assert "collective.num_channels" in params

    def test_no_megatron_keys_emitted(self):
        """extract_params does not emit Megatron-specific keys for CollectiveWorkload."""
        sc = make_collective_scenario()
        params = extract_params(sc)
        megatron_keys = [k for k in params if k.startswith("training.") or k.startswith("model.")]
        assert megatron_keys == []

    def test_all_collective_types_emit_correct_param(self):
        """workload.collective_type matches CollectiveType.value for each variant."""
        for ct in ["AllReduce", "AllGather", "ReduceScatter", "AllToAll"]:
            sc = make_collective_scenario(collective_type=ct)
            params = extract_params(sc)
            assert params["workload.collective_type"] == ct
