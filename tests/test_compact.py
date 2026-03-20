"""Tests that compact and non-compact DAGs produce identical replay results.

The invariant: compacting consecutive compute-only sublayers into a single
fused node changes the graph structure but must NOT change the simulated
wall-clock time. This holds because:
  - populate_dag sums individual kernel durations for fused nodes
  - comm nodes are identical in both modes (only comm-free sublayers are fused)
  - the dependency structure is preserved at the slot/comm boundaries
"""

import pytest

from simulon.backend.analytical import AnalyticalBackend
from simulon.config.common import DType
from simulon.config.dc import (
    ClusterSpec,
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    KernelRun,
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
# Synthetic GPU spec — covers every kernel emitted by attn+mlp sublayers
# Omitting "tp" from params so the spec matches any tp value via partial lookup
# ---------------------------------------------------------------------------

_KERNEL_TIMES_MS: dict[str, float] = {
    "layernorm":   0.10,
    "attn_qkv":    0.50,
    "attn_flash":  1.00,
    "attn_proj":   0.50,
    "mlp_linear1": 0.80,
    "mlp_act":     0.20,
    "mlp_linear2": 0.80,
}

_KERNEL_PARAMS = {
    "hidden_size": 512,
    "seq_len":     128,
    "batch_size":  1,
    "dtype":       "bf16",
}


def _make_gpu_spec() -> GPUSpec:
    return GPUSpec(
        name="SyntheticGPU",
        memory_capacity_gb=80.0,
        kernel_runs=[
            KernelRun(kernel=k, params=_KERNEL_PARAMS, times_ms=[v])
            for k, v in _KERNEL_TIMES_MS.items()
        ],
    )


def _make_datacenter(gpus_per_node: int = 1, num_nodes: int = 1) -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=num_nodes),
        node=NodeSpec(gpus_per_node=gpus_per_node, gpu=_make_gpu_spec()),
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
    tp: int = 1,
    pp: int = 1,
    num_gpus: int = 1,
    num_layers: int = 2,
) -> MegatronWorkload:
    return MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name="test-model",
            hidden_size=512,
            num_layers=num_layers,
            num_heads=8,
            vocab_size=32000,
        ),
        parallelism=MegatronParallelism(tp=tp, pp=pp),
        training=MegatronTraining(
            num_gpus=num_gpus,
            global_batch_size=1,
            micro_batch_size=1,
            sequence_length=128,
            dtype=DType.bf16,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompactNonCompactEquivalence:
    """Compact and non-compact DAGs must produce the same SimulationResult."""

    def test_pure_compute_total_time(self):
        """Pure-compute DAG (tp=1, pp=1, dp=1): compact fuses all kernels into one node per slot,
        but the summed duration must equal the non-compact critical path."""
        sc = ScenarioConfig(
            datacenter=_make_datacenter(gpus_per_node=1, num_nodes=1),
            workload=_make_workload(tp=1, pp=1, num_gpus=1, num_layers=2),
        )
        backend = AnalyticalBackend()
        _, result_normal = backend.simulate(sc, compact=False)
        _, result_compact = backend.simulate(sc, compact=True)

        # Compact must actually run and produce a non-zero time
        assert result_normal.total_time_ms > 0
        assert result_compact.total_time_ms == pytest.approx(result_normal.total_time_ms, rel=1e-5)

    def test_mixed_compute_comm_total_time(self):
        """Mixed DAG (tp=2, pp=1, dp=1): bwd_wg sublayers are fused, fwd/bwd_ig are not.
        Total wall-clock time must be identical."""
        sc = ScenarioConfig(
            datacenter=_make_datacenter(gpus_per_node=2, num_nodes=1),
            workload=_make_workload(tp=2, pp=1, num_gpus=2, num_layers=2),
        )
        backend = AnalyticalBackend()
        _, result_normal = backend.simulate(sc, compact=False)
        _, result_compact = backend.simulate(sc, compact=True)

        assert result_normal.total_time_ms > 0
        assert result_compact.total_time_ms == pytest.approx(result_normal.total_time_ms, rel=1e-5)

    def test_per_gpu_times_match(self):
        """Each GPU's individual completion time must be identical between compact and non-compact."""
        sc = ScenarioConfig(
            datacenter=_make_datacenter(gpus_per_node=2, num_nodes=1),
            workload=_make_workload(tp=2, pp=1, num_gpus=2, num_layers=2),
        )
        backend = AnalyticalBackend()
        _, result_normal = backend.simulate(sc, compact=False)
        _, result_compact = backend.simulate(sc, compact=True)

        assert set(result_compact.per_gpu_times_ms) == set(result_normal.per_gpu_times_ms)
        for rank, t_normal in result_normal.per_gpu_times_ms.items():
            assert result_compact.per_gpu_times_ms[rank] == pytest.approx(t_normal, rel=1e-5), (
                f"GPU {rank}: compact={result_compact.per_gpu_times_ms[rank]:.4f}ms "
                f"vs normal={t_normal:.4f}ms"
            )

    def test_compact_reduces_node_count(self):
        """Compact mode must produce fewer compute nodes than non-compact (sanity check that
        fusion actually happened)."""
        sc = ScenarioConfig(
            datacenter=_make_datacenter(gpus_per_node=1, num_nodes=1),
            workload=_make_workload(tp=1, pp=1, num_gpus=1, num_layers=4),
        )
        backend = AnalyticalBackend()
        dag_normal = backend.run_trace(sc, compact=False)
        dag_compact = backend.run_trace(sc, compact=True)

        assert len(dag_compact.compute_nodes) < len(dag_normal.compute_nodes), (
            f"Expected fewer compute nodes in compact mode, got "
            f"compact={len(dag_compact.compute_nodes)} vs normal={len(dag_normal.compute_nodes)}"
        )
        # Comm nodes must be identical (compaction never touches comm-bearing sublayers)
        assert len(dag_compact.comm_nodes) == len(dag_normal.comm_nodes)
