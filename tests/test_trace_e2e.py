"""End-to-end integration test for trace-driven DAG simulation."""

import json
import tempfile
from pathlib import Path

from simulon.backend.analytical import simulate as run_simulation
from simulon.backend.dag.replayer import replay
from simulon.config.dc import (
    DatacenterConfig,
    DatacenterMeta,
    GPUSpec,
    NICSpec,
    NodeSpec,
    ScaleOutSpec,
    ScaleUpSpec,
    SwitchSpec,
    TopologySpec,
    TopologyType,
)
from simulon.config.scenario import NcclConfig, ScenarioConfig
from simulon.config.workload import MegatronWorkload


def _make_trace_file(
    path: Path, *, rank: int, world_size: int, pipeline_stage: int, events: list[dict]
) -> None:
    trace = {
        "trace_format_version": "1.0",
        "rank": rank,
        "world_size": world_size,
        "pipeline_stage": pipeline_stage,
        "events": events,
    }
    path.write_text(json.dumps(trace))


def _make_datacenter() -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test_cluster"),
        num_nodes=2,
        node=NodeSpec(
            gpus_per_node=2,
            gpu=GPUSpec(name="H100", memory_capacity_gb=80.0),
            scale_up=ScaleUpSpec(switch=SwitchSpec(port_speed="2880Gbps", latency="0.000025ms")),
            scale_out=ScaleOutSpec(
                nic=NICSpec(speed="400Gbps", latency="0.005ms"),
                topology=TopologySpec(type=TopologyType.fat_tree, params={"k": 4}),
            ),
        ),
    )


def _make_workload(traces_dir: str | None = None) -> MegatronWorkload:
    return MegatronWorkload(
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
            "global-batch-size": 4,
            "num_gpus": 4,
        },
        traces_dir=traces_dir,
    )


def test_e2e_trace_driven_simulation():
    """Trace-driven MegatronWorkload produces a replayable DAG via AnalyticalBackend."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        trace_0 = tmp_path / "trace_rank_0.json"
        _make_trace_file(
            trace_0,
            rank=0,
            world_size=4,
            pipeline_stage=0,
            events=[
                {
                    "type": "slot_begin",
                    "timestamp_ms": 0.0,
                    "metadata": {"microbatch_id": 0, "phase": "fwd"},
                },
                {
                    "type": "collective",
                    "timestamp_ms": 5.0,
                    "metadata": {
                        "collective_type": "AllReduce",
                        "group_ranks": [0, 1],
                        "bytes": 1_048_576,
                    },
                },
                {"type": "slot_end", "timestamp_ms": 15.0, "metadata": {}},
            ],
        )

        trace_1 = tmp_path / "trace_rank_2.json"
        _make_trace_file(
            trace_1,
            rank=2,
            world_size=4,
            pipeline_stage=1,
            events=[
                {
                    "type": "slot_begin",
                    "timestamp_ms": 0.0,
                    "metadata": {"microbatch_id": 0, "phase": "fwd"},
                },
                {
                    "type": "collective",
                    "timestamp_ms": 5.0,
                    "metadata": {
                        "collective_type": "AllReduce",
                        "group_ranks": [2, 3],
                        "bytes": 1_048_576,
                    },
                },
                {"type": "slot_end", "timestamp_ms": 15.0, "metadata": {}},
            ],
        )

        dc = _make_datacenter()
        wl = _make_workload(traces_dir=str(tmp_path))
        scenario = ScenarioConfig(
            datacenter=dc,
            workload=wl,
            collective=NcclConfig(library="nccl", algorithm="ring", num_channels=1),
        )

        dag, result = run_simulation(scenario)

        # 1. Total iteration time is positive
        assert result.total_time_ms > 0

        # 2. DAG has compute nodes
        assert len(dag.compute_nodes) > 0

        # 3. All compute nodes have duration_ms pre-populated (from trace)
        for cn in dag.compute_nodes:
            assert cn.duration_ms is not None and cn.duration_ms > 0

        # 4. DAG can be replayed without errors
        replay_result = replay(dag)
        assert replay_result.total_time_ms > 0
