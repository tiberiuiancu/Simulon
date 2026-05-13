"""Integration tests for the simulon experiment tracking package (MLflow path)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from simulon.backend.analytical import AnalyticalBackend
from simulon.backend.dag.replayer import SimulationResult
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
from simulon.config.scenario import NcclConfig, ScenarioConfig
from simulon.config.workload import (
    LLMSpec,
    MegatronDeprecatedWorkload,
    MegatronParallelism,
    MegatronTraining,
)
from simulon.tracking.factory import get_trackers
from simulon.tracking.mlflow_tracker import MLflowTracker
from simulon.tracking.params import extract_metrics, extract_params


# ---------------------------------------------------------------------------
# Shared builders (mirrors test_e2e.py helpers)
# ---------------------------------------------------------------------------

_TRACKING_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "WANDB_PROJECT",
    "WANDB_API_KEY",
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
    tp: int = 1,
    pp: int = 1,
    num_gpus: int = 4,
    num_layers: int = 2,
) -> MegatronDeprecatedWorkload:
    return MegatronDeprecatedWorkload(
        framework="megatron-deprecated",
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
            global_batch_size=8,
            micro_batch_size=1,
            sequence_length=128,
            dtype=DType.bf16,
        ),
    )


def _make_scenario(**kwargs) -> ScenarioConfig:
    return ScenarioConfig(datacenter=_make_datacenter(), workload=_make_workload(**kwargs))


# ---------------------------------------------------------------------------
# factory.py — get_trackers()
# ---------------------------------------------------------------------------


def test_get_trackers_empty_with_no_env(monkeypatch):
    """get_trackers() returns an empty list when no tracking env vars are present."""
    for var in _TRACKING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert get_trackers() == []


def test_get_trackers_returns_mlflow(monkeypatch, tmp_path):
    """Setting MLFLOW_TRACKING_URI causes get_trackers() to return exactly one MLflowTracker."""
    for var in _TRACKING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    trackers = get_trackers()
    assert len(trackers) == 1
    assert isinstance(trackers[0], MLflowTracker)


def test_get_trackers_present_but_empty_string(monkeypatch):
    """An empty-string MLFLOW_TRACKING_URI still triggers MLflowTracker creation (presence, not truthiness)."""
    for var in _TRACKING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    trackers = get_trackers()
    assert len(trackers) == 1
    assert isinstance(trackers[0], MLflowTracker)


# ---------------------------------------------------------------------------
# params.py — extract_params()
# ---------------------------------------------------------------------------


def test_extract_params_inline_dc():
    """extract_params() includes datacenter.gpu and correct collective/workload keys for inline DC."""
    sc = _make_scenario(tp=2, pp=1, num_gpus=4)
    params = extract_params(sc)

    assert params["datacenter.gpu"] == "H100"
    assert params["collective.library"] == "nccl"
    assert params["collective.algorithm"] == "ring"
    assert params["collective.num_channels"] == 1
    assert params["workload.tp"] == 2
    assert params["workload.pp"] == 1
    assert params["model.name"] == "test-model"
    assert "datacenter.config_path" not in params


def test_extract_params_path_dc():
    """When datacenter is a Path, extract_params() logs datacenter.config_path instead of datacenter.gpu."""
    wl = _make_workload()
    sc = ScenarioConfig(datacenter=Path("templates/dc/example.yaml"), workload=wl)
    params = extract_params(sc)

    assert params["datacenter.config_path"] == "templates/dc/example.yaml"
    assert "datacenter.gpu" not in params


def test_extract_params_model_as_string():
    """When the workload model is a plain string, extract_params() sets model.name to that string."""
    wl = MegatronDeprecatedWorkload(
        framework="megatron-deprecated",
        model="llama-7b",
        parallelism=MegatronParallelism(),
        training=MegatronTraining(
            num_gpus=4,
            global_batch_size=8,
            micro_batch_size=1,
            sequence_length=128,
        ),
    )
    sc = ScenarioConfig(datacenter=_make_datacenter(), workload=wl)
    params = extract_params(sc)

    assert params["model.name"] == "llama-7b"


# ---------------------------------------------------------------------------
# params.py — extract_metrics()
# ---------------------------------------------------------------------------


def test_extract_metrics():
    """extract_metrics() includes all mandatory keys and per-type exposed_comm keys."""
    result = SimulationResult(
        total_time_ms=100.0,
        compute_ms=60.0,
        exposed_comm_ms=20.0,
        exposed_comm_by_type={"AllReduce": 12.0, "PP_Send": 8.0},
        bubble_ms=20.0,
        overlapped_comm_ms=5.0,
    )
    metrics = extract_metrics(result)

    assert metrics["total_time_ms"] == pytest.approx(100.0)
    assert metrics["compute_ms"] == pytest.approx(60.0)
    assert metrics["exposed_comm_ms"] == pytest.approx(20.0)
    assert metrics["bubble_ms"] == pytest.approx(20.0)
    assert metrics["overlapped_comm_ms"] == pytest.approx(5.0)
    assert metrics["exposed_comm.AllReduce_ms"] == pytest.approx(12.0)
    assert metrics["exposed_comm.PP_Send_ms"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Full integration test — MLflowTracker context manager + MlflowClient verify
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mlflow_full_run(tmp_path, monkeypatch):
    """End-to-end: simulate a scenario, log to a local MLflow store, and verify the run via MlflowClient."""
    tracking_uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    # Build a minimal but real scenario.
    sc = _make_scenario(tp=1, pp=1, num_gpus=4, num_layers=2)

    # Run the simulation.
    dag, result = AnalyticalBackend().simulate(sc)

    # Write a tiny artifact so we can check artifact logging.
    artifact_path = tmp_path / "scenario.yaml"
    artifact_path.write_text("# test scenario\n")

    # Log everything through the tracker.
    import mlflow as _mlflow
    _mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "simulon-test"
    experiment_id = _mlflow.create_experiment(experiment_name)
    # Set the experiment BEFORE entering the context manager so that
    # start_run() (called by __enter__) logs into the right experiment.
    _mlflow.set_experiment(experiment_name)

    tracker = MLflowTracker()
    with tracker:
        tracker.log_params(extract_params(sc))
        tracker.log_metrics(extract_metrics(result))
        tracker.log_artifact(artifact_path)

    # Verify via MlflowClient (avoids pandas dependency of mlflow.search_runs).
    client = _mlflow.MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(experiment_ids=[experiment_id])

    assert len(runs) == 1, f"Expected 1 run, got {len(runs)}"

    run = runs[0]
    assert run.data.params["workload.tp"] == "1"
    assert run.data.params["collective.library"] == "nccl"
    assert float(run.data.metrics["total_time_ms"]) > 0
    assert "compute_ms" in run.data.metrics

    artifacts = client.list_artifacts(run.info.run_id)
    assert len(artifacts) >= 1
