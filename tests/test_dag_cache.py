"""Unit tests for DAG caching (simulon.backend.dag.cache + DAGTracerConfig.cache_dir)."""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from simulon.backend.dag import cache as dag_cache
from simulon.backend.dag.nodes import ExecutionDAG
from simulon.backend.dag.tracer import DAGTracerConfig
from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
from simulon.config.workload import LLMSpec, MegatronParallelism, MegatronTraining, MegatronWorkload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_workload(tp: int = 1, pp: int = 1) -> MegatronWorkload:
    return MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name="tiny",
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            ffn_hidden_size=128,
            vocab_size=512,
        ),
        parallelism=MegatronParallelism(tp=tp, pp=pp),
        training=MegatronTraining(
            num_gpus=tp * pp,
            global_batch_size=tp * pp,
            micro_batch_size=1,
            sequence_length=128,
        ),
    )


def _tracer(cache_dir: Path) -> MegatronDAGTracer:
    return MegatronDAGTracer(DAGTracerConfig(cache_dir=cache_dir))


def _trace(workload: MegatronWorkload, cache_dir: Path) -> ExecutionDAG:
    # datacenter is unused by trace(), so pass None
    return _tracer(cache_dir).trace(workload, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


def test_cache_key_stable():
    """Same inputs → same key on repeated calls."""
    workload = _small_workload()
    model = LLMSpec(hidden_size=64, num_layers=2, num_heads=2, ffn_hidden_size=128, vocab_size=512)
    cfg = DAGTracerConfig(cache_dir=None)
    k1 = dag_cache._cache_key(workload, model, cfg)
    k2 = dag_cache._cache_key(workload, model, cfg)
    assert k1 == k2


def test_cache_key_differs_on_model_change():
    workload = _small_workload()
    cfg = DAGTracerConfig(cache_dir=None)
    m1 = LLMSpec(hidden_size=64, num_layers=2, num_heads=2, ffn_hidden_size=128, vocab_size=512)
    m2 = LLMSpec(hidden_size=128, num_layers=2, num_heads=2, ffn_hidden_size=256, vocab_size=512)
    assert dag_cache._cache_key(workload, m1, cfg) != dag_cache._cache_key(workload, m2, cfg)


def test_cache_key_differs_on_parallelism_change():
    cfg = DAGTracerConfig(cache_dir=None)
    model = LLMSpec(hidden_size=64, num_layers=2, num_heads=2, ffn_hidden_size=128, vocab_size=512)
    w1 = _small_workload(tp=1, pp=2)
    w2 = _small_workload(tp=2, pp=1)
    assert dag_cache._cache_key(w1, model, cfg) != dag_cache._cache_key(w2, model, cfg)


def test_cache_key_differs_on_algorithm_change():
    workload = _small_workload()
    model = LLMSpec(hidden_size=64, num_layers=2, num_heads=2, ffn_hidden_size=128, vocab_size=512)
    cfg_ring = DAGTracerConfig(cache_dir=None, algorithm="ring")
    cfg_tree = DAGTracerConfig(cache_dir=None, algorithm="tree")
    assert dag_cache._cache_key(workload, model, cfg_ring) != dag_cache._cache_key(workload, model, cfg_tree)


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------


def test_load_returns_none_on_miss(tmp_path):
    result = dag_cache.load(tmp_path, "nonexistent_key")
    assert result is None


def test_save_and_load_roundtrip(tmp_path):
    dag = ExecutionDAG()
    dag_cache.save(tmp_path, "testkey", dag)
    loaded = dag_cache.load(tmp_path, "testkey")
    assert loaded is not None
    assert isinstance(loaded, ExecutionDAG)


def test_save_creates_npz_file(tmp_path):
    dag = ExecutionDAG()
    dag_cache.save(tmp_path, "abc123", dag)
    assert (tmp_path / "abc123.npz").exists()


def test_load_returns_none_on_corrupt_file(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    result = dag_cache.load(tmp_path, "bad")
    assert result is None


def test_save_creates_cache_dir_if_missing(tmp_path):
    subdir = tmp_path / "a" / "b" / "c"
    dag_cache.save(subdir, "key", ExecutionDAG())
    assert subdir.is_dir()


# ---------------------------------------------------------------------------
# DAGTracer integration
# ---------------------------------------------------------------------------


def test_cache_miss_then_hit(tmp_path):
    """First call builds and caches; second call returns cached DAG."""
    workload = _small_workload(tp=2, pp=1)
    dag1 = _trace(workload, tmp_path)
    dag2 = _trace(workload, tmp_path)

    assert len(dag1.compute_nodes) == len(dag2.compute_nodes)
    assert len(dag1.comm_nodes) == len(dag2.comm_nodes)
    assert len(dag1.edges) == len(dag2.edges)


def test_cache_file_written_after_build(tmp_path):
    workload = _small_workload()
    _trace(workload, tmp_path)
    pkl_files = list(tmp_path.glob("*.npz"))
    assert len(pkl_files) == 1


def test_different_workloads_produce_different_cache_files(tmp_path):
    _trace(_small_workload(tp=1, pp=1), tmp_path)
    _trace(_small_workload(tp=1, pp=2), tmp_path)
    pkl_files = list(tmp_path.glob("*.npz"))
    assert len(pkl_files) == 2


def test_same_workload_reuses_single_cache_file(tmp_path):
    workload = _small_workload()
    _trace(workload, tmp_path)
    _trace(workload, tmp_path)
    pkl_files = list(tmp_path.glob("*.npz"))
    assert len(pkl_files) == 1


def test_cache_disabled_when_none(tmp_path):
    """cache_dir=None → no files written, trace still works."""
    workload = _small_workload()
    tracer = MegatronDAGTracer(DAGTracerConfig(cache_dir=None))
    dag = tracer.trace(workload, None)  # type: ignore[arg-type]
    assert len(list(tmp_path.glob("*.npz"))) == 0
    assert len(dag.compute_nodes) > 0


def test_cache_hit_is_faster(tmp_path):
    """Cache hit should be measurably faster than a cold build."""
    import time

    workload = _small_workload(tp=2, pp=2)

    t0 = time.perf_counter()
    _trace(workload, tmp_path)
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    _trace(workload, tmp_path)
    warm = time.perf_counter() - t0

    assert warm < cold


def test_string_and_inline_model_share_cache(tmp_path):
    """A workload with model='llama-7b' and an equivalent inline LLMSpec should hit the same cache entry."""
    import yaml

    template = Path("templates/model/llama-7b.yaml")
    if not template.exists():
        pytest.skip("llama-7b template not found")

    with open(template) as f:
        spec_data = yaml.safe_load(f)
    inline_spec = LLMSpec.model_validate(spec_data)

    num_gpus = 4
    w_str = MegatronWorkload(
        framework="megatron",
        model="llama-7b",
        parallelism=MegatronParallelism(tp=2, pp=2),
        training=MegatronTraining(num_gpus=num_gpus, global_batch_size=num_gpus, micro_batch_size=1, sequence_length=128),
    )
    w_inline = MegatronWorkload(
        framework="megatron",
        model=inline_spec,
        parallelism=MegatronParallelism(tp=2, pp=2),
        training=MegatronTraining(num_gpus=num_gpus, global_batch_size=num_gpus, micro_batch_size=1, sequence_length=128),
    )

    _trace(w_str, tmp_path)
    _trace(w_inline, tmp_path)

    # Both should map to the same cache file
    assert len(list(tmp_path.glob("*.npz"))) == 1
