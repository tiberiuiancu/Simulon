"""DAG caching: content-addressable on-disk cache for ExecutionDAG objects.

The cache key is a hash of all inputs that determine DAG structure:
  - Resolved LLMSpec (model template fully expanded)
  - MegatronParallelism
  - MegatronTraining
  - DAGTracerConfig (num_channels, algorithm, dtype_bytes)

GPU spec and network speeds are intentionally excluded — they only affect
populate_dag() and the simulation backend, not DAG structure.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulon.backend.dag.nodes import ExecutionDAG
    from simulon.backend.dag.tracer import DAGTracerConfig
    from simulon.config.workload import LLMSpec, MegatronWorkload

logger = logging.getLogger(__name__)


def _cache_key(workload: MegatronWorkload, resolved_model: LLMSpec, cfg: DAGTracerConfig) -> str:
    """Return a hex digest that uniquely identifies this DAG's structure."""
    payload = {
        "model": resolved_model.model_dump(),
        "parallelism": workload.parallelism.model_dump(),
        "training": workload.training.model_dump(),
        "tracer": asdict(cfg),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def load(cache_dir: Path, key: str) -> ExecutionDAG | None:
    path = cache_dir / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            dag = pickle.load(f)
        logger.info("DAG cache hit: %s", path.name)
        return dag
    except Exception as exc:
        logger.warning("DAG cache read failed (%s), rebuilding", exc)
        return None


def save(cache_dir: Path, key: str, dag: ExecutionDAG) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.pkl"
    try:
        with path.open("wb") as f:
            pickle.dump(dag, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("DAG cached: %s", path.name)
    except Exception as exc:
        logger.warning("DAG cache write failed: %s", exc)
