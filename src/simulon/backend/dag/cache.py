"""DAG caching: content-addressable on-disk cache for ExecutionDAG objects.

The cache key is a hash of all inputs that determine DAG structure:
  - Resolved LLMSpec (model template fully expanded)
  - MegatronParallelism
  - MegatronTraining
  - DAGTracerConfig (num_channels, algorithm, dtype_bytes)

GPU spec and network speeds are intentionally excluded — they only affect
populate_dag() and the simulation backend, not DAG structure.

Serialisation format
--------------------
DAGs are stored as .npz files (zip of numpy arrays) rather than pickle.
For a large DAG (millions of nodes) this is significantly faster because
numpy saves/loads arrays as a near-memcpy operation with no per-object
Python overhead.

Layout inside the .npz:

  compute_int   – int32 (N_c, 6): node_id, gpu_rank, kernel_code,
                  layer_id, microbatch_id, pipeline_stage, phase_code
                  NOTE: 7 columns despite the name "6" – see _COMPUTE_INT_COLS
  compute_float – float32 (N_c, 3): duration_ms, start_ms, finish_ms
                  (NaN where the original value is None)

  comm_int      – int32/int64 (N_m, 7): node_id, src_gpu, dst_gpu,
                  bytes (int64), collective_code, layer_id, phase_code,
                  flow_id
  comm_float    – float32 (N_m, 3): duration_ms, start_ms, finish_ms
  pfids_flat    – int32 (total parent_flow_ids across all comm nodes)
  pfids_offsets – int32 (N_m + 1): CSR row-pointer into pfids_flat

  edges         – int32 (N_e, 2): src_node_id, dst_node_id

  vocab_json    – single-element object array holding a JSON bytestring
                  {"kernels": [...], "phases": [...], "collectives": [...]}
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulon.backend.dag.nodes import CommNode, ComputeNode, ExecutionDAG
    from simulon.backend.dag.tracer import DAGTracerConfig
    from simulon.config.workload import LLMSpec, MegatronWorkload

logger = logging.getLogger(__name__)

_NAN32 = np.float32("nan")


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _build_vocab(items: list[str]) -> dict[str, int]:
    seen: dict[str, int] = {}
    for s in items:
        if s not in seen:
            seen[s] = len(seen)
    return seen


def _f32(v: float | None) -> np.float32:
    return _NAN32 if v is None else np.float32(v)


def _to_npz(dag: ExecutionDAG) -> dict[str, np.ndarray]:
    """Encode an ExecutionDAG as a dict of numpy arrays."""
    from simulon.backend.dag.nodes import ComputeNode, CommNode

    cn = dag.compute_nodes
    mn = dag.comm_nodes
    ed = dag.edges

    # Build string vocabularies from actual data
    kernel_vocab  = _build_vocab([n.kernel           for n in cn])
    phase_vocab   = _build_vocab([n.phase            for n in cn] +
                                 [n.phase            for n in mn])
    coll_vocab    = _build_vocab([n.collective_type  for n in mn])

    vocab = {
        "kernels":     list(kernel_vocab.keys()),
        "phases":      list(phase_vocab.keys()),
        "collectives": list(coll_vocab.keys()),
    }

    # --- compute_nodes ---
    N_c = len(cn)
    # columns: node_id, gpu_rank, kernel_code, layer_id, microbatch_id, pipeline_stage, phase_code
    c_int = np.empty((N_c, 7), dtype=np.int32)
    c_flt = np.empty((N_c, 3), dtype=np.float32)
    for i, n in enumerate(cn):
        c_int[i] = (n.node_id, n.gpu_rank, kernel_vocab[n.kernel],
                    n.layer_id, n.microbatch_id, n.pipeline_stage,
                    phase_vocab[n.phase])
        c_flt[i] = (_f32(n.duration_ms), _f32(n.start_ms), _f32(n.finish_ms))

    # --- comm_nodes ---
    N_m = len(mn)
    # columns: node_id, src_gpu, dst_gpu, bytes(i64), collective_code, layer_id, phase_code, flow_id
    # bytes can exceed int32, so store as int64; embed in object array → use two arrays instead
    m_int   = np.empty((N_m, 7), dtype=np.int32)   # all except bytes
    m_bytes = np.empty(N_m,      dtype=np.int64)
    m_flt   = np.empty((N_m, 3), dtype=np.float32)

    pfids_flat: list[int] = []
    pfids_offsets = np.empty(N_m + 1, dtype=np.int32)
    pfids_offsets[0] = 0

    for i, n in enumerate(mn):
        m_int[i]   = (n.node_id, n.src_gpu, n.dst_gpu,
                      coll_vocab[n.collective_type], n.layer_id,
                      phase_vocab[n.phase], n.flow_id)
        m_bytes[i] = n.bytes
        m_flt[i]   = (_f32(n.duration_ms), _f32(n.start_ms), _f32(n.finish_ms))
        pfids_flat.extend(n.parent_flow_ids)
        pfids_offsets[i + 1] = len(pfids_flat)

    # --- edges ---
    N_e = len(ed)
    edges = np.empty((N_e, 2), dtype=np.int32)
    for i, e in enumerate(ed):
        edges[i] = (e.src_node_id, e.dst_node_id)

    vocab_arr = np.empty(1, dtype=object)
    vocab_arr[0] = json.dumps(vocab).encode()

    return {
        "compute_int":    c_int,
        "compute_float":  c_flt,
        "comm_int":       m_int,
        "comm_bytes":     m_bytes,
        "comm_float":     m_flt,
        "pfids_flat":     np.array(pfids_flat, dtype=np.int32),
        "pfids_offsets":  pfids_offsets,
        "edges":          edges,
        "vocab_json":     vocab_arr,
    }


def _from_npz(arrays: dict[str, np.ndarray]) -> ExecutionDAG:
    """Reconstruct an ExecutionDAG from the numpy array dict."""
    from simulon.backend.dag.nodes import CommNode, ComputeNode, DAGEdge, ExecutionDAG

    vocab = json.loads(arrays["vocab_json"][0])
    kernels    = vocab["kernels"]
    phases     = vocab["phases"]
    collectives = vocab["collectives"]

    def _opt_f(v: np.float32) -> float | None:
        return None if math.isnan(v) else float(v)

    c_int = arrays["compute_int"]
    c_flt = arrays["compute_float"]
    compute_nodes = [
        ComputeNode(
            node_id=int(c_int[i, 0]),
            gpu_rank=int(c_int[i, 1]),
            kernel=kernels[c_int[i, 2]],
            layer_id=int(c_int[i, 3]),
            microbatch_id=int(c_int[i, 4]),
            pipeline_stage=int(c_int[i, 5]),
            phase=phases[c_int[i, 6]],
            duration_ms=_opt_f(c_flt[i, 0]),
            start_ms=_opt_f(c_flt[i, 1]),
            finish_ms=_opt_f(c_flt[i, 2]),
        )
        for i in range(len(c_int))
    ]

    m_int          = arrays["comm_int"]
    m_bytes        = arrays["comm_bytes"]
    m_flt          = arrays["comm_float"]
    pfids_flat     = arrays["pfids_flat"]
    pfids_offsets  = arrays["pfids_offsets"]
    comm_nodes = [
        CommNode(
            node_id=int(m_int[i, 0]),
            src_gpu=int(m_int[i, 1]),
            dst_gpu=int(m_int[i, 2]),
            bytes=int(m_bytes[i]),
            collective_type=collectives[m_int[i, 3]],
            layer_id=int(m_int[i, 4]),
            phase=phases[m_int[i, 5]],
            flow_id=int(m_int[i, 6]),
            parent_flow_ids=pfids_flat[pfids_offsets[i]:pfids_offsets[i + 1]].tolist(),
            duration_ms=_opt_f(m_flt[i, 0]),
            start_ms=_opt_f(m_flt[i, 1]),
            finish_ms=_opt_f(m_flt[i, 2]),
        )
        for i in range(len(m_int))
    ]

    e = arrays["edges"]
    edges = [DAGEdge(src_node_id=int(e[i, 0]), dst_node_id=int(e[i, 1])) for i in range(len(e))]

    return ExecutionDAG(compute_nodes=compute_nodes, comm_nodes=comm_nodes, edges=edges)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(cache_dir: Path, key: str) -> ExecutionDAG | None:
    path = cache_dir / f"{key}.npz"
    if not path.exists():
        return None
    try:
        arrays = dict(np.load(path, allow_pickle=True))
        dag = _from_npz(arrays)
        logger.info("  DAG cache hit: %s", path.name)
        return dag
    except Exception as exc:
        logger.warning("DAG cache read failed (%s), rebuilding", exc)
        return None


def save(cache_dir: Path, key: str, dag: ExecutionDAG) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    try:
        arrays = _to_npz(dag)
        np.savez(path, **arrays)
        logger.info("  DAG cached: %s", path.name)
    except Exception as exc:
        logger.warning("DAG cache write failed: %s", exc)
