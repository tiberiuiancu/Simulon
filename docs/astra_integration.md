# Architecture: DAG Trace Extractor

> **Note:** The original ASTRA-Sim C++ integration (pybind11 bindings, CMakeLists,
> `astra_converter/`, `csrc/astra_bindings/`) was removed in favour of a pure-Python
> DAG extractor. This document describes the current architecture.

---

## Overview

simulon produces a **GPU-agnostic execution DAG** from a `ScenarioConfig`. The DAG
is a dependency graph of compute kernels and P2P network flows. It carries no timing
— durations are injected later from GPU profiling data, making the DAG replayable on
any target hardware.

---

## Architecture

```
ScenarioConfig (DatacenterConfig + MegatronWorkload)
    │
    ▼
MegatronDAGTracer.trace()
    │
    ├─ PipelineScheduler   — 1F1B schedule per pipeline stage
    │
    ├─ LayerExpander       — per sublayer: AG → kernels → RS (Megatron SP pattern)
    │
    └─ CCLDecomposer.decompose()  (default: DefaultCCLDecomposer → decompose_collective)
           │
           ├─ ring.py      — AllGather / ReduceScatter / AllReduce / AllToAll
           └─ nvls.py      — NVLS AllReduce (intra-node NVLink Switch)
    │
    ▼
ExecutionDAG
    ├─ compute_nodes: list[ComputeNode]
    ├─ comm_nodes:    list[CommNode]      ← one entry per P2PFlow
    └─ edges:         list[DAGEdge]
```

---

## Key modules

### `simulon.collective`

Pure-Python collective decomposition. Each function returns a flat list of `P2PFlow`
objects with explicit `parent_flow_ids` / `child_flow_ids` dependency chains.

| Function | Algorithm | Notes |
|---|---|---|
| `ring_reduce_scatter` | Ring | `nsteps = N-1`, `chunk_size = data // N // C` |
| `ring_all_gather` | Ring | Same structure, reversed direction |
| `ring_all_reduce` | Ring | RS then AG; AG step-0 parents = RS final-step flows |
| `ring_all_to_all` | Direct | `N*(N-1)` independent flows |
| `nvls_all_reduce` | NVLS | 4 chunks × (gather → NVSwitch → scatter) |

Top-level dispatcher: `decompose_collective(collective_type, group_ranks, data_size, ...)`

### `simulon.backend.dag`

| Module | Responsibility |
|---|---|
| `nodes.py` | `ComputeNode`, `CommNode`, `DAGEdge`, `ExecutionDAG` (with `to_dict` / `to_json`) |
| `pipeline.py` | `PipelineScheduler` — 1F1B warmup / steady-state / cooldown schedule |
| `layer_expander.py` | Expands one sublayer (attn or mlp) into: `AllGather → kernels → ReduceScatter` for fwd/bwd_ig; `kernels` only for bwd_wg |
| `tracer.py` | `DAGTracer` (ABC) + `DAGTracerConfig` |
| `megatron_tracer.py` | `MegatronDAGTracer` — iterates all GPU ranks × pipeline slots × layers × sublayers, fills comm stubs via `CCLDecomposer`, adds PP_Send nodes at stage boundaries |

### `simulon.backend.analytical`

`AnalyticalBackend` is a thin wrapper around `MegatronDAGTracer`:

```python
backend = AnalyticalBackend()
dag = backend.run_trace(scenario)      # returns ExecutionDAG
result = backend.run(scenario)         # returns dict with counts + dag
dag, sim = backend.simulate(scenario)  # also runs replay → SimulationResult
```

---

## DAG node reference

### `ComputeNode`

```python
@dataclass
class ComputeNode:
    node_id: int
    gpu_rank: int
    kernel: str          # layernorm | attn_qkv | attn_flash | attn_proj |
                         # mlp_linear1 | mlp_act | mlp_linear2
    layer_id: int
    microbatch_id: int
    pipeline_stage: int
    phase: str           # fwd | bwd_ig | bwd_wg
```

No `duration` field — GPU-agnostic by design.

### `CommNode`

```python
@dataclass
class CommNode:
    node_id: int
    src_gpu: int
    dst_gpu: int
    bytes: int
    collective_type: str   # AllGather | ReduceScatter | AllReduce | PP_Send
    layer_id: int
    phase: str
    flow_id: int
    parent_flow_ids: list[int]
```

One `CommNode` per `P2PFlow` from `decompose_collective`. `parent_flow_ids` encodes
the intra-collective dependency chain (e.g. ring step s depends on step s-1).

### `DAGEdge`

```python
@dataclass
class DAGEdge:
    src_node_id: int
    dst_node_id: int
```

Encodes inter-node ordering (e.g. AllGather must complete before first kernel).

---

## Per-sublayer execution pattern

Megatron sequence parallelism pattern, implemented in `LayerExpander`:

```
fwd / bwd_ig:
    AllGather(TP) → layernorm → [attn or mlp kernels] → ReduceScatter(TP)

bwd_wg:
    [weight gradient kernels only]   (no TP collective — weight grads reduced by DP AllReduce)

PP boundaries:
    PP_Send CommNode at each pipeline stage boundary
    fwd: stage i → i+1
    bwd: stage i+1 → i
```

---

## Global rank mapping

```
global_rank = dp_rank * (tp * pp) + pp_stage * tp + tp_rank
```

TP group for a given `(dp_rank, pp_stage)`:
```
[global_rank(dp_rank, pp_stage, r) for r in range(tp)]
```

---

## Analytical network model

The DAG replayer (`backend/dag/replayer.py`) models flow timing with a **LogGP-style** formula:

```
transfer_ns = L + 2*o + G*(bytes - 1)
           ≈  latency_ns + bytes / bandwidth
```

With per-link serialisation for contention. `replay(dag, datacenter)` returns a
`SimulationResult` with `total_time_ms` and per-GPU breakdowns.
