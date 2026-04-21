# backend/dag — Core Simulation Engine

13 files, ~2400 lines. Most complex package in simulon.

## OVERVIEW

Builds, populates, replays, and exports the execution DAG — the dependency graph of compute kernels and P2P flows representing one LLM training iteration across all GPUs.

## STRUCTURE

```
dag/
├── nodes.py             # ComputeNode, CommNode, DAGEdge, ExecutionDAG dataclasses
├── tracer.py            # DAGTracer ABC + DAGTracerConfig
├── megatron_tracer.py   # MegatronDAGTracer — main builder (704 lines, highest complexity)
├── layer_expander.py    # Sublayer → kernel stubs + comm stubs per phase
├── pipeline.py          # PipelineScheduler ABC, OneFOneBScheduler, make_scheduler
├── collective_tracer.py # Standalone collective-only DAG builder
├── populate.py          # Inject GPU kernel timing + network BW/latency into nodes
├── replayer.py          # Critical-path topological walk → SimulationResult
├── chrome_trace.py      # Chrome/Perfetto trace JSON export
├── goal_trace.py        # GOAL format export (for ATLAS/LogGOPSim)
├── cache.py             # Content-addressable DAG cache (.npz, numpy arrays)
├── _progress.py         # Progress bar logging utility
└── __init__.py          # Public API: 17 exports
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add kernel type | `layer_expander.py` | Add to expansion patterns for attn/mlp/moe phases |
| New pipeline schedule | `pipeline.py` | Subclass `PipelineScheduler`, register in `make_scheduler` |
| New parallelism dim | `megatron_tracer.py` | Modify 4-level loop (dp,pp,ep,tp) + `ParallelGroups` |
| New export format | Add new file | Follow `chrome_trace.py` pattern: read ExecutionDAG, emit |
| Modify timing injection | `populate.py` | `populate_dag` (compute) / `populate_network` (comm) |
| Debug replay timing | `replayer.py` | Kahn's topo sort → per-GPU interval merging |
| Cache issues | `cache.py` | Key = SHA256(model+parallelism+training+tracerconfig) |

## DATA FLOW (within this package)

```
MegatronDAGTracer.trace()
  ├─ LayerExpander.expand_sublayer()  → compute stubs + comm stubs (flow_id=-1)
  ├─ CCLDecomposer.decompose()        → replace stubs with P2P flows
  ├─ OneFOneBScheduler.schedule_for_stage() → fwd/bwd slot ordering
  └─ ExecutionDAG (structure only)
       ↓
populate_dag()    → ComputeNode.duration_ms (from GPUSpec kernel_runs)
populate_network() → CommNode.duration_ms (latency + bytes/bandwidth)
       ↓
replay()          → SimulationResult (total_time, compute, exposed_comm, bubble)
       ↓
to_chrome_trace() → JSON for Perfetto
```

## CONVENTIONS

- **Stub-and-replace** — `LayerExpander` emits comm stubs with `flow_id=-1`; `MegatronDAGTracer` replaces with real P2P flows
- **Compact mode** — fuses sequential compute-only sublayers into single nodes; `fused_kernels` list tracks originals
- **Two dependency systems** — `DAGEdge` for sequential compute deps; `CommNode.parent_flow_ids` for collective flow deps
- **Cache excludes GPU/network** — only structure (model+parallelism+training) is keyed; timing is always re-injected
- **`_` prefix** — all internal helpers: `_emit_compute_node`, `_parse_speed`, `_merge_intervals`

## ANTI-PATTERNS

- **megatron_tracer.py is 704 lines** — the 4-level nested loop (dp,pp,ep,tp) at lines 306-652 is the hardest code in the project. Modify with extreme care
- **Parameter sync required** — parameter count formulas in `megatron_tracer.py` lines 64-105 MUST match `profiling/kernels.py` lines 530-542
- **Only 1F1B schedule** — `make_scheduler` only supports `"1f1b"`. New schedules need both the scheduler AND wiring in megatron_tracer
- **No congestion model** — `populate_network` is purely analytical: `duration = latency + bytes/bandwidth`
