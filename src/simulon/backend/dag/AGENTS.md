# backend/dag — Core Simulation Engine

13 files, ~2400 lines. Most complex package in simulon.

## OVERVIEW

Builds, populates, replays, and exports the execution DAG — the dependency graph of compute kernels and P2P flows representing one LLM training iteration across all GPUs.

## STRUCTURE

```
dag/
├── nodes.py             # ComputeNode, CommNode, DAGEdge, ExecutionDAG dataclasses
├── tracer.py            # DAGTracer ABC + DAGTracerConfig
├── trace_tracer.py      # MegatronDagTracer — builds DAG from real GPU execution traces
├── trace_parser.py      # TraceFileParser — validates and loads trace JSON files
├── layer_expander.py    # Sublayer → kernel stubs + comm stubs per phase
├── pipeline.py          # PipelineScheduler ABC, OneFOneBScheduler, make_scheduler
├── collective_tracer.py # Standalone collective-only DAG builder
├── network_populate.py  # Inject network BW/latency into CommNodes
├── replayer.py          # Critical-path topological walk → SimulationResult
├── chrome_trace.py      # Chrome/Perfetto trace JSON export
├── goal_trace.py        # GOAL format export (for ATLAS/LogGOPSim)
├── _progress.py         # Progress bar logging utility
└── __init__.py          # Public API: 17 exports
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add kernel type | `layer_expander.py` | Add to expansion patterns for attn/mlp/moe phases |
| New pipeline schedule | `pipeline.py` | Subclass `PipelineScheduler`, register in `make_scheduler` |
| New parallelism dim | `trace_tracer.py` | Modify rank formula + `ParallelGroups` |
| New export format | Add new file | Follow `chrome_trace.py` pattern: read ExecutionDAG, emit |
| Modify timing injection | `network_populate.py` | `populate_network()` → CommNode.duration_ms (latency + bytes/bandwidth) |
| Debug replay timing | `replayer.py` | Kahn's topo sort → per-GPU interval merging |
| Trace parsing issues | `trace_parser.py` | Validates and loads trace JSON files |

## DATA FLOW (within this package)

```
TraceFileParser
  ├─ MegatronDagTracer.trace()
  │   ├─ LayerExpander.expand_sublayer()  → compute stubs + comm stubs (flow_id=-1)
  │   ├─ CCLDecomposer.decompose()        → replace stubs with P2P flows
  │   ├─ OneFOneBScheduler.schedule_for_stage() → fwd/bwd slot ordering
  │   └─ ExecutionDAG (structure only)
  │        ↓
  populate_network() → CommNode.duration_ms (latency + bytes/bandwidth)
  │        ↓
  replay()          → SimulationResult (total_time, compute, exposed_comm, bubble)
  │        ↓
  to_chrome_trace() → JSON for Perfetto
```

## CONVENTIONS

- **Stub-and-replace** — `LayerExpander` emits comm stubs with `flow_id=-1`; `MegatronDagTracer` replaces with real P2P flows
- **Compact mode** — fuses sequential compute-only sublayers into single nodes; `fused_kernels` list tracks originals
- **Two dependency systems** — `DAGEdge` for sequential compute deps; `CommNode.parent_flow_ids` for collective flow deps
- **`_` prefix** — all internal helpers: `_emit_compute_node`, `_parse_speed`, `_merge_intervals`

## ANTI-PATTERNS

- **Trace-driven complexity** — `trace_tracer.py` builds DAG from real GPU traces; trace file validation and path resolution are key failure points
- **Only 1F1B schedule** — `make_scheduler` only supports `"1f1b"`. New schedules need both the scheduler AND wiring in trace_tracer
- **No congestion model** — `populate_network` is purely analytical: `duration = latency + bytes/bandwidth`
