# simulon

AI cluster simulator for LLM training workloads. Given a datacenter config and a
workload config, simulon generates a **GPU-agnostic execution DAG** — a dependency
graph of compute kernels and P2P network flows — that can be replayed on any GPU
by injecting profiling data.

---

## What it does

1. **Execution DAG extraction** — parses a `MegatronWorkload` config (model
   architecture + parallelism strategy) and produces an `ExecutionDAG`: a dependency
   graph of `ComputeNode` (kernel ops) and `CommNode` (P2P flows) records representing
   one training iteration across all GPUs, including 1F1B pipeline scheduling.

2. **Collective decomposition** — ring AllGather, ReduceScatter, AllReduce, and AllToAll
   are decomposed into individual `P2PFlow` records with explicit `parent_flow_ids` /
   `child_flow_ids` dependency chains, matching MockNccl semantics.

3. **GPU profiling** — the `simulon profile gpu` CLI benchmarks transformer kernels
   on the local GPU and writes a hardware template YAML with per-kernel timing data.
   This data is injected into a DAG replay to produce GPU-specific timing estimates.

4. **Chrome/Perfetto trace export** — `simulon simulate` replays the DAG and writes a
   Chrome trace JSON readable in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`.

---

## Project structure

```
simulon/
├── src/simulon/
│   ├── config/
│   │   ├── common.py        # DType, Cost
│   │   ├── dc.py            # DatacenterConfig, GPUSpec, KernelRun, ...
│   │   ├── workload.py      # MegatronWorkload, InferenceWorkload, LLMSpec
│   │   └── scenario.py      # ScenarioConfig (datacenter + workload + collective)
│   ├── collective/
│   │   ├── common.py        # P2PFlow dataclass
│   │   ├── ring.py          # ring_reduce_scatter / all_gather / all_reduce / all_to_all
│   │   ├── tree.py          # tree_all_reduce (stub)
│   │   ├── collnet.py       # collnet_direct / collnet_chain (stubs)
│   │   ├── nvls.py          # nvls_all_reduce (stub — intra-node NVLink Switch)
│   │   └── decompose.py     # decompose_collective() top-level dispatcher
│   ├── backend/
│   │   ├── base.py          # Backend ABC
│   │   ├── analytical.py    # AnalyticalBackend — thin wrapper around MegatronDAGTracer
│   │   └── dag/
│   │       ├── nodes.py          # ComputeNode, CommNode, DAGEdge, ExecutionDAG
│   │       ├── pipeline.py       # PipelineScheduler ABC, OneFOneBScheduler, make_scheduler
│   │       ├── layer_expander.py # per-sublayer kernel + comm stub expansion
│   │       ├── tracer.py         # DAGTracer (ABC) + DAGTracerConfig
│   │       ├── megatron_tracer.py # MegatronDAGTracer — assembles full multi-GPU DAG
│   │       ├── populate.py       # injects GPU kernel timing into DAG nodes
│   │       ├── replayer.py       # critical-path replay → SimulationResult
│   │       └── chrome_trace.py   # Chrome/Perfetto trace export
│   ├── workload/
│   │   ├── trace.py         # WorkloadTrace, CommOp, ComputeOp (legacy types)
│   │   └── megatron.py      # generate_megatron_trace() — legacy stub (use MegatronDAGTracer)
│   ├── profiling/
│   │   └── kernels.py       # benchmark_kernels() — CUDA event timing
│   └── cli/
│       └── __init__.py      # `simulon simulate`, `simulon profile gpu`
├── templates/
│   ├── gpu/                 # GPU hardware profiles (YAML)
│   ├── cpu/                 # CPU profiles
│   ├── nic/                 # NIC profiles
│   ├── switch/              # Switch profiles
│   └── model/               # LLM architecture profiles
├── examples/
│   └── scenario_96gpu.yaml  # Example 96-GPU scenario config
├── docs/spec/               # Config format specifications
└── tests/
    ├── test_collective.py   # Collective decomposition unit tests
    ├── test_dag.py          # DAG nodes, pipeline scheduler, LayerExpander
    ├── test_e2e.py          # MegatronDAGTracer + AnalyticalBackend integration
    ├── test_moe.py          # MoE/EP layer expansion and DAG tracing
    ├── test_step.py         # DP gradient sync step phase
    ├── test_legacy_compare.py # Parity record: tracer vs legacy workload model
    └── test_scenario.py     # Config serialisation round-trip
```

---

## Installation

Requires Python 3.11+. Uses [uv](https://github.com/astral-sh/uv). Pure Python — no
build step required.

```bash
uv sync
```

---

## Quick start

### 1. Write a scenario YAML

```yaml
# scenario.yaml
datacenter:
  datacenter:
    name: my-cluster
  cluster:
    num_nodes: 1
  node:
    gpus_per_node: 4
    gpu:
      name: H100
      memory_capacity_gb: 80.0
  network:
    scale_up:
      switch:
        port_speed: 2880Gbps
        latency: 0.000025ms
    scale_out:
      nic:
        speed: 400Gbps
        latency: 0.005ms

collective:
  library: nccl
  algorithm: ring
  num_channels: 1

workload:
  framework: megatron
  model:
    name: llama-7b
    hidden_size: 4096
    num_layers: 32
    num_heads: 32
    ffn_hidden_size: 11008
    vocab_size: 32000
  parallelism:
    tp: 2
    pp: 2
    num_microbatches: 4
    pipeline_schedule: 1f1b
  training:
    num_gpus: 4
    global_batch_size: 4
    micro_batch_size: 1
    sequence_length: 2048
```

### 2. Run the simulation

```bash
simulon simulate scenario.yaml -o trace.json
```

Output:
```
Trace written to trace.json
  GPUs: 4  |  Total: 612.4 ms
  Load in https://ui.perfetto.dev or chrome://tracing
```

Add `-v` to also print per-GPU timing breakdown:

```bash
simulon simulate scenario.yaml -v
```

### 3. Use the Python API directly

```python
from simulon.backend.analytical import AnalyticalBackend
from simulon.config.scenario import ScenarioConfig
import yaml, json

with open("scenario.yaml") as f:
    sc = ScenarioConfig.model_validate(yaml.safe_load(f))

backend = AnalyticalBackend()
dag, result = backend.simulate(sc)

print(f"Total: {result.total_time_ms:.1f} ms")
print(f"compute_nodes: {len(dag.compute_nodes)}")
print(f"comm_nodes:    {len(dag.comm_nodes)}")
```

### 4. Profile a GPU

```bash
simulon profile gpu \
  --name H100-SXM5-80GB \
  --vendor nvidia --memory-capacity-gb 80 --tdp-w 700 \
  --hidden-size 4096 --num-heads 32 --ffn-hidden-size 16384 \
  --seq-len 2048 --batch-size 1 --vocab-size 32000 \
  --dtype bf16 --tp 1 --epoch-num 20 \
  --output templates/gpu/h100.yaml
```

Run with different `--tp`, `--seq-len`, or `--hidden-size` values to build a richer
profile. The command appends new `kernel_runs` entries to the existing file.

---

## DAG node types

**`ComputeNode`** — a single kernel invocation on one GPU:

| Field | Description |
|---|---|
| `node_id` | Unique node ID across the DAG |
| `gpu_rank` | Global GPU rank |
| `kernel` | `layernorm` \| `attn_qkv` \| `attn_flash` \| `attn_proj` \| `mlp_linear1` \| `mlp_act` \| `mlp_linear2` \| `moe_norm` \| `moe_route` \| `moe_expert` |
| `layer_id` | Transformer layer index |
| `microbatch_id` | Pipeline micro-batch index |
| `pipeline_stage` | PP stage |
| `phase` | `fwd` \| `bwd_ig` \| `bwd_wg` |

**`CommNode`** — one P2P flow from a collective decomposition:

| Field | Description |
|---|---|
| `node_id` | Unique node ID |
| `src_gpu`, `dst_gpu` | Sender and receiver global ranks |
| `bytes` | Transfer size in bytes |
| `collective_type` | `AllGather` \| `ReduceScatter` \| `AllReduce` \| `AllToAll` \| `PP_Send` |
| `flow_id` | Unique flow ID within the DAG |
| `parent_flow_ids` | Flow IDs that must complete before this flow starts |

**`DAGEdge`** — dependency between any two nodes:

```json
{ "src_node_id": 5, "dst_node_id": 6 }
```

---

## Collective decomposition

The `simulon.collective` package decomposes collectives into P2P flows independently
of the DAG tracer. The algorithm is taken from the scenario's `collective` block.

```python
from simulon.collective import decompose_collective

result, next_flow_id = decompose_collective(
    collective_type="AllReduce",   # AllGather | ReduceScatter | AllReduce | AllToAll
    group_ranks=[0, 1, 2, 3],
    data_size=1024 * 1024,         # bytes
    num_channels=2,
    algorithm="ring",              # ring | tree | collnet_direct | collnet_chain | nvls | nvls_tree
)

print(f"{len(result.flows)} flows")
for flow in result.flows[:3]:
    print(f"  flow {flow.flow_id}: {flow.src} → {flow.dst}, parents={flow.parent_flow_ids}")
```

---

## Workload config

See [`docs/spec/config-workload.md`](docs/spec/config-workload.md) for the full
specification. Key `parallelism` fields for DAG tracing:

```yaml
parallelism:
  tp: 4                    # tensor parallel degree
  pp: 4                    # pipeline parallel degree
  ep: 2                    # expert parallel degree (MoE only)
  dp: 4                    # data parallel (derived if omitted)
  num_microbatches: 8      # override pipeline micro-batch count (derived if omitted)
  pipeline_schedule: 1f1b  # pipeline schedule (default: 1f1b)
```

---

## Running tests

```bash
uv run pytest
```
