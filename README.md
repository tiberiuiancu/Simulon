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

2. **Collective decomposition** — ring AllGather, ReduceScatter, AllReduce, AllToAll
   and NVLS AllReduce are decomposed into individual `P2PFlow` records with explicit
   `parent_flow_ids` / `child_flow_ids` dependency chains, matching MockNccl semantics.

3. **GPU profiling** — the `simulon profile gpu` CLI benchmarks transformer kernels
   on the local GPU and writes a hardware template YAML with per-kernel timing data.
   This data can be injected into a DAG replay to produce GPU-specific timing estimates.

4. **Workload trace generation** — the legacy `generate_megatron_trace()` API produces
   a per-representative-GPU `WorkloadTrace` (ordered `CommOp`/`ComputeOp` sequence).
   Useful for analytical modelling; the DAG is the preferred output for new work.

---

## Project structure

```
simulon/
├── src/simulon/
│   ├── config/
│   │   ├── common.py        # DType, Cost
│   │   ├── dc.py            # DatacenterConfig, GPUSpec, KernelRun, ...
│   │   ├── workload.py      # MegatronWorkload, InferenceWorkload, LLMSpec
│   │   └── scenario.py      # ScenarioConfig (datacenter + workload)
│   ├── collective/
│   │   ├── common.py        # P2PFlow dataclass
│   │   ├── ring.py          # ring_reduce_scatter / all_gather / all_reduce / all_to_all
│   │   ├── nvls.py          # nvls_all_reduce (intra-node NVLink Switch)
│   │   └── decompose.py     # decompose_collective() top-level dispatcher
│   ├── backend/
│   │   ├── base.py          # Backend ABC
│   │   ├── astra_sim.py     # AstraSimBackend → DAGTracer wrapper
│   │   └── dag/
│   │       ├── nodes.py     # ComputeNode, CommNode, DAGEdge, ExecutionDAG
│   │       ├── pipeline.py  # PipelineScheduler (1F1B)
│   │       ├── layer_expander.py  # per-sublayer kernel + comm stub expansion
│   │       └── tracer.py    # DAGTracer — assembles full multi-GPU DAG
│   ├── workload/
│   │   ├── trace.py         # WorkloadTrace, CommOp, ComputeOp (legacy)
│   │   └── megatron.py      # generate_megatron_trace() (legacy)
│   ├── profiling/
│   │   └── kernels.py       # benchmark_kernels() — CUDA event timing
│   └── cli/
│       └── __init__.py      # `simulon trace`, `simulon simulate`, `simulon profile gpu`
├── templates/
│   ├── gpu/                 # GPU hardware profiles (YAML)
│   ├── cpu/                 # CPU profiles
│   ├── nic/                 # NIC profiles
│   ├── switch/              # Switch profiles
│   └── model/               # LLM architecture profiles
├── examples/
│   └── scenario.yaml        # Example scenario config
├── docs/spec/               # Config format specifications
└── tests/
    ├── test_collective.py   # Collective decomposition (ring + NVLS)
    ├── test_dag.py          # DAG nodes, PipelineScheduler, LayerExpander
    ├── test_e2e.py          # DAGTracer + AstraSimBackend integration
    ├── test_megatron.py     # Legacy workload trace generation
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

### 1. Extract an execution DAG from a scenario file

```bash
simulon trace examples/scenario.yaml --output dag.json
```

Output:
```
DAG written to dag.json
  compute_nodes: 1280
  comm_nodes:    1032
  edges:         1408
```

### 2. Write a scenario YAML

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
      topology:
        type: fat_tree
        params:
          k: 4

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
  training:
    num_gpus: 4
    global_batch_size: 4
    micro_batch_size: 1
    sequence_length: 2048
```

### 3. Use the Python API directly

```python
from simulon.backend.dag import DAGTracer, DAGTracerConfig
from simulon.config.scenario import ScenarioConfig
import yaml, json

with open("scenario.yaml") as f:
    sc = ScenarioConfig.model_validate(yaml.safe_load(f))

dag = DAGTracer(DAGTracerConfig(num_channels=1, algorithm="ring")).trace(
    sc.workload, sc.datacenter
)

print(f"compute_nodes: {len(dag.compute_nodes)}")
print(f"comm_nodes:    {len(dag.comm_nodes)}")
print(f"edges:         {len(dag.edges)}")

with open("dag.json", "w") as f:
    f.write(dag.to_json())
```

### 4. Inspect the DAG JSON

The output has three flat arrays:

```json
{
  "compute_nodes": [
    {
      "node_id": 1, "gpu_rank": 0, "kernel": "layernorm",
      "layer_id": 0, "microbatch_id": 0, "pipeline_stage": 0, "phase": "fwd"
    }
  ],
  "comm_nodes": [
    {
      "node_id": 6, "src_gpu": 0, "dst_gpu": 1, "bytes": 8388608,
      "collective_type": "AllGather", "layer_id": 0, "phase": "fwd",
      "flow_id": 0, "parent_flow_ids": []
    }
  ],
  "edges": [
    { "src_node_id": 0, "dst_node_id": 1 }
  ]
}
```

`compute_nodes` carry no duration — inject kernel timing from a GPU profile to replay
on a specific target.

### 5. Profile a GPU

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

**`ComputeNode`** — a single kernel invocation on one GPU, no duration:

| Field | Description |
|---|---|
| `node_id` | Unique node ID across the DAG |
| `gpu_rank` | Global GPU rank |
| `kernel` | `layernorm` \| `attn_qkv` \| `attn_flash` \| `attn_proj` \| `mlp_linear1` \| `mlp_act` \| `mlp_linear2` |
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
| `collective_type` | `AllGather` \| `ReduceScatter` \| `AllReduce` \| `PP_Send` |
| `flow_id` | Unique flow ID within the DAG |
| `parent_flow_ids` | Flow IDs that must complete before this flow starts |

**`DAGEdge`** — dependency between any two nodes:

```json
{ "src_node_id": 5, "dst_node_id": 6 }
```

---

## Collective decomposition

The `simulon.collective` package decomposes collectives into P2P flows independently
of the DAG tracer:

```python
from simulon.collective import decompose_collective

result, next_flow_id, next_node_id = decompose_collective(
    collective_type="AllReduce",   # AllGather | ReduceScatter | AllReduce | AllToAll
    group_ranks=[0, 1, 2, 3],
    data_size=1024 * 1024,         # bytes
    num_channels=2,
    algorithm="ring",              # ring | nvls
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
  dp: 4                    # data parallel (derived if omitted)
  num_microbatches: 8      # override pipeline micro-batch count (derived if omitted)
```

---

## Running tests

```bash
uv run pytest
```
