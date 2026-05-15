# simulon

AI cluster simulator for LLM training workloads. Given a datacenter config and a
workload config, simulon generates a **GPU-agnostic execution DAG** — a dependency
graph of compute kernels and P2P network flows — that can be replayed on any GPU
by injecting profiling data.

---

## What it does

1. **Trace-driven DAG extraction** — `MegatronWorkload` reads real GPU execution traces
   emitted by the instrumented Megatron-LM fork and builds an `ExecutionDAG`: a
   dependency graph of `ComputeNode` (kernel ops) and `CommNode` (P2P flows) records
   representing one training iteration across all GPUs. The tracer infers trace file
   paths from the datacenter's `traces_dir` if one is configured.

2. **Collective decomposition** — ring AllGather, ReduceScatter, AllReduce, and AllToAll
   are decomposed into individual `P2PFlow` records with explicit `parent_flow_ids` /
   `child_flow_ids` dependency chains, matching MockNccl semantics.

3. **Chrome/Perfetto trace export** — `simulon simulate` replays the DAG and writes a
   Chrome trace JSON readable in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`.

4. **Network bandwidth population** — `simulon simulate` populates each `CommNode` with
   real network bandwidth data from NCCL profiles and decomposes collectives using the
   configured CCL library and algorithm.

5. **GOAL trace integration** — `simulon simulate` can emit GOAL-compatible JSON traces
   alongside Chrome traces for integration with external network simulation tools.

---

## Project structure

```
simulon/
├── src/simulon/
│   ├── config/
│   │   ├── common.py        # DType, Cost
│   │   ├── dc.py            # DatacenterConfig, GPUSpec, KernelRun, ...
│   │   ├── workload.py      # MegatronWorkload, CollectiveWorkload
│   │   ├── scenario.py      # ScenarioConfig (datacenter + workload + collective)
│   │   ├── resolve.py       # Factory functions for template resolution
│   │   └── nccl_profile.py  # NCCLProfile, NCCLEntry
│   ├── collective/
│   │   ├── common.py        # P2PFlow dataclass
│   │   ├── ring.py          # ring_reduce_scatter / all_gather / all_reduce / all_to_all
│   │   ├── tree.py          # tree_all_reduce (stub)
│   │   ├── collnet.py       # collnet_direct / collnet_chain (stubs)
│   │   ├── nvls.py          # nvls_all_reduce (stub — intra-node NVLink Switch)
│   │   ├── decompose.py     # decompose_collective() top-level dispatcher
│   │   └── calbusbw.py      # bus bandwidth calculation
│   ├── backend/
│   │   ├── base.py          # Backend ABC
│   │   ├── analytical.py    # AnalyticalBackend — dispatches MegatronDagTracer
│   │   └── dag/
│   │       ├── nodes.py          # ComputeNode, CommNode, DAGEdge, ExecutionDAG
│   │       ├── pipeline.py       # PipelineScheduler ABC, OneFOneBScheduler, make_scheduler
│   │       ├── layer_expander.py # per-sublayer kernel + comm stub expansion
│   │       ├── tracer.py         # DAGTracer (ABC) + DAGTracerConfig
│   │       ├── trace_tracer.py   # MegatronDagTracer — builds DAG from real GPU execution traces
│   │       ├── trace_parser.py   # TraceFileParser — validates and loads trace JSON files
│   │       ├── network_populate.py # injects GPU kernel timing and network bandwidth into DAG nodes
│   │       ├── replayer.py       # critical-path replay → SimulationResult
│   │       ├── chrome_trace.py   # Chrome/Perfetto trace export
│   │       ├── goal_trace.py     # GOAL-compatible trace export
│   │       ├── collective_tracer.py # Collective-level DAG tracer
│   │       └── _progress.py      # progress bar utility
│   ├── cli/
│   │   ├── __init__.py      # `simulon simulate`, `simulon trace generate`
│   │   ├── trace.py         # Trace generation sub-command
│   │   └── install.py       # `simulon install apex` / `simulon install deepgemm`
│   ├── cost.py              # Cost estimation
│   ├── energy.py            # Energy estimation
│   └── tracking/            # Experiment tracking
├── templates/
│   ├── gpu/                 # GPU hardware profiles (YAML)
│   ├── cpu/                 # CPU profiles
│   ├── nic/                 # NIC profiles
│   ├── switch/              # Switch profiles
│   └── model/               # LLM architecture profiles
├── examples/
│   ├── llama3_8b_training_new.yaml    # Trace-driven (framework: megatron)
│   ├── gpt_oss_20b_training_new.yaml  # Trace-driven (framework: megatron)
│   ├── gpt_oss_20b_training_single_node.yaml  # Trace-driven (framework: megatron)
│   ├── deepseek_v3_training.yaml      # (framework: megatron-deprecated)
│   ├── gpt_oss_5b_training.yaml       # (framework: megatron-deprecated)
│   ├── gpt_oss_20b_training.yaml      # (framework: megatron-deprecated)
│   ├── llama7b_32gpu_training.yaml    # (framework: megatron-deprecated)
│   └── qwen3_30b_training.yaml        # (framework: megatron-deprecated)
├── docs/spec/               # Config format specifications
└── tests/
    ├── test_collective.py       # Collective decomposition unit tests
    ├── test_collective_workload.py # Collective workload config tests
    ├── test_chrome_trace.py     # Chrome trace export tests
    ├── test_trace_tracer.py     # MegatronDagTracer unit tests
    ├── test_trace_parser.py     # TraceFileParser tests
    ├── test_trace_e2e.py        # End-to-end trace-driven simulation
    ├── test_trace_refactor.py   # Trace refactoring tests
    ├── test_goal_trace.py       # GOAL trace export tests
    ├── test_replayer.py         # DAG replayer tests
    ├── test_node_template.py    # Node template tests
    ├── test_tree_nvls_calbusbw.py # Tree/NVLS bandwidth tests
    └── utils.py                 # Test utilities
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
  config:
    num-layers: 32
    hidden-size: 4096
    num-attention-heads: 32
    ffn-hidden-size: 11008
    vocab-size: 32000
    tensor-model-parallel-size: 2
    pipeline-model-parallel-size: 2
    micro-batch-size: 1
    global-batch-size: 4
    seq-length: 2048
    num_gpus: 4
    dtype: bf16
```

### 2. Generate a trace and simulate

```bash
# 2a. Run the instrumented Megatron-LM fork to produce execution traces
simulon trace generate scenario.yaml -o traces/

# 2b. Simulate using the generated traces
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
specification. See the `_new.yaml` examples in `examples/` for trace-driven configs:

- [`examples/llama3_8b_training_new.yaml`](examples/llama3_8b_training_new.yaml)
- [`examples/gpt_oss_20b_training_new.yaml`](examples/gpt_oss_20b_training_new.yaml)
- [`examples/gpt_oss_20b_training_single_node.yaml`](examples/gpt_oss_20b_training_single_node.yaml)

All new workloads use `framework: megatron` with a flat `config` dictionary. The
deprecated `megatron-deprecated` framework (nested `model`/`parallelism`/`training`
blocks) exists only in older example files.

---

## Running tests

```bash
uv run pytest
```
