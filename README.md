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
   paths from the workload's `traces_dir` if one is configured.

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
├── simulon/
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
│   │   ├── network.py       # Network backend abstractions
│   │   └── dag/
│   │       ├── nodes.py             # ComputeNode, CommNode, DAGEdge, ExecutionDAG
│   │       ├── tracer.py            # DAGTracerConfig + DAGTracer (ABC)
│   │       ├── trace_tracer.py      # MegatronDagTracer — builds DAG from real GPU traces
│   │       ├── trace_parser.py      # TraceFileParser — validates and loads trace JSON
│   │       ├── network_populate.py  # injects GPU kernel timing and network bandwidth into DAG nodes
│   │       ├── collective_populate.py # collective-level DAG population
│   │       ├── collective_tracer.py # Collective-level DAG tracer
│   │       ├── replayer.py          # critical-path replay → SimulationResult
│   │       ├── chrome_trace.py      # Chrome/Perfetto trace export
│   │       ├── goal_trace.py        # GOAL-compatible trace export
│   │       └── _progress.py         # progress bar utility
│   ├── cli/
│   │   ├── __init__.py      # `simulon simulate`, `simulon trace generate`
│   │   ├── simulate.py      # Simulation sub-command
│   │   ├── trace.py         # Trace generation sub-command
│   │   ├── compare.py       # Compare simulation results
│   │   ├── install.py       # `simulon install apex` / `simulon install deepgemm`
│   │   └── utils.py         # CLI utilities
│   ├── cost.py              # Cost estimation
│   ├── energy.py            # Energy estimation
│   └── tracking/            # Experiment tracking (W&B, MLflow)
│       ├── base.py          # Tracker ABC
│       ├── factory.py       # Tracker factory
│       ├── wandb_tracker.py # Weights & Biases tracker
│       ├── mlflow_tracker.py # MLflow tracker
│       ├── params.py        # Tracking parameters
│       └── env.py           # Environment-based tracker selection
├── templates/
│   ├── gpu/                 # GPU hardware profiles (YAML) + bundled traces
│   ├── cpu/                 # CPU profiles
│   ├── nic/                 # NIC profiles
│   ├── switch/              # Switch profiles
│   ├── node/                # Node templates (snellius-h100-4g, dgx-h100, nvl72-h100, ...)
│   └── workload/            # Workload templates (inheritable Megatron-LM configs)
├── examples/
│   ├── llama3_8b_training.yaml       # Trace-driven (framework: megatron)
│   ├── gpt_oss_20b_training.yaml     # Trace-driven (framework: megatron)
│   └── collective_validation.yaml     # Collective-level validation
├── experiments/              # Validation and use-case experiments
│   ├── validate_e2e/        # End-to-end validation (GPT-OSS, Qwen3)
│   ├── validate_bridge/     # Megatron-Bridge external validation
│   ├── validate_simccl/     # SimCCL network validation
│   ├── usecase_workload_tuning/ # Workload tuning (Qwen3-32B)
│   ├── usecase_system_tuning/   # System tuning (node size, link BW sweep)
│   └── usecase_energy/      # Energy estimation
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
    ├── test_energy.py           # Energy model tests
    ├── test_node_template.py    # Node template tests
    ├── test_tree_nvls_calbusbw.py # Tree/NVLS bandwidth tests
    ├── test_cli_env.py          # CLI environment tests
    └── utils.py                 # Test utilities
```

---

## Installation

Requires Python 3.11+. Uses [uv](https://github.com/astral-sh/uv).

### Standard (core simulation only)

```bash
uv sync
```

This installs the pure-Python simulator. The C++ extension (`simulon._mocknccl`) is optional and only needed for advanced topology queries.

### With C++ extension

```bash
uv sync
python setup.py build_ext --inplace
```

### Development (CUDA required)

For running the instrumented Megatron-LM fork and the test suite, install the extra dependencies:

```bash
uv sync --extra dev
```

The `dev` extra pulls in PyTorch, pytest, the Hugging Face stack (`datasets`, `transformers`), and `transformer_engine[pytorch]`. A CUDA-capable environment is required.

> **Note:** `transformer_engine[pytorch]` may require `--no-build-isolation` because it compiles against your local PyTorch headers. If `uv sync --extra dev` fails on this package, install it manually:
> ```bash
> uv pip install --no-build-isolation transformer_engine[pytorch]
> ```

### Additional GPU-dependent components (optional)

These cannot be declared in `pyproject.toml` because they must be built against your local CUDA / PyTorch toolchain. Install them manually when needed:

```bash
# NVIDIA Apex (layer-norm fusion, gradient scaling, etc.)
simulon install apex

# Flash Attention 3 (Hopper-optimized)
simulon install flash-attn-hopper

# DeepGEMM (DeepSeek MoE kernels — optional)
simulon install deepgemm
```

---

## Quick start

The repository ships with pre-generated GPU execution traces under `templates/gpu/h100/traces/`. You can run a simulation immediately using one of the experiment scenarios that references bundled traces:

```bash
# Simulate GPT-OSS 20B (16 GPUs, TP=1, PP=4, EP=4) using a bundled trace
simulon simulate experiments/validate_e2e/configs/gptoss-bf16/scenario.yaml -v
```

This reads the scenario YAML, resolves the `snellius-h100-4g` node template (which includes measured NCCL bandwidth profiles), loads the compute trace from `templates/gpu/h100/traces/validate-e2e-gptoss-bf16/`, builds the execution DAG, and replays it.

Output:
```
Iteration wall time:  742.560 ms
  Compute:          328.029 ms   44.2%
  Exposed comm:     300.378 ms   40.5%
    AllToAll:                256.058 ms
    ReduceScatter:            26.895 ms
    PP_Send:                  13.652 ms
    AllGather:                11.471 ms
    AllReduce:                 0.850 ms
  Bubble:           114.152 ms   15.4%

Throughput:           5,516.1 tokens/s
  MFU:                12.7%
```

Add `--chrome output/trace.json` to write a Chrome Trace timeline for visual inspection in [Perfetto](https://ui.perfetto.dev):

```bash
simulon simulate experiments/validate_e2e/configs/gptoss-bf16/scenario.yaml --chrome output/trace.json
```
