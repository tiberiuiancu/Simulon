# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-21
**Commit:** 88f3d13
**Branch:** main

## OVERVIEW

AI cluster simulator for LLM training. Transforms a datacenter + workload YAML into a GPU-agnostic execution DAG (compute kernels + P2P flows), injects profiled kernel timings, and replays to produce per-GPU timing estimates with Chrome/Perfetto trace export. Pure Python core + optional C++ pybind11 extension (MockNccl).

## STRUCTURE

```
simulon/
├── src/simulon/              # Main package (src-layout)
│   ├── config/               # Pydantic v2 config models, foundation layer, highest fan-out
│   ├── collective/           # CCL decomposition: collective → P2PFlow list
│   ├── backend/
│   │   ├── analytical.py     # AnalyticalBackend, orchestrates trace→populate→replay
│   │   └── dag/              # Core engine: DAG tracing, populate, replay, export (13 files)
│   ├── cli/                  # Typer CLI: simulate, profile gpu/node, install
│   ├── profiling/            # GPU kernel benchmarking (CUDA event timing) + lookup
│   ├── tracking/             # Experiment tracking: MLflow, WandB (ABC + factory)
│   ├── cost.py               # CAPEX/OPEX cost model
│   └── energy.py             # Per-component power + PUE energy model
├── templates/                # YAML hardware/model profiles (gpu/, cpu/, nic/, switch/, model/, node/)
├── csrc/                     # MockNccl C++17 extension (pybind11), optional
├── vendor/aicb/              # Vendored AICB reference, NOT imported, zero code coupling
├── tests/                    # pytest suite (25+ files, ~80k lines including fixtures)
├── experiments/validation/   # SimCCL validation vs real NCCL (git submodule: nccl-tests)
├── examples/                 # 4 scenario YAMLs (llama7b, gpt-oss-20b, deepseek-v3, qwen3)
├── docs/spec/                # Config format specs (dc, workload, scenario)
└── scripts/                  # SLURM profiling script (profile_h100.sh)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new kernel type | `backend/dag/layer_expander.py` + `profiling/kernels.py` | Expansion patterns + benchmark function |
| Add collective algorithm | `collective/{algo}.py` + `collective/decompose.py` registry | Register in `_ALGORITHM_MAP` |
| New parallelism dimension | `backend/dag/megatron_tracer.py` | 4-level loop (dp,pp,ep,tp) + `ParallelGroups` |
| New pipeline schedule | `backend/dag/pipeline.py` | Subclass `PipelineScheduler`, register in `make_scheduler` |
| New config field | `config/workload.py` or `config/dc.py` | Pydantic BaseModel, update YAML spec in `docs/spec/` |
| GPU template creation | `simulon profile gpu` CLI | Outputs to `templates/gpu/*.yaml` |
| Node template from nccl-tests | `simulon profile node` CLI | Parses JSON or runs nccl-tests live |
| New export format | `backend/dag/` (add file like `chrome_trace.py`) | Read ExecutionDAG, emit format |
| New experiment tracker | `tracking/` , subclass `ExperimentTracker` | Register in `tracking/factory.py` |
| Cost/energy model changes | `cost.py`, `energy.py` | Standalone modules, config-dependent |
| Template resolution | `config/resolve.py` | Factory functions: `resolve_gpu_spec`, `resolve_node_spec`, etc. |

## DATA FLOW

```
ScenarioConfig (YAML)
  → MegatronDAGTracer.trace()     # structure only, no timing
  → populate_dag()                # inject ComputeNode.duration_ms from GPUSpec
  → populate_network()            # inject CommNode.duration_ms from link BW/latency
  → replay()                      # topological walk → SimulationResult
  → to_chrome_trace()             # export to Perfetto
```

Orchestrated by `AnalyticalBackend.simulate()`.

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `AnalyticalBackend` | Class | `backend/analytical.py` | Top-level orchestrator: trace→populate→replay |
| `MegatronDAGTracer` | Class | `backend/dag/megatron_tracer.py` | Builds ExecutionDAG from MegatronWorkload |
| `ExecutionDAG` | Dataclass | `backend/dag/nodes.py` | Container: compute_nodes + comm_nodes + edges |
| `ComputeNode` / `CommNode` | Dataclass | `backend/dag/nodes.py` | DAG node types |
| `LayerExpander` | Class | `backend/dag/layer_expander.py` | Sublayer → kernel stubs + comm stubs |
| `OneFOneBScheduler` | Class | `backend/dag/pipeline.py` | 1F1B pipeline schedule |
| `populate_dag` | Function | `backend/dag/populate.py` | Injects kernel timing from GPUSpec |
| `populate_network` | Function | `backend/dag/populate.py` | Injects comm timing from link params |
| `replay` | Function | `backend/dag/replayer.py` | Critical-path walk → SimulationResult |
| `decompose_collective` | Function | `collective/decompose.py` | Dispatcher: collective → P2PFlow list |
| `CCLDecomposer` | Protocol | `collective/__init__.py` | Interface for collective decomposition |
| `NCCLDecomposer` | Class | `collective/__init__.py` | NCCL implementation (delegates to registry) |
| `ScenarioConfig` | Pydantic | `config/scenario.py` | Top-level: datacenter + workload + collective |
| `DatacenterConfig` | Pydantic | `config/dc.py` | Cluster topology + hardware specs |
| `MegatronWorkload` | Pydantic | `config/workload.py` | Model + parallelism + training params |
| `GPUSpec` | Pydantic | `config/dc.py` | GPU hardware: kernel_runs timing data |
| `lookup_kernel_time` | Function | `profiling/lookup.py` | Nearest-neighbor match on (kernel, params) |
| `benchmark_kernels` | Function | `profiling/kernels.py` | CUDA event timing for transformer kernels |
| `cal_busbw` | Function | `collective/calbusbw.py` | Bandwidth calibration from NCCL profile |

## CONVENTIONS

- **Pydantic v2 everywhere**, all configs are `BaseModel`, validated + serialized via `model_validate`/`model_dump`
- **src-layout**, imports are `from simulon.config.dc import DatacenterConfig`, not relative
- **`uv` only**, `uv sync`, `uv run pytest`. Never global `python` or `pip`
- **No linting enforced**, no ruff/flake8/mypy config. Developer discipline only
- **Stubs raise NotImplementedError** — collnet_direct, collnet_chain collectives; RCCLDecomposer. Never delete stubs
- **Templates resolved from CWD**, `templates/gpu/<name>.yaml`. Run CLI from repo root
- **DAG cache**, `~/.cache/simulon/dag/` by default. Key excludes GPU/network spec (structure-only)
- **Numpy .npz serialization**, DAG cache uses numpy arrays for fast I/O on large DAGs
- **Factory functions in `config/resolve.py`**, lazy resolution of GPU spec, node spec, NCCL profile
- **`_` prefix for internal helpers**, `_parse_speed`, `_emit_compute_node`, etc.
- **Inline test fixtures**, no `conftest.py`. Each test file defines its own `make_*` factories
- **Parameter sync**, `profiling/kernels.py` parameter formulas MUST match `backend/dag/megatron_tracer.py`

## ANTI-PATTERNS (THIS PROJECT)

- **Only AdamW optimizer modeled**, `megatron_tracer.py:286`. Extend there for others
- **MoE token distribution assumed uniform**, vendor/aicb MockedMegatron. In practice, tokens are NOT evenly split across experts
- **vendor/aicb is dead code**, zero Python imports from simulon. Reference material only. Safe to remove
- **No CI/CD**, no GitHub Actions, no pre-commit hooks. All testing is manual
- **`workload/` package is empty**, legacy module, unused

## COMMANDS

```bash
uv sync                                          # install deps
uv sync --extra cpp && python setup.py build_ext --inplace  # build C++ extension
uv run pytest                                    # run all tests
uv run pytest tests/test_dag.py::test_name       # single test
simulon simulate scenario.yaml -o trace.json     # run simulation
simulon simulate scenario.yaml -v                # verbose per-GPU breakdown
simulon profile gpu --name H100 ...              # profile local GPU
simulon profile node --name H100-node ...        # generate node template from nccl-tests
simulon install apex                             # install NVIDIA Apex
```

## NOTES

- **Compact mode** (`compact=True`) fuses consecutive compute-only sublayers into single nodes. Tracks originals in `fused_kernels` list. Reduces node count for large DAGs
- **Flow-based dependencies**, `CommNode.parent_flow_ids` encode collective decomposition deps (separate from `DAGEdge`)
- **Stub-and-replace pattern**, `LayerExpander` creates comm stubs with `flow_id=-1`; `MegatronDAGTracer` replaces with actual P2P flows from `CCLDecomposer`
- **Analytical network model**, no congestion. Each flow: `duration = latency + bytes/bandwidth`. Intra-node (NVLink) vs inter-node (NIC) handled separately
- **NCCL profile bandwidth calibration**, `cal_busbw()` selects algorithm + overrides raw link BW when nccl profile is available
- **Rank ordering**, tp-cp-ep-dp-pp (cp = context parallel, placeholder)
- **Git submodule**, `experiments/validation/simccl/nccl-tests` points to NVIDIA/nccl-tests
- **Python 3.13**, specified in `.python-version`
