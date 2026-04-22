# Multi-Workload Support in Simulon (TIB-104)

## TL;DR

> **Quick Summary**: Extend Simulon to support multiple concurrent named workloads per scenario, with greedy node placement, start offsets, `after_finish` dependencies, and merged-DAG replay. Backward-compatible with existing single-workload configs.
>
> **Deliverables**:
> - `templates/workload/` directory with example workload templates
> - `ScenarioConfig` with named `workloads` list and backward-compat `workload` alias
> - `WorkloadInstance`, `StartConfig` pydantic models
> - Node placement engine with GPU budget validation
> - Dependency resolver with cycle detection
> - `SimulationOutput` return type for `simulate()`
> - `replay()` with first-class `start_offsets` support
> - DAG merge utility (`merge_dags()`)
> - GOAL and Chrome trace export with idle-block generation
> - Multi-workload CLI output
> - Test suite covering config validation, DAG merge, and two-workload integration
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: Config models → Node placement → Trace per workload → Merge DAGs → Replay with offsets → CLI update

---

## Context

### Original Request
Simulon currently supports a single workload per scenario. ATLAHS (TIB-104) requires modeling network congestion under multiple concurrent workloads. This ticket adds multi-workload config support, independent tracing per workload, merged-DAG concurrent replay, and unified trace export.

### Interview Summary
**Key Discussions**:
- Workload templates directory (`templates/workload/`) following GPU template convention
- Named workload instances with optional `start` block (`offset_ms`, `after_finish`)
- Backward compat: singular `workload` expands to single-element `workloads` list
- Greedy contiguous node placement, each workload spans ≥1 full node
- GPU budget validation at config parse time
- DAG tracing per workload against its node slice
- Effective start time = max(offset_ms, finish_time(dep) for dep in after_finish)
- Dependency cycle detection at config time
- Merged-DAG architecture: trace independently → merge → replay once → extract per-workload stats
- First-class waiting offsets in replayer (no fake idle compute nodes)
- GOAL/Chrome export generates idle blocks from offsets for ATLAHS compatibility
- `simulate()` returns `SimulationOutput` with backward-compatible `__iter__`
- Energy/cost computed on merged DAG (correct for concurrent execution)
- All three workload types in config; InferenceWorkload simulation raises NotImplementedError
- CollectiveWorkload constrained to node slice via new `num_gpus` param
- Tests added by separate agent after implementation

**Research Findings**:
- `ScenarioConfig` currently has single `workload: Union[Path, WorkloadConfig]`
- `AnalyticalBackend.simulate()` returns `tuple[ExecutionDAG, SimulationResult]` (25 call sites)
- Energy is non-additive for shared clusters; merged DAG approach is correct
- `replay()` is a pure scheduler (Kahn's algorithm) that could support offsets
- GOAL export exists at `src/simulon/backend/dag/goal_trace.py`
- Node IDs monotonic within trace, GPU ranks contiguous 0..num_gpus-1
- `populate_network` must run on merged DAG with full datacenter
- `compute_energy()` uses `max(finish_ms)` across all nodes — requires merged DAG for correctness

### Metis Review
**Identified Gaps** (addressed):
- InferenceWorkload has no tracer → excluded from simulation scope
- CollectiveWorkload uses full cluster → constrained to node slice
- Idle blocks inflating metrics → first-class offsets in replayer instead
- CLI output shape → multi-workload summary + per-workload summaries
- Cache invalidation → cache disabled for multi-workload or key includes all states
- GOAL rank contiguity → enforced by greedy contiguous placement
- Flow ID collision in merge → offset per workload in merge utility

---

## Work Objectives

### Core Objective
Enable Simulon to simulate multiple concurrent workloads on a shared datacenter, with independent tracing, dependency-aware scheduling, and correct aggregate energy/cost metrics.

### Concrete Deliverables
- `templates/workload/gpt-oss-20b-16gpu.yaml` (example workload template)
- `examples/multi_workload.yaml` (example two-workload scenario for CLI testing)
- `src/simulon/config/scenario.py` — `WorkloadInstance`, `StartConfig`, updated `ScenarioConfig`
- `src/simulon/config/workload.py` — `CollectiveWorkload.num_gpus` field
- `src/simulon/config/placement.py` — Node placement engine (new file)
- `src/simulon/backend/dag/merge.py` — DAG merge utility (new file)
- `src/simulon/backend/dag/replayer.py` — `start_offsets` parameter
- `src/simulon/backend/dag/goal_trace.py` — `start_offsets` parameter for idle blocks
- `src/simulon/backend/dag/chrome_trace.py` — `start_offsets` parameter for idle events
- `src/simulon/backend/analytical.py` — Multi-workload `simulate()` and `run()`
- `src/simulon/energy.py` — Handle merged DAG (no changes likely, but verify)
- `src/simulon/cli/__init__.py` — Multi-workload summary output

### Definition of Done
- [ ] `uv run pytest tests/test_scenario.py` passes (backward compat)
- [ ] Two-workload scenario runs end-to-end: `backend.simulate(sc)` returns `SimulationOutput` with `by_workload` dict
- [ ] GOAL export of two-workload scenario produces valid ATLAHS input
- [ ] Chrome trace of two-workload scenario shows both workloads in Perfetto
- [ ] GPU over-allocation raises `ValidationError`
- [ ] Dependency cycle raises `ValidationError`

### Must Have
- Named workload instances in ScenarioConfig
- Backward-compatible singular `workload` alias
- Greedy contiguous node placement with GPU budget validation
- `after_finish` dependency resolution and cycle detection
- Merged-DAG replay with correct concurrent behavior
- Per-workload SimulationResult extraction
- Backward-compatible `simulate()` return type
- GOAL and Chrome trace export with idle time representation

### Must NOT Have (Guardrails)
- No inference tracer implementation
- No `after_start` dependency type
- No dynamic scheduling or bin-packing
- No fractional-node workloads
- No dynamic workload injection at runtime
- No per-workload separate trace files
- No cross-workload data-flow edges
- No heterogeneous GPU assignments
- No mid-workload preemption

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: Tests-after (implement first, then add tests with separate agent)
- **Framework**: pytest (`uv run pytest`)
- **Agent-Executed QA**: Every task includes concrete QA scenarios

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Config validation**: Bash (`python -c "..."`) — Parse YAML, assert ValidationError on bad input
- **DAG merge**: Bash (`python -c "..."`) — Merge two DAGs, assert node counts and ID offsets
- **Replay**: Bash (`python -c "..."`) — Replay merged DAG, assert total_time and per-workload times
- **CLI**: Bash (`simulon simulate ...`) — Run command, assert output contains expected strings
- **Trace export**: Bash (`python -c "..."`) — Export GOAL/Chrome, assert file content

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — config models + placement + templates):
├── Task 1: Workload templates directory + example template
├── Task 2: StartConfig + WorkloadInstance models
├── Task 3: ScenarioConfig refactor with backward-compat alias
├── Task 4: Node placement engine
├── Task 5: GPU budget validation + dependency cycle detection
├── Task 6: CollectiveWorkload.num_gpus field
└── Task 7: AnalyticalBackend._resolve_workloads() helper

Wave 2 (Core engine — tracing + merge + replay):
├── Task 8: Trace each workload against node slice
├── Task 9: DAG merge utility (merge_dags)
├── Task 10: replay() with start_offsets parameter
├── Task 11: populate_network on merged DAG
└── Task 12: SimulationOutput dataclass

Wave 3 (Export + CLI + integration):
├── Task 13: GOAL trace export with start_offsets
├── Task 14: Chrome trace export with start_offsets
├── Task 15: AnalyticalBackend.simulate() multi-workload flow
├── Task 16: AnalyticalBackend.run() multi-workload flow
├── Task 17: CLI multi-workload summary output
└── Task 18: Energy/cost on merged DAG verification

Wave 4 (Tests + final QA):
├── Task 19: Config validation tests
├── Task 20: DAG merge unit tests
├── Task 21: Two-workload integration test
├── Task 22: Backward compat test (singular workload)
└── Task 23: CLI end-to-end test

Wave FINAL (4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
├── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T2-T3 → T4-T5 → T8 → T9 → T10 → T15 → T21 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 7 (Wave 1)
```

### Dependency Matrix (abbreviated)

- **T1-T7**: None → T8, T9, T15
- **T8**: T4, T7 → T9, T15
- **T9**: T8 → T10, T13, T14, T15
- **T10**: T9 → T15
- **T11**: T9 → T15
- **T12**: T9 → T15
- **T13-T14**: T9 → T15
- **T15**: T8-T14 → T16-T18, T21
- **T16-T18**: T15 → T19-T23
- **T19-T23**: T15 → F1-F4

### Agent Dispatch Summary

- **Wave 1**: 7 tasks → `quick` (T1-T7 are config/model changes, low complexity)
- **Wave 2**: 5 tasks → `deep` (T8-T12 are core engine changes, high complexity)
- **Wave 3**: 6 tasks → `unspecified-high` + `visual-engineering` (T13-T18 are integration/export)
- **Wave 4**: 5 tasks → `simulon-test-writer` (T19-T23 are tests)
- **FINAL**: 4 tasks → `oracle`, `unspecified-high`, `unspecified-high`, `deep`

---

## TODOs

- [x] 1. **Workload templates directory + example template**

  **What to do**:
  - Create `templates/workload/` directory
  - Create `templates/workload/gpt-oss-20b-16gpu.yaml` containing a full `MegatronWorkload` config
  - Follow the same YAML structure as `templates/gpu/*.yaml` (standalone, self-contained)
  - Include: framework, model (inline or reference), parallelism, training
  - Add a second example template if useful (e.g., `llama-7b-8gpu.yaml`)

  **Must NOT do**:
  - Do not add template resolution logic (templates are loaded directly as YAML paths)
  - Do not create a template registry or discovery mechanism

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-7)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `templates/gpu/h100.yaml` — Template YAML structure pattern
  - `templates/model/llama-7b.yaml` — Model template structure
  - `examples/scenario_96gpu.yaml` — Full scenario YAML showing workload inline

  **Acceptance Criteria**:
  - [ ] `templates/workload/` directory exists
  - [ ] At least one valid `MegatronWorkload` YAML template exists
  - [ ] Template can be loaded via `yaml.safe_load()` and validated with `MegatronWorkload.model_validate()`

  **QA Scenarios**:
  ```
  Scenario: Load workload template
    Tool: Bash (python -c)
    Preconditions: templates/workload/gpt-oss-20b-16gpu.yaml exists
    Steps:
      1. python -c "import yaml; from simulon.config.workload import MegatronWorkload; d=yaml.safe_load(open('templates/workload/gpt-oss-20b-16gpu.yaml')); w=MegatronWorkload.model_validate(d); print(w.training.num_gpus)"
    Expected Result: Validation succeeds, prints expected GPU count
    Evidence: .sisyphus/evidence/task-1-load-template.txt
  ```

  **Commit**: YES — Wave 1
  - Message: `feat(templates): add workload template directory with example`
  - Files: `templates/workload/*`

- [x] 2. **StartConfig + WorkloadInstance models**

  **What to do**:
  - Create `StartConfig` model with `offset_ms: float = 0.0` and `after_finish: list[str] = []`
  - Create `WorkloadInstance` model with `name: str`, `workload: Union[Path, WorkloadConfig]`, `start: StartConfig = Field(default_factory=StartConfig)`
  - Both in `src/simulon/config/scenario.py` (or new file if cleaner)
  - Use Pydantic v2 `Field` and `BeforeValidator` where needed
  - Ensure `after_finish` names are validated to exist within the same scenario (at scenario level, not in StartConfig itself)

  **Must NOT do**:
  - Do not add `after_start` dependency type (out of scope)
  - Do not implement scheduling logic in these models

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3-7)
  - **Blocks**: Task 3 (ScenarioConfig refactor), Task 5 (validation)
  - **Blocked By**: None

  **References**:
  - `src/simulon/config/scenario.py` — Existing ScenarioConfig pattern
  - `src/simulon/config/dc.py` — BeforeValidator pattern (`_coerce_node`)

  **Acceptance Criteria**:
  - [ ] `WorkloadInstance.model_validate({"name": "job_a", "workload": "path/to.yaml"})` succeeds
  - [ ] `WorkloadInstance.model_validate({"name": "job_b", "workload": {...}, "start": {"offset_ms": 100}})` succeeds
  - [ ] `after_finish` defaults to `[]`

  **QA Scenarios**:
  ```
  Scenario: Validate WorkloadInstance
    Tool: Bash (python -c)
    Preconditions: simulon package importable
    Steps:
      1. python -c "from simulon.config.scenario import WorkloadInstance, StartConfig; s=StartConfig(offset_ms=50.0, after_finish=['a']); w=WorkloadInstance(name='b', workload='templates/workload/gpt-oss-20b-16gpu.yaml', start=s); print(w.start.offset_ms)"
    Expected Result: Prints 50.0
    Evidence: .sisyphus/evidence/task-2-workload-instance.txt
  ```

  **Commit**: YES — Wave 1
  - Message: `feat(config): add WorkloadInstance and StartConfig models`
  - Files: `src/simulon/config/scenario.py`

- [x] 3. **ScenarioConfig refactor with backward-compat alias**

  **What to do**:
  - Replace `workload: Union[Path, WorkloadConfig]` with `workloads: list[WorkloadInstance]`
  - Add backward-compat: keep `workload` as a field that expands to a single-element `workloads` list via `model_validator(mode='before')`
  - The alias should produce a `WorkloadInstance` with `name="default"` or auto-generated name, `start=StartConfig()` (no offsets, no dependencies)
  - Ensure existing YAMLs with singular `workload:` still parse correctly
  - Add `_validate_workloads()` model validator for cross-workload checks (names unique, etc.)

  **Must NOT do**:
  - Do not break existing `ScenarioConfig.model_validate(yaml.load(...))` calls in tests
  - Do not remove the `collective` field (shared across workloads)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (depends on Task 2 for models)
  - **Parallel Group**: Wave 1 (with Tasks 1-2, 4-7)
  - **Blocks**: Task 5 (validation uses workloads list), Task 7 (resolve_workloads)
  - **Blocked By**: Task 2

  **References**:
  - `src/simulon/config/scenario.py` — Current ScenarioConfig
  - `tests/test_scenario.py` — Existing config round-trip tests

  **Acceptance Criteria**:
  - [ ] Old YAML with `workload:` parses correctly, produces `workloads` with one element
  - [ ] New YAML with `workloads:` list parses correctly
  - [ ] `tests/test_scenario.py` still passes (backward compat)

  **QA Scenarios**:
  ```
  Scenario: Backward compat singular workload
    Tool: Bash (python -c)
    Preconditions: tests/test_scenario.py passes
    Steps:
      1. uv run pytest tests/test_scenario.py -v
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-3-backward-compat.txt

  Scenario: New multi-workload config
    Tool: Bash (python -c)
    Preconditions: WorkloadInstance exists
    Steps:
      1. python -c "from simulon.config.scenario import ScenarioConfig; sc=ScenarioConfig.model_validate({'datacenter': {'datacenter': {'name': 'test'}, 'cluster': {'num_nodes': 2}, 'node': {'gpus_per_node': 4}}, 'workloads': [{'name': 'a', 'workload': {'framework': 'megatron', 'model': {'name': 'test', 'hidden_size': 256, 'num_layers': 2, 'num_heads': 2, 'ffn_hidden_size': 512, 'vocab_size': 1000}, 'parallelism': {'tp': 1, 'pp': 1}, 'training': {'num_gpus': 4, 'global_batch_size': 4, 'micro_batch_size': 1, 'sequence_length': 512, 'iterations': 1}}}]}); print(len(sc.workloads))"
    Expected Result: Prints 1
    Evidence: .sisyphus/evidence/task-3-multi-config.txt
  ```

  **Commit**: YES — Wave 1
  - Message: `feat(config): refactor ScenarioConfig for multi-workload with backward compat`
  - Files: `src/simulon/config/scenario.py`

- [x] 4. **Node placement engine**

  **What to do**:
  - Create `src/simulon/config/placement.py` with placement logic
  - Function signature: `place_workloads(workloads: list[WorkloadInstance], datacenter: DatacenterConfig) -> dict[str, NodeSlice]`
  - `NodeSlice` dataclass: `start_node: int`, `end_node: int`, `start_gpu_rank: int`, `end_gpu_rank: int`, `num_gpus: int`
  - Greedy assignment in declaration order: assign the next available contiguous nodes
  - Each workload must span at least one full node (no fractional-node workloads)
  - Node-aligned demand: `demand_gpus = ceil(raw_num_gpus / gpus_per_node) * gpus_per_node`. Example: 6 GPUs requested on 8-GPU/node cluster → allocated 8 GPUs (1 node), 2 stranded.
  - For `MegatronWorkload`, raw GPU demand = `training.num_gpus`
  - For `InferenceWorkload`, raw GPU demand = determine from inference config (if supported)
  - For `CollectiveWorkload`, raw GPU demand = `collective.num_gpus` (new field, see Task 6) or full cluster if absent
  - Compute `total_cluster_gpus = datacenter.cluster.num_nodes * datacenter.node.gpus_per_node`

  **Must NOT do**:
  - Do not implement bin-packing or optimization
  - Do not support fractional-node workloads

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-3, 5-7)
  - **Blocks**: Task 5 (validation uses placements), Task 8 (tracing uses slices)
  - **Blocked By**: None

  **References**:
  - `src/simulon/config/dc.py` — DatacenterConfig, ClusterSpec, NodeSpec
  - `src/simulon/config/workload.py` — MegatronWorkload.training.num_gpus

  **Acceptance Criteria**:
  - [ ] Two workloads of 4 GPUs each on 8-GPU cluster get non-overlapping slices
  - [ ] Workload declaration order determines placement
  - [ ] Node slices are contiguous and node-aligned

  **QA Scenarios**:
  ```
  Scenario: Place two workloads
    Tool: Bash (python -c)
    Preconditions: placement.py exists
    Steps:
      1. python -c "from simulon.config.placement import place_workloads; ..."
    Expected Result: Returns non-overlapping slices, first workload at node 0, second at node 1
    Evidence: .sisyphus/evidence/task-4-placement.txt
  ```

  **Commit**: YES — Wave 1
  - Message: `feat(config): add greedy node placement engine`
  - Files: `src/simulon/config/placement.py`

- [x] 5. **GPU budget validation + dependency cycle detection**

  **What to do**:
  - Add Pydantic `model_validator(mode='after')` on `ScenarioConfig` for:
    1. **GPU budget**: Sum GPU demand across all workloads ≤ `datacenter.cluster.num_nodes * datacenter.node.gpus_per_node`. Raise `ValidationError` with clear message showing demand vs capacity.
    2. **Dependency cycle detection**: Build dependency graph from `after_finish` lists, run DFS to detect cycles. Raise `ValidationError` with cycle path.
    3. **Missing dependency names**: Verify all names in `after_finish` exist in `workloads`. Raise `ValidationError` with missing name.
    4. **Duplicate workload names**: Verify all workload names are unique.
  - For GPU demand: `MegatronWorkload` uses `training.num_gpus`; `CollectiveWorkload` uses `num_gpus` (Task 6) or full cluster; `InferenceWorkload` uses inference-derived GPU count (if supported in config)

  **Must NOT do**:
  - Do not validate at simulation time — all validation must be at Pydantic config parse time
  - Do not implement general graph algorithms library; simple DFS is sufficient

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (depends on Tasks 2-3)
  - **Parallel Group**: Wave 1 (with Tasks 1-4, 6-7)
  - **Blocks**: Task 8 (simulation assumes valid config)
  - **Blocked By**: Tasks 2-3

  **References**:
  - `src/simulon/config/scenario.py` — ScenarioConfig validation pattern
  - `src/simulon/config/dc.py` — DatacenterConfig fields

  **Acceptance Criteria**:
  - [ ] Over-allocation raises ValidationError with message showing demand and capacity
  - [ ] Cycle in after_finish raises ValidationError with cycle path
  - [ ] Missing dependency name raises ValidationError
  - [ ] Duplicate workload names raise ValidationError

  **QA Scenarios**:
  ```
  Scenario: GPU over-allocation
    Tool: Bash (python -c)
    Steps:
      1. python -c "from simulon.config.scenario import ScenarioConfig; ScenarioConfig.model_validate({'datacenter': ..., 'workloads': [{'name': 'a', 'workload': ...}, {'name': 'b', 'workload': ...}]})"  # demand > capacity
    Expected Result: Raises ValidationError
    Evidence: .sisyphus/evidence/task-5-overalloc.txt

  Scenario: Dependency cycle
    Tool: Bash (python -c)
    Steps:
      1. python -c "from simulon.config.scenario import ScenarioConfig; ScenarioConfig.model_validate({'datacenter': ..., 'workloads': [{'name': 'a', 'start': {'after_finish': ['b']}}, {'name': 'b', 'start': {'after_finish': ['a']}}]})"
    Expected Result: Raises ValidationError mentioning cycle a → b → a
    Evidence: .sisyphus/evidence/task-5-cycle.txt
  ```

  **Commit**: YES — Wave 1
  - Message: `feat(config): add GPU budget and dependency validation`
  - Files: `src/simulon/config/scenario.py`

- [x] 6. **CollectiveWorkload.num_gpus field**

  **What to do**:
  - Add `num_gpus: Optional[int] = None` to `CollectiveWorkload` in `src/simulon/config/workload.py`
  - When `num_gpus` is set, it determines the GPU demand for placement
  - When `num_gpus` is None, behavior is backward-compatible (uses full cluster, but this is invalid in multi-workload — validation will catch it)
  - Update any code that derives GPU count from CollectiveWorkload to check `num_gpus` first

  **Must NOT do**:
  - Do not change the existing full-cluster behavior when `num_gpus` is absent and scenario has single workload
  - Do not add `num_nodes` field (use `num_gpus` and derive nodes from datacenter)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-5, 7)
  - **Blocks**: Task 4 (placement uses num_gpus), Task 5 (validation uses num_gpus)
  - **Blocked By**: None

  **References**:
  - `src/simulon/config/workload.py` — CollectiveWorkload definition

  **Acceptance Criteria**:
  - [ ] `CollectiveWorkload` accepts `num_gpus` field
  - [ ] Backward compat: `CollectiveWorkload` without `num_gpus` still works for single-workload

  **QA Scenarios**:
  ```
  Scenario: CollectiveWorkload with num_gpus
    Tool: Bash (python -c)
    Steps:
      1. python -c "from simulon.config.workload import CollectiveWorkload; w=CollectiveWorkload(framework='collective', collective_type='allreduce', message_size_bytes=1024, num_gpus=8); print(w.num_gpus)"
    Expected Result: Prints 8
    Evidence: .sisyphus/evidence/task-6-collective-num-gpus.txt
  ```

  **Commit**: YES — Wave 1
  - Message: `feat(config): add num_gpus to CollectiveWorkload`
  - Files: `src/simulon/config/workload.py`

- [x] 7. **AnalyticalBackend._resolve_workloads() helper**

  **What to do**:
  - Add `_resolve_workloads(scenario: ScenarioConfig) -> list[tuple[str, WorkloadConfig, NodeSlice]]` to `AnalyticalBackend`
  - This helper:
    1. Calls `place_workloads()` to get node slices
    2. Resolves any `Path` workloads to `WorkloadConfig` objects
    3. Returns a list of (workload_name, resolved_workload, node_slice) tuples
  - Also add `_slice_datacenter(datacenter, node_slice)` helper to create a sub-datacenter config for tracing

  **Must NOT do**:
  - Do not modify the public API of AnalyticalBackend yet
  - Do not implement tracing logic here

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (depends on Tasks 3-4)
  - **Parallel Group**: Wave 1 (with Tasks 1-6)
  - **Blocks**: Task 8 (tracing uses resolved workloads and sliced datacenters)
  - **Blocked By**: Tasks 3-4

  **References**:
  - `src/simulon/backend/analytical.py` — AnalyticalBackend class
  - `src/simulon/config/resolve.py` — Template resolution patterns

  **Acceptance Criteria**:
  - [ ] `_resolve_workloads()` returns resolved workloads with node slices
  - [ ] `_slice_datacenter()` produces valid DatacenterConfig with correct num_nodes

  **QA Scenarios**:
  ```
  Scenario: Resolve workloads
    Tool: Bash (python -c)
    Steps:
      1. python -c "from simulon.backend.analytical import AnalyticalBackend; backend=AnalyticalBackend(); ..."
    Expected Result: Returns list of tuples with correct types
    Evidence: .sisyphus/evidence/task-7-resolve.txt
  ```

  **Commit**: YES — Wave 1
  - Message: `feat(backend): add workload resolution and datacenter slicing helpers`
  - Files: `src/simulon/backend/analytical.py`

- [x] 8. **Trace each workload against node slice**

  **What to do**:
  - Keep `run_trace()` backward-compatible: it returns a single `ExecutionDAG` for single-workload scenarios (existing behavior)
  - Add new private method `_trace_all_workloads(scenario) -> list[tuple[str, ExecutionDAG]]` for internal multi-workload use
  - `_trace_all_workloads` logic:
    1. Call `_resolve_workloads()` to get (name, workload, slice) tuples
    2. For each workload: slice datacenter, call appropriate tracer, collect ExecutionDAG
    3. For `MegatronWorkload`: call `MegatronDAGTracer.trace(workload, sliced_datacenter)`
    4. For `CollectiveWorkload`: call `build_collective_dag(workload, sliced_datacenter, ...)`
    5. For `InferenceWorkload`: raise `NotImplementedError` with clear message
    6. Return list of (workload_name, dag) tuples
  - `run_trace()` delegates to `_trace_all_workloads()` and returns `[0][1]` for single-workload, maintaining existing signature and return type

  **Must NOT do**:
  - Do not merge DAGs here (Task 9 handles merging)
  - Do not implement inference tracing

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO — depends on Wave 1 completion
  - **Parallel Group**: Wave 2 (with Tasks 9-12)
  - **Blocks**: Task 9 (merging traced DAGs)
  - **Blocked By**: Tasks 4, 7

  **References**:
  - `src/simulon/backend/analytical.py` — Current run_trace implementation
    - Line 84+: `run_trace()` method with `isinstance(scenario.workload, MegatronWorkload)` dispatch
  - `src/simulon/backend/dag/megatron_tracer.py` — `MegatronDAGTracer.trace(workload, datacenter)`
  - `src/simulon/backend/dag/populate.py` — `populate_dag()` and `populate_network()`

  **Acceptance Criteria**:
  - [ ] Two Megatron workloads traced independently produce two separate ExecutionDAGs
  - [ ] Each DAG's GPU ranks are relative to 0..num_gpus-1 for that workload
  - [ ] `InferenceWorkload` raises `NotImplementedError`

  **QA Scenarios**:
  ```
  Scenario: Trace two workloads
    Tool: Bash (python -c)
    Steps:
      1. python -c "from simulon.backend.analytical import AnalyticalBackend; ..."
    Expected Result: Returns two ExecutionDAGs with distinct compute_nodes
    Evidence: .sisyphus/evidence/task-8-trace-two.txt
  ```

  **Commit**: YES — Wave 2
  - Message: `feat(backend): trace each workload independently against node slice`
  - Files: `src/simulon/backend/analytical.py`

- [x] 9. **DAG merge utility (merge_dags)**

  **What to do**:
  - Create `src/simulon/backend/dag/merge.py` with `merge_dags(dags: list[tuple[str, ExecutionDAG]]) -> ExecutionDAG`
  - For each workload DAG in order:
    1. Compute `node_id_offset` = max node_id in accumulated merged DAG + 1 (or 0 for first)
    2. Compute `gpu_rank_offset` based on accumulated GPU count from previous workloads
    3. Compute `flow_id_offset` = max flow_id in accumulated merged DAG + 1
    4. Offset all `node_id`, `gpu_rank`, `src_gpu`, `dst_gpu`, `flow_id` fields
    5. Offset all `DAGEdge.src_node_id` and `dst_node_id`
    6. Offset all `CommNode.parent_flow_ids`
    7. Append compute_nodes, comm_nodes, edges to merged DAG
  - Return a single `ExecutionDAG` with globally unique IDs
  - Add a helper to track which nodes belong to which workload (e.g., `node_id_to_workload` dict)

  **Must NOT do**:
  - Do not add cross-workload edges (workloads are independent)
  - Do not modify the input DAGs in place (create new nodes or deep copy)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 10-12 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 10 (replay needs merged DAG), Task 13 (GOAL export), Task 14 (Chrome export), Task 15 (simulate)
  - **Blocked By**: Task 8

  **References**:
  - `src/simulon/backend/dag/nodes.py` — ExecutionDAG, ComputeNode, CommNode, DAGEdge dataclasses
  - `src/simulon/backend/dag/replayer.py` — How node IDs and flow IDs are used in replay

  **Acceptance Criteria**:
  - [ ] Merged DAG node IDs are globally unique and monotonic
  - [ ] Merged DAG GPU ranks are globally unique and contiguous
  - [ ] Merged DAG flow IDs are globally unique
  - [ ] All edges and parent_flow_ids correctly reference offset IDs
  - [ ] `node_id_to_workload` dict correctly maps every node to its workload name

  **QA Scenarios**:
  ```
  Scenario: Merge two simple DAGs
    Tool: Bash (python -c)
    Steps:
      1. Create two ExecutionDAGs with known node IDs, GPU ranks, flow IDs
      2. Call merge_dags()
      3. Assert merged node count = sum of individual node counts
      4. Assert no overlapping node IDs, GPU ranks, or flow IDs
      5. Assert edges reference valid offset IDs
    Expected Result: All assertions pass
    Evidence: .sisyphus/evidence/task-9-merge.txt
  ```

  **Commit**: YES — Wave 2
  - Message: `feat(dag): add DAG merge utility for multi-workload`
  - Files: `src/simulon/backend/dag/merge.py`

- [x] 10. **replay() with start_offsets parameter**

  **What to do**:
  - Modify `replay(dag: ExecutionDAG, start_offsets: Optional[dict[int, float]] = None)` in `src/simulon/backend/dag/replayer.py`
  - `start_offsets`: map from global GPU rank to effective start time in ms
  - In the replayer initialization:
    - If `start_offsets` is provided, initialize `per_gpu_finish[gpu_rank] = start_offsets.get(gpu_rank, 0.0)`
    - Otherwise, initialize all GPUs at 0.0 (backward compat)
  - The rest of Kahn's algorithm remains unchanged — nodes simply start after their offset
  - Update `SimulationResult` if needed (no changes likely needed)
  - Ensure `_summarize()` is unchanged (no fake idle nodes to filter)

  **Must NOT do**:
  - Do not add fake idle ComputeNodes to the DAG
  - Do not change the topological sort logic
  - Do not break backward compat (default start_offsets=None means all GPUs start at 0)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9, 11-12 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 15 (simulate uses replay)
  - **Blocked By**: None (this is independent of merge, but both feed into simulate)

  **References**:
  - `src/simulon/backend/dag/replayer.py` — Current replay() implementation
    - Lines 1-303: Kahn's algorithm, per_gpu_finish initialization, _summarize
  - `tests/test_replayer.py` — Existing replay tests

  **Acceptance Criteria**:
  - [ ] `replay(dag)` without start_offsets behaves identically to before
  - [ ] `replay(dag, start_offsets={0: 100.0, 1: 100.0})` delays GPU 0 and 1 by 100ms
  - [ ] `tests/test_replayer.py` still passes

  **QA Scenarios**:
  ```
  Scenario: Replay with start offsets
    Tool: Bash (python -c)
    Steps:
      1. Build a simple DAG with one ComputeNode on GPU 0 (duration=50ms)
      2. replay(dag, start_offsets={0: 100.0})
      3. Assert node.start_ms == 100.0 and node.finish_ms == 150.0
    Expected Result: Pass
    Evidence: .sisyphus/evidence/task-10-replay-offsets.txt

  Scenario: Backward compat replay
    Tool: Bash (python -c)
    Steps:
      1. Run existing replay tests: uv run pytest tests/test_replayer.py -v
    Expected Result: All pass
    Evidence: .sisyphus/evidence/task-10-backward-compat.txt
  ```

  **Commit**: YES — Wave 2
  - Message: `feat(dag): add start_offsets support to replay`
  - Files: `src/simulon/backend/dag/replayer.py`

- [x] 11. **populate_network on merged DAG**

  **What to do**:
  - Ensure `populate_network()` in `src/simulon/backend/dag/populate.py` works correctly on a merged DAG
  - `populate_network` uses `datacenter` to compute link bandwidths and latencies
  - Since merged DAG has globally offset GPU ranks, `populate_network` must map these ranks to the correct physical links in the full datacenter
  - For intra-node links (NVLink): GPU ranks within the same node map to NVLink
  - For inter-node links (NIC): GPU ranks on different nodes map to NIC
  - The current implementation likely already handles this correctly if GPU ranks map to physical ranks in the datacenter
  - Verify by reading the current `populate_network()` implementation and checking if rank-to-node mapping is correct for offset ranks

  **Must NOT do**:
  - Do not call populate_network per-workload (must be called once on merged DAG with full datacenter)
  - Do not modify populate_network unless necessary

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9-10, 12 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 15 (simulate needs populated merged DAG)
  - **Blocked By**: None (verification task)

  **References**:
  - `src/simulon/backend/dag/populate.py` — populate_network() implementation
  - `src/simulon/config/dc.py` — DatacenterConfig, network topology

  **Acceptance Criteria**:
  - [ ] `populate_network()` works on merged DAG without modification, OR
  - [ ] Minimal modification made if rank-to-node mapping needs adjustment
  - [ ] Two workloads on different nodes get correct inter-node bandwidth for cross-node comms

  **QA Scenarios**:
  ```
  Scenario: populate_network on merged DAG
    Tool: Bash (python -c)
    Steps:
      1. Build merged DAG with two workloads on different node slices
      2. Call populate_network(merged_dag, full_datacenter)
      3. Inspect CommNode.duration_ms for cross-node vs intra-node comms
    Expected Result: Cross-node comms have lower bandwidth (higher duration) than intra-node
    Evidence: .sisyphus/evidence/task-11-populate-network.txt
  ```

  **Commit**: YES — Wave 2 (if changes needed)
  - Message: `fix(dag): ensure populate_network works with offset GPU ranks`
  - Files: `src/simulon/backend/dag/populate.py`

- [x] 12. **SimulationOutput dataclass**

  **What to do**:
  - Create `SimulationOutput` dataclass in `src/simulon/backend/analytical.py` or `src/simulon/backend/dag/nodes.py`
  - Fields:
    ```python
    @dataclass
    class SimulationOutput:
        dag: ExecutionDAG                    # merged DAG
        result: SimulationResult             # aggregate result from merged DAG replay
        by_workload: dict[str, SimulationResult]  # per-workload results
        
        def __iter__(self):
            return iter((self.dag, self.result))
        
        def __getitem__(self, idx: int):
            return (self.dag, self.result)[idx]
    ```
  - The `__iter__` method ensures backward compatibility: `dag, result = backend.simulate(...)` still works
  - The `__getitem__` method ensures index access works: `backend.simulate(sc)[0]` returns the DAG
  - Users can access per-workload results via `output.by_workload["job_a"]`

  **Must NOT do**:
  - Do not break the tuple unpacking contract for single-workload scenarios
  - Do not add fields that would make `__iter__` ambiguous

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9-11 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 15 (simulate returns SimulationOutput)
  - **Blocked By**: None

  **References**:
  - `src/simulon/backend/analytical.py` — Current simulate() return type
  - `src/simulon/backend/dag/replayer.py` — SimulationResult dataclass

  **Acceptance Criteria**:
  - [ ] `SimulationOutput` dataclass exists with correct fields
  - [ ] `dag, result = output` tuple unpacking works
  - [ ] `output.by_workload` is accessible

  **QA Scenarios**:
  ```
  Scenario: SimulationOutput backward compat
    Tool: Bash (python -c)
    Steps:
      1. Create SimulationOutput with dummy dag, result, by_workload
      2. dag, result = output
      3. Assert dag == output.dag and result == output.result
    Expected Result: Pass
    Evidence: .sisyphus/evidence/task-12-simulation-output.txt
  ```

  **Commit**: YES — Wave 2
  - Message: `feat(backend): add SimulationOutput dataclass for multi-workload results`
  - Files: `src/simulon/backend/dag/nodes.py` or `src/simulon/backend/analytical.py`

- [x] 13. **GOAL trace export with start_offsets**

  **What to do**:
  - Modify `dag_to_goal(dag, start_offsets=None)` in `src/simulon/backend/dag/goal_trace.py` to accept `start_offsets: dict[int, float]`
  - For each GPU rank with non-zero offset:
    1. Emit an idle `calc` block: `c_idle_{rank}: calc <duration_ns>` where duration_ns = offset_ms * 1_000_000
    2. Wire dependency: all subsequent events for that rank must `require c_idle_{rank}`
    3. This ensures ATLAHS respects the waiting time before scheduling actual work
  - The idle block is emitted in the rank's block before any compute or comm events
  - Ensure existing `dag_to_goal()` behavior unchanged when `start_offsets=None`
  - Update `write_goal_trace()` to accept and pass through `start_offsets`

  **Must NOT do**:
  - Do not modify the DAG (add fake nodes)
  - Do not change the GOAL format itself (ATLAHS compatibility)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 14-18 in Wave 3)
  - **Parallel Group**: Wave 3
  - **Blocks**: None (export only)
  - **Blocked By**: Task 9 (needs merged DAG), Task 10 (needs replay to get timing)

  **References**:
  - `src/simulon/backend/dag/goal_trace.py` — Current GOAL export implementation
  - `tests/test_goal_trace.py` — Existing GOAL tests

  **Acceptance Criteria**:
  - [ ] GOAL output contains idle `calc` block for GPUs with non-zero start offset
  - [ ] Idle block duration equals offset in nanoseconds
  - [ ] Existing GOAL tests pass with `start_offsets=None`

  **QA Scenarios**:
  ```
  Scenario: GOAL export with start offsets
    Tool: Bash (python -c)
    Steps:
      1. Build simple DAG with one node on GPU 0
      2. goal = dag_to_goal(dag, start_offsets={0: 100.0})
      3. Assert goal contains 'calc 100000000.0' (or similar) before the actual compute
    Expected Result: Idle block present with correct duration
    Evidence: .sisyphus/evidence/task-13-goal-offsets.txt
  ```

  **Commit**: YES — Wave 3
  - Message: `feat(goal): support start_offsets in GOAL trace export`
  - Files: `src/simulon/backend/dag/goal_trace.py`

- [x] 14. **Chrome trace export with start_offsets**

  **What to do**:
  - Modify `to_chrome_trace(dag, ...)` in `src/simulon/backend/dag/chrome_trace.py` to accept `start_offsets: dict[int, float]`
  - For each GPU rank with a non-zero offset, emit an idle event at the beginning of that rank's trace
  - Chrome trace event format:
    ```json
    {"name": "idle", "ph": "X", "ts": 0, "dur": <offset_us>, "pid": <rank>, "tid": <tid>}
    ```
  - Use a distinct color/category for idle events (e.g., "grey")
  - Ensure existing Chrome trace export unchanged when `start_offsets=None`
  - **Chrome trace rank decoding design (Option B)**: Add `workload_labels: dict[int, str] | None = None` parameter. If provided, process label is `"GPU {gpu} | {workload_name}"` and skip `_decode_rank()`. If None, use existing `_decode_rank()` logic with `tp/pp/dp/ep` params (backward compat).
  - The `workload_labels` dict maps global GPU rank to workload name, produced by the merge utility.

  **Must NOT do**:
  - Do not modify the DAG
  - Do not create separate trace files per workload
  - Do not remove existing `_decode_rank()` logic

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 13, 15-18 in Wave 3)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 9, Task 10

  **References**:
  - `src/simulon/backend/dag/chrome_trace.py` — Current Chrome trace export
  - `tests/test_e2e.py` — Chrome trace assertions

  **Acceptance Criteria**:
  - [ ] Chrome trace JSON contains idle events for GPUs with non-zero offset
  - [ ] Idle events visible in Perfetto UI with correct duration
  - [ ] All workload compute/comm events still present and correctly timed

  **QA Scenarios**:
  ```
  Scenario: Chrome trace with start offsets
    Tool: Bash (python -c)
    Steps:
      1. Build DAG, call to_chrome_trace(dag, start_offsets={0: 100.0})
      2. Parse JSON, find event with name="idle" and pid=0
      3. Assert event.dur == 100000.0 (us)
    Expected Result: Idle event present
    Evidence: .sisyphus/evidence/task-14-chrome-offsets.txt
  ```

  **Commit**: YES — Wave 3
  - Message: `feat(chrome): support start_offsets in Chrome trace export`
  - Files: `src/simulon/backend/dag/chrome_trace.py`

- [ ] 15. **AnalyticalBackend.simulate() multi-workload flow**

  **What to do**:
  - **Pre-refactoring**: Extract existing single-workload simulate logic into `_simulate_single_workload()` private method BEFORE adding multi-workload logic. Keep `simulate()` as a clean dispatcher:
    ```python
    def simulate(self, scenario, ...):
        if len(scenario.workloads) == 1:
            return self._simulate_single_workload(...)
        else:
            return self._simulate_multi_workload(...)
    ```
  - **Cache handling**: Disable DAG cache for multi-workload scenarios (`len(workloads) > 1`). Cache key complexity not worth the risk for v1. Single-workload caching remains unchanged.
    - *Note*: Verify current cache implementation location during execution. Cache may be in `cache.py` or inline in `analytical.py`. If no cache exists yet, this is a no-op.
  - **Per-workload stat extraction**: Add `summarize_subset(dag, node_ids)` helper in `replayer.py` that computes a `SimulationResult` from a subset of nodes. Use this to build `by_workload` results by passing each workload's node IDs.
  - **Multi-workload flow** (`_simulate_multi_workload`):
    1. Call `_resolve_workloads()` to get (name, workload, slice) tuples
    2. For each workload:
       - Slice datacenter
       - Trace: `_trace_all_workloads()` → ExecutionDAG
       - Populate: `populate_dag()` with workload config and GPU spec
       - Populate network: `populate_network(dag, sliced_datacenter, ...)` to set CommNode durations
       - Replay independently to get finish_time → store (name, dag, finish_time, slice)
    3. Resolve dependencies: compute effective_start_ms for each workload
       - Topological sort of dependency graph from `after_finish`
       - effective_start = max(offset_ms, max(finish_time(dep) for dep in after_finish))
    4. Build `start_offsets` dict: for each GPU in workload slice, offset = effective_start_ms
    5. Merge all DAGs into one using `merge_dags()`
    6. Call `populate_network()` on merged DAG with FULL datacenter
    7. Call `replay(merged_dag, start_offsets=start_offsets)` → aggregate SimulationResult
    8. Extract per-workload results using `summarize_subset()` on each workload's node set
    9. Return `SimulationOutput(dag=merged_dag, result=aggregate, by_workload=per_workload)`
  - For single-workload scenarios (backward compat), `_simulate_single_workload()` behavior identical to before

  **Must NOT do**:
  - Do not break backward compatibility for single-workload scenarios
  - Do not implement inference tracing

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO — depends on Wave 2
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 16 (run uses simulate), Task 18 (energy uses simulate output)
  - **Blocked By**: Tasks 8-12

  **References**:
  - `src/simulon/backend/analytical.py` — Current simulate() implementation (lines 120-186)
  - `src/simulon/backend/dag/populate.py` — populate_dag(), populate_network()
  - `src/simulon/backend/dag/replayer.py` — replay()
  - `src/simulon/backend/dag/merge.py` — merge_dags()

  **Acceptance Criteria**:
  - [ ] Two-workload scenario returns SimulationOutput with both aggregate and per-workload results
  - [ ] Single-workload scenario backward compatible (dag, result = simulate(sc))
  - [ ] Per-workload total_time_ms matches isolation run (modulo offset)
  - [ ] Aggregate total_time_ms = max(workload effective_start + workload_duration)
  - [ ] Multi-workload scenarios bypass DAG cache (no stale single-workload cache hits)
  - [ ] `summarize_subset()` helper exists and produces correct SimulationResult for node subsets

  **QA Scenarios**:
  ```
  Scenario: Simulate two workloads
    Tool: Bash (python -c)
    Steps:
      1. Build two-workload scenario with 8-GPU datacenter, two 4-GPU workloads
      2. output = backend.simulate(sc)
      3. Assert len(output.by_workload) == 2
      4. Assert output.result.total_time_ms >= max(w.total_time_ms for w in output.by_workload.values())
    Expected Result: Pass
    Evidence: .sisyphus/evidence/task-15-simulate-two.txt

  Scenario: Backward compat single workload
    Tool: Bash (python -c)
    Steps:
      1. dag, result = backend.simulate(single_workload_scenario)
      2. Assert isinstance(dag, ExecutionDAG) and isinstance(result, SimulationResult)
    Expected Result: Pass
    Evidence: .sisyphus/evidence/task-15-backward-compat.txt
  ```

  **Commit**: YES — Wave 3
  - Message: `feat(backend): multi-workload simulate() with merged DAG replay`
  - Files: `src/simulon/backend/analytical.py`

- [x] 16. **AnalyticalBackend.run() multi-workload flow**

  **What to do**:
  - Update `run()` in `src/simulon/backend/analytical.py` to support multi-workload scenarios
  - `run()` currently calls `run_trace()` and returns a dict with status, node counts, etc.
  - For multi-workload:
    1. Call the updated `run_trace()` which traces all workloads
    2. Return a dict with per-workload DAG info and merged DAG info
    3. Example return shape:
       ```python
       {
           "status": "success",
           "workloads": {
               "job_a": {"compute_nodes": N, "comm_nodes": M},
               "job_b": {"compute_nodes": N, "comm_nodes": M},
           },
           "merged": {"compute_nodes": N, "comm_nodes": M},
       }
       ```
  - For single-workload, keep existing return shape for backward compat

  **Must NOT do**:
  - Do not change the single-workload return shape
  - Do not call simulate() from run() (run() should remain lightweight, no replay)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 13-15, 17-18 in Wave 3)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 8 (run_trace updated)

  **References**:
  - `src/simulon/backend/analytical.py` — Current run() implementation
  - `tests/test_e2e.py` — run() assertions

  **Acceptance Criteria**:
  - [ ] Multi-workload run() returns dict with per-workload and merged node counts
  - [ ] Single-workload run() return shape unchanged
  - [ ] tests/test_e2e.py passes

  **QA Scenarios**:
  ```
  Scenario: run() with two workloads
    Tool: Bash (python -c)
    Steps:
      1. result = backend.run(two_workload_scenario)
      2. Assert result["status"] == "success"
      3. Assert "workloads" in result and len(result["workloads"]) == 2
    Expected Result: Pass
    Evidence: .sisyphus/evidence/task-16-run-two.txt
  ```

  **Commit**: YES — Wave 3
  - Message: `feat(backend): multi-workload run() return shape`
  - Files: `src/simulon/backend/analytical.py`

- [ ] 17. **CLI multi-workload summary output**

  **What to do**:
  - Update `src/simulon/cli/__init__.py` simulate command to handle multi-workload output
  - Current CLI does: `dag, result = backend.simulate(...)` and prints single summary
  - New behavior:
    1. `output = backend.simulate(...)` (SimulationOutput)
    2. Print aggregate summary (merged DAG total time, compute/comm/bubble)
    3. Print per-workload summaries:
       - Workload name
       - Total time, compute time, comm time, bubble
       - Throughput (if MegatronWorkload)
       - Start offset and effective start time
    4. `--verbose` / `-v` prints per-workload per-GPU breakdowns
  - Extract `_print_summary()`, `_print_collective_summary()`, `_print_energy_summary()`, `_print_cost_summary()` to new `src/simulon/cli/output.py` module as part of this task (CLI file is 755 lines, needs cleanup)
  - Update `_print_summary()` to accept a SimulationResult and optional workload name
  - Add `_print_multi_workload_summary()` helper
  - Ensure `--chrome`, `--goal`, `--energy`, `--cost` flags work with multi-workload:
    - `--chrome`: one merged trace (Task 14)
    - `--goal`: one merged GOAL file (Task 13)
    - `--energy`: computed on merged DAG (Task 18)
    - `--cost`: computed on merged DAG (Task 18)
  - Create `examples/multi_workload.yaml` with two small Megatron workloads (e.g., 2 layers, small hidden size) for CLI testing and demonstration

  **Must NOT do**:
  - Do not add per-workload separate trace files
  - Do not change CLI argument parsing (no new flags needed)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 13-16, 18 in Wave 3)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Tasks 13-16

  **References**:
  - `src/simulon/cli/__init__.py` — Current CLI simulate command
    - Lines 79+: `dag, result = backend.simulate(...)`
    - Lines 100+: `_print_summary()`, `_print_collective_summary()`
  - `src/simulon/cli/__init__.py` — Energy/cost printing

  **Acceptance Criteria**:
  - [ ] `simulon simulate multi_workload.yaml` prints aggregate + per-workload summaries
  - [ ] Single-workload CLI output unchanged
  - [ ] `--chrome` produces one valid trace file
  - [ ] `--goal` produces one valid GOAL file

  **QA Scenarios**:
  ```
  Scenario: CLI multi-workload output
    Tool: Bash (simulon CLI)
    Steps:
      1. Create examples/multi_workload.yaml with two workloads
      2. simulon simulate examples/multi_workload.yaml
      3. Assert stdout contains both workload names and aggregate total time
    Expected Result: Both workloads listed, aggregate shown
    Evidence: .sisyphus/evidence/task-17-cli-output.txt

  Scenario: CLI backward compat
    Tool: Bash (simulon CLI)
    Steps:
      1. simulon simulate examples/scenario_96gpu.yaml
      2. Assert output format unchanged from before
    Expected Result: Single summary, no workload names
    Evidence: .sisyphus/evidence/task-17-cli-compat.txt
  ```

  **Commit**: YES — Wave 3
  - Message: `feat(cli): multi-workload summary output`
  - Files: `src/simulon/cli/__init__.py`

- [ ] 18. **Energy/cost on merged DAG verification**

  **What to do**:
  - Verify that `compute_energy(dag, scenario)` works correctly on a merged DAG
  - The energy model computes `total_time_ms = max(finish_ms)` across all nodes — this is correct for concurrent execution
  - GPU utilization is computed as `avg_active_ms / (num_cluster_gpus * total_time_ms)` — with merged DAG, this naturally accounts for idle GPUs
  - CAPEX is computed from the full datacenter hardware — correct (shared hardware, not double-counted)
  - Cost amortization uses `run_duration_hours` from the merged DAG — correct
  - If `compute_energy` or `compute_cost` need adjustments for merged DAGs (e.g., handling empty node sets, different workload types), make minimal changes
  - Primarily a verification task — may require no code changes

  **Must NOT do**:
  - Do not rewrite the energy model
  - Do not compute per-workload energy (full-cluster only, per guardrail)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 13-17 in Wave 3)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 15 (needs simulate output)

  **References**:
  - `src/simulon/energy.py` — compute_energy()
  - `src/simulon/cost.py` — compute_cost()
  - `src/simulon/cli/__init__.py` — Energy/cost CLI integration

  **Acceptance Criteria**:
  - [ ] `compute_energy(merged_dag, scenario)` returns EnergyResult with correct total_time_ms
  - [ ] Energy for two concurrent workloads is less than sum of individual energies (due to shared time and hardware)
  - [ ] Cost CAPEX is not double-counted

  **QA Scenarios**:
  ```
  Scenario: Energy on merged DAG
    Tool: Bash (python -c)
    Steps:
      1. Simulate two workloads concurrently
      2. energy = compute_energy(output.dag, scenario)
      3. energy_single = compute_energy(single_dag, scenario)  # for comparison
      4. Assert energy.run_duration_hours >= max(single durations)
      5. Assert energy.run_duration_hours <= sum(single durations)
    Expected Result: Concurrent energy duration is between max and sum
    Evidence: .sisyphus/evidence/task-18-energy-merged.txt
  ```

  **Commit**: YES — Wave 3 (if changes needed)
  - Message: `fix(energy): verify energy/cost correctness on merged DAG`
  - Files: `src/simulon/energy.py` or `src/simulon/cost.py`

- [ ] 19. **Config validation tests**

  **What to do**:
  - Add tests for multi-workload config validation:
    1. **GPU over-allocation**: Two workloads requesting more GPUs than cluster capacity → ValidationError
    2. **Dependency cycle**: A depends on B, B depends on A → ValidationError
    3. **Missing dependency**: A depends on C, but C not in workloads → ValidationError
    4. **Duplicate names**: Two workloads named "job_a" → ValidationError
    5. **Backward compat alias**: Old YAML with singular `workload` parses correctly, produces single-element workloads list
    6. **Valid multi-workload**: Two workloads with no conflicts → parses successfully
  - Add to `tests/test_scenario.py` or create `tests/test_multi_workload_config.py`
  - Use inline factories (no conftest.py), following existing test conventions

  **Must NOT do**:
  - Do not modify existing tests unless necessary for backward compat
  - Do not add slow integration tests here (unit tests only)

  **Recommended Agent Profile**:
  - **Category**: `simulon-test-writer`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 20-23 in Wave 4)
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: Tasks 2-6 (config models and validation)

  **References**:
  - `tests/test_scenario.py` — Existing config round-trip tests
  - `src/simulon/config/scenario.py` — Validation logic

  **Acceptance Criteria**:
  - [ ] All 6 validation scenarios have passing tests
  - [ ] Tests run with `uv run pytest tests/test_multi_workload_config.py -v`

  **QA Scenarios**:
  ```
  Scenario: Run validation tests
    Tool: Bash (pytest)
    Steps:
      1. uv run pytest tests/test_multi_workload_config.py -v
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-19-validation-tests.txt
  ```

  **Commit**: YES — Wave 4
  - Message: `test(config): add multi-workload config validation tests`
  - Files: `tests/test_multi_workload_config.py`

- [ ] 20. **DAG merge unit tests**

  **What to do**:
  - Add tests for `merge_dags()`:
    1. **Node ID uniqueness**: Two DAGs merged have no overlapping node IDs
    2. **GPU rank offset**: Second DAG's GPU ranks start after first DAG's max rank
    3. **Flow ID uniqueness**: No overlapping flow IDs
    4. **Edge preservation**: All edges reference valid offset node IDs
    5. **Parent flow IDs**: CommNode parent_flow_ids reference valid offset flow IDs
    6. **Workload mapping**: node_id_to_workload correctly maps all nodes
    7. **Empty DAG handling**: Merging with an empty DAG works
    8. **Three DAGs**: Merging three DAGs in sequence works
  - Create `tests/test_dag_merge.py`
  - Build simple DAGs programmatically (inline factories)

  **Must NOT do**:
  - Do not depend on real tracer output (build minimal DAGs by hand)
  - Do not test replay here (that's Task 21)

  **Recommended Agent Profile**:
  - **Category**: `simulon-test-writer`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 19, 21-23 in Wave 4)
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: Task 9 (merge utility)

  **References**:
  - `src/simulon/backend/dag/merge.py` — merge_dags()
  - `src/simulon/backend/dag/nodes.py` — ExecutionDAG, ComputeNode, CommNode, DAGEdge

  **Acceptance Criteria**:
  - [ ] All 8 merge scenarios have passing tests
  - [ ] Tests run with `uv run pytest tests/test_dag_merge.py -v`

  **QA Scenarios**:
  ```
  Scenario: Run merge tests
    Tool: Bash (pytest)
    Steps:
      1. uv run pytest tests/test_dag_merge.py -v
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-20-merge-tests.txt
  ```

  **Commit**: YES — Wave 4
  - Message: `test(dag): add DAG merge utility tests`
  - Files: `tests/test_dag_merge.py`

- [ ] 21. **Two-workload integration test**

  **What to do**:
  - Add end-to-end integration test for two concurrent Megatron workloads:
    1. Build a scenario with 8-GPU datacenter (2 nodes × 4 GPUs)
    2. Workload A: 4 GPUs, Llama-7b-like config, no offset
    3. Workload B: 4 GPUs, same model, offset=0 (concurrent)
    4. Run `backend.simulate(sc)`
    5. Assert:
       - `output.by_workload` has 2 entries
       - Each per-workload result has total_time_ms > 0
       - Aggregate `output.result.total_time_ms` ≈ max(per-workload times) (concurrent, so should be close to max)
       - Merged DAG has correct node count = sum of individual node counts
    6. Test with dependency: Workload B has `after_finish: [A]`
       - Assert B's effective start ≈ A's finish time
       - Aggregate total_time ≈ A_time + B_time (sequential)
  - Add to `tests/test_multi_workload_integration.py`
  - Use inline factories for datacenter and workload configs

  **Must NOT do**:
  - Do not use real model YAMLs (build minimal configs inline)
  - Do not test more than 2 workloads (keep test fast)

  **Recommended Agent Profile**:
  - **Category**: `simulon-test-writer`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 19-20, 22-23 in Wave 4)
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: Tasks 9-15

  **References**:
  - `tests/test_e2e.py` — Existing integration test patterns
  - `tests/test_compact.py` — Comparison test pattern (normal vs compact)

  **Acceptance Criteria**:
  - [ ] Concurrent two-workload test passes
  - [ ] Sequential two-workload test (after_finish dependency) passes
  - [ ] `uv run pytest tests/test_multi_workload_integration.py -v` passes

  **QA Scenarios**:
  ```
  Scenario: Run integration tests
    Tool: Bash (pytest)
    Steps:
      1. uv run pytest tests/test_multi_workload_integration.py -v
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-21-integration-tests.txt
  ```

  **Commit**: YES — Wave 4
  - Message: `test(integration): add two-workload concurrent and sequential tests`
  - Files: `tests/test_multi_workload_integration.py`

- [ ] 22. **Backward compat test (singular workload)**

  **What to do**:
  - Ensure all existing tests still pass with the refactored ScenarioConfig:
    1. `uv run pytest tests/test_scenario.py` — config round-trip
    2. `uv run pytest tests/test_e2e.py` — end-to-end single workload
    3. `uv run pytest tests/test_replayer.py` — replay unchanged
    4. `uv run pytest tests/test_goal_trace.py` — GOAL export unchanged
    5. `uv run pytest tests/test_compact.py` — compact mode unchanged
    6. `uv run pytest tests/test_collective_workload.py` — collective workload unchanged
  - Fix any breakage caused by ScenarioConfig refactor or simulate() return type change
  - The `SimulationOutput.__iter__` and `__getitem__` tricks should handle most simulate() call sites
  - Search for `extract_metrics` usage across all tests and ensure they receive `output.result` not the tuple
  - Check `tests/test_tracking_mlflow.py` — uses `extract_metrics(result)` which takes SimulationResult directly

  **Must NOT do**:
  - Do not disable or skip existing tests
  - Do not change test assertions unless the behavior change is intentional

  **Recommended Agent Profile**:
  - **Category**: `simulon-test-writer`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 19-21, 23 in Wave 4)
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: Tasks 3, 12, 15

  **References**:
  - All files in `tests/` directory
  - `src/simulon/backend/analytical.py` — simulate() and run()

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/` passes (or at minimum, no regressions in existing tests)
  - [ ] Any test changes are documented in commit message

  **QA Scenarios**:
  ```
  Scenario: Run full test suite
    Tool: Bash (pytest)
    Steps:
      1. uv run pytest tests/ -v --tb=short
    Expected Result: All existing tests pass, new tests pass
    Evidence: .sisyphus/evidence/task-22-full-suite.txt
  ```

  **Commit**: YES — Wave 4
  - Message: `test(compat): ensure backward compatibility with existing tests`
  - Files: `tests/*` (any fixes needed)

- [ ] 23. **CLI end-to-end test**

  **What to do**:
  - Add CLI-level test for multi-workload scenario:
    1. Create a temporary YAML file with two workloads
    2. Run `simulon simulate temp.yaml` and capture stdout
    3. Assert stdout contains expected workload names and aggregate total
    4. Run `simulon simulate temp.yaml --chrome temp.json` and assert file is valid JSON
    5. Run `simulon simulate temp.yaml --goal temp.goal` and assert file contains expected GOAL syntax
    6. Clean up temp files
  - Add to `tests/test_multi_workload_cli.py`
  - Use `subprocess.run` or `typer.testing.CliRunner` to invoke CLI

  **Must NOT do**:
  - Do not test with large models (keep runtime < 5 seconds)
  - Do not test all CLI flags (focus on core multi-workload path)

  **Recommended Agent Profile**:
  - **Category**: `simulon-test-writer`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 19-22 in Wave 4)
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: Tasks 13-17 (export + CLI)

  **References**:
  - `tests/test_e2e.py` — May have CLI usage patterns
  - `src/simulon/cli/__init__.py` — CLI implementation

  **Acceptance Criteria**:
  - [ ] CLI test passes
  - [ ] Temp files cleaned up after test

  **QA Scenarios**:
  ```
  Scenario: Run CLI test
    Tool: Bash (pytest)
    Steps:
      1. uv run pytest tests/test_multi_workload_cli.py -v
    Expected Result: Pass
    Evidence: .sisyphus/evidence/task-23-cli-test.txt
  ```

  **Commit**: YES — Wave 4
  - Message: `test(cli): add multi-workload CLI end-to-end test`
  - Files: `tests/test_multi_workload_cli.py`

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run pytest` + `tsc --noEmit` (if applicable) + linter. Review all changed files for: `as any`/`@ts-ignore`, empty catches, `console.log` in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (features working together, not isolation). Test edge cases: empty state, invalid input, rapid actions. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(config): add multi-workload config models and placement engine`
- **Wave 2**: `feat(dag): add DAG merge utility and replay start_offsets`
- **Wave 3**: `feat(backend): multi-workload simulate/run and trace export`
- **Wave 4**: `test(multi-workload): add validation, merge, and integration tests`

## Success Criteria

### Verification Commands
```bash
# Config validation
uv run pytest tests/test_scenario.py

# Full suite
uv run pytest

# CLI end-to-end
simulon simulate examples/multi_workload.yaml --goal out.goal --chrome out.json
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] CLI produces valid multi-workload output
- [ ] GOAL export valid for ATLAHS
- [ ] Chrome trace viewable in Perfetto
