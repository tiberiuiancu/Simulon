# tests — Test Suite

25+ test files. pytest framework, no conftest.py.

## OVERVIEW

Each test file is self-contained with inline fixtures and `make_*` factory functions. No shared conftest.py — each module defines its own test data builders.

## STRUCTURE

Key test files by domain:
- `test_dag.py` — DAGEdge, pipeline scheduler, LayerExpander
- `test_e2e.py` — Full pipeline: MegatronDAGTracer + AnalyticalBackend
- `test_collective.py` — Ring decomposition, flow counts, dependency chains
- `test_replayer.py` — Critical-path replay, timing breakdown (parametrized)
- `test_moe.py` — MoE/EP layer expansion and DAG tracing
- `test_step.py` — DP gradient sync step phase
- `test_compact.py` — Compact vs non-compact DAG equivalence
- `test_scenario.py` — Config serialization round-trip
- `test_chrome_trace.py` — Trace export with test classes
- `test_lookup.py` — Kernel timing lookup with autouse cache-clearing fixture
- `test_cli_profile.py` — CLI profiling with `tmp_path`
- `test_energy_cost.py` — Power + cost models
- `test_kernels.py` — Kernel profiling (requires torch)
- `test_tracking_mlflow.py` — MLflow tracking (`@pytest.mark.slow`)

## CONVENTIONS

- **`make_*` factories** — `make_datacenter()`, `make_workload()`, `_gpu()`, `_compute()`, `_comm()`, `_dag()`. Each test file defines its own
- **No conftest.py** — all fixtures inline. `@pytest.fixture` and `@pytest.fixture(autouse=True)` used locally
- **`@pytest.mark.parametrize`** — heavily used for data-driven tests
- **`@pytest.mark.slow`** — marks long-running tests (e.g., MLflow tracking)
- **`utils.py`** — `requires_torch` and `requires_cuda` skip markers for GPU-dependent tests
- **Test classes** — `TestFusedKernelsArg`, `TestCollectiveWorkloadConfig`, etc. for logical grouping (not unittest.TestCase)
- **`_` prefix helpers** — `_gpu()`, `_compute()`, `_comm()` are shorthand builders within test files

## ANTI-PATTERNS

- **Some test files are very large** — `test_collective_workload.py` (17k lines), `test_goal_trace.py` (12k lines) contain massive inline fixture data
- **No coverage config** — no pytest-cov or coverage.py setup
- **No CI integration** — tests run manually via `uv run pytest`

