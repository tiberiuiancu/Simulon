# tests — Test Suite

11 test files. pytest framework, no conftest.py.

## OVERVIEW

Each test file is self-contained with inline fixtures and `make_*` factory functions. No shared conftest.py — each module defines its own test data builders.

## STRUCTURE

Key test files by domain:
- `test_chrome_trace.py` — Chrome/Perfetto trace JSON export
- `test_collective.py` — Ring decomposition, flow counts, dependency chains
- `test_collective_workload.py` — CollectiveWorkload end-to-end simulation
- `test_goal_trace.py` — GOAL format export (for ATLAS/LogGOPSim)
- `test_node_template.py` — NodeSpec YAML template resolution
- `test_replayer.py` — Critical-path replay, timing breakdown (parametrized)
- `test_trace_e2e.py` — Trace-driven DAG + AnalyticalBackend integration
- `test_trace_parser.py` — TraceFileParser validation and JSON loading
- `test_trace_refactor.py` — Trace model and collector unit tests
- `test_trace_tracer.py` — MegatronDagTracer from real GPU traces
- `test_tree_nvls_calbusbw.py` — Tree/NVLS algorithms and bandwidth calibration

## CONVENTIONS

- **`make_*` factories** — `make_datacenter()`, `make_workload()`, `_gpu()`, `_compute()`, `_comm()`, `_dag()`. Each test file defines its own
- **No conftest.py** — all fixtures inline. `@pytest.fixture` and `@pytest.fixture(autouse=True)` used locally
- **`@pytest.mark.parametrize`** — heavily used for data-driven tests
- **`@pytest.mark.slow`** — marks long-running tests (e.g., MLflow tracking)
- **`utils.py`** — `requires_torch` and `requires_cuda` skip markers for GPU-dependent tests
- **Test classes** — `TestFusedKernelsArg`, `TestCollectiveWorkloadConfig`, etc. for logical grouping (not unittest.TestCase)
- **`_` prefix helpers** — `_gpu()`, `_compute()`, `_comm()` are shorthand builders within test files

## ANTI-PATTERNS

- **Some test files contain large inline fixture data** — `test_collective_workload.py` (~390 lines), `test_goal_trace.py` (~315 lines) both embed sizeable test data
- **No coverage config** — no pytest-cov or coverage.py setup
- **No CI integration** — tests run manually via `uv run pytest`

