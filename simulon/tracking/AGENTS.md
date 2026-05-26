# tracking — Experiment Tracking

6 files. ABC + factory pattern for MLflow and Weights & Biases integration.

## OVERVIEW

Logs simulation parameters and results to experiment tracking platforms. Activated by environment variables, not config.

## STRUCTURE

```
tracking/
├── base.py             # ExperimentTracker ABC — log_params, log_metrics, log_artifact, end_run
├── factory.py          # get_trackers() — returns active trackers based on env vars
├── mlflow_tracker.py   # MLflowTracker implementation
├── wandb_tracker.py    # WandBTracker implementation
├── params.py           # Parameter extraction helpers for logging
└── __init__.py         # Exports: ExperimentTracker, get_trackers
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| New tracker backend | Subclass `ExperimentTracker` in new file | Register in `factory.py` |
| Add logged parameters | `params.py` | Extract from ScenarioConfig / SimulationResult |
| Change activation logic | `factory.py` | Environment variable detection |

## CONVENTIONS

- **Env-var activated** — trackers auto-enable when `MLFLOW_TRACKING_URI` or `WANDB_API_KEY` are set
- **ABC interface** — `ExperimentTracker` defines: `log_params`, `log_metrics`, `log_artifact`, `end_run`
- **Factory returns list** — `get_trackers()` returns all active trackers (can be multiple simultaneously)
- **Lazy imports** — mlflow/wandb imported inside tracker classes, not at module level (optional deps)
