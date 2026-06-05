from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from simulon.tracking.base import ExperimentTracker

logger = logging.getLogger(__name__)


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker.

    All configuration is read from MLflow's own environment variables:
      MLFLOW_TRACKING_URI     — tracking server URI (default: local ./mlruns)
      MLFLOW_EXPERIMENT_NAME  — experiment to log into (default: "Default")
      MLFLOW_RUN_NAME         — optional human-readable run name
    """

    def start_run(self) -> None:
        import mlflow

        try:
            mlflow.start_run()
            run = mlflow.active_run()
            if run is None:
                logger.warning("MLflow run was not started (active_run() returned None).")
        except Exception as exc:
            logger.warning("Failed to start MLflow run: %s", exc)

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        import mlflow

        try:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        except Exception as exc:
            logger.warning("Failed to log parameters to MLflow: %s", exc)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        import mlflow

        try:
            mlflow.log_metrics(metrics)
        except Exception as exc:
            logger.warning("Failed to log metrics to MLflow: %s", exc)

    def log_artifact(self, path: Path) -> None:
        import mlflow

        try:
            mlflow.log_artifact(str(path))
        except Exception as exc:
            logger.warning("Failed to log artifact '%s' to MLflow: %s", path, exc)

    def end_run(self) -> None:
        import mlflow

        with contextlib.suppress(Exception):
            mlflow.end_run()
