from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from simulon.tracking.base import ExperimentTracker

logger = logging.getLogger(__name__)


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker.

    All configuration is read from MLflow's own environment variables:
      MLFLOW_TRACKING_URI     — tracking server URI (default: local ./mlruns)
      MLFLOW_EXPERIMENT_NAME  — experiment to log into (default: "Default")
      MLFLOW_RUN_NAME        — optional human-readable run name (default: "simulon")
    """

    def start_run(self) -> None:
        import mlflow

        try:
            experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "simulon")
            run_name = os.environ.get("MLFLOW_RUN_NAME", "simulon")

            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if exp is None:
                logger.warning(
                    "MLflow experiment %r not found on server. Skipping tracking.", experiment_name
                )
                return
            mlflow.start_run(experiment_id=exp.experiment_id, run_name=run_name)
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
