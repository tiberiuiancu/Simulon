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
            experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Default")
            run_name = os.environ.get("MLFLOW_RUN_NAME", "simulon")
            logger.info("MLflow: experiment_name=%r  run_name=%r", experiment_name, run_name)

            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            exp_id = client.create_experiment(experiment_name) if exp is None else exp.experiment_id
            logger.info("MLflow: resolved experiment_id=%s", exp_id)

            mlflow.start_run(experiment_id=exp_id, run_name=run_name)
            run = mlflow.active_run()
            if run is None:
                logger.warning("MLflow run was not started (active_run() returned None).")
            else:
                logger.info("MLflow: run started id=%s", run.info.run_id)
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
