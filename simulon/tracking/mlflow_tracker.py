from __future__ import annotations

from pathlib import Path

from simulon.tracking.base import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker.

    All configuration is read from MLflow's own environment variables:
      MLFLOW_TRACKING_URI     — tracking server URI (default: local ./mlruns)
      MLFLOW_EXPERIMENT_NAME  — experiment to log into (default: "Default")
      MLFLOW_RUN_NAME         — optional human-readable run name
    """

    def start_run(self) -> None:
        import mlflow

        mlflow.start_run()

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        import mlflow

        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        import mlflow

        mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        import mlflow

        mlflow.log_artifact(str(path))

    def end_run(self) -> None:
        import mlflow

        mlflow.end_run()
