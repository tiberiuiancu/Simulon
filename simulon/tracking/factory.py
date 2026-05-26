from __future__ import annotations

import os

from simulon.tracking.base import ExperimentTracker


def get_trackers() -> list[ExperimentTracker]:
    """Return zero or more active trackers based on environment variables.

    MLflow is activated when any of MLFLOW_TRACKING_URI or MLFLOW_EXPERIMENT_NAME is set.
    W&B is activated when any of WANDB_PROJECT or WANDB_API_KEY is set.
    Multiple trackers may be active simultaneously.
    """
    trackers: list[ExperimentTracker] = []

    if any(
        os.environ.get(k) is not None for k in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME")
    ):
        from simulon.tracking.mlflow_tracker import MLflowTracker

        trackers.append(MLflowTracker())

    if any(os.environ.get(k) is not None for k in ("WANDB_PROJECT", "WANDB_API_KEY")):
        from simulon.tracking.wandb_tracker import WandbTracker

        trackers.append(WandbTracker())

    return trackers
