from __future__ import annotations

import logging
import os

from simulon.tracking.base import ExperimentTracker

logger = logging.getLogger(__name__)


def get_trackers() -> list[ExperimentTracker]:
    """Return zero or more active trackers based on environment variables.

    MLflow is activated when any of MLFLOW_TRACKING_URI or MLFLOW_EXPERIMENT_NAME is set.
    W&B is activated when any of WANDB_PROJECT or WANDB_API_KEY is set.
    Multiple trackers may be active simultaneously.
    """
    trackers: list[ExperimentTracker] = []

    mlflow_envs = {k: v for k, v in os.environ.items() if k.startswith("MLFLOW_")}
    if mlflow_envs:
        logger.info("MLflow env vars: %s", mlflow_envs)
    else:
        logger.info("No MLFLOW_* env vars found.")

    if any(
        os.environ.get(k) is not None for k in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME")
    ):
        from simulon.tracking.mlflow_tracker import MLflowTracker

        trackers.append(MLflowTracker())

    if any(os.environ.get(k) is not None for k in ("WANDB_PROJECT", "WANDB_API_KEY")):
        from simulon.tracking.wandb_tracker import WandbTracker

        trackers.append(WandbTracker())

    return trackers
