from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from simulon.tracking.base import ExperimentTracker

logger = logging.getLogger(__name__)


class WandbTracker(ExperimentTracker):
    def __init__(self) -> None:
        self._run: Any | None = None

    def start_run(self) -> None:
        try:
            import wandb

            entity = os.environ.get("WANDB_ENTITY")
            project = os.environ.get("WANDB_PROJECT", "simulon")
            run_name = os.environ.get("WANDB_RUN_NAME")

            init_kwargs: dict[str, str | bool] = {"project": project}
            if entity:
                init_kwargs["entity"] = entity
            if run_name:
                init_kwargs["name"] = run_name

            self._run = wandb.init(**init_kwargs)
        except Exception as exc:
            logger.warning("Failed to start W&B run: %s", exc)

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        try:
            if self._run is not None:
                import wandb

                wandb.config.update(params)
        except Exception as exc:
            logger.warning("Failed to log parameters to W&B: %s", exc)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        try:
            if self._run is not None:
                import wandb

                wandb.log(metrics)
        except Exception as exc:
            logger.warning("Failed to log metrics to W&B: %s", exc)

    def log_artifact(self, path: Path) -> None:
        try:
            if self._run is not None:
                import wandb

                wandb.save(str(path))
        except Exception as exc:
            logger.warning("Failed to log artifact '%s' to W&B: %s", path, exc)

    def end_run(self) -> None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass
