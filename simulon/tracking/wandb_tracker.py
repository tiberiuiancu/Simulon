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

    def pull_metrics(
        self,
        workload_hash: str | None = None,
        config_filters: dict[str, str | int | float | bool] | None = None,
        run_name_prefix: str | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            import wandb

            entity = os.environ.get("WANDB_ENTITY")
            project = os.environ.get("WANDB_PROJECT", "simulon")
            api = wandb.Api()

            filters: dict[str, object] = {"state": "finished"}
            if run_name is not None:
                filters["display_name"] = run_name
            elif run_name_prefix is not None:
                filters["display_name"] = {"$regex": f"^{run_name_prefix}"}
            if workload_hash is not None:
                filters["config.workload_hash"] = workload_hash
            if config_filters:
                for k, v in config_filters.items():
                    filters[f"config.{k}"] = v
            runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
            for run in runs:
                if run_name is not None and run.display_name != run_name:
                    continue
                if run_name_prefix is not None and not run.display_name.startswith(run_name_prefix):
                    continue
                return dict(run.summary)
        except Exception as exc:
            logger.warning("Failed to pull metrics from W&B: %s", exc)
        return None

    def fetch_runs(self, prefix: str | None = None) -> list[dict[str, Any]]:
        """Fetch all finished runs whose display name starts with *prefix*.

        Returns a list of dicts with ``display_name``, ``config``, ``summary``.
        """
        try:
            import wandb

            entity = os.environ.get("WANDB_ENTITY")
            project = os.environ.get("WANDB_PROJECT", "simulon")
            api = wandb.Api()
            filters: dict[str, object] = {"state": "finished"}
            if prefix:
                filters["display_name"] = {"$regex": f"^{prefix}"}
            runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
            return [
                {
                    "display_name": run.display_name,
                    "config": dict(run.config),
                    "summary": dict(run.summary),
                }
                for run in runs
            ]
        except Exception as exc:
            logger.warning("Failed to fetch runs from W&B: %s", exc)
        return []
