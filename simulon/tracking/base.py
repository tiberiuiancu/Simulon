from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ExperimentTracker(ABC):
    """Abstract base class for experiment trackers (MLflow, W&B, etc.)."""

    @abstractmethod
    def start_run(self) -> None:
        """Begin a new tracked run."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        """Log scalar/string hyperparameters."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log numeric result metrics."""
        ...

    @abstractmethod
    def log_artifact(self, path: Path) -> None:
        """Upload a file artifact (YAML, JSON, etc.)."""
        ...

    @abstractmethod
    def end_run(self) -> None:
        """Finalize the run."""
        ...

    def pull_metrics(
        self,
        workload_hash: str | None = None,
        config_filters: dict[str, str | int | float | bool] | None = None,
        run_name_prefix: str | None = None,
    ) -> dict[str, Any] | None:
        """Pull metrics for a finished run.

        Parameters
        ----------
        workload_hash: str | None
            Match runs whose wandb config contains this workload hash.
        config_filters: dict[str, ...] | None
            Additional flat config keys that must match.
        run_name_prefix: str | None
            Match runs whose display name starts with this prefix.

        Returns
        -------
        dict[str, Any] | None
            The first matching run's summary metrics, or None if no match.
        """
        return None

    def fetch_runs(self, prefix: str | None = None) -> list[dict[str, Any]]:
        """Fetch all finished runs whose display name starts with *prefix*.

        Returns a list of dicts with ``display_name``, ``config``, ``summary``.
        """
        return []

    def __enter__(self) -> ExperimentTracker:
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_run()
