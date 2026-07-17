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
        run_name: str | None = None,
    ) -> dict[str, Any] | None:
        return None

    def fetch_runs(self, prefix: str | None = None) -> list[dict[str, Any]]:
        """Fetch all finished runs whose display name starts with *prefix*.

        Returns a list of dicts with ``display_name``, ``config``, ``summary``.
        """
        return []

    def has_run(self, run_name: str | None = None, workload_hash: str | None = None) -> bool:
        """Return True if a finished run matching *run_name* and/or *workload_hash* exists."""
        return False

    def __enter__(self) -> ExperimentTracker:
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_run()
