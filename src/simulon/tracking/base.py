from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class ExperimentTracker(ABC):
    """Abstract base class for experiment trackers (MLflow, W&B, etc.)."""

    @abstractmethod
    def start_run(self) -> None:
        """Begin a new tracked run."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Union[str, int, float, bool]]) -> None:
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

    def __enter__(self) -> ExperimentTracker:
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_run()
