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
        workload_hash: str,
        config_filters: dict[str, str | int | float | bool] | None = None,
    ) -> dict[str, Any] | None:
        """Pull metrics for a finished run matching *workload_hash*.

        Optionally also match on *config_filters* — a flat dict of wandb
        config keys (e.g. ``{"datacenter.node.scale_out.nic.speed": "10Gbps"}``)
        that must all equal the run's logged config.  This is needed when
        the same workload is simulated against different datacenter configs
        (e.g. varying network bandwidth).

        Returns ``None`` when the tracker backend does not support querying
        or when no matching run is found.
        """
        return None

    def __enter__(self) -> ExperimentTracker:
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_run()
