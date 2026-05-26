from __future__ import annotations

from pathlib import Path

from simulon.tracking.base import ExperimentTracker


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker — not yet implemented."""

    def start_run(self) -> None:
        raise NotImplementedError("W&B tracking is not yet implemented.")

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        raise NotImplementedError("W&B tracking is not yet implemented.")

    def log_metrics(self, metrics: dict[str, float]) -> None:
        raise NotImplementedError("W&B tracking is not yet implemented.")

    def log_artifact(self, path: Path) -> None:
        raise NotImplementedError("W&B tracking is not yet implemented.")

    def end_run(self) -> None:
        raise NotImplementedError("W&B tracking is not yet implemented.")
