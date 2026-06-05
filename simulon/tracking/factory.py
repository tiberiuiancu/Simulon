from __future__ import annotations

import os

from simulon.tracking.base import ExperimentTracker


def get_trackers() -> list[ExperimentTracker]:
    """Return active W&B tracker based on environment variables."""
    trackers: list[ExperimentTracker] = []

    if os.environ.get("WANDB_API_KEY") is not None:
        from simulon.tracking.wandb_tracker import WandbTracker

        trackers.append(WandbTracker())

    return trackers
