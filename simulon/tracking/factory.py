from __future__ import annotations

import os
from pathlib import Path

from simulon.tracking.base import ExperimentTracker


def get_trackers(scenario_path: str | Path | None = None) -> list[ExperimentTracker]:
    if scenario_path is not None:
        from simulon.tracking.env import load_cascading_tracking_env

        load_cascading_tracking_env(scenario_path)

    trackers: list[ExperimentTracker] = []

    if os.environ.get("WANDB_API_KEY") is not None:
        from simulon.tracking.wandb_tracker import WandbTracker

        trackers.append(WandbTracker())

    return trackers
