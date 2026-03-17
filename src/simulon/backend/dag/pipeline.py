from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class ScheduleSlot:
    microbatch_id: int
    pipeline_stage: int
    direction: Literal["fwd", "bwd"]


class PipelineScheduler(ABC):
    """Abstract base for pipeline schedules."""

    def __init__(self, pp: int, num_microbatches: int):
        self.pp = pp
        self.num_microbatches = num_microbatches

    @abstractmethod
    def schedule_for_stage(self, stage: int) -> list[ScheduleSlot]:
        """Return the ordered list of slots for a given pipeline stage."""
        ...


class OneFOneBScheduler(PipelineScheduler):
    """Standard 1F1B pipeline schedule.

    Each stage runs:
      - warmup:       (pp - stage - 1) forward microbatches
      - steady state: alternating fwd/bwd pairs
      - cooldown:     remaining backward microbatches
    """

    def schedule_for_stage(self, stage: int) -> list[ScheduleSlot]:
        pp = self.pp
        nm = self.num_microbatches
        warmup = pp - stage - 1
        slots: list[ScheduleSlot] = []

        for mb in range(warmup):
            slots.append(ScheduleSlot(microbatch_id=mb, pipeline_stage=stage, direction="fwd"))

        fwd_mb = warmup
        bwd_mb = 0
        for _ in range(nm - warmup):
            slots.append(ScheduleSlot(microbatch_id=fwd_mb, pipeline_stage=stage, direction="fwd"))
            slots.append(ScheduleSlot(microbatch_id=bwd_mb, pipeline_stage=stage, direction="bwd"))
            fwd_mb += 1
            bwd_mb += 1

        for mb in range(bwd_mb, nm):
            slots.append(ScheduleSlot(microbatch_id=mb, pipeline_stage=stage, direction="bwd"))

        return slots


_SCHEDULERS: dict[str, type[PipelineScheduler]] = {
    "1f1b": OneFOneBScheduler,
}


def make_scheduler(schedule: str, pp: int, num_microbatches: int) -> PipelineScheduler:
    cls = _SCHEDULERS.get(schedule)
    if cls is None:
        raise ValueError(f"Unknown pipeline schedule {schedule!r}. Supported: {list(_SCHEDULERS)}")
    return cls(pp, num_microbatches)
