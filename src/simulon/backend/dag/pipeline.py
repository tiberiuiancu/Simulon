from dataclasses import dataclass
from typing import Literal


@dataclass
class ScheduleSlot:
    microbatch_id: int
    pipeline_stage: int
    direction: Literal["fwd", "bwd"]


class PipelineScheduler:
    """1F1B pipeline schedule."""

    def __init__(self, pp: int, num_microbatches: int):
        self.pp = pp
        self.num_microbatches = num_microbatches

    def schedule_for_stage(self, stage: int) -> list[ScheduleSlot]:
        """Generate 1F1B schedule slots for a given pipeline stage.

        warmup = pp - stage - 1 forward passes
        then steady-state 1F1B
        then cooldown backwards
        """
        pp = self.pp
        nm = self.num_microbatches
        warmup = pp - stage - 1
        slots: list[ScheduleSlot] = []

        # Warmup: warmup forward microbatches
        for mb in range(warmup):
            slots.append(ScheduleSlot(microbatch_id=mb, pipeline_stage=stage, direction="fwd"))

        # Steady state: 1F1B pairs
        # After warmup, we have nm - warmup microbatches left for forward
        # In steady state we alternate fwd/bwd
        fwd_mb = warmup
        bwd_mb = 0
        steady = nm - warmup
        for _ in range(steady):
            slots.append(ScheduleSlot(microbatch_id=fwd_mb, pipeline_stage=stage, direction="fwd"))
            slots.append(ScheduleSlot(microbatch_id=bwd_mb, pipeline_stage=stage, direction="bwd"))
            fwd_mb += 1
            bwd_mb += 1

        # Cooldown: remaining backward passes
        for mb in range(bwd_mb, nm):
            slots.append(ScheduleSlot(microbatch_id=mb, pipeline_stage=stage, direction="bwd"))

        return slots


