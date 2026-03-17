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

    def schedule_steady_state_for_stage(self, stage: int) -> list[ScheduleSlot]:
        """Return only the steady-state 1F1B slots for this stage (no warmup or cooldown).

        In steady state every stage alternates fwd/bwd. Stage k's steady state
        starts at fwd microbatch = pp - stage - 1 (the first mb not in warmup).

        All stages run the same number of 1F1B pairs, bounded by the stage with
        the most warmup (stage 0, warmup = pp - 1):
            steady = nm - (pp - 1)
        This ensures no stage runs a longer wind-down tail that would show up as
        a cooldown bubble in the simulation.
        """
        pp = self.pp
        nm = self.num_microbatches
        warmup = pp - stage - 1
        steady = nm - (pp - 1)  # same for every stage
        slots: list[ScheduleSlot] = []
        fwd_mb = warmup
        bwd_mb = 0
        for _ in range(steady):
            slots.append(ScheduleSlot(microbatch_id=fwd_mb, pipeline_stage=stage, direction="fwd"))
            slots.append(ScheduleSlot(microbatch_id=bwd_mb, pipeline_stage=stage, direction="bwd"))
            fwd_mb += 1
            bwd_mb += 1
        return slots
