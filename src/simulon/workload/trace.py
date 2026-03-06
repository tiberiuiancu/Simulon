from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class CommOpType(str, Enum):
    all_reduce = "all_reduce"
    all_gather = "all_gather"
    reduce_scatter = "reduce_scatter"
    broadcast = "broadcast"
    isend = "isend"
    irecv = "irecv"
    all_to_all = "all_to_all"


class ParallelGroup(str, Enum):
    tp = "tp"
    dp = "dp"
    pp = "pp"
    ep = "ep"


class CommOp(BaseModel):
    op: Literal["comm"] = "comm"
    comm_type: CommOpType
    group: ParallelGroup
    group_size: int
    size_bytes: int
    stage: str
    additional: str = ""
    src: Optional[int] = None
    dst: Optional[int] = None


class ComputeOp(BaseModel):
    op: Literal["compute"] = "compute"
    kernel: str
    input_shapes: list[tuple[int, ...]]
    compute_time_us: Optional[float] = None
    stage: str


TraceOp = Annotated[Union[CommOp, ComputeOp], Field(discriminator="op")]


class WorkloadTrace(BaseModel):
    framework: str
    model_name: str
    num_gpus: int
    tp: int
    pp: int
    dp: int
    ep: int
    iterations: int
    ops: list[TraceOp]
