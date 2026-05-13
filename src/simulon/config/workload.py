from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import DType


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InferencePhase(str, Enum):
    prefill = "prefill"
    decode = "decode"


class RoutingStrategy(str, Enum):
    round_robin = "RoundRobin"
    random = "Random"


# ---------------------------------------------------------------------------
# Shared model spec
# ---------------------------------------------------------------------------


class LLMSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    from_: Optional[str] = Field(None, alias="from")
    name: Optional[str] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    swiglu: bool = False
    num_experts: Optional[int] = None
    top_k: Optional[int] = None
    gflops_per_train_token: Optional[float] = None


# ---------------------------------------------------------------------------
# Megatron-LM workload
# ---------------------------------------------------------------------------


class MegatronParallelism(BaseModel):
    tp: int = 1
    pp: int = 1
    ep: int = 1
    dp: Optional[int] = None  # derived as num_gpus / (tp * pp * ep) if omitted
    sp: bool = False
    vpp: int = 1
    distributed_optimizer: bool = False
    num_microbatches: Optional[int] = None
    pipeline_schedule: str = "1f1b"
    cp: int = 1

    @field_validator("cp")
    @classmethod
    def _validate_cp(cls, value: int) -> int:
        if value > 1:
            raise ValueError("Context Parallelism > 1 not supported")
        return value


class MegatronTraining(BaseModel):
    num_gpus: int
    global_batch_size: int
    micro_batch_size: int
    sequence_length: int
    dtype: DType = DType.bf16
    flash_attention: bool = False
    iterations: int = 1


class MegatronDeprecatedWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: Literal["megatron-deprecated"]
    model: Union[str, LLMSpec]
    parallelism: MegatronParallelism
    training: MegatronTraining
    megatron_args: dict[str, Any] | None = None


class MegatronWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: Literal["megatron"]
    config: dict[str, Any]


# ---------------------------------------------------------------------------
# Inference workload
# ---------------------------------------------------------------------------


class InferenceParallelism(BaseModel):
    tp: int = 1
    pp: int = 1
    ep: int = 1
    dp: Optional[int] = None  # derived as num_gpus / (tp * pp * ep) if omitted


class InferenceRun(BaseModel):
    num_gpus: int
    phase: InferencePhase = InferencePhase.decode
    batch_size: int
    seq_length: int
    dtype: DType = DType.bf16
    flash_attention: bool = False
    routing_strategy: RoutingStrategy = RoutingStrategy.round_robin


class InferenceWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: Literal["inference"]
    model: Union[str, LLMSpec]
    parallelism: InferenceParallelism
    inference: InferenceRun


# ---------------------------------------------------------------------------
# Collective workload
# ---------------------------------------------------------------------------


class CollectiveType(str, Enum):
    AllReduce = "AllReduce"
    AllGather = "AllGather"
    ReduceScatter = "ReduceScatter"
    AllToAll = "AllToAll"


class CollectiveWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: Literal["collective"]
    collective_type: CollectiveType
    message_size_bytes: int = Field(..., gt=0)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

WorkloadConfig = Annotated[
    Union[MegatronDeprecatedWorkload, MegatronWorkload, InferenceWorkload, CollectiveWorkload],
    Field(discriminator="framework"),
]
