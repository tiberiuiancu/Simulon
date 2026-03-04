from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

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
    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[str] = Field(None, alias="from")
    name: Optional[str] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    swiglu: bool = False
    moe: bool = False
    num_experts: Optional[int] = None
    top_k: Optional[int] = None


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


class MegatronTraining(BaseModel):
    num_gpus: int
    global_batch_size: int
    micro_batch_size: int
    sequence_length: int
    dtype: DType = DType.bf16
    flash_attention: bool = False
    iterations: int = 1


class MegatronWorkload(BaseModel):
    framework: Literal["megatron"]
    model: Union[str, LLMSpec]
    parallelism: MegatronParallelism
    training: MegatronTraining


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
    framework: Literal["inference"]
    model: Union[str, LLMSpec]
    parallelism: InferenceParallelism
    inference: InferenceRun


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

WorkloadConfig = Annotated[
    Union[MegatronWorkload, InferenceWorkload],
    Field(discriminator="framework"),
]
