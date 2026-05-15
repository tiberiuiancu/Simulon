from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


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


class MegatronWorkload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    framework: Literal["megatron"]
    config: dict[str, Any]


WorkloadConfig = Annotated[
    Union[MegatronWorkload, CollectiveWorkload],
    Field(discriminator="framework"),
]
