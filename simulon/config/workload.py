from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CollectiveType(StrEnum):
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

    @field_validator("config", mode="before")
    @classmethod
    def _snake_to_kebab(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Recursively convert snake_case dict keys to kebab-case so downstream
        code can assume a single key style.
        """

        def _convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k.replace("_", "-"): _convert(val) for k, val in obj.items()}
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            return obj

        return _convert(v)


WorkloadConfig = Annotated[MegatronWorkload | CollectiveWorkload, Field(discriminator="framework")]
