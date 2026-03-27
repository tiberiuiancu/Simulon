from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, BeforeValidator, Field


class Cost(BaseModel):
    value: float
    min: Optional[float] = None
    max: Optional[float] = None


# A cost field accepts either a plain float or a Cost object.
CostField = Union[float, Cost]


class DType(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"
    fp8 = "fp8"


# ---------------------------------------------------------------------------
# Power models
# ---------------------------------------------------------------------------


class ConstantPowerModel(BaseModel):
    """Draws a fixed wattage regardless of utilisation."""
    type: Literal["constant"] = "constant"
    tdp_w: float


class LinearPowerModel(BaseModel):
    """Interpolates linearly between idle power at 0% utilisation and TDP at 100%."""
    type: Literal["linear"] = "linear"
    tdp_w: float
    idle_power_w: float


def _default_power_model_type(v: object) -> object:
    """Inject ``type: 'constant'`` when the type discriminator is absent.

    Per spec, ``type`` defaults to ``constant`` if omitted in YAML/config.
    """
    if isinstance(v, dict) and "type" not in v:
        return {**v, "type": "constant"}
    return v


PowerModel = Annotated[
    Annotated[Union[ConstantPowerModel, LinearPowerModel], Field(discriminator="type")],
    BeforeValidator(_default_power_model_type),
]
