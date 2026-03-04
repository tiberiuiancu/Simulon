from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel


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
