from pathlib import Path
from typing import Union

from pydantic import BaseModel

from .dc import DatacenterConfig
from .workload import WorkloadConfig


class ScenarioConfig(BaseModel):
    datacenter: Union[Path, DatacenterConfig]
    workload: Union[Path, WorkloadConfig]
