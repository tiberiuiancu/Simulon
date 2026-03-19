from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from simulon.backend.dag.nodes import ExecutionDAG
from simulon.config.dc import DatacenterConfig


@dataclass
class DAGTracerConfig:
    num_channels: int = 1
    algorithm: str = "ring"   # ring | tree | collnet_direct | collnet_chain | nvls | nvls_tree
    dtype_bytes: int = 2  # bf16
    cache_dir: Optional[Path] = field(default_factory=lambda: Path.home() / ".cache" / "simulon" / "dag")  # set to None to disable


class DAGTracer(ABC):
    @abstractmethod
    def trace(self, workload, datacenter: DatacenterConfig) -> ExecutionDAG: ...
