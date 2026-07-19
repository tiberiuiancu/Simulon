from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from simulon.backend.dag.nodes import ExecutionDAG
from simulon.config.dc import DatacenterConfig


@dataclass
class DAGTracerConfig:
    num_channels: int = 1
    algorithm: str = "ring"  # ring | tree | collnet_direct | collnet_chain | nvls | nvls_tree
    overlap_async_collectives: bool = False


class DAGTracer(ABC):
    @abstractmethod
    def trace(self, workload, datacenter: DatacenterConfig) -> ExecutionDAG: ...
