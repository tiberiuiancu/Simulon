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
    # Under sequence parallelism the traced ReduceScatter/AllGather are TP collectives on the
    # critical path; only with SP off can a traced RS/AG be the (async) distributed-optimizer
    # grad sync. The async inference in _add_collective needs to know which world it is in.
    sequence_parallel: bool = False
    # The run's distributed-optimizer overlap flags: they decide whether a RECORDED
    # dist_opt_* collective is asynchronous (hidden) or synchronous (exposed).
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False


class DAGTracer(ABC):
    @abstractmethod
    def trace(self, workload, datacenter: DatacenterConfig) -> ExecutionDAG: ...
