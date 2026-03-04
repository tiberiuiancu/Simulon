from abc import ABC, abstractmethod

from simulon.config.dc import DatacenterConfig
from simulon.config.workload import WorkloadConfig


class Backend(ABC):
    @abstractmethod
    def run(self, dc: DatacenterConfig, workload: WorkloadConfig) -> dict:
        """Run the simulation and return results."""
        ...
