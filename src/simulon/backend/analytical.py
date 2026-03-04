from simulon.config.dc import DatacenterConfig
from simulon.config.workload import WorkloadConfig

from .base import Backend


class AnalyticalBackend(Backend):
    def run(self, dc: DatacenterConfig, workload: WorkloadConfig) -> dict:
        raise NotImplementedError
