from abc import ABC, abstractmethod
from typing import Protocol

from simulon.backend.dag import ExecutionDAG
from simulon.config.scenario import ScenarioConfig


class BackendResult(Protocol):
    total_time_ms: float
    # Optional summary metadata may be present on concrete results.


class Backend(ABC):
    @abstractmethod
    def run(self, scenario: ScenarioConfig) -> dict[str, object]:
        """Run the simulation and return results."""
        ...

    @abstractmethod
    def simulate(
        self,
        scenario: ScenarioConfig,
        compact: bool = False,
        ignore_oom: bool = False,
        ignore_missing: bool = False,
    ) -> tuple[ExecutionDAG, BackendResult]:
        """Build and replay the simulation DAG."""
        ...

    @staticmethod
    def _get_trackers():
        """Return active experiment trackers based on environment variables."""
        from simulon.tracking import get_trackers
        return get_trackers()
