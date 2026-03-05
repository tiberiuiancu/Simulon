from abc import ABC, abstractmethod

from simulon.config.scenario import ScenarioConfig


class Backend(ABC):
    @abstractmethod
    def run(self, scenario: ScenarioConfig) -> dict:
        """Run the simulation and return results."""
        ...
