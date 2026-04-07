from abc import ABC, abstractmethod

from simulon.config.scenario import ScenarioConfig


class Backend(ABC):
    @abstractmethod
    def run(self, scenario: ScenarioConfig) -> dict:
        """Run the simulation and return results."""
        ...

    @staticmethod
    def _get_trackers():
        """Return active experiment trackers based on environment variables."""
        from simulon.tracking import get_trackers
        return get_trackers()
