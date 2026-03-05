from simulon.config.scenario import ScenarioConfig

from .base import Backend


class AnalyticalBackend(Backend):
    def run(self, scenario: ScenarioConfig) -> dict:
        raise NotImplementedError
