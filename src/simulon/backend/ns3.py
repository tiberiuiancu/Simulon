from simulon.config.scenario import ScenarioConfig

from .base import Backend


class NS3Backend(Backend):
    def run(self, scenario: ScenarioConfig) -> dict:
        raise NotImplementedError
