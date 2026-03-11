"""Analytical network backend for ASTRA-Sim.

This backend uses ASTRA-Sim with an analytical (queueing-theory based) network model
for fast approximate simulation.
"""

from simulon.config.scenario import ScenarioConfig

from .astra_sim import AstraSimBackend


class AnalyticalBackend(AstraSimBackend):
    """ASTRA-Sim backend with analytical network simulation.

    Uses a fast analytical network model based on queueing theory for
    approximate but quick simulation results.
    """

    def __init__(self):
        """Initialize ASTRA-Sim with analytical network backend."""
        super().__init__(network_backend="analytical")
