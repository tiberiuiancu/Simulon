"""NS-3 network backend for ASTRA-Sim.

This backend uses ASTRA-Sim with NS-3 packet-level network simulation for
detailed and accurate network modeling.
"""

from simulon.config.scenario import ScenarioConfig

from .astra_sim import AstraSimBackend


class NS3Backend(AstraSimBackend):
    """ASTRA-Sim backend with NS-3 packet-level network simulation.

    Uses the NS-3 network simulator for detailed packet-level simulation,
    providing higher accuracy at the cost of slower simulation speed.
    """

    def __init__(self):
        """Initialize ASTRA-Sim with NS-3 network backend."""
        super().__init__(network_backend="ns3")
