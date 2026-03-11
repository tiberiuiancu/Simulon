"""Simulation backend for simulon.

Simulon uses ASTRA-Sim as the simulation engine. The only difference between
configurations is the network simulation backend:

- network_backend="analytical": Fast analytical network model (queueing theory)
- network_backend="ns3": Detailed packet-level network simulation (not yet implemented)

Usage:
    backend = AstraSimBackend(network_backend="analytical")
    results = backend.run(scenario)

For convenience, you can also use:
    backend = AnalyticalBackend()  # Same as AstraSimBackend(network_backend="analytical")
    backend = NS3Backend()          # Same as AstraSimBackend(network_backend="ns3")
"""

from .analytical import AnalyticalBackend
from .astra_sim import AstraSimBackend, NetworkBackend
from .base import Backend
from .ns3 import NS3Backend

__all__ = [
    "Backend",
    "AstraSimBackend",
    "AnalyticalBackend",
    "NS3Backend",
    "NetworkBackend",
]
