"""Simulation backends for simulon.

All backends are based on ASTRA-Sim, differing only in their network simulation approach:
- AnalyticalBackend: Fast analytical network model (queueing theory)
- NS3Backend: Detailed packet-level network simulation using NS-3

You can also use AstraSimBackend directly with the network_backend parameter.
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
