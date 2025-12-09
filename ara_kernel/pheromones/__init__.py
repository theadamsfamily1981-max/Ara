"""
Ara Kernel Pheromones
======================

Pheromone-based coordination for agent swarms.
Implements a shared signal bus that agents can emit to and read from.

The insight: coordination through tiny shared signals, not central micromanagement.
"""

from .bus_local import LocalPheromoneBus, Pheromone, PheromoneScope

__all__ = [
    "LocalPheromoneBus",
    "Pheromone",
    "PheromoneScope",
]
