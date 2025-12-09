"""
Ara Hive System
================

Pheromone-based coordination for Ara's agent swarm.

The insight: coordination through tiny shared signals, not central micromanagement.

Components:
- Pheromone: A tiny signal (kind, key, strength, ttl)
- PheromoneStore: Shared state for all agents
- HiveAgent: Base class for worker agents
- QueenHiveAdapter: How Ara-Core steers the hive

Pheromone kinds:
- global: System-wide mode (PUBLISHING_EXPANSION, MAINTENANCE, SAFE_MODE)
- priority: High-value targets to swarm toward
- alarm: Things to avoid / backoff from
- reward: What strategies worked (for learning)
- role: Agent role assignments

Usage:
    from ara.hive import PheromoneStore, HiveAgent, QueenHiveAdapter

    store = PheromoneStore()
    adapter = QueenHiveAdapter(store)

    # Queen sets direction
    adapter.set_global_mode("PUBLISHING_EXPANSION", focus="KDP_QUANTUM_GAPS")

    # Workers sense and react
    agent = HiveAgent("worker_01", "kdp_scanner", store, tools)
    agent.tick()
"""

from .pheromones import Pheromone, PheromoneKind
from .store import PheromoneStore
from .agent import HiveAgent, GenericWorker
from .queen import QueenHiveAdapter
from .node import HiveNode, HiveConfig

__all__ = [
    "Pheromone",
    "PheromoneKind",
    "PheromoneStore",
    "HiveAgent",
    "GenericWorker",
    "QueenHiveAdapter",
    "HiveNode",
    "HiveConfig",
]
