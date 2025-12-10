"""
Ara Swarm Module
================

Stupidly-cubed micro-agent swarm with hypervector hive state
and quantum-inspired control plane.

"So dumb we can run millions of them."

Components:
- MicroAgent: Tiny state, pattern match, emit votes + pheromone deltas
- HiveState: Hypervector-of-hypervectors encoding swarm state
- ControlPlane: Quantum-inspired steering (state vector over modes)
- SwarmSimulation: Main runner
- MathSearchSwarm: Configured for mathematical search
"""

from ara.swarm.swarm_sim import (
    # Core types
    MicroAgent,
    HiveState,
    ControlPlane,
    SwarmMode,
    # Simulations
    SwarmSimulation,
    MathSearchSwarm,
    # Hypervector ops
    random_hv,
    bind,
    bundle,
    similarity,
    permute,
    # Constants
    D,
    MODE_BASIS,
)

__all__ = [
    "MicroAgent",
    "HiveState",
    "ControlPlane",
    "SwarmMode",
    "SwarmSimulation",
    "MathSearchSwarm",
    "random_hv",
    "bind",
    "bundle",
    "similarity",
    "permute",
    "D",
    "MODE_BASIS",
]
