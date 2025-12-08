"""
BANOS Antifragility Kernel v1.4

Implements the governing physics for soul-shard swarms:
- Intrinsic: Convex response to stress (HDC bundling)
- Inherited: Heterogeneous archetype distribution
- Induced: Structured stress injection + reputation

Three layers of antifragility ensuring the network's expected
performance increases with stress.
"""

from .core import (
    ShardArchetype,
    ShardConfig,
    ShardState,
    AntifragilityMetrics,
    AntifragileEngine,
)

__all__ = [
    "ShardArchetype",
    "ShardConfig",
    "ShardState",
    "AntifragilityMetrics",
    "AntifragileEngine",
]
