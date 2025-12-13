"""
Hierarchical Memory Module
==========================

GUTC-based hierarchical memory with Γ coupling for cross-level communication.

Levels:
    L1 (fast): Chat buffer, token-level dynamics
    L2 (medium): Session summaries, sentence/utterance scale
    L3 (slow): Long-term themes, identity, values

Γ Coupling:
    - Ascending: Only surprise propagates up (novel/unexpected)
    - Descending: Priors damp out-of-character moves

See: docs/theory/CRITICAL_CAPACITY_RUNTIME_POLICY.md
"""

from .hierarchy import (
    HierarchicalMemory,
    HierarchicalMemoryState,
    MemoryEntry,
)

__all__ = [
    "HierarchicalMemory",
    "HierarchicalMemoryState",
    "MemoryEntry",
]
