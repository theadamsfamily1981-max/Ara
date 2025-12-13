#!/usr/bin/env python3
"""
Hierarchical Memory with Γ Coupling
====================================

Implements GUTC-based hierarchical memory with ascending/descending Γ coupling.

Theory (from GUTC):
- L1 (fast): Chat buffer, token-level dynamics
- L2 (medium): Session summaries, sentence/utterance scale
- L3 (slow): Long-term themes, identity, values

Γ Coupling:
- Ascending (Γ_asc): Only surprise propagates up
  - When prediction error exceeds threshold, compress and store at higher level
- Descending (Γ_desc): Priors damp out-of-character moves
  - L3 values constrain L1 generation via logit biasing

This implements the GUTC manifold constraint:
    Γ*_asc(l) · Π^(l)_sens = Γ*_desc(l+1) · Π^(l+1)_prior

Usage:
    memory = HierarchicalMemory()

    # On each turn
    dampening = memory.update(
        user_turn="...",
        model_prediction="...",
        surprise_score=0.4,
        l3_priors=["be helpful", "avoid harm"]
    )

    # Use dampening to adjust generation
    if dampening > 0.3:
        # Apply stronger grounding, lower temperature
        pass
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import deque
import hashlib
import time


# =============================================================================
# Memory State
# =============================================================================

@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    content: str
    timestamp: float
    surprise_score: float = 0.0
    embedding: Optional[np.ndarray] = None
    source_level: int = 1
    tags: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Generate unique ID from content hash."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class HierarchicalMemoryState:
    """State of the three-level memory hierarchy."""
    L1: deque = field(default_factory=lambda: deque(maxlen=100))  # Chat buffer (fast)
    L2: List[MemoryEntry] = field(default_factory=list)  # Session summaries (medium)
    L3: List[MemoryEntry] = field(default_factory=list)  # Long-term themes (slow)


# =============================================================================
# Hierarchical Memory
# =============================================================================

class HierarchicalMemory:
    """
    Three-level hierarchical memory with GUTC Γ coupling.

    Implements ascending (surprise → up) and descending (priors → down) coupling
    as specified in the GUTC hierarchical framework.

    Levels:
        L1: Fast - recent conversation turns (chat buffer)
        L2: Medium - session summaries, notable interactions
        L3: Slow - identity, values, long-term themes

    Γ Coupling Parameters:
        gamma_asc: Strength of ascending surprise propagation (0-1)
        gamma_desc: Strength of descending prior influence (0-1)
    """

    def __init__(
        self,
        gamma_asc: float = 0.5,
        gamma_desc: float = 0.5,
        surprise_threshold: float = 0.3,
        l2_consolidation_interval: int = 10,
        l3_consolidation_interval: int = 50,
    ):
        """
        Initialize hierarchical memory.

        Args:
            gamma_asc: Ascending coupling strength (surprise → up)
            gamma_desc: Descending coupling strength (priors → down)
            surprise_threshold: Minimum surprise for L2 propagation
            l2_consolidation_interval: Turns between L2 consolidation
            l3_consolidation_interval: L2 entries before L3 consolidation
        """
        self.gamma_asc = gamma_asc
        self.gamma_desc = gamma_desc
        self.surprise_threshold = surprise_threshold
        self.l2_consolidation_interval = l2_consolidation_interval
        self.l3_consolidation_interval = l3_consolidation_interval

        self.state = HierarchicalMemoryState()
        self._turn_count = 0

    def update(
        self,
        user_turn: str,
        model_prediction: str,
        surprise_score: float,
        l3_priors: Optional[List[str]] = None,
    ) -> float:
        """
        Update memory hierarchy with new turn.

        Args:
            user_turn: User's message
            model_prediction: Model's predicted/generated response
            surprise_score: Scalar [0, 1] measuring prediction error
            l3_priors: Optional list of L3 constraints (identity/values)

        Returns:
            dampening: Scalar [0, 1] to apply to generation
                       (higher = more prior constraint)
        """
        self._turn_count += 1
        timestamp = time.time()

        # === L1: Always store full conversation ===
        entry = MemoryEntry(
            content=user_turn,
            timestamp=timestamp,
            surprise_score=surprise_score,
            source_level=1,
        )
        self.state.L1.append(entry)

        # === Ascending Γ: Propagate surprise to L2 ===
        if surprise_score > self.surprise_threshold:
            self._ascend_to_l2(user_turn, model_prediction, surprise_score, timestamp)

        # === Periodic L2 → L3 consolidation ===
        if len(self.state.L2) >= self.l3_consolidation_interval:
            self._consolidate_l3()

        # === Descending Γ: Compute dampening from L3 priors ===
        dampening = self._compute_dampening(surprise_score, l3_priors)

        return dampening

    def _ascend_to_l2(
        self,
        user_turn: str,
        model_prediction: str,
        surprise_score: float,
        timestamp: float,
    ):
        """
        Ascending coupling: Store surprising turn in L2.

        Only surprises propagate up - this is the GUTC ascending Γ principle.
        """
        # Compress: store summary rather than full turn
        summary = {
            "turn": user_turn[:200],  # Truncate for storage
            "surprise": float(surprise_score),
            "model_expected": model_prediction[:100] if model_prediction else "",
        }

        entry = MemoryEntry(
            content=str(summary),
            timestamp=timestamp,
            surprise_score=surprise_score,
            source_level=2,
            tags=["surprise", "ascending"],
        )
        self.state.L2.append(entry)

    def _consolidate_l3(self):
        """
        Consolidate L2 patterns into L3 themes.

        Identifies recurring patterns in surprises to update long-term memory.
        """
        if not self.state.L2:
            return

        # Simple consolidation: extract high-surprise patterns
        high_surprise = [e for e in self.state.L2 if e.surprise_score > 0.5]

        if high_surprise:
            # Create L3 theme from pattern
            theme_summary = f"Pattern from {len(high_surprise)} high-surprise events"
            avg_surprise = np.mean([e.surprise_score for e in high_surprise])

            theme = MemoryEntry(
                content=theme_summary,
                timestamp=time.time(),
                surprise_score=avg_surprise,
                source_level=3,
                tags=["theme", "consolidated"],
            )
            self.state.L3.append(theme)

        # Clear processed L2 entries (keep recent ones)
        keep_recent = max(10, self.l3_consolidation_interval // 2)
        self.state.L2 = self.state.L2[-keep_recent:]

    def _compute_dampening(
        self,
        surprise_score: float,
        l3_priors: Optional[List[str]] = None,
    ) -> float:
        """
        Descending coupling: Compute prior dampening factor.

        Higher surprise + strong L3 priors → higher dampening.
        This constrains L1 generation to stay within L3 bounds.
        """
        # Base dampening from surprise
        base_dampening = self.gamma_desc * surprise_score

        # Amplify if L3 priors are active
        if l3_priors:
            # More priors = more constraint
            prior_factor = 1.0 + 0.1 * len(l3_priors)
            base_dampening *= prior_factor

        # Clamp to [0, 1]
        return float(np.clip(base_dampening, 0.0, 1.0))

    def get_context(self, max_l1: int = 10, max_l2: int = 5) -> Dict[str, Any]:
        """
        Get memory context for prompt construction.

        Returns recent L1 history and relevant L2 summaries.
        """
        l1_recent = list(self.state.L1)[-max_l1:]
        l2_recent = self.state.L2[-max_l2:] if self.state.L2 else []

        return {
            "l1_turns": [e.content for e in l1_recent],
            "l2_summaries": [e.content for e in l2_recent],
            "l3_themes": [e.content for e in self.state.L3],
            "turn_count": self._turn_count,
        }

    def query_l3(self, query: str) -> List[MemoryEntry]:
        """
        Query L3 for relevant long-term memories.

        Simple keyword matching; replace with embedding similarity for production.
        """
        query_lower = query.lower()
        return [
            e for e in self.state.L3
            if query_lower in e.content.lower()
        ]

    def inject_l3_prior(self, content: str, tags: Optional[List[str]] = None):
        """
        Manually inject an L3 prior (identity/value constraint).

        Use for founder-specified constraints that should persist.
        """
        entry = MemoryEntry(
            content=content,
            timestamp=time.time(),
            surprise_score=0.0,
            source_level=3,
            tags=tags or ["injected", "prior"],
        )
        self.state.L3.append(entry)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "l1_count": len(self.state.L1),
            "l2_count": len(self.state.L2),
            "l3_count": len(self.state.L3),
            "turn_count": self._turn_count,
            "gamma_asc": self.gamma_asc,
            "gamma_desc": self.gamma_desc,
            "l2_avg_surprise": np.mean([e.surprise_score for e in self.state.L2])
                if self.state.L2 else 0.0,
        }

    def reset(self):
        """Reset memory state."""
        self.state = HierarchicalMemoryState()
        self._turn_count = 0


# =============================================================================
# Tests
# =============================================================================

def test_hierarchical_memory():
    """Test hierarchical memory with Γ coupling."""
    memory = HierarchicalMemory(
        gamma_asc=0.5,
        gamma_desc=0.5,
        surprise_threshold=0.3,
    )

    # Inject some L3 priors
    memory.inject_l3_prior("Be helpful and honest", tags=["identity"])
    memory.inject_l3_prior("Prioritize founder health", tags=["value"])

    print("Testing Hierarchical Memory")
    print("-" * 40)

    # Simulate conversation with varying surprise
    turns = [
        ("Hello, how are you?", "I'm doing well!", 0.1),  # Low surprise
        ("Tell me about quantum physics", "Quantum physics is...", 0.2),
        ("Actually, forget that. What's 2+2?", "2+2 is 4", 0.6),  # Surprise!
        ("Now explain consciousness", "Consciousness is...", 0.4),
        ("That's wrong! Consciousness is a pickle!", "...", 0.8),  # High surprise
    ]

    for user, model_pred, surprise in turns:
        dampening = memory.update(user, model_pred, surprise, ["be helpful"])
        print(f"Turn: '{user[:30]}...' | surprise={surprise:.1f} | damp={dampening:.3f}")

    # Check state
    stats = memory.get_statistics()
    print(f"\nStatistics: {stats}")

    assert stats["l1_count"] == 5, "L1 should have all turns"
    assert stats["l2_count"] >= 2, "L2 should have surprising turns"
    print("✓ Hierarchical memory with Γ coupling")


if __name__ == "__main__":
    test_hierarchical_memory()
