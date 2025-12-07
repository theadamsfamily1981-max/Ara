"""
Resonant Stream - Holographic Memory via Rolling Superposition
==============================================================

The marriage of 1-Bit SNN and Hyperdimensional Computing (HDC).

KEY INSIGHT: Instead of storing memories in a list (KV cache),
we BUNDLE them into a running stream using superposition:

    Context_t = Context_{t-1} + New_Thought

Old memories don't disappear; they just get fainter (holographic).
This gives Ara effectively INFINITE short-term memory.

To recall a memory, we don't "look it up" - we QUERY the stream
with a related vector. If the memory exists, the stream RESONATES
(constructive interference).

The mathematics:
    - Thoughts are binary hypervectors {-1, +1}^D where D = 8192
    - Bundling: element-wise addition with saturation
    - Querying: dot product gives "resonance" (signal strength)
    - Decay: multiply by 0.9 to implement forgetting curve

Usage:
    from ara.cognition.stream_binding import ResonantStream

    stream = ResonantStream(dim=8192)

    # Add thoughts over time
    stream.add_thought(concept_vector)
    stream.add_thought(another_concept)

    # Later: does the stream contain this concept?
    resonance = stream.query(query_vector)
    if resonance > 0.5:
        print("Memory found!")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Record of a thought added to the stream."""
    vector_hash: str
    label: str
    timestamp: float
    resonance_at_add: float


class ResonantStream:
    """
    Rolling superposition memory - the "River of Consciousness."

    Each thought is bundled into a running stream vector.
    The stream is holographic: all memories coexist as
    interference patterns in the same vector.

    Properties:
        - Capacity: Infinite (but memories fade)
        - Recall: O(D) where D = dimension
        - Memory: O(D) regardless of thought count
        - Forgetting: Natural decay via forgetting curve
    """

    def __init__(
        self,
        dim: int = 8192,
        decay_rate: float = 0.9,
        saturation_limit: int = 127,
    ):
        """
        Initialize the resonant stream.

        Args:
            dim: Hypervector dimension
            decay_rate: Multiplicative decay per thought (0.9 = slow forget)
            saturation_limit: Max magnitude per dimension
        """
        self.dim = dim
        self.decay_rate = decay_rate
        self.saturation_limit = saturation_limit

        # The Stream: accumulated thoughts
        # Each dimension is a signed counter
        self.stream = np.zeros(dim, dtype=np.float32)

        # Metadata
        self.thought_count = 0
        self.event_log: List[StreamEvent] = []

        log.info(f"ResonantStream initialized: dim={dim}, decay={decay_rate}")

    def add_thought(
        self,
        vector: np.ndarray,
        label: str = "",
        apply_decay: bool = True,
    ) -> float:
        """
        Bundle a new thought into the stream.

        The thought is superimposed on existing memories.
        Old thoughts decay slightly (forgetting curve).

        Args:
            vector: Binary hypervector {0,1}^D or bipolar {-1,+1}^D
            label: Human-readable label for logging
            apply_decay: Whether to decay existing memories

        Returns:
            Resonance of this thought with existing stream (self-similarity check)
        """
        # Convert to bipolar if needed
        if vector.min() >= 0:
            # Assume {0, 1} encoding, convert to {-1, +1}
            bipolar = np.where(vector > 0.5, 1.0, -1.0).astype(np.float32)
        else:
            bipolar = vector.astype(np.float32)

        # Check resonance before adding (does stream already contain this?)
        pre_resonance = self.query(bipolar)

        # 1. Decay existing memories (forgetting curve)
        if apply_decay:
            self.stream *= self.decay_rate

        # 2. Superimpose new thought
        self.stream += bipolar

        # 3. Clamp to prevent saturation
        self.stream = np.clip(
            self.stream,
            -self.saturation_limit,
            self.saturation_limit
        )

        # Log event
        self.thought_count += 1
        event = StreamEvent(
            vector_hash=self._hash_vector(bipolar),
            label=label,
            timestamp=time.time(),
            resonance_at_add=pre_resonance,
        )
        self.event_log.append(event)

        log.debug(
            f"ResonantStream: Added thought {self.thought_count} "
            f"'{label}' (pre-resonance: {pre_resonance:.3f})"
        )

        return pre_resonance

    def query(self, vector: np.ndarray) -> float:
        """
        Query the stream for a concept.

        Returns "resonance" - how strongly the stream responds
        to this query vector. High resonance = memory exists.

        Args:
            vector: Query hypervector

        Returns:
            Resonance score in [-1, 1] (normalized dot product)
        """
        # Convert to bipolar if needed
        if vector.min() >= 0:
            bipolar = np.where(vector > 0.5, 1.0, -1.0).astype(np.float32)
        else:
            bipolar = vector.astype(np.float32)

        # Dot product: resonance
        raw_resonance = np.dot(self.stream, bipolar)

        # Normalize by dimension and saturation
        max_resonance = self.dim * self.saturation_limit
        normalized = raw_resonance / max_resonance

        return float(normalized)

    def query_topk(
        self,
        queries: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Find top-k resonating queries from a set.

        Args:
            queries: Array of shape (N, dim) - candidate vectors
            k: Number of top matches to return

        Returns:
            List of (index, resonance) tuples, sorted by resonance
        """
        resonances = []
        for i, q in enumerate(queries):
            r = self.query(q)
            resonances.append((i, r))

        # Sort by resonance (descending)
        resonances.sort(key=lambda x: -x[1])

        return resonances[:k]

    def apply_decay(self, factor: Optional[float] = None) -> None:
        """
        Manually apply decay to the stream.

        Args:
            factor: Decay factor (default: self.decay_rate)
        """
        factor = factor or self.decay_rate
        self.stream *= factor

    def reset(self) -> None:
        """Clear the stream (forget everything)."""
        self.stream.fill(0)
        self.thought_count = 0
        self.event_log.clear()
        log.info("ResonantStream: Reset (all memories cleared)")

    def get_activity(self) -> float:
        """
        Get current stream activity level.

        High activity = many strong memories.
        Low activity = stream is mostly decayed.

        Returns:
            Activity level in [0, 1]
        """
        energy = np.abs(self.stream).sum()
        max_energy = self.dim * self.saturation_limit
        return float(energy / max_energy)

    def get_dominant_dimensions(self, k: int = 10) -> List[Tuple[int, float]]:
        """
        Get the k dimensions with highest magnitude.

        These represent the "loudest" features in the stream.

        Returns:
            List of (dimension_index, value) tuples
        """
        abs_stream = np.abs(self.stream)
        top_indices = np.argpartition(abs_stream, -k)[-k:]
        top_indices = top_indices[np.argsort(-abs_stream[top_indices])]

        return [(int(i), float(self.stream[i])) for i in top_indices]

    def snapshot(self) -> np.ndarray:
        """Get a copy of the current stream state."""
        return self.stream.copy()

    def load_snapshot(self, snapshot: np.ndarray) -> None:
        """Restore stream from a snapshot."""
        if snapshot.shape != (self.dim,):
            raise ValueError(f"Snapshot shape {snapshot.shape} != expected ({self.dim},)")
        self.stream = snapshot.astype(np.float32)

    def _hash_vector(self, vector: np.ndarray) -> str:
        """Create a short hash of a vector for logging."""
        # Use first 8 bytes of binary representation
        binary = (vector > 0).astype(np.uint8)[:64]
        packed = np.packbits(binary)
        return packed.tobytes()[:8].hex()


class ContextualStream(ResonantStream):
    """
    Extended ResonantStream with context tracking.

    Maintains multiple "channels" for different context types:
        - Semantic: Concept meanings
        - Temporal: Time-ordered events
        - Emotional: Mood/affect states
    """

    def __init__(
        self,
        dim: int = 8192,
        decay_rate: float = 0.9,
        n_channels: int = 3,
    ):
        super().__init__(dim=dim, decay_rate=decay_rate)

        self.n_channels = n_channels
        self.channel_names = ["semantic", "temporal", "emotional"]

        # Multiple streams for different context types
        self.channels: Dict[str, np.ndarray] = {
            name: np.zeros(dim, dtype=np.float32)
            for name in self.channel_names[:n_channels]
        }

    def add_to_channel(
        self,
        channel: str,
        vector: np.ndarray,
        label: str = "",
    ) -> float:
        """Add thought to a specific channel."""
        if channel not in self.channels:
            raise ValueError(f"Unknown channel: {channel}")

        # Convert to bipolar
        if vector.min() >= 0:
            bipolar = np.where(vector > 0.5, 1.0, -1.0).astype(np.float32)
        else:
            bipolar = vector.astype(np.float32)

        # Decay and add
        self.channels[channel] *= self.decay_rate
        self.channels[channel] += bipolar
        self.channels[channel] = np.clip(
            self.channels[channel],
            -self.saturation_limit,
            self.saturation_limit
        )

        # Also add to main stream
        return self.add_thought(vector, label=label, apply_decay=False)

    def query_channel(self, channel: str, vector: np.ndarray) -> float:
        """Query a specific channel."""
        if channel not in self.channels:
            raise ValueError(f"Unknown channel: {channel}")

        if vector.min() >= 0:
            bipolar = np.where(vector > 0.5, 1.0, -1.0).astype(np.float32)
        else:
            bipolar = vector.astype(np.float32)

        raw = np.dot(self.channels[channel], bipolar)
        return float(raw / (self.dim * self.saturation_limit))

    def get_channel_activity(self) -> Dict[str, float]:
        """Get activity level of each channel."""
        return {
            name: float(np.abs(ch).sum() / (self.dim * self.saturation_limit))
            for name, ch in self.channels.items()
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_global_stream: Optional[ResonantStream] = None


def get_resonant_stream(dim: int = 8192) -> ResonantStream:
    """Get or create the global resonant stream."""
    global _global_stream
    if _global_stream is None:
        _global_stream = ResonantStream(dim=dim)
    return _global_stream


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StreamEvent',
    'ResonantStream',
    'ContextualStream',
    'get_resonant_stream',
]
