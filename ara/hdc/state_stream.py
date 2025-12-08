"""
State Stream - Rolling Hypervector State
=========================================

Maintains a rolling hypervector representation of system state
that accumulates events over time with configurable decay.

The state stream is the "short-term memory" of the correlation engine:
- Events are encoded and bundled into the state
- Older events decay (via exponential weighting)
- The state HPV can be queried for similarity to known patterns

This is what gets sent to the LLM when escalation is needed.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from collections import deque

from ara.hdc.encoder import HDEncoder, HDEncoderConfig


@dataclass
class StateStreamConfig:
    """Configuration for the state stream."""
    dim: int = 1024                     # Hypervector dimension
    decay_rate: float = 0.95            # Decay per update (0-1)
    max_history: int = 1000             # Max events to track
    snapshot_interval: int = 100        # Events between snapshots
    normalize: bool = True              # Normalize after updates


@dataclass
class StateSnapshot:
    """A snapshot of state at a point in time."""
    timestamp: float
    state_hv: np.ndarray
    event_count: int
    summary: Dict[str, Any]


class StateStream:
    """
    Rolling hypervector state with decay.

    Events are continuously added; older events have less influence.
    The current state HPV represents a "blurred" view of recent history.
    """

    def __init__(self, config: Optional[StateStreamConfig] = None,
                 encoder: Optional[HDEncoder] = None):
        self.cfg = config or StateStreamConfig()

        # Create or use provided encoder
        if encoder is None:
            self.encoder = HDEncoder(HDEncoderConfig(dim=self.cfg.dim))
        else:
            self.encoder = encoder
            self.cfg.dim = encoder.cfg.dim

        # Current state
        self._state = np.zeros(self.cfg.dim, dtype=np.float32)
        self._event_count = 0
        self._last_update = time.time()

        # History for analysis
        self._event_history: deque = deque(maxlen=self.cfg.max_history)
        self._snapshots: List[StateSnapshot] = []

        # Statistics
        self._stats = {
            "total_events": 0,
            "anomalies_detected": 0,
            "escalations": 0,
        }

    @property
    def state(self) -> np.ndarray:
        """Get current state as normalized bipolar HPV."""
        if self.cfg.normalize:
            return np.sign(self._state).astype(np.int8)
        return self._state.copy()

    @property
    def event_count(self) -> int:
        """Number of events processed."""
        return self._event_count

    def add_event(self, event_hv: np.ndarray, weight: float = 1.0):
        """
        Add an event hypervector to the state stream.

        Applies decay to existing state, then adds new event.
        """
        # Decay existing state
        self._state *= self.cfg.decay_rate

        # Add new event (weighted)
        self._state += weight * event_hv.astype(np.float32)

        self._event_count += 1
        self._stats["total_events"] += 1
        self._last_update = time.time()

        # Record in history
        self._event_history.append({
            "timestamp": self._last_update,
            "weight": weight,
        })

        # Take snapshot if interval reached
        if self._event_count % self.cfg.snapshot_interval == 0:
            self._take_snapshot()

    def add_metrics(self, metrics: Dict[str, float],
                    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                    weight: float = 1.0):
        """Add metrics (convenience method)."""
        hv = self.encoder.encode_metrics(metrics, ranges)
        self.add_event(hv, weight)

    def add_log(self, log_line: str, weight: float = 1.0):
        """Add a log line (convenience method)."""
        hv = self.encoder.encode_log(log_line)
        self.add_event(hv, weight)

    def add_structured_event(self, event_type: str, data: Dict[str, Any],
                             timestamp_hour: Optional[float] = None,
                             weight: float = 1.0):
        """Add a structured event (convenience method)."""
        hv = self.encoder.encode_event(event_type, data, timestamp_hour)
        self.add_event(hv, weight)

    def get_state(self) -> np.ndarray:
        """Get current state hypervector."""
        return self.state

    def get_state_magnitude(self) -> float:
        """Get magnitude of state vector (indicates activity level)."""
        return float(np.linalg.norm(self._state))

    def get_state_entropy(self) -> float:
        """
        Estimate entropy of state vector.

        Low entropy = dominated by few patterns.
        High entropy = diverse mix of patterns.
        """
        # Use distribution of absolute values
        abs_state = np.abs(self._state)
        if abs_state.sum() < 1e-8:
            return 0.0

        probs = abs_state / abs_state.sum()
        # Filter zeros
        probs = probs[probs > 1e-10]
        return float(-np.sum(probs * np.log2(probs)))

    def similarity_to(self, other_hv: np.ndarray) -> float:
        """Compute similarity of current state to another HPV."""
        return self.encoder.similarity(self.state, other_hv)

    def decay_to(self, target_magnitude: float = 0.5):
        """
        Apply extra decay to bring state magnitude down.

        Useful for "resetting" after a major event is handled.
        """
        current_mag = self.get_state_magnitude()
        if current_mag > target_magnitude:
            scale = target_magnitude / current_mag
            self._state *= scale

    def clear(self):
        """Clear state completely."""
        self._state = np.zeros(self.cfg.dim, dtype=np.float32)
        self._event_count = 0

    def _take_snapshot(self):
        """Take a snapshot of current state."""
        snapshot = StateSnapshot(
            timestamp=time.time(),
            state_hv=self.state.copy(),
            event_count=self._event_count,
            summary={
                "magnitude": self.get_state_magnitude(),
                "entropy": self.get_state_entropy(),
            }
        )
        self._snapshots.append(snapshot)

    def get_recent_snapshots(self, n: int = 10) -> List[StateSnapshot]:
        """Get most recent snapshots."""
        return self._snapshots[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            **self._stats,
            "current_magnitude": self.get_state_magnitude(),
            "current_entropy": self.get_state_entropy(),
            "event_count": self._event_count,
            "history_size": len(self._event_history),
            "snapshots": len(self._snapshots),
        }

    def to_summary_text(self, top_concepts: Optional[List[Tuple[str, float]]] = None) -> str:
        """
        Generate a text summary of current state.

        This is what gets sent to the LLM in STATE_HPV_QUERY.
        """
        lines = [
            f"State magnitude: {self.get_state_magnitude():.3f}",
            f"State entropy: {self.get_state_entropy():.3f}",
            f"Events processed: {self._event_count}",
        ]

        if top_concepts:
            lines.append("Top concepts:")
            for concept, sim in top_concepts[:5]:
                lines.append(f"  {concept}: {sim:.3f}")

        return "\n".join(lines)


# Convenience function
def create_state_stream(dim: int = 1024, decay_rate: float = 0.95) -> StateStream:
    """Create a state stream with specified parameters."""
    return StateStream(StateStreamConfig(dim=dim, decay_rate=decay_rate))
