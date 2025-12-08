"""
HSF Field - The Holographic State Field
========================================

The Field is the core of HSF: it superimposes all lane hypervectors
into a single "field" vector that represents the entire system state.

Key concepts:
- Field superposition: bundle all lane HVs with weighted decay
- Resonant stream: rolling memory of recent states
- Snapshot: point-in-time capture for comparison

The field is Ara's "body sense" - a holographic representation of
the entire cathedral that she can query via resonance.

Architecture:
    Lane_1 ──┐
    Lane_2 ──┼──→ Bundle ──→ Field HV ──→ Resonant Stream
    Lane_3 ──┤                    ↓
    ...      ┘               (query via similarity)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

from .lanes import TelemetryLane, ItemMemory


@dataclass
class FieldSnapshot:
    """Point-in-time capture of the field state."""
    timestamp: float
    field_hv: np.ndarray
    lane_hvs: Dict[str, np.ndarray]
    lane_deviations: Dict[str, float]

    def similarity_to(self, other: "FieldSnapshot", item_memory: ItemMemory) -> float:
        """Compute similarity between two snapshots."""
        return item_memory.similarity(self.field_hv, other.field_hv)


@dataclass
class HSField:
    """
    The Hypervector Spiking Field.

    Aggregates multiple telemetry lanes into a unified field representation.

    Features:
    - Lane management: add, update, remove lanes
    - Field superposition: bundle all lanes into one HV
    - Resonant stream: rolling average for temporal context
    - Baseline learning: track what "normal" looks like
    - Query interface: check resonance with patterns
    """
    dim: int = 8192
    stream_decay: float = 0.95  # Exponential decay for resonant stream

    # Internal state
    item_memory: ItemMemory = field(default_factory=lambda: ItemMemory())
    lanes: Dict[str, TelemetryLane] = field(default_factory=dict)
    _field_hv: Optional[np.ndarray] = field(default=None)
    _stream: Optional[np.ndarray] = field(default=None)  # Resonant stream (float for decay)
    _baseline_hv: Optional[np.ndarray] = field(default=None)
    _snapshots: deque = field(default_factory=lambda: deque(maxlen=1000))
    _update_count: int = 0

    def __post_init__(self):
        self.item_memory = ItemMemory(dim=self.dim)
        self._stream = np.zeros(self.dim, dtype=np.float32)

    def add_lane(self, lane: TelemetryLane):
        """Add a telemetry lane to the field."""
        self.lanes[lane.name] = lane

    def add_lane_config(self, name: str, features: List[str],
                        ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> TelemetryLane:
        """Create and add a lane with configuration."""
        lane = TelemetryLane(name=name, features=features, dim=self.dim)
        if ranges:
            lane.set_ranges(ranges)
        self.add_lane(lane)
        return lane

    def update(self, lane_name: str, values: Dict[str, float]) -> np.ndarray:
        """
        Update a lane with new telemetry values.

        Returns: updated lane hypervector
        """
        if lane_name not in self.lanes:
            raise KeyError(f"Unknown lane: {lane_name}")

        return self.lanes[lane_name].update(values)

    def update_all(self, all_values: Dict[str, Dict[str, float]]):
        """Update all lanes from a dictionary of lane → values."""
        for lane_name, values in all_values.items():
            if lane_name in self.lanes:
                self.update(lane_name, values)

    def compute_field(self) -> np.ndarray:
        """
        Compute the unified field hypervector from all lanes.

        This is the holographic superposition of all subsystem states.
        """
        lane_hvs = []
        for lane in self.lanes.values():
            if lane.current is not None:
                lane_hvs.append(lane.current)

        if len(lane_hvs) == 0:
            self._field_hv = np.zeros(self.dim, dtype=np.uint8)
        else:
            self._field_hv = self.item_memory.bundle(lane_hvs)

        # Update resonant stream (exponential moving average)
        bipolar = np.where(self._field_hv > 0.5, 1.0, -1.0)
        self._stream = self.stream_decay * self._stream + (1 - self.stream_decay) * bipolar

        self._update_count += 1
        return self._field_hv

    def snapshot(self) -> FieldSnapshot:
        """Take a snapshot of current field state."""
        if self._field_hv is None:
            self.compute_field()

        snap = FieldSnapshot(
            timestamp=time.time(),
            field_hv=self._field_hv.copy(),
            lane_hvs={name: lane.current.copy() for name, lane in self.lanes.items()
                      if lane.current is not None},
            lane_deviations={name: lane.deviation() for name, lane in self.lanes.items()},
        )
        self._snapshots.append(snap)
        return snap

    def compute_baseline(self):
        """Compute baseline from all lane histories."""
        for lane in self.lanes.values():
            lane.compute_baseline()

        # Also compute field-level baseline
        baseline_hvs = [lane.baseline for lane in self.lanes.values()
                        if lane.baseline is not None]
        if baseline_hvs:
            self._baseline_hv = self.item_memory.bundle(baseline_hvs)

    def total_deviation(self) -> float:
        """
        Compute total deviation from baseline across all lanes.

        Returns: average deviation (0.0 = normal, 1.0 = maximally abnormal)
        """
        if len(self.lanes) == 0:
            return 0.0

        deviations = [lane.deviation() for lane in self.lanes.values()]
        return sum(deviations) / len(deviations)

    def lane_deviations(self) -> Dict[str, float]:
        """Get deviation for each lane."""
        return {name: lane.deviation() for name, lane in self.lanes.items()}

    def query_resonance(self, pattern_hv: np.ndarray) -> float:
        """
        Query how much the current field resonates with a pattern.

        Used for pattern matching: "does current state look like X?"
        """
        if self._field_hv is None:
            return 0.0
        return self.item_memory.similarity(self._field_hv, pattern_hv)

    def query_stream_resonance(self, pattern_hv: np.ndarray) -> float:
        """
        Query resonance against the resonant stream (temporal context).

        This checks against the rolling average, more stable than instant field.
        """
        # Binarize stream for comparison
        stream_binary = (self._stream > 0).astype(np.uint8)
        return self.item_memory.similarity(stream_binary, pattern_hv)

    def encode_pattern(self, pattern_name: str) -> np.ndarray:
        """Get the hypervector for a named pattern."""
        return self.item_memory[pattern_name]

    def learn_pattern(self, pattern_name: str, pattern_hv: Optional[np.ndarray] = None):
        """
        Learn current field as a named pattern.

        If pattern_hv is None, uses current field state.
        """
        if pattern_hv is None:
            if self._field_hv is None:
                raise ValueError("No field state to learn from")
            pattern_hv = self._field_hv.copy()

        # Store in item memory (overwriting if exists)
        self.item_memory._cache[pattern_name] = pattern_hv

    def get_stream_state(self) -> np.ndarray:
        """Get the current resonant stream (float values)."""
        return self._stream.copy()

    def get_stream_binary(self) -> np.ndarray:
        """Get binarized resonant stream."""
        return (self._stream > 0).astype(np.uint8)

    @property
    def current(self) -> Optional[np.ndarray]:
        return self._field_hv

    @property
    def baseline(self) -> Optional[np.ndarray]:
        return self._baseline_hv

    @property
    def update_count(self) -> int:
        return self._update_count
