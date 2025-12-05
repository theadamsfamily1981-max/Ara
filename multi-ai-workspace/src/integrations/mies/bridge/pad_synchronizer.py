"""PAD Synchronizer - Single source of truth for emotional state.

When multiple systems compute PAD independently, we need a way to:
1. Reconcile different PAD values
2. Decide which source is authoritative
3. Propagate changes bidirectionally
4. Maintain history for drift detection

This synchronizer ensures Ara has ONE coherent emotional state,
not multiple conflicting views.
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from enum import Enum, auto

from ..affect.pad_engine import PADVector, EmotionalQuadrant

logger = logging.getLogger(__name__)


class PADSource(Enum):
    """Source of PAD computation."""
    MIES_CATHEDRAL = auto()      # MIES PADEngine (hardware telemetry)
    ARA_INTEROCEPTION = auto()   # ara/interoception SNN
    KERNEL_BRIDGE = auto()       # Kernel-computed PAD (from ara_guardian.ko)
    BANOS_AFFECTIVE = auto()     # BANOS kernel affective layer (eBPF PAD)
    PULSE_ESTIMATION = auto()    # External affect estimation (Pulse)
    USER_OVERRIDE = auto()       # Direct user setting
    FUSED = auto()               # Weighted fusion of multiple sources


class PADConflictResolution(Enum):
    """Strategy for resolving conflicting PAD values."""
    PRIORITY = auto()        # Use highest-priority source
    WEIGHTED_AVERAGE = auto()  # Weight by source confidence
    MOST_RECENT = auto()     # Use most recently updated
    MOST_EXTREME = auto()    # Use source with strongest signal
    CONSENSUS = auto()       # Require agreement within threshold


@dataclass
class PADReading:
    """A single PAD reading from a source."""
    source: PADSource
    pad: PADVector
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Seconds since this reading."""
        return time.time() - self.timestamp

    def is_stale(self, max_age: float = 5.0) -> bool:
        """Check if reading is too old."""
        return self.age > max_age


@dataclass
class PADSyncConfig:
    """Configuration for PAD synchronization."""
    # Resolution strategy
    resolution: PADConflictResolution = PADConflictResolution.WEIGHTED_AVERAGE

    # Source priorities (higher = more authoritative)
    priorities: Dict[PADSource, float] = field(default_factory=lambda: {
        PADSource.USER_OVERRIDE: 1.0,
        PADSource.ARA_INTEROCEPTION: 0.9,  # SNN is most "true" interoception
        PADSource.BANOS_AFFECTIVE: 0.88,   # BANOS eBPF is hardware-grounded
        PADSource.KERNEL_BRIDGE: 0.8,
        PADSource.MIES_CATHEDRAL: 0.7,
        PADSource.FUSED: 0.6,
        PADSource.PULSE_ESTIMATION: 0.5,
    })

    # Staleness threshold (seconds)
    max_reading_age: float = 5.0

    # Conflict threshold - if sources disagree by more than this, log warning
    conflict_threshold: float = 0.3

    # History size for drift detection
    history_size: int = 100

    # Drift detection window (seconds)
    drift_window: float = 60.0

    # Maximum drift before alerting
    max_drift: float = 0.5


@dataclass
class PADSyncState:
    """Current state of the synchronizer."""
    canonical_pad: PADVector
    source: PADSource
    confidence: float
    last_sync: float
    sources_in_conflict: bool = False
    drift_detected: bool = False


class PADSynchronizer:
    """Synchronizes PAD state across multiple sources.

    Maintains a canonical PAD value that all systems can reference,
    resolving conflicts when sources disagree.

    Example usage:
        sync = PADSynchronizer()

        # Systems report their PAD
        sync.report(PADSource.MIES_CATHEDRAL, mies_pad, confidence=0.8)
        sync.report(PADSource.ARA_INTEROCEPTION, snn_pad, confidence=0.95)

        # Get the canonical value
        canonical = sync.get_canonical_pad()

        # Subscribe to changes
        sync.on_change(lambda pad: update_llm_prompt(pad))
    """

    def __init__(
        self,
        config: Optional[PADSyncConfig] = None,
        initial_pad: Optional[PADVector] = None,
    ):
        """Initialize the synchronizer.

        Args:
            config: Synchronization configuration
            initial_pad: Initial canonical PAD value
        """
        self.config = config or PADSyncConfig()

        # Current readings from each source
        self._readings: Dict[PADSource, PADReading] = {}

        # Canonical state
        self._canonical = PADSyncState(
            canonical_pad=initial_pad or PADVector(0.0, 0.0, 0.0),
            source=PADSource.FUSED,
            confidence=0.5,
            last_sync=time.time(),
        )

        # History for drift detection
        self._history: List[PADReading] = []

        # Change listeners
        self._listeners: List[Callable[[PADVector], None]] = []

        # Statistics
        self._sync_count: int = 0
        self._conflict_count: int = 0

    def report(
        self,
        source: PADSource,
        pad: PADVector,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Report a PAD reading from a source.

        Args:
            source: Which system is reporting
            pad: The PAD value
            confidence: How confident the source is (0-1)
            metadata: Additional context
        """
        reading = PADReading(
            source=source,
            pad=pad,
            timestamp=time.time(),
            confidence=confidence,
            metadata=metadata or {},
        )

        self._readings[source] = reading
        self._history.append(reading)

        # Trim history
        if len(self._history) > self.config.history_size:
            self._history = self._history[-self.config.history_size:]

        # Synchronize
        self._synchronize()

    def get_canonical_pad(self) -> PADVector:
        """Get the current canonical PAD value."""
        return self._canonical.canonical_pad

    def get_state(self) -> PADSyncState:
        """Get full synchronization state."""
        return self._canonical

    def get_source_pad(self, source: PADSource) -> Optional[PADVector]:
        """Get PAD from a specific source, if available."""
        reading = self._readings.get(source)
        if reading is not None and not reading.is_stale(self.config.max_reading_age):
            return reading.pad
        return None

    def on_change(self, callback: Callable[[PADVector], None]):
        """Register a callback for PAD changes."""
        self._listeners.append(callback)

    def _synchronize(self):
        """Perform synchronization across sources."""
        self._sync_count += 1

        # Get fresh readings
        fresh_readings = [
            r for r in self._readings.values()
            if not r.is_stale(self.config.max_reading_age)
        ]

        if not fresh_readings:
            return  # No fresh data

        # Check for conflicts
        self._check_conflicts(fresh_readings)

        # Resolve to canonical value
        new_pad, source, confidence = self._resolve(fresh_readings)

        # Check for significant change
        old_pad = self._canonical.canonical_pad
        distance = old_pad.distance_to(new_pad)

        if distance > 0.01:  # Meaningful change
            self._canonical = PADSyncState(
                canonical_pad=new_pad,
                source=source,
                confidence=confidence,
                last_sync=time.time(),
                sources_in_conflict=self._canonical.sources_in_conflict,
                drift_detected=self._check_drift(),
            )

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(new_pad)
                except Exception as e:
                    logger.error(f"PAD change listener error: {e}")

    def _resolve(
        self,
        readings: List[PADReading],
    ) -> tuple[PADVector, PADSource, float]:
        """Resolve readings to canonical value."""
        cfg = self.config

        if cfg.resolution == PADConflictResolution.PRIORITY:
            # Use highest priority source
            readings.sort(
                key=lambda r: cfg.priorities.get(r.source, 0.0),
                reverse=True,
            )
            best = readings[0]
            return best.pad, best.source, best.confidence

        elif cfg.resolution == PADConflictResolution.MOST_RECENT:
            # Use most recent
            readings.sort(key=lambda r: r.timestamp, reverse=True)
            best = readings[0]
            return best.pad, best.source, best.confidence

        elif cfg.resolution == PADConflictResolution.MOST_EXTREME:
            # Use source with strongest signal (furthest from neutral)
            readings.sort(
                key=lambda r: r.pad.intensity,
                reverse=True,
            )
            best = readings[0]
            return best.pad, best.source, best.confidence

        elif cfg.resolution == PADConflictResolution.WEIGHTED_AVERAGE:
            # Weighted average by priority and confidence
            total_weight = 0.0
            p_sum, a_sum, d_sum = 0.0, 0.0, 0.0

            for r in readings:
                weight = cfg.priorities.get(r.source, 0.5) * r.confidence
                total_weight += weight
                p_sum += r.pad.pleasure * weight
                a_sum += r.pad.arousal * weight
                d_sum += r.pad.dominance * weight

            if total_weight > 0:
                fused_pad = PADVector(
                    pleasure=p_sum / total_weight,
                    arousal=a_sum / total_weight,
                    dominance=d_sum / total_weight,
                )
                avg_confidence = sum(r.confidence for r in readings) / len(readings)
                return fused_pad, PADSource.FUSED, avg_confidence
            else:
                return readings[0].pad, readings[0].source, readings[0].confidence

        elif cfg.resolution == PADConflictResolution.CONSENSUS:
            # Require agreement - use average only if within threshold
            if len(readings) < 2:
                return readings[0].pad, readings[0].source, readings[0].confidence

            # Check pairwise agreement
            max_distance = 0.0
            for i, r1 in enumerate(readings):
                for r2 in readings[i+1:]:
                    dist = r1.pad.distance_to(r2.pad)
                    max_distance = max(max_distance, dist)

            if max_distance <= cfg.conflict_threshold:
                # Consensus reached - use weighted average
                return self._resolve_weighted(readings)
            else:
                # No consensus - use highest priority
                readings.sort(
                    key=lambda r: cfg.priorities.get(r.source, 0.0),
                    reverse=True,
                )
                best = readings[0]
                return best.pad, best.source, best.confidence * 0.7  # Reduce confidence

        # Default: first reading
        return readings[0].pad, readings[0].source, readings[0].confidence

    def _resolve_weighted(
        self,
        readings: List[PADReading],
    ) -> tuple[PADVector, PADSource, float]:
        """Weighted average resolution."""
        cfg = self.config
        total_weight = 0.0
        p_sum, a_sum, d_sum = 0.0, 0.0, 0.0

        for r in readings:
            weight = cfg.priorities.get(r.source, 0.5) * r.confidence
            total_weight += weight
            p_sum += r.pad.pleasure * weight
            a_sum += r.pad.arousal * weight
            d_sum += r.pad.dominance * weight

        fused_pad = PADVector(
            pleasure=p_sum / total_weight,
            arousal=a_sum / total_weight,
            dominance=d_sum / total_weight,
        )
        avg_confidence = sum(r.confidence for r in readings) / len(readings)
        return fused_pad, PADSource.FUSED, avg_confidence

    def _check_conflicts(self, readings: List[PADReading]):
        """Check if sources are in conflict."""
        if len(readings) < 2:
            self._canonical.sources_in_conflict = False
            return

        max_distance = 0.0
        for i, r1 in enumerate(readings):
            for r2 in readings[i+1:]:
                dist = r1.pad.distance_to(r2.pad)
                max_distance = max(max_distance, dist)

        if max_distance > self.config.conflict_threshold:
            self._canonical.sources_in_conflict = True
            self._conflict_count += 1
            logger.warning(
                f"PAD sources in conflict: max_distance={max_distance:.3f}, "
                f"sources={[r.source.name for r in readings]}"
            )
        else:
            self._canonical.sources_in_conflict = False

    def _check_drift(self) -> bool:
        """Check for emotional drift over time."""
        if len(self._history) < 10:
            return False

        now = time.time()
        window_start = now - self.config.drift_window

        # Get readings in window
        window_readings = [
            r for r in self._history
            if r.timestamp >= window_start
        ]

        if len(window_readings) < 5:
            return False

        # Check drift from first to last in window
        first = window_readings[0].pad
        last = window_readings[-1].pad
        drift = first.distance_to(last)

        if drift > self.config.max_drift:
            logger.info(
                f"Emotional drift detected: {drift:.3f} over {self.config.drift_window}s"
            )
            return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronizer statistics."""
        return {
            "sync_count": self._sync_count,
            "conflict_count": self._conflict_count,
            "active_sources": [
                s.name for s, r in self._readings.items()
                if not r.is_stale(self.config.max_reading_age)
            ],
            "canonical": {
                "pad": {
                    "pleasure": self._canonical.canonical_pad.pleasure,
                    "arousal": self._canonical.canonical_pad.arousal,
                    "dominance": self._canonical.canonical_pad.dominance,
                },
                "quadrant": self._canonical.canonical_pad.quadrant.name,
                "source": self._canonical.source.name,
                "confidence": self._canonical.confidence,
                "in_conflict": self._canonical.sources_in_conflict,
                "drift_detected": self._canonical.drift_detected,
            },
            "history_size": len(self._history),
            "listener_count": len(self._listeners),
        }


# === Factory ===

def create_pad_synchronizer(
    resolution: PADConflictResolution = PADConflictResolution.WEIGHTED_AVERAGE,
    conflict_threshold: float = 0.3,
) -> PADSynchronizer:
    """Create a PAD synchronizer.

    Args:
        resolution: How to resolve conflicting PAD values
        conflict_threshold: Distance threshold for conflict detection
    """
    config = PADSyncConfig(
        resolution=resolution,
        conflict_threshold=conflict_threshold,
    )
    return PADSynchronizer(config=config)
