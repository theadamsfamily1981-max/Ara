"""
HSF Zone Quantization
======================

Discretize the continuous hypervector field into actionable zones.

Each subsystem has reference hypervectors for each zone:
- GOOD: Normal operating state (learned from baseline)
- WARM: Slightly elevated, worth watching
- WEIRD: Significant deviation, reflex territory
- CRITICAL: Emergency, escalate immediately

Zone detection is just dot products + winner-take-all:
extremely cheap in 1-bit hardware.

The zones give us:
1. A discrete state for reflex table lookup
2. Hysteresis to prevent oscillation
3. Clear escalation thresholds
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum
from collections import deque


class Zone(IntEnum):
    """
    Operating zones for a subsystem.

    IntEnum so we can use as array indices and compare severity.
    """
    GOOD = 0      # Normal, homeostatic
    WARM = 1      # Elevated, monitoring
    WEIRD = 2     # Anomalous, reflex territory
    CRITICAL = 3  # Emergency, escalate


@dataclass
class ZoneThresholds:
    """Similarity thresholds for zone classification."""
    good_min: float = 0.7      # > this → GOOD
    warm_min: float = 0.4      # > this → WARM
    weird_min: float = 0.2     # > this → WEIRD
    # Below weird_min → CRITICAL


@dataclass
class ZoneState:
    """Current zone state for a subsystem."""
    zone: Zone
    confidence: float          # How strongly we're in this zone
    similarity_to_good: float  # Raw similarity to baseline
    time_in_zone: int          # Ticks spent in current zone
    previous_zone: Zone        # For transition detection


@dataclass
class ZoneQuantizer:
    """
    Quantizes a hypervector field into discrete zones.

    Uses reference hypervectors for each zone, learned from data
    or synthesized from domain knowledge.

    Features:
    - Hysteresis to prevent oscillation at boundaries
    - Confidence scoring
    - Transition detection
    """
    dim: int = 8192
    thresholds: ZoneThresholds = field(default_factory=ZoneThresholds)
    hysteresis: float = 0.05  # Must exceed threshold by this much to transition

    # Reference hypervectors for each zone (learned or set)
    _ref_good: Optional[np.ndarray] = None
    _ref_warm: Optional[np.ndarray] = None
    _ref_weird: Optional[np.ndarray] = None
    _ref_critical: Optional[np.ndarray] = None

    # Current state
    _current_zone: Zone = Zone.GOOD
    _time_in_zone: int = 0
    _previous_zone: Zone = Zone.GOOD

    def set_baseline(self, baseline_hv: np.ndarray):
        """
        Set the GOOD reference from a learned baseline.

        Other zones are derived via perturbation or learned separately.
        """
        self._ref_good = baseline_hv.copy()

        # Synthesize other references by progressive perturbation
        # WARM: ~15% bits flipped from GOOD
        rng = np.random.default_rng(42)
        flip_mask = rng.random(self.dim) < 0.15
        self._ref_warm = baseline_hv.copy()
        self._ref_warm[flip_mask] = 1 - self._ref_warm[flip_mask]

        # WEIRD: ~30% bits flipped
        flip_mask = rng.random(self.dim) < 0.30
        self._ref_weird = baseline_hv.copy()
        self._ref_weird[flip_mask] = 1 - self._ref_weird[flip_mask]

        # CRITICAL: ~45% bits flipped (approaching random)
        flip_mask = rng.random(self.dim) < 0.45
        self._ref_critical = baseline_hv.copy()
        self._ref_critical[flip_mask] = 1 - self._ref_critical[flip_mask]

    def learn_zone_reference(self, zone: Zone, hv: np.ndarray):
        """Directly set a zone reference from observed data."""
        if zone == Zone.GOOD:
            self._ref_good = hv.copy()
        elif zone == Zone.WARM:
            self._ref_warm = hv.copy()
        elif zone == Zone.WEIRD:
            self._ref_weird = hv.copy()
        elif zone == Zone.CRITICAL:
            self._ref_critical = hv.copy()

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine-ish similarity for binary vectors."""
        if a is None or b is None:
            return 0.0
        matches = np.sum(a == b)
        return (2 * matches - self.dim) / self.dim

    def classify(self, hv: np.ndarray) -> ZoneState:
        """
        Classify a hypervector into a zone.

        Uses similarity to GOOD baseline as primary signal,
        with hysteresis to prevent oscillation.
        """
        if self._ref_good is None:
            # No baseline yet, assume GOOD
            return ZoneState(
                zone=Zone.GOOD,
                confidence=1.0,
                similarity_to_good=1.0,
                time_in_zone=self._time_in_zone,
                previous_zone=self._previous_zone,
            )

        # Compute similarity to GOOD baseline
        sim_good = self._similarity(hv, self._ref_good)

        # Determine raw zone from thresholds
        if sim_good >= self.thresholds.good_min:
            raw_zone = Zone.GOOD
        elif sim_good >= self.thresholds.warm_min:
            raw_zone = Zone.WARM
        elif sim_good >= self.thresholds.weird_min:
            raw_zone = Zone.WEIRD
        else:
            raw_zone = Zone.CRITICAL

        # Apply hysteresis: require exceeding threshold by margin to change
        new_zone = self._current_zone

        if raw_zone > self._current_zone:
            # Getting worse: easier to escalate (no hysteresis down)
            new_zone = raw_zone
        elif raw_zone < self._current_zone:
            # Getting better: require hysteresis to de-escalate
            threshold = self._get_threshold_for_zone(raw_zone)
            if sim_good >= threshold + self.hysteresis:
                new_zone = raw_zone

        # Update state
        if new_zone != self._current_zone:
            self._previous_zone = self._current_zone
            self._current_zone = new_zone
            self._time_in_zone = 0
        else:
            self._time_in_zone += 1

        # Compute confidence: how far into this zone are we?
        confidence = self._compute_confidence(sim_good, new_zone)

        return ZoneState(
            zone=new_zone,
            confidence=confidence,
            similarity_to_good=sim_good,
            time_in_zone=self._time_in_zone,
            previous_zone=self._previous_zone,
        )

    def _get_threshold_for_zone(self, zone: Zone) -> float:
        """Get the minimum threshold for a zone."""
        if zone == Zone.GOOD:
            return self.thresholds.good_min
        elif zone == Zone.WARM:
            return self.thresholds.warm_min
        elif zone == Zone.WEIRD:
            return self.thresholds.weird_min
        else:
            return -1.0  # CRITICAL has no lower bound

    def _compute_confidence(self, sim_good: float, zone: Zone) -> float:
        """
        Compute confidence in zone classification.

        Higher confidence = further from zone boundaries.
        """
        if zone == Zone.GOOD:
            # Distance from WARM threshold
            return min(1.0, (sim_good - self.thresholds.good_min) / 0.3 + 0.5)
        elif zone == Zone.WARM:
            # In the middle of WARM zone
            mid = (self.thresholds.good_min + self.thresholds.warm_min) / 2
            dist = abs(sim_good - mid) / (self.thresholds.good_min - self.thresholds.warm_min)
            return max(0.3, 1.0 - dist)
        elif zone == Zone.WEIRD:
            mid = (self.thresholds.warm_min + self.thresholds.weird_min) / 2
            dist = abs(sim_good - mid) / (self.thresholds.warm_min - self.thresholds.weird_min)
            return max(0.3, 1.0 - dist)
        else:  # CRITICAL
            # Further from weird threshold = more confident
            return min(1.0, (self.thresholds.weird_min - sim_good) / 0.2 + 0.5)

    def reset(self):
        """Reset zone state (e.g., after maintenance window)."""
        self._current_zone = Zone.GOOD
        self._time_in_zone = 0
        self._previous_zone = Zone.GOOD


@dataclass
class MultiZoneQuantizer:
    """
    Manages zone quantizers for multiple subsystems.

    Provides:
    - Per-subsystem local zones
    - Global aggregate zone
    - Cross-subsystem correlation detection
    """
    subsystems: Dict[str, ZoneQuantizer] = field(default_factory=dict)
    _global_zone: Zone = Zone.GOOD

    def add_subsystem(self, name: str, dim: int = 8192,
                      thresholds: Optional[ZoneThresholds] = None) -> ZoneQuantizer:
        """Add a subsystem with its own zone quantizer."""
        quantizer = ZoneQuantizer(
            dim=dim,
            thresholds=thresholds or ZoneThresholds(),
        )
        self.subsystems[name] = quantizer
        return quantizer

    def set_baseline(self, name: str, baseline_hv: np.ndarray):
        """Set baseline for a subsystem."""
        if name in self.subsystems:
            self.subsystems[name].set_baseline(baseline_hv)

    def classify(self, name: str, hv: np.ndarray) -> ZoneState:
        """Classify a subsystem's current state."""
        if name not in self.subsystems:
            raise KeyError(f"Unknown subsystem: {name}")
        return self.subsystems[name].classify(hv)

    def classify_all(self, hvs: Dict[str, np.ndarray]) -> Dict[str, ZoneState]:
        """Classify all provided subsystem states."""
        return {
            name: self.classify(name, hv)
            for name, hv in hvs.items()
            if name in self.subsystems
        }

    def compute_global_zone(self, states: Dict[str, ZoneState]) -> Zone:
        """
        Compute aggregate global zone from all subsystem states.

        Policy: worst zone wins, but weighted by confidence.
        """
        if not states:
            return Zone.GOOD

        # Simple policy: max zone
        worst = max(state.zone for state in states.values())

        # If multiple subsystems are in the same bad zone, escalate further
        count_at_worst = sum(1 for s in states.values() if s.zone == worst)
        if count_at_worst >= 2 and worst < Zone.CRITICAL:
            # Multiple subsystems failing together → escalate
            worst = Zone(min(worst + 1, Zone.CRITICAL))

        self._global_zone = worst
        return worst

    def detect_correlation(self, states: Dict[str, ZoneState]) -> List[Tuple[str, str]]:
        """
        Detect subsystems that are correlated (failing together).

        Returns list of (subsystem_a, subsystem_b) pairs that
        transitioned to worse zones at the same time.
        """
        correlations = []
        names = list(states.keys())

        for i, name_a in enumerate(names):
            for name_b in names[i+1:]:
                state_a = states[name_a]
                state_b = states[name_b]

                # Both recently transitioned to worse zone
                if (state_a.zone > state_a.previous_zone and
                    state_b.zone > state_b.previous_zone and
                    state_a.time_in_zone < 3 and
                    state_b.time_in_zone < 3):
                    correlations.append((name_a, name_b))

        return correlations

    @property
    def global_zone(self) -> Zone:
        return self._global_zone
