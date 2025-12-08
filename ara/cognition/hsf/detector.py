"""
HSF Anomaly Detector
=====================

Ultra-cheap anomaly detection via hypervector resonance.

The detector:
1. Maintains a library of known anomaly patterns
2. Checks current field against baselines
3. Checks resonance with known bad patterns
4. Reports deviations and pattern matches

This is the "smell test" - fast, always-on, cheap.
It doesn't explain WHY something is wrong, just that it IS.

For diagnostics, escalate to higher-level reasoning (LLM/Architect).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
import time

from .field import HSField
from .lanes import ItemMemory


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    NORMAL = auto()      # Within expected range
    ELEVATED = auto()    # Slightly unusual, worth watching
    WARNING = auto()     # Significant deviation, may need attention
    CRITICAL = auto()    # Major anomaly, needs immediate attention


@dataclass
class AnomalyPattern:
    """
    A known anomaly pattern to check against.

    Can be:
    - Learned from actual incidents
    - Synthesized from domain knowledge
    - Discovered via clustering of past anomalies
    """
    name: str
    description: str
    severity: AnomalySeverity
    pattern_hv: np.ndarray
    threshold: float = 0.3  # Similarity threshold for match

    def matches(self, field_hv: np.ndarray, item_memory: ItemMemory) -> Tuple[bool, float]:
        """Check if this pattern matches the given field."""
        similarity = item_memory.similarity(field_hv, self.pattern_hv)
        return similarity > self.threshold, similarity


@dataclass
class AnomalyReport:
    """Report from anomaly scan."""
    timestamp: float
    total_deviation: float
    severity: AnomalySeverity
    lane_deviations: Dict[str, float]
    pattern_matches: List[Tuple[AnomalyPattern, float]]  # (pattern, similarity)
    worst_lane: Optional[str]
    summary: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.name,
            "total_deviation": self.total_deviation,
            "lane_deviations": self.lane_deviations,
            "pattern_matches": [(p.name, s) for p, s in self.pattern_matches],
            "worst_lane": self.worst_lane,
            "summary": self.summary,
        }


@dataclass
class AnomalyDetector:
    """
    Hypervector-based anomaly detector.

    Ultra-cheap, always-on anomaly detection via:
    1. Baseline deviation: how far is current from learned normal?
    2. Pattern resonance: does current match any known bad patterns?
    """
    field: HSField
    patterns: List[AnomalyPattern] = field(default_factory=list)

    # Thresholds for severity levels
    elevated_threshold: float = 0.15
    warning_threshold: float = 0.30
    critical_threshold: float = 0.50

    def add_pattern(self, pattern: AnomalyPattern):
        """Add a known anomaly pattern."""
        self.patterns.append(pattern)

    def create_pattern(self, name: str, description: str,
                       severity: AnomalySeverity = AnomalySeverity.WARNING,
                       threshold: float = 0.3) -> AnomalyPattern:
        """Create and add a pattern from current field state."""
        if self.field.current is None:
            raise ValueError("No field state to create pattern from")

        pattern = AnomalyPattern(
            name=name,
            description=description,
            severity=severity,
            pattern_hv=self.field.current.copy(),
            threshold=threshold,
        )
        self.add_pattern(pattern)
        return pattern

    def synthesize_pattern(self, name: str, description: str,
                           lane_states: Dict[str, Dict[str, float]],
                           severity: AnomalySeverity = AnomalySeverity.WARNING,
                           threshold: float = 0.3) -> AnomalyPattern:
        """
        Synthesize a pattern from specified lane states.

        Useful for creating patterns from domain knowledge without
        needing to observe actual incidents.
        """
        # Temporarily update lanes with specified states
        original_states = {}
        for lane_name, values in lane_states.items():
            if lane_name in self.field.lanes:
                lane = self.field.lanes[lane_name]
                original_states[lane_name] = lane.current
                lane.update(values)

        # Capture field state
        pattern_hv = self.field.compute_field().copy()

        # Restore original states
        for lane_name, original in original_states.items():
            self.field.lanes[lane_name]._current = original

        # Recompute field with original states
        self.field.compute_field()

        pattern = AnomalyPattern(
            name=name,
            description=description,
            severity=severity,
            pattern_hv=pattern_hv,
            threshold=threshold,
        )
        self.add_pattern(pattern)
        return pattern

    def scan(self) -> AnomalyReport:
        """
        Perform anomaly scan on current field state.

        Returns: AnomalyReport with findings
        """
        if self.field.current is None:
            self.field.compute_field()

        # Get lane deviations
        lane_devs = self.field.lane_deviations()
        total_dev = self.field.total_deviation()

        # Find worst lane
        worst_lane = None
        worst_dev = 0.0
        for lane, dev in lane_devs.items():
            if dev > worst_dev:
                worst_dev = dev
                worst_lane = lane

        # Determine severity from deviation
        if total_dev >= self.critical_threshold:
            severity = AnomalySeverity.CRITICAL
        elif total_dev >= self.warning_threshold:
            severity = AnomalySeverity.WARNING
        elif total_dev >= self.elevated_threshold:
            severity = AnomalySeverity.ELEVATED
        else:
            severity = AnomalySeverity.NORMAL

        # Check pattern matches
        pattern_matches = []
        for pattern in self.patterns:
            matches, similarity = pattern.matches(
                self.field.current, self.field.item_memory
            )
            if matches:
                pattern_matches.append((pattern, similarity))
                # Upgrade severity if pattern is worse
                if pattern.severity.value > severity.value:
                    severity = pattern.severity

        # Sort matches by similarity (descending)
        pattern_matches.sort(key=lambda x: x[1], reverse=True)

        # Generate summary
        summary = self._generate_summary(severity, total_dev, worst_lane,
                                         worst_dev, pattern_matches)

        return AnomalyReport(
            timestamp=time.time(),
            total_deviation=total_dev,
            severity=severity,
            lane_deviations=lane_devs,
            pattern_matches=pattern_matches,
            worst_lane=worst_lane,
            summary=summary,
        )

    def _generate_summary(self, severity: AnomalySeverity, total_dev: float,
                          worst_lane: Optional[str], worst_dev: float,
                          pattern_matches: List[Tuple[AnomalyPattern, float]]) -> str:
        """Generate human-readable summary."""
        if severity == AnomalySeverity.NORMAL:
            return "All systems nominal."

        parts = []

        # Severity header
        if severity == AnomalySeverity.CRITICAL:
            parts.append("CRITICAL: Major anomaly detected.")
        elif severity == AnomalySeverity.WARNING:
            parts.append("WARNING: Significant deviation from baseline.")
        else:
            parts.append("ELEVATED: Slightly unusual activity.")

        # Worst lane
        if worst_lane and worst_dev > self.elevated_threshold:
            parts.append(f"Highest deviation in {worst_lane} lane ({worst_dev:.1%}).")

        # Pattern matches
        if pattern_matches:
            top_pattern, sim = pattern_matches[0]
            parts.append(f"Matches pattern '{top_pattern.name}' ({sim:.1%} similarity).")

        return " ".join(parts)

    def quick_check(self) -> Tuple[AnomalySeverity, float]:
        """
        Ultra-fast anomaly check - just deviation and severity.

        For tight loops where full scan is too expensive.
        """
        if self.field.current is None:
            self.field.compute_field()

        total_dev = self.field.total_deviation()

        if total_dev >= self.critical_threshold:
            return AnomalySeverity.CRITICAL, total_dev
        elif total_dev >= self.warning_threshold:
            return AnomalySeverity.WARNING, total_dev
        elif total_dev >= self.elevated_threshold:
            return AnomalySeverity.ELEVATED, total_dev
        else:
            return AnomalySeverity.NORMAL, total_dev


def create_default_patterns(field: HSField) -> List[AnomalyPattern]:
    """
    Create default anomaly patterns based on domain knowledge.

    These are synthetic patterns representing common failure modes.
    """
    patterns = []
    dim = field.dim

    # Create deterministic "bad" patterns using seeded RNG
    def make_pattern(name: str, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=dim, dtype=np.uint8)

    patterns.append(AnomalyPattern(
        name="thermal_crisis",
        description="GPU thermal runaway pattern",
        severity=AnomalySeverity.CRITICAL,
        pattern_hv=make_pattern("thermal_crisis", 42),
        threshold=0.25,
    ))

    patterns.append(AnomalyPattern(
        name="network_storm",
        description="Network congestion/DDoS pattern",
        severity=AnomalySeverity.CRITICAL,
        pattern_hv=make_pattern("network_storm", 43),
        threshold=0.25,
    ))

    patterns.append(AnomalyPattern(
        name="memory_exhaustion",
        description="Service memory leak pattern",
        severity=AnomalySeverity.WARNING,
        pattern_hv=make_pattern("memory_exhaustion", 44),
        threshold=0.25,
    ))

    patterns.append(AnomalyPattern(
        name="cascade_failure",
        description="Multi-system cascade failure",
        severity=AnomalySeverity.CRITICAL,
        pattern_hv=make_pattern("cascade_failure", 45),
        threshold=0.30,
    ))

    return patterns
