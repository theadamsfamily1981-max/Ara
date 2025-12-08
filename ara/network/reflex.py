"""
Ara Network Reflex Engine - HD Template Classification
======================================================

Fast, unconscious flow classification using HD templates.

The reflex layer runs closest to the network (eBPF/SmartNIC) and makes
instant decisions without consulting the full HTC. It uses pre-computed
HD templates that are periodically updated by the Sovereign loop.

Classifications:
- GOOD: Prioritize, no throttling
- BACKGROUND: Normal handling
- SUSPICIOUS: Tag and potentially throttle
- DANGER: Immediate action (block/alert)

The templates (h_good, h_bad) are learned by the HTC over time
and pushed down to the reflex layer.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from ara.hd.ops import cosine, DIM
from ara.io.types import HV


class FlowLabel(str, Enum):
    """Flow classification labels."""
    GOOD = "good"              # Prioritize
    BACKGROUND = "background"  # Normal handling
    SUSPICIOUS = "suspicious"  # Monitor closely
    DANGER = "danger"          # Immediate action


@dataclass
class ClassificationResult:
    """Result of flow classification."""
    label: FlowLabel
    confidence: float
    sim_good: float
    sim_bad: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "sim_good": round(self.sim_good, 4),
            "sim_bad": round(self.sim_bad, 4),
            "timestamp": self.timestamp.isoformat(),
        }


class ReflexEngine:
    """
    HD template-based flow classifier.

    Uses pre-computed templates (h_good, h_bad) for instant classification.
    Templates are updated periodically by the Sovereign loop based on
    HTC learning.
    """

    def __init__(
        self,
        h_good: Optional[HV] = None,
        h_bad: Optional[HV] = None,
        thresh_good: float = 0.3,
        thresh_bad: float = 0.3,
        thresh_danger: float = 0.5,
    ):
        """
        Initialize the reflex engine.

        Args:
            h_good: Template for "good" flows
            h_bad: Template for "bad" flows
            thresh_good: Similarity threshold for GOOD classification
            thresh_bad: Similarity threshold for SUSPICIOUS classification
            thresh_danger: Similarity threshold for DANGER classification
        """
        self.h_good = h_good
        self.h_bad = h_bad
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        self.thresh_danger = thresh_danger

        # Statistics
        self._classifications: int = 0
        self._by_label: Dict[FlowLabel, int] = {l: 0 for l in FlowLabel}
        self._last_update: Optional[datetime] = None

    def update_templates(
        self,
        h_good: Optional[HV] = None,
        h_bad: Optional[HV] = None,
    ) -> None:
        """
        Update classification templates.

        Called by Sovereign when HTC learns new patterns.
        """
        if h_good is not None:
            self.h_good = h_good
        if h_bad is not None:
            self.h_bad = h_bad
        self._last_update = datetime.utcnow()

    def update_thresholds(
        self,
        thresh_good: Optional[float] = None,
        thresh_bad: Optional[float] = None,
        thresh_danger: Optional[float] = None,
    ) -> None:
        """Update classification thresholds."""
        if thresh_good is not None:
            self.thresh_good = thresh_good
        if thresh_bad is not None:
            self.thresh_bad = thresh_bad
        if thresh_danger is not None:
            self.thresh_danger = thresh_danger

    def classify(self, h_flow: HV) -> ClassificationResult:
        """
        Classify a flow HV.

        Decision logic:
        1. If sim(h_flow, h_bad) > thresh_danger: DANGER
        2. If sim(h_flow, h_bad) > thresh_bad: SUSPICIOUS
        3. If sim(h_flow, h_good) > thresh_good: GOOD
        4. Otherwise: BACKGROUND
        """
        self._classifications += 1

        # Compute similarities
        sim_good = 0.0
        sim_bad = 0.0

        if self.h_good is not None:
            sim_good = cosine(h_flow, self.h_good)

        if self.h_bad is not None:
            sim_bad = cosine(h_flow, self.h_bad)

        # Classification logic
        if sim_bad > self.thresh_danger:
            label = FlowLabel.DANGER
            confidence = sim_bad
        elif sim_bad > self.thresh_bad:
            label = FlowLabel.SUSPICIOUS
            confidence = sim_bad
        elif sim_good > self.thresh_good:
            label = FlowLabel.GOOD
            confidence = sim_good
        else:
            label = FlowLabel.BACKGROUND
            confidence = 1.0 - max(sim_good, sim_bad)

        self._by_label[label] += 1

        return ClassificationResult(
            label=label,
            confidence=confidence,
            sim_good=sim_good,
            sim_bad=sim_bad,
        )

    def classify_batch(self, hvs: List[HV]) -> List[ClassificationResult]:
        """Classify a batch of flow HVs."""
        return [self.classify(hv) for hv in hvs]

    def is_ready(self) -> bool:
        """Check if templates are loaded."""
        return self.h_good is not None or self.h_bad is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        return {
            "total_classifications": self._classifications,
            "by_label": {l.value: c for l, c in self._by_label.items()},
            "templates_loaded": {
                "good": self.h_good is not None,
                "bad": self.h_bad is not None,
            },
            "thresholds": {
                "good": self.thresh_good,
                "bad": self.thresh_bad,
                "danger": self.thresh_danger,
            },
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._classifications = 0
        self._by_label = {l: 0 for l in FlowLabel}


# =============================================================================
# Template Builders
# =============================================================================

def build_good_flow_template(
    good_flows: List[HV],
) -> HV:
    """
    Build a template for "good" flows from examples.

    Uses bundling to create a superposition of known good patterns.
    """
    from ara.hd.ops import bundle

    if not good_flows:
        return np.zeros(DIM, dtype=np.uint8)

    return bundle(good_flows)


def build_bad_flow_template(
    bad_flows: List[HV],
) -> HV:
    """
    Build a template for "bad" flows from examples.

    Uses bundling to create a superposition of known bad patterns.
    """
    from ara.hd.ops import bundle

    if not bad_flows:
        return np.zeros(DIM, dtype=np.uint8)

    return bundle(bad_flows)


# =============================================================================
# Factory
# =============================================================================

_reflex_engine: Optional[ReflexEngine] = None


def get_reflex_engine() -> ReflexEngine:
    """Get the global reflex engine."""
    global _reflex_engine
    if _reflex_engine is None:
        _reflex_engine = ReflexEngine()
    return _reflex_engine


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'FlowLabel',
    'ClassificationResult',
    'ReflexEngine',
    'get_reflex_engine',
    'build_good_flow_template',
    'build_bad_flow_template',
]
