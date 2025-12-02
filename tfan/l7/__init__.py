"""
L7: Predictive Structural Control via Temporal Topology

This module implements predictive instability detection using temporal
persistent homology (TPH). Instead of reacting when topo_gap/EPR-CV are
already bad, L7 predicts failure from the *rate* of topological change.

Core idea:
- Take SNN spike traces over time
- Run temporal persistent homology over sliding windows
- Derive structural rate (Ṡ) = how quickly topology is changing
- Use Ṡ as a predictor of future instability
- Trigger AEPO + L3 changes *proactively*

Integration:
    CLV ← structural_dynamics (from L7)
        ↓
    SemanticOptimizer/AdaptiveController
        ↓
    Proactive mode switch (conservative backend, lower temperature)

Usage:
    from tfan.l7 import (
        TemporalTopologyTracker,
        compute_structural_rate,
        get_predictive_alert,
    )

    tracker = TemporalTopologyTracker(window_size=50)
    tracker.update(spike_counts, topology_features)

    if tracker.get_structural_rate() > threshold:
        # Proactive intervention before failure
        switch_to_conservative_mode()
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging
import math

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("tfan.l7.temporal_topology")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TemporalTopologyConfig:
    """Configuration for temporal topology tracking."""
    # Sliding window size for TPH
    window_size: int = 50

    # How often to compute full TPH (every N updates)
    tph_compute_interval: int = 5

    # Thresholds for structural rate alerts
    structural_rate_warning: float = 0.3   # Elevated change rate
    structural_rate_critical: float = 0.6  # Imminent instability

    # Wasserstein distance parameters
    wasserstein_p: int = 2  # L2 distance

    # Smoothing for rate computation
    rate_smoothing: float = 0.2

    # Prediction horizon (how far ahead to warn)
    prediction_horizon_steps: int = 10

    # Feature weights for combined rate
    betti_weight: float = 0.4
    spectrum_weight: float = 0.3
    gap_weight: float = 0.3


class StructuralAlert(Enum):
    """Alert levels for structural dynamics."""
    STABLE = "stable"           # Normal operation
    ELEVATED = "elevated"       # Increased dynamics, monitor closely
    WARNING = "warning"         # Approaching instability
    CRITICAL = "critical"       # Imminent failure, act now


# =============================================================================
# TOPOLOGICAL FEATURES
# =============================================================================

@dataclass
class TopologySnapshot:
    """Snapshot of topological features at a point in time."""
    timestamp: float
    betti_0: float = 0.0        # Connected components
    betti_1: float = 0.0        # Loops/holes
    betti_2: float = 0.0        # Voids
    spectral_gap: float = 0.0   # λ₂ of graph Laplacian
    topo_gap: float = 0.0       # Topology gap metric
    epr_cv: float = 0.0         # EPR coefficient of variation
    spike_rate: float = 0.0     # Mean spike rate

    def to_vector(self) -> List[float]:
        """Convert to feature vector for distance computation."""
        return [
            self.betti_0,
            self.betti_1,
            self.betti_2,
            self.spectral_gap,
            self.topo_gap,
            self.epr_cv,
            self.spike_rate,
        ]

    @classmethod
    def from_vector(cls, vec: List[float], timestamp: float) -> "TopologySnapshot":
        """Create from feature vector."""
        return cls(
            timestamp=timestamp,
            betti_0=vec[0] if len(vec) > 0 else 0,
            betti_1=vec[1] if len(vec) > 1 else 0,
            betti_2=vec[2] if len(vec) > 2 else 0,
            spectral_gap=vec[3] if len(vec) > 3 else 0,
            topo_gap=vec[4] if len(vec) > 4 else 0,
            epr_cv=vec[5] if len(vec) > 5 else 0,
            spike_rate=vec[6] if len(vec) > 6 else 0,
        )


@dataclass
class StructuralDynamics:
    """Computed dynamics from temporal topology."""
    structural_rate: float = 0.0      # Ṡ = rate of topological change
    betti_rate: float = 0.0           # Rate of Betti number change
    spectrum_rate: float = 0.0        # Rate of spectral change
    gap_rate: float = 0.0             # Rate of gap change
    alert_level: StructuralAlert = StructuralAlert.STABLE
    predicted_instability_steps: int = -1  # Steps until predicted instability (-1 = none)
    confidence: float = 0.0           # Prediction confidence


# =============================================================================
# TEMPORAL TOPOLOGY TRACKER
# =============================================================================

class TemporalTopologyTracker:
    """
    Tracks topological features over time and computes structural dynamics.

    This is the core L7 component that predicts instability before it happens.
    """

    def __init__(self, config: Optional[TemporalTopologyConfig] = None):
        """
        Initialize temporal topology tracker.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or TemporalTopologyConfig()

        # Sliding window of topology snapshots
        self._history: List[TopologySnapshot] = []
        self._max_history = self.config.window_size * 2

        # Computed dynamics
        self._current_dynamics = StructuralDynamics()

        # Rate history for trend analysis
        self._rate_history: List[float] = []
        self._max_rate_history = 100

        # Update counter
        self._updates = 0

        logger.info(
            f"TemporalTopologyTracker initialized: "
            f"window={self.config.window_size}, interval={self.config.tph_compute_interval}"
        )

    def update(
        self,
        betti_0: float = 0.0,
        betti_1: float = 0.0,
        betti_2: float = 0.0,
        spectral_gap: float = 0.0,
        topo_gap: float = 0.0,
        epr_cv: float = 0.0,
        spike_rate: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> StructuralDynamics:
        """
        Update tracker with new topology snapshot.

        Args:
            betti_0: Connected components count
            betti_1: Loops/holes count
            betti_2: Voids count
            spectral_gap: λ₂ of graph Laplacian
            topo_gap: Topology gap metric
            epr_cv: EPR coefficient of variation
            spike_rate: Mean spike rate
            timestamp: Optional timestamp (uses monotonic counter if None)

        Returns:
            Current structural dynamics
        """
        if timestamp is None:
            timestamp = float(self._updates)

        # Create snapshot
        snapshot = TopologySnapshot(
            timestamp=timestamp,
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            spectral_gap=spectral_gap,
            topo_gap=topo_gap,
            epr_cv=epr_cv,
            spike_rate=spike_rate,
        )

        # Add to history
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        self._updates += 1

        # Compute dynamics at interval
        if self._updates % self.config.tph_compute_interval == 0:
            self._compute_dynamics()

        return self._current_dynamics

    def update_from_dict(self, features: Dict[str, float]) -> StructuralDynamics:
        """Update from feature dictionary."""
        return self.update(
            betti_0=features.get("betti_0", 0),
            betti_1=features.get("betti_1", 0),
            betti_2=features.get("betti_2", 0),
            spectral_gap=features.get("spectral_gap", 0),
            topo_gap=features.get("topo_gap", 0),
            epr_cv=features.get("epr_cv", 0),
            spike_rate=features.get("spike_rate", 0),
            timestamp=features.get("timestamp"),
        )

    def _compute_dynamics(self):
        """Compute structural dynamics from history."""
        cfg = self.config

        if len(self._history) < 2:
            return

        # Get recent window
        window = self._history[-cfg.window_size:]
        if len(window) < 2:
            return

        # Compute rates for each feature
        betti_rate = self._compute_feature_rate(window, ["betti_0", "betti_1", "betti_2"])
        spectrum_rate = self._compute_feature_rate(window, ["spectral_gap"])
        gap_rate = self._compute_feature_rate(window, ["topo_gap", "epr_cv"])

        # Combined structural rate
        raw_rate = (
            cfg.betti_weight * betti_rate +
            cfg.spectrum_weight * spectrum_rate +
            cfg.gap_weight * gap_rate
        )

        # Smooth the rate
        prev_rate = self._current_dynamics.structural_rate
        smoothed_rate = (
            cfg.rate_smoothing * prev_rate +
            (1 - cfg.rate_smoothing) * raw_rate
        )

        # Determine alert level
        if smoothed_rate >= cfg.structural_rate_critical:
            alert = StructuralAlert.CRITICAL
        elif smoothed_rate >= cfg.structural_rate_warning:
            alert = StructuralAlert.WARNING
        elif smoothed_rate >= cfg.structural_rate_warning * 0.5:
            alert = StructuralAlert.ELEVATED
        else:
            alert = StructuralAlert.STABLE

        # Predict instability
        predicted_steps, confidence = self._predict_instability(smoothed_rate)

        # Update current dynamics
        self._current_dynamics = StructuralDynamics(
            structural_rate=smoothed_rate,
            betti_rate=betti_rate,
            spectrum_rate=spectrum_rate,
            gap_rate=gap_rate,
            alert_level=alert,
            predicted_instability_steps=predicted_steps,
            confidence=confidence,
        )

        # Track rate history
        self._rate_history.append(smoothed_rate)
        if len(self._rate_history) > self._max_rate_history:
            self._rate_history.pop(0)

        logger.debug(
            f"L7 dynamics: Ṡ={smoothed_rate:.4f}, alert={alert.value}, "
            f"predicted_steps={predicted_steps}"
        )

    def _compute_feature_rate(
        self,
        window: List[TopologySnapshot],
        feature_names: List[str],
    ) -> float:
        """Compute rate of change for a set of features."""
        if not NUMPY_AVAILABLE or len(window) < 2:
            # Fallback: simple difference
            start = window[0]
            end = window[-1]
            total_diff = 0
            for name in feature_names:
                v0 = getattr(start, name, 0)
                v1 = getattr(end, name, 0)
                total_diff += abs(v1 - v0)
            dt = end.timestamp - start.timestamp
            return total_diff / max(dt, 1e-6) if dt > 0 else 0

        # Compute Wasserstein-like distance between start and end of window
        start_vec = []
        end_vec = []
        for name in feature_names:
            start_vec.append(getattr(window[0], name, 0))
            end_vec.append(getattr(window[-1], name, 0))

        start_arr = np.array(start_vec)
        end_arr = np.array(end_vec)

        # L2 distance normalized by time
        dist = np.linalg.norm(end_arr - start_arr)
        dt = window[-1].timestamp - window[0].timestamp

        return dist / max(dt, 1e-6)

    def _predict_instability(self, current_rate: float) -> Tuple[int, float]:
        """
        Predict steps until instability based on rate trends.

        Returns:
            (predicted_steps, confidence) - (-1, 0) if no instability predicted
        """
        cfg = self.config

        if current_rate < cfg.structural_rate_warning * 0.5:
            return -1, 0.0  # No instability predicted

        # Simple linear extrapolation
        if len(self._rate_history) >= 5:
            recent = self._rate_history[-5:]
            if NUMPY_AVAILABLE:
                # Fit trend
                x = np.arange(len(recent))
                slope = np.polyfit(x, recent, 1)[0]

                if slope > 0:
                    # Predict when rate will hit critical
                    steps_to_critical = (cfg.structural_rate_critical - current_rate) / slope
                    if 0 < steps_to_critical < cfg.prediction_horizon_steps * 2:
                        confidence = min(1.0, current_rate / cfg.structural_rate_critical)
                        return int(steps_to_critical), confidence

        # Fallback: immediate warning if already high
        if current_rate >= cfg.structural_rate_critical:
            return 0, 0.9
        elif current_rate >= cfg.structural_rate_warning:
            return cfg.prediction_horizon_steps, 0.6

        return -1, 0.0

    def get_structural_rate(self) -> float:
        """Get current structural rate (Ṡ)."""
        return self._current_dynamics.structural_rate

    def get_alert_level(self) -> StructuralAlert:
        """Get current alert level."""
        return self._current_dynamics.alert_level

    def get_dynamics(self) -> StructuralDynamics:
        """Get full structural dynamics."""
        return self._current_dynamics

    def get_state(self) -> Dict[str, Any]:
        """Get tracker state for monitoring."""
        return {
            "structural_rate": self._current_dynamics.structural_rate,
            "betti_rate": self._current_dynamics.betti_rate,
            "spectrum_rate": self._current_dynamics.spectrum_rate,
            "gap_rate": self._current_dynamics.gap_rate,
            "alert_level": self._current_dynamics.alert_level.value,
            "predicted_instability_steps": self._current_dynamics.predicted_instability_steps,
            "confidence": self._current_dynamics.confidence,
            "history_length": len(self._history),
            "updates": self._updates,
        }

    def get_rate_history(self) -> List[float]:
        """Get structural rate history for analysis."""
        return self._rate_history.copy()

    def reset(self):
        """Reset tracker state."""
        self._history.clear()
        self._rate_history.clear()
        self._current_dynamics = StructuralDynamics()
        self._updates = 0
        logger.info("TemporalTopologyTracker reset")


# =============================================================================
# CLV INTEGRATION
# =============================================================================

@dataclass
class L7CLVExtension:
    """
    Extension to CLV with L7 structural dynamics.

    This adds predictive capability to the Cognitive Load Vector.
    """
    structural_dynamics: float = 0.0    # Ṡ normalized to [0, 1]
    predicted_risk: float = 0.0         # Risk from prediction
    alert_level: str = "stable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structural_dynamics": self.structural_dynamics,
            "predicted_risk": self.predicted_risk,
            "alert_level": self.alert_level,
        }


def compute_l7_clv_extension(
    dynamics: StructuralDynamics,
    config: Optional[TemporalTopologyConfig] = None,
) -> L7CLVExtension:
    """
    Compute CLV extension from L7 dynamics.

    This integrates L7 predictions into the CLV framework.
    """
    cfg = config or TemporalTopologyConfig()

    # Normalize structural rate to [0, 1]
    normalized_rate = min(1.0, dynamics.structural_rate / cfg.structural_rate_critical)

    # Compute predicted risk
    if dynamics.predicted_instability_steps >= 0:
        # Higher risk if instability is imminent
        urgency = 1.0 - (dynamics.predicted_instability_steps / (cfg.prediction_horizon_steps * 2))
        predicted_risk = urgency * dynamics.confidence
    else:
        predicted_risk = 0.0

    return L7CLVExtension(
        structural_dynamics=normalized_rate,
        predicted_risk=predicted_risk,
        alert_level=dynamics.alert_level.value,
    )


# =============================================================================
# PROACTIVE CONTROL HOOKS
# =============================================================================

class ProactiveController:
    """
    Uses L7 predictions to trigger proactive control actions.

    This is the "act before failure" component.
    """

    def __init__(
        self,
        tracker: TemporalTopologyTracker,
        warning_threshold: float = 0.3,
        critical_threshold: float = 0.6,
    ):
        """
        Initialize proactive controller.

        Args:
            tracker: L7 temporal topology tracker
            warning_threshold: Ṡ threshold for warning actions
            critical_threshold: Ṡ threshold for critical actions
        """
        self.tracker = tracker
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # Action history
        self._actions_taken: List[Dict[str, Any]] = []
        self._max_action_history = 50

    def check_and_recommend(self) -> Dict[str, Any]:
        """
        Check current dynamics and recommend proactive actions.

        Returns:
            Dict with recommended actions and reasoning
        """
        dynamics = self.tracker.get_dynamics()

        recommendations = {
            "timestamp": datetime.utcnow().isoformat(),
            "structural_rate": dynamics.structural_rate,
            "alert_level": dynamics.alert_level.value,
            "actions": [],
            "reasoning": [],
        }

        if dynamics.alert_level == StructuralAlert.CRITICAL:
            recommendations["actions"].extend([
                {"action": "switch_backend", "target": "pgu_verified"},
                {"action": "lower_temperature", "delta": -0.3},
                {"action": "reduce_entropy", "factor": 0.5},
                {"action": "enable_conservative_mask", "enabled": True},
            ])
            recommendations["reasoning"].append(
                f"Critical: Ṡ={dynamics.structural_rate:.3f} indicates imminent instability"
            )

        elif dynamics.alert_level == StructuralAlert.WARNING:
            recommendations["actions"].extend([
                {"action": "lower_temperature", "delta": -0.1},
                {"action": "reduce_entropy", "factor": 0.8},
            ])
            recommendations["reasoning"].append(
                f"Warning: Ṡ={dynamics.structural_rate:.3f} approaching critical threshold"
            )

            if dynamics.predicted_instability_steps >= 0:
                recommendations["reasoning"].append(
                    f"Predicted instability in {dynamics.predicted_instability_steps} steps "
                    f"(confidence: {dynamics.confidence:.2f})"
                )

        elif dynamics.alert_level == StructuralAlert.ELEVATED:
            recommendations["actions"].append(
                {"action": "increase_monitoring", "interval_factor": 0.5}
            )
            recommendations["reasoning"].append(
                f"Elevated: Ṡ={dynamics.structural_rate:.3f}, monitoring closely"
            )

        # Record action
        self._actions_taken.append(recommendations)
        if len(self._actions_taken) > self._max_action_history:
            self._actions_taken.pop(0)

        return recommendations

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get history of recommended actions."""
        return self._actions_taken.copy()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global tracker instance
_global_tracker: Optional[TemporalTopologyTracker] = None


def get_topology_tracker() -> TemporalTopologyTracker:
    """Get global temporal topology tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TemporalTopologyTracker()
    return _global_tracker


def compute_structural_rate(
    betti_0: float = 0.0,
    betti_1: float = 0.0,
    spectral_gap: float = 0.0,
    topo_gap: float = 0.0,
    epr_cv: float = 0.0,
) -> float:
    """
    Convenience function to update tracker and get structural rate.

    Example:
        rate = compute_structural_rate(betti_0=5, spectral_gap=0.8)
        if rate > 0.5:
            trigger_proactive_control()
    """
    tracker = get_topology_tracker()
    dynamics = tracker.update(
        betti_0=betti_0,
        betti_1=betti_1,
        spectral_gap=spectral_gap,
        topo_gap=topo_gap,
        epr_cv=epr_cv,
    )
    return dynamics.structural_rate


def get_predictive_alert() -> Tuple[str, int, float]:
    """
    Get current predictive alert status.

    Returns:
        (alert_level, predicted_steps, confidence)
    """
    tracker = get_topology_tracker()
    dynamics = tracker.get_dynamics()
    return (
        dynamics.alert_level.value,
        dynamics.predicted_instability_steps,
        dynamics.confidence,
    )


def should_act_proactively() -> bool:
    """Check if proactive action is recommended."""
    tracker = get_topology_tracker()
    alert = tracker.get_alert_level()
    return alert in [StructuralAlert.WARNING, StructuralAlert.CRITICAL]


__all__ = [
    "TemporalTopologyConfig",
    "StructuralAlert",
    "TopologySnapshot",
    "StructuralDynamics",
    "TemporalTopologyTracker",
    "L7CLVExtension",
    "compute_l7_clv_extension",
    "ProactiveController",
    "get_topology_tracker",
    "compute_structural_rate",
    "get_predictive_alert",
    "should_act_proactively",
]
