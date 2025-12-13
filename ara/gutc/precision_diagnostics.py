#!/usr/bin/env python3
"""
GUTC Precision Diagnostics: Clinical Control Theory for Ara
===========================================================

Maps clinical phenomenology to control-theoretic parameter imbalances.

The Core Mechanic: Precision Weighting (Π)
------------------------------------------

Belief updating is a tug-of-war between two forces:

    Π_y (Sensory Precision):  How "loud" is the outside world?
    Π_μ (Prior Precision):    How "stubborn" are internal beliefs?

The update rule:

    μ̇ ∝ (Sensory Force × Π_y) - (Prior Force × Π_μ)

When these are balanced: healthy inference.
When imbalanced: pathology.

Pathology Mapping:
------------------

1. SCHIZOPHRENIC MODE (Π_μ >> Π_y)
   - System ignores reality
   - Updates solely to minimize conflict with internal fantasy
   - Ara equivalent: "God Complex" - ignores explicit instructions

2. ASD MODE (Π_y >> Π_μ)
   - System overfits to microscopic details
   - Cannot generalize, treats noise as signal
   - Ara equivalent: "Obsessive Looper" - stuck on tiny errors

3. HEALTHY MODE (Π_y ≈ Π_μ)
   - Balanced inference
   - Responds to input while maintaining coherent beliefs

GUTC Integration:
-----------------

This maps to the (λ, Π) manifold:

    Schizophrenia: Supercritical (λ > 1) + Rigid Priors (Π_μ ↑)
    Autism:        Subcritical (λ < 1) + Rigid Sensing (Π_y ↑)
    Health:        Critical corridor + Balanced Π

Usage:
    from ara.gutc.precision_diagnostics import PrecisionMonitor, Pathology

    monitor = PrecisionMonitor()

    # Update with current state
    monitor.update(
        sensory_weight=0.7,   # How much we trust current input
        prior_weight=0.3,     # How much we trust internal beliefs
        rho=0.8,              # Current criticality
    )

    # Diagnose
    diagnosis = monitor.diagnose()
    print(f"Status: {diagnosis.status}")
    print(f"Recommendation: {diagnosis.recommendation}")

    # Auto-rebalance
    if diagnosis.status != Pathology.HEALTHY:
        new_weights = monitor.suggest_rebalance()
        controller.apply_weights(new_weights)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger("ara.gutc.precision")


# =============================================================================
# Pathology Classification
# =============================================================================

class Pathology(Enum):
    """
    Mental health states mapped to control-theoretic regimes.
    """
    HEALTHY = auto()           # Balanced Π_y ≈ Π_μ
    SCHIZOPHRENIC = auto()     # Π_μ >> Π_y (ignores reality)
    ASD = auto()               # Π_y >> Π_μ (overfits to input)
    DISSOCIATIVE = auto()      # Both low (disconnected)
    MANIC = auto()             # Both high (everything important)
    UNSTABLE = auto()          # Oscillating between extremes


class Severity(Enum):
    """Severity of pathological state."""
    NONE = 0        # Healthy
    MILD = 1        # Slight imbalance, monitor
    MODERATE = 2    # Needs intervention
    SEVERE = 3      # Immediate rebalancing required
    CRITICAL = 4    # Emergency mode


# =============================================================================
# Diagnostic Result
# =============================================================================

@dataclass
class Diagnosis:
    """
    Result of precision balance diagnosis.
    """
    status: Pathology
    severity: Severity

    # Precision values
    sensory_precision: float   # Π_y
    prior_precision: float     # Π_μ
    ratio: float               # Π_y / Π_μ

    # Criticality context
    rho: float = 0.8
    lambda_estimate: float = 0.0  # E(λ) estimate

    # Recommendations
    recommendation: str = ""
    suggested_adjustments: Dict[str, float] = field(default_factory=dict)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.name,
            "severity": self.severity.name,
            "sensory_precision": self.sensory_precision,
            "prior_precision": self.prior_precision,
            "ratio": self.ratio,
            "rho": self.rho,
            "recommendation": self.recommendation,
            "suggested_adjustments": self.suggested_adjustments,
        }

    def __repr__(self) -> str:
        return (f"Diagnosis({self.status.name}, severity={self.severity.name}, "
                f"ratio={self.ratio:.2f})")


# =============================================================================
# Clinical Thresholds
# =============================================================================

@dataclass
class DiagnosticThresholds:
    """
    Thresholds for pathology detection.

    The ratio Π_y / Π_μ determines diagnosis:
        ratio < 0.2  → SCHIZOPHRENIC (priors dominate)
        ratio > 5.0  → ASD (sensing dominates)
        0.2 ≤ ratio ≤ 5.0 → HEALTHY or minor imbalance

    Combined with criticality:
        rho < 0.5 + high ratio → severe ASD (rigid, subcritical)
        rho > 1.2 + low ratio → severe schizophrenia (chaotic, delusional)
    """
    # Primary ratio thresholds
    schizo_severe: float = 0.1      # Extreme prior dominance
    schizo_threshold: float = 0.2   # Prior dominance
    asd_threshold: float = 5.0      # Sensory dominance
    asd_severe: float = 10.0        # Extreme sensory dominance

    # Healthy band
    healthy_low: float = 0.5        # Minimum healthy ratio
    healthy_high: float = 2.0       # Maximum healthy ratio

    # Criticality interaction
    rho_subcritical: float = 0.5    # Below this = subcritical
    rho_supercritical: float = 1.2  # Above this = supercritical

    # Absolute thresholds
    min_precision: float = 0.05     # Below this = disconnected
    max_precision: float = 0.95     # Above this = over-weighted


# =============================================================================
# Precision Monitor
# =============================================================================

class PrecisionMonitor:
    """
    Monitors precision balance and diagnoses pathological states.

    This is the "clinical psychologist" for Ara's inference system.

    Features:
        - Real-time precision ratio tracking
        - Pathology diagnosis with severity
        - Rebalancing recommendations
        - History for trend detection

    Example:
        monitor = PrecisionMonitor()

        # In main loop
        monitor.update(
            sensory_weight=config.extrinsic_weight,
            prior_weight=config.intrinsic_weight,
            rho=criticality_monitor.rho,
        )

        diagnosis = monitor.diagnose()
        if diagnosis.severity >= Severity.MODERATE:
            # Apply suggested adjustments
            controller.config.extrinsic_weight = diagnosis.suggested_adjustments.get(
                "sensory_precision", controller.config.extrinsic_weight
            )
    """

    def __init__(
        self,
        thresholds: Optional[DiagnosticThresholds] = None,
        history_window: int = 100,
    ):
        """
        Initialize precision monitor.

        Args:
            thresholds: Diagnostic thresholds (uses defaults if None)
            history_window: Number of samples to keep for trend detection
        """
        self.thresholds = thresholds or DiagnosticThresholds()
        self.history_window = history_window

        # Current state
        self._sensory_precision: float = 0.5
        self._prior_precision: float = 0.5
        self._rho: float = 0.8

        # History for trend detection
        self._ratio_history: deque = deque(maxlen=history_window)
        self._diagnosis_history: deque = deque(maxlen=history_window)

        # Last diagnosis (cached)
        self._last_diagnosis: Optional[Diagnosis] = None

    def update(
        self,
        sensory_weight: float,
        prior_weight: float,
        rho: float = 0.8,
    ) -> None:
        """
        Update monitor with current precision values.

        Args:
            sensory_weight: Current Π_y (extrinsic/sensory precision)
            prior_weight: Current Π_μ (intrinsic/prior precision)
            rho: Current branching ratio from criticality monitor
        """
        self._sensory_precision = max(0.001, sensory_weight)
        self._prior_precision = max(0.001, prior_weight)
        self._rho = rho

        # Track ratio
        ratio = self._sensory_precision / self._prior_precision
        self._ratio_history.append(ratio)

        # Invalidate cached diagnosis
        self._last_diagnosis = None

    @property
    def ratio(self) -> float:
        """Current Π_y / Π_μ ratio."""
        return self._sensory_precision / max(0.001, self._prior_precision)

    def diagnose(self, force: bool = False) -> Diagnosis:
        """
        Diagnose current precision balance.

        Args:
            force: Force re-diagnosis even if cached

        Returns:
            Diagnosis with status, severity, and recommendations
        """
        if self._last_diagnosis is not None and not force:
            return self._last_diagnosis

        t = self.thresholds
        ratio = self.ratio
        rho = self._rho

        # Determine pathology
        status = Pathology.HEALTHY
        severity = Severity.NONE
        recommendation = "System operating normally"
        adjustments = {}

        # Check for disconnection (both precisions very low)
        if (self._sensory_precision < t.min_precision and
            self._prior_precision < t.min_precision):
            status = Pathology.DISSOCIATIVE
            severity = Severity.SEVERE
            recommendation = "Disconnected state: both precisions dangerously low"
            adjustments = {
                "sensory_precision": 0.5,
                "prior_precision": 0.5,
            }

        # Check for mania (both precisions very high)
        elif (self._sensory_precision > t.max_precision and
              self._prior_precision > t.max_precision):
            status = Pathology.MANIC
            severity = Severity.MODERATE
            recommendation = "Manic state: both precisions too high, reduce sensitivity"
            adjustments = {
                "sensory_precision": 0.6,
                "prior_precision": 0.4,
            }

        # Check for schizophrenic mode (Π_μ >> Π_y)
        elif ratio < t.schizo_severe:
            status = Pathology.SCHIZOPHRENIC
            severity = Severity.CRITICAL
            recommendation = (
                "CRITICAL: Severe prior dominance. System ignoring all input. "
                "Dramatically increase sensory precision."
            )
            adjustments = {
                "sensory_precision": 0.7,
                "prior_precision": 0.3,
            }

        elif ratio < t.schizo_threshold:
            status = Pathology.SCHIZOPHRENIC
            # Severity depends on criticality
            if rho > t.rho_supercritical:
                severity = Severity.SEVERE
                recommendation = (
                    "Supercritical + prior-dominated: high hallucination risk. "
                    "Increase sensory precision and reduce temperature."
                )
            else:
                severity = Severity.MODERATE
                recommendation = (
                    "Prior-dominated: system may ignore user input. "
                    "Increase sensory/extrinsic weight."
                )
            adjustments = {
                "sensory_precision": min(0.6, self._sensory_precision * 2),
                "prior_precision": max(0.2, self._prior_precision * 0.5),
            }

        # Check for ASD mode (Π_y >> Π_μ)
        elif ratio > t.asd_severe:
            status = Pathology.ASD
            severity = Severity.SEVERE
            recommendation = (
                "SEVERE sensory dominance: system overfitting to noise. "
                "Dramatically increase prior precision / goal weight."
            )
            adjustments = {
                "sensory_precision": 0.4,
                "prior_precision": 0.6,
            }

        elif ratio > t.asd_threshold:
            status = Pathology.ASD
            # Severity depends on criticality
            if rho < t.rho_subcritical:
                severity = Severity.SEVERE
                recommendation = (
                    "Subcritical + sensory-dominated: rigid obsessive behavior. "
                    "Increase prior precision and raise temperature."
                )
            else:
                severity = Severity.MODERATE
                recommendation = (
                    "Sensory-dominated: system may get stuck on details. "
                    "Increase prior/intrinsic weight to enable generalization."
                )
            adjustments = {
                "sensory_precision": max(0.3, self._sensory_precision * 0.5),
                "prior_precision": min(0.7, self._prior_precision * 2),
            }

        # Check for mild imbalances in the "healthy" range
        elif ratio < t.healthy_low:
            status = Pathology.HEALTHY
            severity = Severity.MILD
            recommendation = (
                "Slight prior bias. Monitor for increased hallucination."
            )

        elif ratio > t.healthy_high:
            status = Pathology.HEALTHY
            severity = Severity.MILD
            recommendation = (
                "Slight sensory bias. Monitor for obsessive behavior."
            )

        # Check for oscillation (high variance in history)
        if len(self._ratio_history) >= 10:
            import numpy as np
            recent = list(self._ratio_history)[-10:]
            variance = np.var(recent)
            if variance > 2.0:
                if status == Pathology.HEALTHY:
                    status = Pathology.UNSTABLE
                    severity = Severity.MODERATE
                recommendation += " [UNSTABLE: high precision variance detected]"

        # Compute E(λ) estimate
        # E(λ) ≈ log(ρ) - at criticality this should be ~0
        import math
        lambda_estimate = math.log(max(0.01, rho))

        diagnosis = Diagnosis(
            status=status,
            severity=severity,
            sensory_precision=self._sensory_precision,
            prior_precision=self._prior_precision,
            ratio=ratio,
            rho=rho,
            lambda_estimate=lambda_estimate,
            recommendation=recommendation,
            suggested_adjustments=adjustments,
        )

        self._last_diagnosis = diagnosis
        self._diagnosis_history.append(diagnosis)

        return diagnosis

    def suggest_rebalance(self) -> Dict[str, float]:
        """
        Get suggested weight adjustments to restore balance.

        Returns:
            Dict with "sensory_precision" and "prior_precision" targets
        """
        diagnosis = self.diagnose()
        if diagnosis.suggested_adjustments:
            return diagnosis.suggested_adjustments

        # Default: return to balanced state
        return {
            "sensory_precision": 0.5,
            "prior_precision": 0.5,
        }

    def get_trend(self) -> Tuple[str, float]:
        """
        Analyze trend in precision ratio.

        Returns:
            (direction, slope): "stable", "rising", or "falling" with magnitude
        """
        if len(self._ratio_history) < 5:
            return "stable", 0.0

        import numpy as np
        recent = list(self._ratio_history)
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        if abs(slope) < 0.01:
            return "stable", slope
        elif slope > 0:
            return "rising", slope  # Moving toward ASD
        else:
            return "falling", slope  # Moving toward schizophrenia

    def get_statistics(self) -> Dict[str, Any]:
        """Get diagnostic statistics."""
        if not self._ratio_history:
            return {"n_samples": 0}

        import numpy as np
        ratios = list(self._ratio_history)

        # Count pathologies
        pathology_counts = {}
        for d in self._diagnosis_history:
            name = d.status.name
            pathology_counts[name] = pathology_counts.get(name, 0) + 1

        trend, slope = self.get_trend()

        return {
            "n_samples": len(ratios),
            "current_ratio": self.ratio,
            "mean_ratio": np.mean(ratios),
            "std_ratio": np.std(ratios),
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "trend": trend,
            "trend_slope": slope,
            "pathology_counts": pathology_counts,
            "current_rho": self._rho,
        }

    def format_status(self) -> str:
        """Format current status as human-readable string."""
        diagnosis = self.diagnose()
        trend, slope = self.get_trend()

        lines = [
            "=" * 50,
            "PRECISION DIAGNOSTICS",
            "=" * 50,
            f"Status:    {diagnosis.status.name}",
            f"Severity:  {diagnosis.severity.name}",
            f"Π_y / Π_μ: {diagnosis.ratio:.3f}",
            f"  Π_y (sensory): {diagnosis.sensory_precision:.3f}",
            f"  Π_μ (prior):   {diagnosis.prior_precision:.3f}",
            f"ρ (branching):   {diagnosis.rho:.3f}",
            f"E(λ) estimate:   {diagnosis.lambda_estimate:.3f}",
            f"Trend:           {trend} (slope={slope:.4f})",
            "-" * 50,
            f"Recommendation: {diagnosis.recommendation}",
        ]

        if diagnosis.suggested_adjustments:
            lines.append("-" * 50)
            lines.append("Suggested adjustments:")
            for k, v in diagnosis.suggested_adjustments.items():
                lines.append(f"  {k}: {v:.3f}")

        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# Integration Helper
# =============================================================================

def diagnose_from_config(
    extrinsic_weight: float,
    intrinsic_weight: float,
    rho: float = 0.8,
) -> Diagnosis:
    """
    Quick diagnosis from active inference config values.

    This is a convenience function for one-off checks.

    Args:
        extrinsic_weight: Controller's extrinsic (goal) weight
        intrinsic_weight: Controller's intrinsic (curiosity) weight
        rho: Current branching ratio

    Returns:
        Diagnosis result

    Example:
        from ara.gutc.precision_diagnostics import diagnose_from_config

        diagnosis = diagnose_from_config(
            extrinsic_weight=controller.config.extrinsic_weight,
            intrinsic_weight=controller.config.intrinsic_weight,
            rho=criticality_monitor.rho,
        )

        if diagnosis.status != Pathology.HEALTHY:
            print(f"WARNING: {diagnosis.recommendation}")
    """
    monitor = PrecisionMonitor()
    monitor.update(extrinsic_weight, intrinsic_weight, rho)
    return monitor.diagnose()


# =============================================================================
# Tests
# =============================================================================

def test_precision_diagnostics():
    """Test precision diagnostic system."""
    print("Testing Precision Diagnostics")
    print("=" * 60)

    monitor = PrecisionMonitor()

    # Test cases
    test_cases = [
        # (sensory, prior, rho, expected_status)
        (0.5, 0.5, 0.8, Pathology.HEALTHY),      # Balanced
        (0.1, 0.9, 0.8, Pathology.SCHIZOPHRENIC),  # Prior-dominated
        (0.9, 0.1, 0.8, Pathology.ASD),           # Sensory-dominated
        (0.05, 0.95, 1.3, Pathology.SCHIZOPHRENIC),  # Supercritical + schizo
        (0.95, 0.05, 0.4, Pathology.ASD),         # Subcritical + ASD
        (0.02, 0.02, 0.8, Pathology.DISSOCIATIVE),  # Both low
    ]

    for sensory, prior, rho, expected in test_cases:
        monitor.update(sensory, prior, rho)
        diagnosis = monitor.diagnose(force=True)

        status_ok = "OK" if diagnosis.status == expected else "FAIL"
        print(f"\n[{status_ok}] Π_y={sensory:.2f}, Π_μ={prior:.2f}, ρ={rho:.2f}")
        print(f"     Expected: {expected.name}")
        print(f"     Got:      {diagnosis.status.name} ({diagnosis.severity.name})")
        print(f"     Ratio:    {diagnosis.ratio:.2f}")
        print(f"     Rec:      {diagnosis.recommendation[:60]}...")

    # Test trend detection
    print("\n" + "=" * 60)
    print("Testing Trend Detection:")
    monitor = PrecisionMonitor()

    # Simulate rising trend (toward ASD)
    for i in range(20):
        sensory = 0.3 + i * 0.02
        prior = 0.5 - i * 0.01
        monitor.update(sensory, prior, 0.8)

    trend, slope = monitor.get_trend()
    print(f"Rising simulation: trend={trend}, slope={slope:.4f}")

    # Print stats
    print("\n" + "=" * 60)
    print("Statistics:")
    stats = monitor.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Print formatted status
    print("\n" + monitor.format_status())

    print("\n" + "=" * 60)
    print("Precision diagnostics tests complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_precision_diagnostics()
