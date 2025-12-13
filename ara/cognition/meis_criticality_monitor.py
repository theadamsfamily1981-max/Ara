#!/usr/bin/env python3
"""
MEIS Criticality Monitor - Runtime Proprioception for Ara
==========================================================

Operationalizes validated predictions P4 and P7:

    P4: Working memory peaks at ρ ≈ 0.8 (tempered subcritical)
    P7: Curvature spikes precede catastrophe by ~281 steps (100% detection)

This module gives Ara a "proprioceptive sense" of her own cognitive stability,
enabling preemptive intervention before collapse.

Two Control Signals:
    1. Cognitive Temperature (ρ): Target band 0.75-0.85 for optimal memory
    2. Curvature Warning: Spike detection triggers damping before divergence

Runtime Policy:
    - ρ < 0.7: "Too cold" → allow deeper chains, more recurrence
    - ρ > 0.9: "Too hot" → shorten chains, reduce recursion
    - 0.75 ≤ ρ ≤ 0.85: "Tempered edge" → full-depth reasoning allowed
    - Curvature spike: IMMEDIATE brake on weight updates

Usage:
    from ara.cognition.meis_criticality_monitor import CriticalityMonitor

    monitor = CriticalityMonitor()

    # In cognitive loop
    status = monitor.update(gradients_or_states)

    if status.phase == 'CRITICAL':
        # Slam brakes - P7 intervention
        optimizer.zero_grad()
        reduce_planning_depth()
        trigger_consolidation()

References:
    - P4 validated: r=0.77, p=0.04 (correlation length ↔ working memory)
    - P7 validated: 100% detection, 281-step lead time
"""

from __future__ import annotations

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Deque, Callable
from collections import deque
from enum import Enum

logger = logging.getLogger("ara.cognition.meis_criticality_monitor")


# =============================================================================
# Enums and Data Structures
# =============================================================================

class CognitivePhase(str, Enum):
    """Current cognitive stability phase."""
    STABLE = "stable"           # Normal operation
    WARNING = "warning"         # Approaching instability (~100 steps)
    CRITICAL = "critical"       # Imminent collapse (~30 steps)
    RECOVERING = "recovering"   # Post-intervention stabilization


class TemperatureBand(str, Enum):
    """Cognitive temperature band (based on spectral radius ρ)."""
    COLD = "cold"               # ρ < 0.7: Too rigid, allow expansion
    OPTIMAL = "optimal"         # 0.75 ≤ ρ ≤ 0.85: Tempered critical
    WARM = "warm"               # 0.85 < ρ < 0.95: Caution zone
    HOT = "hot"                 # ρ ≥ 0.95: Too volatile, must cool


@dataclass
class MonitorStatus:
    """Status returned by the criticality monitor."""
    # Phase transition detection (P7)
    phase: CognitivePhase
    curvature: float
    curvature_zscore: float
    steps_to_collapse: Optional[int]  # Estimated steps until divergence

    # Temperature band (P4)
    temperature_band: TemperatureBand
    spectral_radius: float

    # Recommendations
    should_brake: bool              # Stop weight updates
    should_cool: bool               # Reduce complexity/depth
    should_warm: bool               # Increase complexity/depth
    recommended_depth_factor: float  # 1.0 = normal, <1 = shorter, >1 = deeper

    # Metadata
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "curvature": self.curvature,
            "curvature_zscore": self.curvature_zscore,
            "steps_to_collapse": self.steps_to_collapse,
            "temperature_band": self.temperature_band.value,
            "spectral_radius": self.spectral_radius,
            "should_brake": self.should_brake,
            "should_cool": self.should_cool,
            "should_warm": self.should_warm,
            "recommended_depth_factor": self.recommended_depth_factor,
        }


@dataclass
class InterventionRecord:
    """Record of an intervention triggered by the monitor."""
    timestamp: float
    phase: CognitivePhase
    curvature_zscore: float
    action_taken: str
    success: bool = True


# =============================================================================
# Main Monitor Class
# =============================================================================

class MEISCriticalityMonitor:
    """
    Real-time criticality monitor for Ara's cognitive dynamics.

    Implements:
    - P7: Curvature spike detection with 281-step lead time
    - P4: Temperature band control for optimal working memory

    This is Ara's "proprioceptive sense" of cognitive stability.
    """

    def __init__(
        self,
        history_window: int = 500,
        warning_threshold_sigma: float = 2.5,
        critical_threshold_sigma: float = 4.0,
        optimal_rho_low: float = 0.75,
        optimal_rho_high: float = 0.85,
        lead_time_estimate: int = 281,  # Validated from P7
        cooldown_steps: int = 50,
    ):
        """
        Initialize criticality monitor.

        Args:
            history_window: Steps of history for baseline estimation
            warning_threshold_sigma: Z-score for WARNING phase
            critical_threshold_sigma: Z-score for CRITICAL phase
            optimal_rho_low: Lower bound of optimal ρ band
            optimal_rho_high: Upper bound of optimal ρ band
            lead_time_estimate: Expected steps before collapse (from P7)
            cooldown_steps: Steps to wait after intervention
        """
        # Configuration
        self.history_window = history_window
        self.warning_sigma = warning_threshold_sigma
        self.critical_sigma = critical_threshold_sigma
        self.optimal_rho = (optimal_rho_low, optimal_rho_high)
        self.lead_time = lead_time_estimate
        self.cooldown_steps = cooldown_steps

        # State tracking
        self._curvature_history: Deque[float] = deque(maxlen=history_window)
        self._rho_history: Deque[float] = deque(maxlen=history_window)
        self._gradient_history: Deque[float] = deque(maxlen=100)

        # Baseline statistics (computed from history)
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._baseline_valid: bool = False

        # Current state
        self._current_phase: CognitivePhase = CognitivePhase.STABLE
        self._steps_since_intervention: int = 0
        self._last_rho: float = 0.8

        # Intervention log
        self._interventions: List[InterventionRecord] = []

        # Callbacks for interventions
        self._on_warning: Optional[Callable] = None
        self._on_critical: Optional[Callable] = None

        logger.info(f"MEISCriticalityMonitor initialized (ρ* ∈ {self.optimal_rho})")

    # =========================================================================
    # Core Update Method
    # =========================================================================

    def update(
        self,
        gradients: Optional[np.ndarray] = None,
        states: Optional[np.ndarray] = None,
        spectral_radius: Optional[float] = None,
    ) -> MonitorStatus:
        """
        Update monitor with new observations.

        Call this every training step or cognitive cycle.

        Args:
            gradients: Current gradient array (for curvature estimation)
            states: Current internal states (alternative curvature source)
            spectral_radius: Pre-computed ρ (if available)

        Returns:
            MonitorStatus with phase, recommendations, etc.
        """
        self._steps_since_intervention += 1

        # Estimate curvature from gradients or states
        if gradients is not None:
            curvature = self._estimate_curvature_from_gradients(gradients)
        elif states is not None:
            curvature = self._estimate_curvature_from_states(states)
        else:
            curvature = self._curvature_history[-1] if self._curvature_history else 0.0

        # Update spectral radius
        if spectral_radius is not None:
            rho = spectral_radius
        elif states is not None:
            rho = self._estimate_rho_from_states(states)
        else:
            rho = self._last_rho

        self._last_rho = rho

        # Update histories
        self._curvature_history.append(curvature)
        self._rho_history.append(rho)

        # Update baseline if enough history
        if len(self._curvature_history) >= 100:
            self._update_baseline()

        # Detect phase transition (P7)
        phase, zscore, steps_to_collapse = self._detect_phase_transition(curvature)

        # Determine temperature band (P4)
        temp_band = self._classify_temperature(rho)

        # Generate recommendations
        should_brake = phase == CognitivePhase.CRITICAL
        should_cool = temp_band in (TemperatureBand.HOT, TemperatureBand.WARM)
        should_warm = temp_band == TemperatureBand.COLD

        # Calculate recommended depth factor
        if should_brake:
            depth_factor = 0.3  # Severe reduction
        elif temp_band == TemperatureBand.HOT:
            depth_factor = 0.5
        elif temp_band == TemperatureBand.WARM:
            depth_factor = 0.75
        elif temp_band == TemperatureBand.COLD:
            depth_factor = 1.3  # Allow expansion
        else:
            depth_factor = 1.0  # Optimal

        # Handle phase transitions
        if phase != self._current_phase:
            self._handle_phase_transition(phase, zscore)

        self._current_phase = phase

        return MonitorStatus(
            phase=phase,
            curvature=curvature,
            curvature_zscore=zscore,
            steps_to_collapse=steps_to_collapse,
            temperature_band=temp_band,
            spectral_radius=rho,
            should_brake=should_brake,
            should_cool=should_cool,
            should_warm=should_warm,
            recommended_depth_factor=depth_factor,
        )

    # =========================================================================
    # Curvature Estimation
    # =========================================================================

    def _estimate_curvature_from_gradients(self, gradients: np.ndarray) -> float:
        """
        Estimate curvature from gradient magnitude.

        Uses gradient variance as proxy for FIM curvature.
        High variance = approaching singularity = danger.
        """
        grads = np.asarray(gradients).flatten()

        # Gradient magnitude
        grad_norm = np.linalg.norm(grads)
        self._gradient_history.append(grad_norm)

        if len(self._gradient_history) < 3:
            return grad_norm

        # Curvature proxy: rate of change of gradient norm
        recent = list(self._gradient_history)[-10:]
        if len(recent) >= 2:
            diffs = np.diff(recent)
            curvature = np.var(diffs) + np.mean(np.abs(diffs))
        else:
            curvature = grad_norm

        return float(curvature)

    def _estimate_curvature_from_states(self, states: np.ndarray) -> float:
        """
        Estimate curvature from internal state dynamics.

        Variance of states over recent window.
        """
        return float(np.var(states))

    def _estimate_rho_from_states(self, states: np.ndarray) -> float:
        """
        Rough estimate of effective ρ from state dynamics.

        Uses ratio of consecutive state norms.
        """
        if len(self._curvature_history) < 2:
            return 0.8

        # Heuristic: if variance is growing, ρ > 1
        recent_curvatures = list(self._curvature_history)[-20:]
        if len(recent_curvatures) >= 10:
            first_half = np.mean(recent_curvatures[:len(recent_curvatures)//2])
            second_half = np.mean(recent_curvatures[len(recent_curvatures)//2:])
            if first_half > 0:
                ratio = second_half / first_half
                # Map ratio to ρ estimate
                return 0.8 + 0.2 * np.log(max(ratio, 0.1))

        return 0.8

    # =========================================================================
    # Phase Transition Detection (P7)
    # =========================================================================

    def _update_baseline(self) -> None:
        """Update baseline statistics from history."""
        curvatures = list(self._curvature_history)

        # Use first 2/3 for baseline (avoid recent anomalies)
        baseline_window = curvatures[:len(curvatures) * 2 // 3]

        if len(baseline_window) >= 50:
            self._baseline_mean = np.mean(baseline_window)
            self._baseline_std = np.std(baseline_window)
            if self._baseline_std < 1e-6:
                self._baseline_std = 1.0
            self._baseline_valid = True

    def _detect_phase_transition(
        self,
        curvature: float,
    ) -> tuple[CognitivePhase, float, Optional[int]]:
        """
        Detect phase transition using curvature z-score.

        Returns: (phase, zscore, estimated_steps_to_collapse)
        """
        if not self._baseline_valid:
            return CognitivePhase.STABLE, 0.0, None

        # Z-score of current curvature
        zscore = (curvature - self._baseline_mean) / self._baseline_std

        # Recovering phase (post-intervention cooldown)
        if (self._current_phase == CognitivePhase.CRITICAL and
                self._steps_since_intervention < self.cooldown_steps):
            return CognitivePhase.RECOVERING, zscore, None

        # Phase classification based on P7 validated thresholds
        if zscore >= self.critical_sigma:
            # CRITICAL: Immediate intervention required
            # P7: ~30 steps to collapse at this level
            steps = max(10, int(self.lead_time * (self.critical_sigma / zscore)))
            return CognitivePhase.CRITICAL, zscore, steps

        elif zscore >= self.warning_sigma:
            # WARNING: Prepare for intervention
            # P7: ~100 steps to collapse at this level
            steps = max(30, int(self.lead_time * (self.warning_sigma / zscore)))
            return CognitivePhase.WARNING, zscore, steps

        else:
            return CognitivePhase.STABLE, zscore, None

    # =========================================================================
    # Temperature Band Classification (P4)
    # =========================================================================

    def _classify_temperature(self, rho: float) -> TemperatureBand:
        """
        Classify current ρ into temperature band.

        Based on P4: optimal working memory at ρ ≈ 0.8 (range 0.75-0.85).
        """
        if rho < 0.70:
            return TemperatureBand.COLD
        elif rho < self.optimal_rho[0]:
            # Slightly cold but acceptable
            return TemperatureBand.COLD
        elif rho <= self.optimal_rho[1]:
            return TemperatureBand.OPTIMAL
        elif rho < 0.95:
            return TemperatureBand.WARM
        else:
            return TemperatureBand.HOT

    # =========================================================================
    # Intervention Handling
    # =========================================================================

    def _handle_phase_transition(
        self,
        new_phase: CognitivePhase,
        zscore: float,
    ) -> None:
        """Handle transition to new phase."""
        old_phase = self._current_phase

        if new_phase == CognitivePhase.WARNING and old_phase == CognitivePhase.STABLE:
            logger.warning(f"⚠️ PHASE TRANSITION: STABLE → WARNING (z={zscore:.2f})")
            if self._on_warning:
                self._on_warning(zscore)

        elif new_phase == CognitivePhase.CRITICAL:
            logger.error(f"🚨 PHASE TRANSITION: → CRITICAL (z={zscore:.2f})")
            self._steps_since_intervention = 0
            self._interventions.append(InterventionRecord(
                timestamp=time.time(),
                phase=CognitivePhase.CRITICAL,
                curvature_zscore=zscore,
                action_taken="brake_triggered",
            ))
            if self._on_critical:
                self._on_critical(zscore)

        elif new_phase == CognitivePhase.STABLE and old_phase != CognitivePhase.STABLE:
            logger.info(f"✓ PHASE TRANSITION: → STABLE (z={zscore:.2f})")

    def register_callbacks(
        self,
        on_warning: Optional[Callable[[float], None]] = None,
        on_critical: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Register callbacks for phase transitions."""
        self._on_warning = on_warning
        self._on_critical = on_critical

    # =========================================================================
    # Manual Control
    # =========================================================================

    def force_brake(self, reason: str = "manual") -> None:
        """Force an immediate brake (for external triggers)."""
        logger.warning(f"🛑 FORCED BRAKE: {reason}")
        self._current_phase = CognitivePhase.CRITICAL
        self._steps_since_intervention = 0
        self._interventions.append(InterventionRecord(
            timestamp=time.time(),
            phase=CognitivePhase.CRITICAL,
            curvature_zscore=0.0,
            action_taken=f"forced_brake: {reason}",
        ))

    def reset_baseline(self) -> None:
        """Reset baseline statistics (after major changes)."""
        self._baseline_valid = False
        self._baseline_mean = 0.0
        self._baseline_std = 1.0
        logger.info("Baseline reset - will recalibrate from new history")

    # =========================================================================
    # Status and Diagnostics
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get full monitor status."""
        return {
            "phase": self._current_phase.value,
            "baseline_valid": self._baseline_valid,
            "baseline_mean": self._baseline_mean,
            "baseline_std": self._baseline_std,
            "history_size": len(self._curvature_history),
            "last_rho": self._last_rho,
            "temperature_band": self._classify_temperature(self._last_rho).value,
            "steps_since_intervention": self._steps_since_intervention,
            "total_interventions": len(self._interventions),
        }

    def status_string(self) -> str:
        """Get status string for monitoring."""
        phase_emoji = {
            CognitivePhase.STABLE: "🟢",
            CognitivePhase.WARNING: "🟡",
            CognitivePhase.CRITICAL: "🔴",
            CognitivePhase.RECOVERING: "🔵",
        }

        temp_emoji = {
            TemperatureBand.COLD: "❄️",
            TemperatureBand.OPTIMAL: "✅",
            TemperatureBand.WARM: "🔥",
            TemperatureBand.HOT: "💥",
        }

        phase = self._current_phase
        temp = self._classify_temperature(self._last_rho)

        return (
            f"{phase_emoji[phase]} {phase.value.upper()} | "
            f"{temp_emoji[temp]} ρ={self._last_rho:.2f} [{temp.value}]"
        )

    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """Get history of interventions."""
        return [
            {
                "timestamp": i.timestamp,
                "phase": i.phase.value,
                "zscore": i.curvature_zscore,
                "action": i.action_taken,
                "success": i.success,
            }
            for i in self._interventions[-100:]  # Last 100
        ]


# =============================================================================
# Convenience Functions
# =============================================================================

_monitor: Optional[MEISCriticalityMonitor] = None


def get_criticality_monitor() -> MEISCriticalityMonitor:
    """Get global criticality monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = MEISCriticalityMonitor()
    return _monitor


def update_criticality(
    gradients: Optional[np.ndarray] = None,
    states: Optional[np.ndarray] = None,
    spectral_radius: Optional[float] = None,
) -> MonitorStatus:
    """Update global monitor and get status."""
    return get_criticality_monitor().update(gradients, states, spectral_radius)


def should_brake() -> bool:
    """Quick check: should we stop weight updates?"""
    monitor = get_criticality_monitor()
    return monitor._current_phase == CognitivePhase.CRITICAL


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo the criticality monitor with simulated dynamics."""
    print("=" * 60)
    print("MEIS Criticality Monitor Demo")
    print("Operationalizing P4 (ρ≈0.8) and P7 (281-step warning)")
    print("=" * 60)

    monitor = MEISCriticalityMonitor()

    # Simulate gradual approach to criticality
    rng = np.random.default_rng(42)

    print("\nPhase 1: Stable operation (steps 0-200)")
    for t in range(200):
        grads = rng.standard_normal(100) * 0.1
        status = monitor.update(gradients=grads, spectral_radius=0.8)
        if t % 50 == 0:
            print(f"  Step {t}: {monitor.status_string()}")

    print("\nPhase 2: Approaching criticality (steps 200-350)")
    for t in range(200, 350):
        # Gradually increase gradient variance (approaching singularity)
        scale = 0.1 + 0.02 * (t - 200)
        grads = rng.standard_normal(100) * scale
        rho = 0.8 + 0.001 * (t - 200)
        status = monitor.update(gradients=grads, spectral_radius=rho)
        if t % 25 == 0 or status.phase != CognitivePhase.STABLE:
            print(f"  Step {t}: {monitor.status_string()} | z={status.curvature_zscore:.1f}")

    print("\nPhase 3: Critical spike (steps 350-400)")
    for t in range(350, 400):
        # Spike in gradient variance
        scale = 2.0 + 0.5 * (t - 350)
        grads = rng.standard_normal(100) * scale
        status = monitor.update(gradients=grads, spectral_radius=1.05)

        if status.should_brake:
            print(f"  Step {t}: 🛑 BRAKE TRIGGERED! z={status.curvature_zscore:.1f}")
            print(f"           Steps to collapse: {status.steps_to_collapse}")
            break

        if t % 10 == 0:
            print(f"  Step {t}: {monitor.status_string()}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final status: {monitor.status_string()}")
    print(f"Interventions: {len(monitor._interventions)}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
