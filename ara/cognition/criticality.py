#!/usr/bin/env python3
"""
Criticality Monitor - Edge-of-Chaos Dynamics for Ara
=====================================================

Implements the Information-Criticality framework:

    E(θ) = 1 - ρ(W)  (edge function, distance from criticality)
    J(θ) ~ |E(θ)|^(-γ)  (Fisher information singularity)

Key insight: Near criticality (E→0), Fisher information (sensitivity/curvature)
diverges. This creates a "reframing" regime where the system is maximally
sensitive to inputs and can rapidly adapt.

MEIS Strategy:
- Phase I (Explore): Push λ toward E≈0 for high sensitivity, rapid learning
- Phase II (Consolidate): Retreat to E<-ε for stable weight updates

The λ parameter (adrenaline) is a global gain that modulates proximity to
the critical manifold.

Theory:
    - Correlation length: ξ(θ) ~ |E(θ)|^(-ν)
    - Auto-correlation: C(τ) ~ τ^(-α) exp(-τ/ξ)
    - Fisher info: J(θ) ~ ξ^(1-α) ~ |E(θ)|^(-γ) where γ = ν(1-α)

Usage:
    from ara.cognition.criticality import CriticalityMonitor

    monitor = CriticalityMonitor()

    # Update with current weight matrix
    state = monitor.update(weight_matrix)

    # Get recommended lambda adjustment
    if state.should_explore:
        increase_lambda()
    elif state.should_consolidate:
        decrease_lambda()

References:
    - Langton (1990) - Computation at the Edge of Chaos
    - Bertschinger & Natschläger (2004) - Real-Time Computation at the Edge of Chaos
    - Toyoizumi et al. (2011) - Beyond the edge of chaos: Amplification and
      temporal integration by recurrent networks
"""

from __future__ import annotations

import time
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from collections import deque

import numpy as np

logger = logging.getLogger("ara.cognition.criticality")


# =============================================================================
# Types and Enums
# =============================================================================

class CriticalityRegime(str, Enum):
    """Current regime relative to criticality."""
    SUBCRITICAL = "subcritical"     # E < -ε, stable, low sensitivity
    CRITICAL = "critical"           # |E| < ε, edge of chaos, high sensitivity
    SUPERCRITICAL = "supercritical" # E > ε, unstable, chaotic


class Phase(str, Enum):
    """MEIS phase for criticality-based control."""
    EXPLORE = "explore"         # Move toward criticality for sensitivity
    CONSOLIDATE = "consolidate" # Retreat from criticality for stability
    MAINTAIN = "maintain"       # Hold current position


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CriticalityState:
    """
    Current state of the system relative to criticality.

    Contains edge function E(θ), estimated curvature, and control signals.
    """
    # Edge function E(θ) = 1 - ρ(W)
    # E < 0: subcritical (stable)
    # E = 0: critical (edge of chaos)
    # E > 0: supercritical (chaotic)
    edge_distance: float = 0.0

    # Spectral radius ρ(W)
    spectral_radius: float = 0.0

    # Estimated Fisher information / curvature
    # High value = high sensitivity, near criticality
    curvature: float = 0.0

    # Regime classification
    regime: CriticalityRegime = CriticalityRegime.SUBCRITICAL

    # Control signals
    recommended_phase: Phase = Phase.MAINTAIN
    lambda_adjustment: float = 0.0  # Suggested change to adrenaline

    # History
    curvature_trend: float = 0.0    # Rate of change of curvature
    stability_score: float = 1.0    # Overall stability (0-1)

    # Metadata
    timestamp: float = field(default_factory=time.time)

    @property
    def should_explore(self) -> bool:
        """Should we move toward criticality?"""
        return self.recommended_phase == Phase.EXPLORE

    @property
    def should_consolidate(self) -> bool:
        """Should we retreat from criticality?"""
        return self.recommended_phase == Phase.CONSOLIDATE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_distance": self.edge_distance,
            "spectral_radius": self.spectral_radius,
            "curvature": self.curvature,
            "regime": self.regime.value,
            "recommended_phase": self.recommended_phase.value,
            "lambda_adjustment": self.lambda_adjustment,
            "stability_score": self.stability_score,
            "timestamp": self.timestamp,
        }


@dataclass
class CriticalityConfig:
    """Configuration for criticality monitoring."""
    # Epsilon for critical region |E| < ε
    epsilon: float = 0.05

    # Target edge distance for exploration
    target_edge_explore: float = 0.02  # Close to but not at criticality

    # Safe edge distance for consolidation
    target_edge_safe: float = -0.1     # Comfortably subcritical

    # Curvature thresholds
    curvature_high: float = 10.0       # Trigger consolidation
    curvature_low: float = 0.5         # Trigger exploration

    # Lambda adjustment rate
    lambda_rate: float = 0.01          # How fast to adjust adrenaline

    # History window
    history_size: int = 100

    # Scaling exponents (physics parameters)
    nu: float = 1.0                    # Correlation length exponent
    alpha: float = 0.5                 # Auto-correlation decay exponent

    @property
    def gamma(self) -> float:
        """Fisher info exponent γ = ν(1-α)"""
        return self.nu * (1.0 - self.alpha)


# =============================================================================
# Core Algorithms
# =============================================================================

def compute_spectral_radius(W: np.ndarray) -> float:
    """
    Compute spectral radius ρ(W) = max|eigenvalue|.

    For RNNs, ρ(W) ≈ 1 is the edge of chaos.
    ρ < 1: subcritical, signals decay
    ρ > 1: supercritical, signals explode
    """
    try:
        eigenvalues = np.linalg.eigvals(W)
        return float(np.max(np.abs(eigenvalues)))
    except np.linalg.LinAlgError:
        logger.warning("Eigenvalue computation failed, using Frobenius norm estimate")
        # Fallback: Frobenius norm / sqrt(n) is a rough upper bound
        n = W.shape[0]
        return float(np.linalg.norm(W, 'fro') / np.sqrt(n))


def compute_edge_distance(spectral_radius: float) -> float:
    """
    Compute edge function E(θ) = 1 - ρ(W).

    E < 0: subcritical
    E = 0: critical
    E > 0: supercritical
    """
    return 1.0 - spectral_radius


def estimate_curvature(
    edge_distance: float,
    gamma: float = 0.5,
    regularize: float = 0.01,
) -> float:
    """
    Estimate Fisher information / curvature from edge distance.

    J(θ) ~ |E(θ)|^(-γ)

    The regularization prevents division by zero near criticality.
    """
    abs_e = abs(edge_distance) + regularize
    return abs_e ** (-gamma)


def estimate_correlation_length(
    edge_distance: float,
    nu: float = 1.0,
    regularize: float = 0.01,
) -> float:
    """
    Estimate correlation length ξ(θ) ~ |E(θ)|^(-ν).

    Near criticality, correlations become long-range.
    """
    abs_e = abs(edge_distance) + regularize
    return abs_e ** (-nu)


def classify_regime(
    edge_distance: float,
    epsilon: float = 0.05,
) -> CriticalityRegime:
    """Classify current regime based on edge distance."""
    if edge_distance < -epsilon:
        return CriticalityRegime.SUBCRITICAL
    elif edge_distance > epsilon:
        return CriticalityRegime.SUPERCRITICAL
    else:
        return CriticalityRegime.CRITICAL


# =============================================================================
# Criticality Monitor
# =============================================================================

class CriticalityMonitor:
    """
    Monitors system criticality and recommends λ adjustments.

    This is the interface between the physics of edge-of-chaos
    and MEIS's control logic.
    """

    def __init__(self, config: Optional[CriticalityConfig] = None):
        self.config = config or CriticalityConfig()

        # Current state
        self._current_state: Optional[CriticalityState] = None

        # History for trend analysis
        self._edge_history: deque = deque(maxlen=self.config.history_size)
        self._curvature_history: deque = deque(maxlen=self.config.history_size)

        # Current λ (adrenaline) - starts at neutral
        self._lambda: float = 1.0

        # Phase state
        self._current_phase: Phase = Phase.MAINTAIN
        self._phase_start_time: float = time.time()
        self._phase_duration: float = 0.0

        logger.info(f"CriticalityMonitor initialized (γ={self.config.gamma:.2f})")

    @property
    def lambda_value(self) -> float:
        """Current adrenaline (λ) value."""
        return self._lambda

    @property
    def current_state(self) -> Optional[CriticalityState]:
        """Get current criticality state."""
        return self._current_state

    def update(
        self,
        W: Optional[np.ndarray] = None,
        spectral_radius: Optional[float] = None,
    ) -> CriticalityState:
        """
        Update criticality state with new weight matrix or spectral radius.

        Args:
            W: Weight matrix (will compute spectral radius)
            spectral_radius: Pre-computed spectral radius (if W not provided)

        Returns:
            Current CriticalityState with control recommendations
        """
        # Compute spectral radius
        if W is not None:
            rho = compute_spectral_radius(W)
        elif spectral_radius is not None:
            rho = spectral_radius
        else:
            # No input - return last state or default
            if self._current_state:
                return self._current_state
            rho = 0.9  # Default subcritical

        # Apply lambda scaling
        rho_scaled = rho * self._lambda

        # Compute edge distance
        edge = compute_edge_distance(rho_scaled)

        # Estimate curvature (Fisher info)
        curvature = estimate_curvature(edge, self.config.gamma)

        # Classify regime
        regime = classify_regime(edge, self.config.epsilon)

        # Update history
        self._edge_history.append(edge)
        self._curvature_history.append(curvature)

        # Compute trend
        curvature_trend = self._compute_trend(self._curvature_history)

        # Compute stability score
        stability = self._compute_stability(edge, curvature)

        # Determine recommended phase and lambda adjustment
        phase, lambda_adj = self._determine_phase(edge, curvature, regime)

        # Build state
        state = CriticalityState(
            edge_distance=edge,
            spectral_radius=rho_scaled,
            curvature=curvature,
            regime=regime,
            recommended_phase=phase,
            lambda_adjustment=lambda_adj,
            curvature_trend=curvature_trend,
            stability_score=stability,
        )

        self._current_state = state

        # Log significant changes
        if regime == CriticalityRegime.CRITICAL:
            logger.debug(f"At criticality: E={edge:.4f}, J={curvature:.2f}")
        elif regime == CriticalityRegime.SUPERCRITICAL:
            logger.warning(f"Supercritical! E={edge:.4f}, recommend consolidation")

        return state

    def apply_lambda_adjustment(self, adjustment: Optional[float] = None) -> float:
        """
        Apply lambda adjustment (or use recommended).

        Returns new lambda value.
        """
        if adjustment is None and self._current_state:
            adjustment = self._current_state.lambda_adjustment

        if adjustment:
            old_lambda = self._lambda
            self._lambda = max(0.5, min(2.0, self._lambda + adjustment))
            logger.debug(f"λ adjusted: {old_lambda:.3f} -> {self._lambda:.3f}")

        return self._lambda

    def set_lambda(self, value: float) -> None:
        """Directly set lambda value."""
        self._lambda = max(0.5, min(2.0, value))

    def _compute_trend(self, history: deque) -> float:
        """Compute trend (rate of change) from history."""
        if len(history) < 2:
            return 0.0

        recent = list(history)[-10:]
        if len(recent) < 2:
            return 0.0

        # Simple linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = np.polyfit(x, y, 1)[0]

        return float(slope)

    def _compute_stability(self, edge: float, curvature: float) -> float:
        """
        Compute stability score (0-1).

        High score = stable, far from criticality
        Low score = unstable, near or past criticality
        """
        # Penalize being near/past criticality
        edge_penalty = 1.0 / (1.0 + np.exp(-10 * (edge + 0.1)))

        # Penalize high curvature
        curvature_penalty = 1.0 / (1.0 + curvature / self.config.curvature_high)

        return float(edge_penalty * curvature_penalty)

    def _determine_phase(
        self,
        edge: float,
        curvature: float,
        regime: CriticalityRegime,
    ) -> Tuple[Phase, float]:
        """
        Determine recommended phase and lambda adjustment.

        Returns:
            (phase, lambda_adjustment)
        """
        # If supercritical - always consolidate (pull back)
        if regime == CriticalityRegime.SUPERCRITICAL:
            return Phase.CONSOLIDATE, -self.config.lambda_rate * 2

        # If curvature too high - consolidate
        if curvature > self.config.curvature_high:
            return Phase.CONSOLIDATE, -self.config.lambda_rate

        # If curvature too low and subcritical - explore
        if curvature < self.config.curvature_low and regime == CriticalityRegime.SUBCRITICAL:
            return Phase.EXPLORE, self.config.lambda_rate

        # If at criticality with moderate curvature - maintain
        if regime == CriticalityRegime.CRITICAL:
            # Fine-tune to stay near target
            if edge > self.config.target_edge_explore:
                return Phase.MAINTAIN, -self.config.lambda_rate * 0.5
            elif edge < -self.config.epsilon / 2:
                return Phase.MAINTAIN, self.config.lambda_rate * 0.5
            return Phase.MAINTAIN, 0.0

        # Default: maintain
        return Phase.MAINTAIN, 0.0

    # =========================================================================
    # Simulation / Demo Methods
    # =========================================================================

    def simulate_rnn_sweep(
        self,
        n_neurons: int = 100,
        lambda_range: Tuple[float, float] = (0.5, 1.5),
        steps: int = 50,
    ) -> Dict[str, List[float]]:
        """
        Simulate the Fisher information spike as λ sweeps through criticality.

        This demonstrates the core theory: J(θ) ~ |E(θ)|^(-γ).

        Returns dict with lambda_values, edge_distances, curvatures.
        """
        logger.info(f"Simulating RNN criticality sweep (n={n_neurons}, λ∈{lambda_range})")

        # Generate random RNN weight matrix with ρ(W) ≈ 0.9
        rng = np.random.default_rng(42)
        W_base = rng.standard_normal((n_neurons, n_neurons)) / np.sqrt(n_neurons)

        # Scale to have spectral radius ≈ 0.9
        rho_base = compute_spectral_radius(W_base)
        W_base = W_base * (0.9 / rho_base)

        # Sweep lambda
        lambda_values = np.linspace(lambda_range[0], lambda_range[1], steps)
        edge_distances = []
        curvatures = []
        spectral_radii = []

        for lam in lambda_values:
            W_scaled = W_base * lam
            rho = compute_spectral_radius(W_scaled)
            edge = compute_edge_distance(rho)
            curv = estimate_curvature(edge, self.config.gamma)

            spectral_radii.append(rho)
            edge_distances.append(edge)
            curvatures.append(curv)

        # Find criticality point
        critical_idx = np.argmin(np.abs(edge_distances))
        critical_lambda = lambda_values[critical_idx]

        logger.info(f"Criticality at λ≈{critical_lambda:.3f}, peak curvature={max(curvatures):.2f}")

        return {
            "lambda_values": lambda_values.tolist(),
            "spectral_radii": spectral_radii,
            "edge_distances": edge_distances,
            "curvatures": curvatures,
            "critical_lambda": critical_lambda,
        }

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get full status for monitoring."""
        state = self._current_state

        return {
            "lambda": self._lambda,
            "current_phase": self._current_phase.value,
            "edge_distance": state.edge_distance if state else None,
            "spectral_radius": state.spectral_radius if state else None,
            "curvature": state.curvature if state else None,
            "regime": state.regime.value if state else None,
            "stability_score": state.stability_score if state else 1.0,
            "history_size": len(self._edge_history),
        }

    def status_string(self) -> str:
        """Get status string for monitoring."""
        state = self._current_state
        if not state:
            return "🔵 Criticality: No data"

        if state.regime == CriticalityRegime.SUPERCRITICAL:
            return f"🔴 SUPERCRITICAL: E={state.edge_distance:.3f}, J={state.curvature:.1f}"
        elif state.regime == CriticalityRegime.CRITICAL:
            return f"🟡 CRITICAL: E={state.edge_distance:.3f}, J={state.curvature:.1f}"
        else:
            return f"🟢 Subcritical: E={state.edge_distance:.3f}, J={state.curvature:.1f}"


# =============================================================================
# Singleton and Convenience
# =============================================================================

_monitor: Optional[CriticalityMonitor] = None


def get_criticality_monitor() -> CriticalityMonitor:
    """Get the global criticality monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = CriticalityMonitor()
    return _monitor


def update_criticality(
    W: Optional[np.ndarray] = None,
    spectral_radius: Optional[float] = None,
) -> CriticalityState:
    """Update criticality state using global monitor."""
    return get_criticality_monitor().update(W, spectral_radius)


def criticality_status() -> str:
    """Get criticality status string."""
    return get_criticality_monitor().status_string()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo the criticality monitor with a simulated sweep."""
    print("=" * 60)
    print("Criticality Monitor Demo - Information Singularity")
    print("=" * 60)

    monitor = CriticalityMonitor()

    # Run simulation
    results = monitor.simulate_rnn_sweep(n_neurons=100, steps=50)

    print(f"\nSweep results:")
    print(f"  λ range: {results['lambda_values'][0]:.2f} to {results['lambda_values'][-1]:.2f}")
    print(f"  Critical λ: {results['critical_lambda']:.3f}")
    print(f"  Peak curvature: {max(results['curvatures']):.2f}")

    # Show regime transitions
    print(f"\nRegime transitions:")
    prev_regime = None
    for i, (lam, edge) in enumerate(zip(results['lambda_values'], results['edge_distances'])):
        regime = classify_regime(edge)
        if regime != prev_regime:
            print(f"  λ={lam:.3f}: {regime.value} (E={edge:.4f})")
            prev_regime = regime

    # Interactive update demo
    print(f"\nInteractive updates:")
    for rho in [0.8, 0.9, 0.95, 1.0, 1.05]:
        state = monitor.update(spectral_radius=rho)
        print(f"  ρ={rho:.2f}: {state.regime.value:12s} E={state.edge_distance:+.3f} "
              f"J={state.curvature:6.2f} → {state.recommended_phase.value}")

    print(f"\nFinal status: {monitor.status_string()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
