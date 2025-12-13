#!/usr/bin/env python3
"""
Criticality Monitor - Edge-of-Chaos Dynamics for Ara
=====================================================

Implements the Information-Criticality framework:

    E(θ) = 1 - ρ(W)  (edge function, distance from criticality)
    g(θ) ~ |E(θ)|^(-γ)  (Fisher information metric singularity)
    R_eff(θ) ~ |E(θ)|^(-β)  (curvature proxy singularity)

where:
    γ = ν(1-α)     (Fisher exponent)
    β = γ + 2      (curvature exponent, diverges faster)

Key insight: Near criticality (E→0), both Fisher information and curvature
diverge, but curvature diverges FASTER (β > γ). This creates a "reframing"
regime where the system is maximally sensitive to inputs.

MEIS Strategy:
- Phase I (Explore): Push λ toward E≈0 for high sensitivity, rapid learning
- Phase II (Consolidate): Retreat to E<-ε for stable weight updates

The λ parameter (adrenaline) is a global gain that modulates proximity to
the critical manifold.

Theory:
    - Correlation length: ξ(θ) ~ |E(θ)|^(-ν)
    - Auto-correlation: C(τ) ~ τ^(-α) exp(-τ/ξ)
    - Fisher metric: g(θ) ~ ξ^(1-α) ~ |E(θ)|^(-γ)
    - Curvature proxy: R_eff(θ) ~ g^(-1)(∂g)² ~ |E(θ)|^(-β)

GEOMETRIC CAVEAT:
    A 1D Riemannian manifold has zero intrinsic scalar curvature.
    R_eff is a "curvature-like sensitivity functional" built from the
    metric and its derivatives, not the true Riemann curvature. In higher-
    dimensional parameter spaces, the full Ricci scalar inherits the same
    leading divergence. For Ara's purposes, R_eff captures "how sharply
    the metric changes" which is the relevant control signal.

Usage:
    from ara.cognition.criticality import CriticalityMonitor

    monitor = CriticalityMonitor()
    state = monitor.update(weight_matrix)

    # Both metrics available:
    print(f"Fisher info: {state.fisher_info}")
    print(f"Curvature proxy: {state.curvature_proxy}")

References:
    - Langton (1990) - Computation at the Edge of Chaos
    - Bertschinger & Natschläger (2004) - Real-Time Computation at the Edge of Chaos
    - Toyoizumi et al. (2011) - Beyond the edge of chaos
    - Amari (1998) - Natural Gradient Works Efficiently in Learning
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


class CriticalityBand(str, Enum):
    """
    Traffic-light bands for MEIS mode selection.

    Maps criticality state to operational guidance:
    - GREEN: Safe for exploration, stable regime
    - AMBER: At criticality, high sensitivity, proceed with care
    - RED: Supercritical/unstable, must consolidate immediately
    """
    GREEN = "green"    # E < -ε/2 : Subcritical, stable, safe for agentic work
    AMBER = "amber"    # -ε/2 ≤ E ≤ ε : Near criticality, heightened awareness
    RED = "red"        # E > ε : Supercritical, must retreat


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

    Contains edge function E(θ), Fisher information, curvature proxy,
    and control signals.

    Two key metrics:
    - fisher_info: g(θ) ~ |E|^(-γ) - the Fisher information metric
    - curvature_proxy: R_eff(θ) ~ |E|^(-β) - curvature-like sensitivity

    Note: β = γ + 2, so curvature_proxy diverges faster than fisher_info.
    """
    # Edge function E(θ) = 1 - ρ(W)
    # E < 0: subcritical (stable)
    # E = 0: critical (edge of chaos)
    # E > 0: supercritical (chaotic)
    edge_distance: float = 0.0

    # Spectral radius ρ(W)
    spectral_radius: float = 0.0

    # Fisher information metric g(θ) ~ |E|^(-γ)
    fisher_info: float = 0.0

    # Curvature proxy R_eff(θ) ~ |E|^(-β) where β = γ + 2
    # This is NOT true Riemann curvature (1D manifolds have R=0)
    # but a sensitivity functional capturing "how sharply g changes"
    curvature_proxy: float = 0.0

    # Legacy alias for backward compatibility
    @property
    def curvature(self) -> float:
        """Alias for curvature_proxy (backward compatibility)."""
        return self.curvature_proxy

    # Regime classification
    regime: CriticalityRegime = CriticalityRegime.SUBCRITICAL

    # Traffic-light band for MEIS (Green/Amber/Red)
    band: CriticalityBand = CriticalityBand.GREEN

    # Control signals
    recommended_phase: Phase = Phase.MAINTAIN
    lambda_adjustment: float = 0.0  # Suggested change to adrenaline
    recommended_meis_mode: str = "support"  # Loose coupling to MEIS

    # History
    curvature_trend: float = 0.0    # Rate of change of curvature_proxy
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
            "fisher_info": self.fisher_info,
            "curvature_proxy": self.curvature_proxy,
            "curvature": self.curvature_proxy,  # Legacy alias
            "regime": self.regime.value,
            "band": self.band.value,
            "recommended_phase": self.recommended_phase.value,
            "recommended_meis_mode": self.recommended_meis_mode,
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

    # Curvature thresholds (for R_eff, not g)
    curvature_high: float = 100.0      # Trigger consolidation (R_eff)
    curvature_low: float = 1.0         # Trigger exploration (R_eff)

    # Lambda adjustment rate
    lambda_rate: float = 0.01          # How fast to adjust adrenaline

    # History window
    history_size: int = 100

    # Scaling exponents (physics parameters)
    nu: float = 1.0                    # Correlation length exponent
    alpha: float = 0.5                 # Auto-correlation decay exponent

    @property
    def gamma(self) -> float:
        """Fisher metric exponent: γ = ν(1-α)"""
        return self.nu * (1.0 - self.alpha)

    @property
    def beta(self) -> float:
        """Curvature proxy exponent: β = γ + 2 (diverges faster than Fisher)"""
        return self.gamma + 2.0


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


def estimate_fisher_info(
    edge_distance: float,
    gamma: float = 0.5,
    regularize: float = 0.01,
) -> float:
    """
    Estimate Fisher information metric g(θ) from edge distance.

    g(θ) ~ |E(θ)|^(-γ)

    where γ = ν(1-α) from correlation scaling.
    The regularization prevents division by zero near criticality.

    Args:
        edge_distance: E(θ) = 1 - ρ(W)
        gamma: Fisher exponent (default 0.5 for ν=1, α=0.5)
        regularize: Small constant to avoid singularity

    Returns:
        Estimated Fisher information metric value
    """
    abs_e = abs(edge_distance) + regularize
    return abs_e ** (-gamma)


def estimate_curvature_proxy(
    edge_distance: float,
    beta: float = 2.5,
    regularize: float = 0.01,
) -> float:
    """
    Estimate curvature proxy R_eff(θ) from edge distance.

    R_eff(θ) ~ |E(θ)|^(-β)

    where β = γ + 2 (curvature diverges faster than Fisher).

    GEOMETRIC NOTE:
    This is NOT the true Riemann scalar curvature (which is zero for 1D
    manifolds). It's a curvature-like sensitivity functional:

        R_eff ~ g^(-1) (∂g)² ~ |E|^(γ) · |E|^(-2γ-2) = |E|^(-γ-2) = |E|^(-β)

    This captures "how sharply the metric changes" which is the relevant
    control signal for MEIS.

    Args:
        edge_distance: E(θ) = 1 - ρ(W)
        beta: Curvature exponent β = γ + 2 (default 2.5 for γ=0.5)
        regularize: Small constant to avoid singularity

    Returns:
        Estimated curvature proxy value
    """
    abs_e = abs(edge_distance) + regularize
    return abs_e ** (-beta)


# Legacy alias
def estimate_curvature(
    edge_distance: float,
    gamma: float = 0.5,
    regularize: float = 0.01,
) -> float:
    """Legacy alias for estimate_fisher_info (backward compatibility)."""
    return estimate_fisher_info(edge_distance, gamma, regularize)


# =============================================================================
# Cheap Fisher Proxy (Empirical Estimation)
# =============================================================================

class FisherProxy:
    """
    Cheap Fisher information proxy from gradient samples.

    Instead of computing the full Fisher information matrix (expensive),
    we estimate Tr(F) from the empirical second moment of gradients:

        Tr(F) ≈ E[||∇ log p(x|θ)||²]
              ≈ (1/B) Σ_i ||g_i||²

    where g_i are per-sample gradients and B is batch size.

    This gives us S(θ), the "sensitivity" used for:
    - Fisher-aware step size: η_eff = η_0 / (1 + k√S)
    - Criticality regularizer: β(log S - log S*)²

    Usage:
        proxy = FisherProxy(ema_decay=0.99)

        # During training, feed gradient samples
        for batch in dataloader:
            grads = compute_gradients(batch)
            fisher_estimate = proxy.update(grads)

        # Get current estimate
        S = proxy.get_sensitivity()
    """

    def __init__(
        self,
        ema_decay: float = 0.99,
        warmup_samples: int = 100,
        clip_max: float = 1e6,
    ):
        """
        Initialize Fisher proxy.

        Args:
            ema_decay: Exponential moving average decay for smoothing
            warmup_samples: Samples before estimate is considered valid
            clip_max: Maximum sensitivity value (prevents explosion)
        """
        self.ema_decay = ema_decay
        self.warmup_samples = warmup_samples
        self.clip_max = clip_max

        # Running statistics
        self._ema_fisher: float = 1.0  # Tr(F) estimate
        self._sample_count: int = 0
        self._history: deque = deque(maxlen=1000)

    def update(self, gradients: np.ndarray) -> float:
        """
        Update Fisher estimate with new gradient sample(s).

        Args:
            gradients: Gradient array, shape (batch, params) or (params,)

        Returns:
            Current Fisher trace estimate
        """
        grads = np.atleast_2d(gradients)

        # Compute ||g||² for each sample
        grad_norms_sq = np.sum(grads ** 2, axis=1)

        # Batch mean
        batch_fisher = float(np.mean(grad_norms_sq))

        # Update EMA
        if self._sample_count == 0:
            self._ema_fisher = batch_fisher
        else:
            self._ema_fisher = (
                self.ema_decay * self._ema_fisher +
                (1 - self.ema_decay) * batch_fisher
            )

        self._sample_count += len(grads)
        self._history.append(batch_fisher)

        # Clip to prevent explosion
        self._ema_fisher = min(self._ema_fisher, self.clip_max)

        return self._ema_fisher

    def update_from_norms(self, grad_norms: np.ndarray) -> float:
        """
        Update from pre-computed gradient norms (more efficient).

        Args:
            grad_norms: Array of ||g_i|| values

        Returns:
            Current Fisher trace estimate
        """
        norms = np.asarray(grad_norms)
        norms_sq = norms ** 2
        batch_fisher = float(np.mean(norms_sq))

        if self._sample_count == 0:
            self._ema_fisher = batch_fisher
        else:
            self._ema_fisher = (
                self.ema_decay * self._ema_fisher +
                (1 - self.ema_decay) * batch_fisher
            )

        self._sample_count += len(norms)
        self._history.append(batch_fisher)
        self._ema_fisher = min(self._ema_fisher, self.clip_max)

        return self._ema_fisher

    def get_sensitivity(self) -> float:
        """
        Get current sensitivity S(θ) = Tr(F).

        Returns 1.0 during warmup to avoid instability.
        """
        if self._sample_count < self.warmup_samples:
            return 1.0
        return self._ema_fisher

    def get_effective_lr(self, base_lr: float, scale_factor: float = 0.1) -> float:
        """
        Compute Fisher-aware effective learning rate.

        η_eff = η_0 / (1 + k√S)

        Near criticality (high S), this reduces the learning rate
        to prevent overshooting in the high-curvature region.

        Args:
            base_lr: Base learning rate η_0
            scale_factor: Scaling factor k

        Returns:
            Effective learning rate
        """
        S = self.get_sensitivity()
        return base_lr / (1.0 + scale_factor * np.sqrt(S))

    def is_warmed_up(self) -> bool:
        """Check if enough samples for valid estimate."""
        return self._sample_count >= self.warmup_samples

    def get_statistics(self) -> Dict[str, Any]:
        """Get proxy statistics."""
        history = list(self._history)
        return {
            "sensitivity": self.get_sensitivity(),
            "sample_count": self._sample_count,
            "warmed_up": self.is_warmed_up(),
            "history_mean": float(np.mean(history)) if history else 0.0,
            "history_std": float(np.std(history)) if history else 0.0,
        }


# =============================================================================
# Criticality Regularizer
# =============================================================================

class CriticalityRegularizer:
    """
    Regularizer for criticality-aware training.

    Adds two terms to the training loss:

        L_total = L_task + α·E² + β·(log S - log S*)²

    where:
    - E = 1 - ρ(W) is the edge distance
    - S = Tr(F) is the Fisher sensitivity
    - S* is the target sensitivity
    - α, β are regularization strengths

    The first term (α·E²) penalizes deviation from criticality (E=0).
    The second term (β·(log S - log S*)²) keeps sensitivity near target.

    Together they implement "criticality-regularized learning" that:
    - Keeps the system near the edge of chaos for maximum expressivity
    - Prevents runaway sensitivity that leads to instability

    Usage:
        reg = CriticalityRegularizer(alpha=0.01, beta=0.1, target_sensitivity=10.0)

        # During training
        loss_task = compute_task_loss(...)
        reg_loss = reg.compute_regularization(
            edge_distance=E,
            sensitivity=S,
        )
        total_loss = loss_task + reg_loss
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.1,
        target_sensitivity: float = 10.0,
        edge_target: float = 0.0,
        warmup_steps: int = 1000,
    ):
        """
        Initialize criticality regularizer.

        Args:
            alpha: Weight for edge distance penalty (α·E²)
            beta: Weight for sensitivity penalty (β·(log S - log S*)²)
            target_sensitivity: Target sensitivity S*
            edge_target: Target edge distance (0 = criticality)
            warmup_steps: Steps before regularization is fully active
        """
        self.alpha = alpha
        self.beta = beta
        self.target_sensitivity = target_sensitivity
        self.edge_target = edge_target
        self.warmup_steps = warmup_steps

        self._step = 0
        self._history: List[Dict[str, float]] = []

    def compute_regularization(
        self,
        edge_distance: float,
        sensitivity: float,
    ) -> float:
        """
        Compute regularization loss.

        L_reg = α·(E - E*)² + β·(log S - log S*)²

        Args:
            edge_distance: Current E(θ) = 1 - ρ(W)
            sensitivity: Current S(θ) = Tr(F)

        Returns:
            Regularization loss value
        """
        self._step += 1

        # Warmup ramp (linear)
        warmup_factor = min(1.0, self._step / self.warmup_steps)

        # Edge penalty: keep near criticality
        edge_loss = self.alpha * (edge_distance - self.edge_target) ** 2

        # Sensitivity penalty: keep log S near log S*
        # Use log scale because sensitivity can span orders of magnitude
        log_S = np.log(max(sensitivity, 1e-8))
        log_S_star = np.log(self.target_sensitivity)
        sensitivity_loss = self.beta * (log_S - log_S_star) ** 2

        # Total regularization
        reg_loss = warmup_factor * (edge_loss + sensitivity_loss)

        # Track history
        self._history.append({
            "step": self._step,
            "edge_loss": edge_loss,
            "sensitivity_loss": sensitivity_loss,
            "total_reg": reg_loss,
            "warmup_factor": warmup_factor,
        })
        if len(self._history) > 10000:
            self._history = self._history[-10000:]

        return float(reg_loss)

    def compute_gradients(
        self,
        edge_distance: float,
        sensitivity: float,
    ) -> Tuple[float, float]:
        """
        Compute gradients of regularization loss.

        Returns (∂L/∂E, ∂L/∂S) for manual gradient injection.
        """
        warmup_factor = min(1.0, self._step / self.warmup_steps)

        # ∂L/∂E = 2α(E - E*)
        dL_dE = warmup_factor * 2 * self.alpha * (edge_distance - self.edge_target)

        # ∂L/∂S = 2β(log S - log S*) / S
        log_S = np.log(max(sensitivity, 1e-8))
        log_S_star = np.log(self.target_sensitivity)
        dL_dS = warmup_factor * 2 * self.beta * (log_S - log_S_star) / max(sensitivity, 1e-8)

        return float(dL_dE), float(dL_dS)

    def get_statistics(self) -> Dict[str, Any]:
        """Get regularizer statistics."""
        if not self._history:
            return {"step": 0, "history_length": 0}

        recent = self._history[-100:]
        return {
            "step": self._step,
            "alpha": self.alpha,
            "beta": self.beta,
            "target_sensitivity": self.target_sensitivity,
            "recent_edge_loss_mean": np.mean([h["edge_loss"] for h in recent]),
            "recent_sensitivity_loss_mean": np.mean([h["sensitivity_loss"] for h in recent]),
            "recent_total_reg_mean": np.mean([h["total_reg"] for h in recent]),
            "warmed_up": self._step >= self.warmup_steps,
        }


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


def classify_band(
    edge_distance: float,
    epsilon: float = 0.05,
) -> CriticalityBand:
    """
    Classify criticality into Green/Amber/Red bands for MEIS.

    Band boundaries:
    - GREEN: E < -ε/2 (comfortably subcritical)
    - AMBER: -ε/2 ≤ E ≤ ε (approaching or at criticality)
    - RED: E > ε (supercritical, unstable)

    These map directly to MEIS operational modes:
    - GREEN → AGENTIC allowed (with consent)
    - AMBER → SUPPORT mode (careful exploration)
    - RED → DAMP mode (must consolidate)
    """
    if edge_distance > epsilon:
        return CriticalityBand.RED
    elif edge_distance >= -epsilon / 2:
        return CriticalityBand.AMBER
    else:
        return CriticalityBand.GREEN


def band_to_meis_mode(band: CriticalityBand) -> str:
    """
    Map criticality band to recommended MEIS mode.

    Returns mode name string for loose coupling.
    """
    if band == CriticalityBand.GREEN:
        return "agentic"  # Safe for autonomous work
    elif band == CriticalityBand.AMBER:
        return "support"  # Careful, heightened awareness
    else:  # RED
        return "damp"     # Must retreat, reduce output


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

        Computes both:
        - Fisher information g(θ) ~ |E|^(-γ)
        - Curvature proxy R_eff(θ) ~ |E|^(-β) where β = γ + 2

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

        # Compute edge distance E(θ) = 1 - ρ(W)
        edge = compute_edge_distance(rho_scaled)

        # Estimate Fisher information g(θ) ~ |E|^(-γ)
        fisher = estimate_fisher_info(edge, self.config.gamma)

        # Estimate curvature proxy R_eff(θ) ~ |E|^(-β) where β = γ + 2
        r_eff = estimate_curvature_proxy(edge, self.config.beta)

        # Classify regime
        regime = classify_regime(edge, self.config.epsilon)

        # Classify band (Green/Amber/Red for MEIS)
        band = classify_band(edge, self.config.epsilon)
        meis_mode = band_to_meis_mode(band)

        # Update history (track curvature proxy for control decisions)
        self._edge_history.append(edge)
        self._curvature_history.append(r_eff)

        # Compute trend
        curvature_trend = self._compute_trend(self._curvature_history)

        # Compute stability score (uses curvature proxy)
        stability = self._compute_stability(edge, r_eff)

        # Determine recommended phase and lambda adjustment (uses R_eff)
        phase, lambda_adj = self._determine_phase(edge, r_eff, regime)

        # Build state with both metrics
        state = CriticalityState(
            edge_distance=edge,
            spectral_radius=rho_scaled,
            fisher_info=fisher,
            curvature_proxy=r_eff,
            regime=regime,
            band=band,
            recommended_phase=phase,
            recommended_meis_mode=meis_mode,
            lambda_adjustment=lambda_adj,
            curvature_trend=curvature_trend,
            stability_score=stability,
        )

        self._current_state = state

        # Log significant changes
        if regime == CriticalityRegime.CRITICAL:
            logger.debug(f"At criticality: E={edge:.4f}, g={fisher:.2f}, R_eff={r_eff:.2f}")
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
    ) -> Dict[str, Any]:
        """
        Simulate Fisher and curvature proxy spikes as λ sweeps through criticality.

        Demonstrates both scaling laws:
        - Fisher info: g(θ) ~ |E(θ)|^(-γ)
        - Curvature proxy: R_eff(θ) ~ |E(θ)|^(-β) where β = γ + 2

        Returns dict with:
        - lambda_values, spectral_radii, edge_distances
        - fisher_infos (g), curvature_proxies (R_eff)
        - critical_lambda, exponents (gamma, beta)
        """
        logger.info(f"Simulating RNN criticality sweep (n={n_neurons}, λ∈{lambda_range})")
        logger.info(f"Exponents: γ={self.config.gamma:.2f}, β={self.config.beta:.2f}")

        # Generate random RNN weight matrix with ρ(W) ≈ 0.9
        rng = np.random.default_rng(42)
        W_base = rng.standard_normal((n_neurons, n_neurons)) / np.sqrt(n_neurons)

        # Scale to have spectral radius ≈ 0.9
        rho_base = compute_spectral_radius(W_base)
        W_base = W_base * (0.9 / rho_base)

        # Sweep lambda
        lambda_values = np.linspace(lambda_range[0], lambda_range[1], steps)
        edge_distances = []
        fisher_infos = []
        curvature_proxies = []
        spectral_radii = []

        for lam in lambda_values:
            W_scaled = W_base * lam
            rho = compute_spectral_radius(W_scaled)
            edge = compute_edge_distance(rho)

            # Compute both metrics
            fisher = estimate_fisher_info(edge, self.config.gamma)
            r_eff = estimate_curvature_proxy(edge, self.config.beta)

            spectral_radii.append(rho)
            edge_distances.append(edge)
            fisher_infos.append(fisher)
            curvature_proxies.append(r_eff)

        # Find criticality point
        critical_idx = np.argmin(np.abs(edge_distances))
        critical_lambda = float(lambda_values[critical_idx])

        # Log results
        logger.info(f"Criticality at λ≈{critical_lambda:.3f}")
        logger.info(f"  Peak Fisher info g: {max(fisher_infos):.2f}")
        logger.info(f"  Peak curvature R_eff: {max(curvature_proxies):.2f}")
        logger.info(f"  R_eff/g ratio at peak: {max(curvature_proxies)/max(fisher_infos):.1f}x")

        return {
            "lambda_values": lambda_values.tolist(),
            "spectral_radii": spectral_radii,
            "edge_distances": edge_distances,
            "fisher_infos": fisher_infos,
            "curvature_proxies": curvature_proxies,
            "curvatures": curvature_proxies,  # Legacy alias
            "critical_lambda": critical_lambda,
            "gamma": self.config.gamma,
            "beta": self.config.beta,
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
            "fisher_info": state.fisher_info if state else None,
            "curvature": state.curvature if state else None,
            "regime": state.regime.value if state else None,
            "band": state.band.value if state else "green",
            "recommended_meis_mode": state.recommended_meis_mode if state else "support",
            "stability_score": state.stability_score if state else 1.0,
            "history_size": len(self._edge_history),
        }

    def status_string(self) -> str:
        """Get status string for monitoring."""
        state = self._current_state
        if not state:
            return "🔵 Criticality: No data"

        # Show both g (Fisher) and R_eff (curvature proxy)
        metrics = f"E={state.edge_distance:.3f}, g={state.fisher_info:.1f}, R={state.curvature_proxy:.1f}"

        # Use band colors (Green/Amber/Red)
        if state.band == CriticalityBand.RED:
            return f"🔴 RED [{state.recommended_meis_mode.upper()}]: {metrics}"
        elif state.band == CriticalityBand.AMBER:
            return f"🟡 AMBER [{state.recommended_meis_mode.upper()}]: {metrics}"
        else:  # GREEN
            return f"🟢 GREEN [{state.recommended_meis_mode.upper()}]: {metrics}"


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
    print("=" * 70)
    print("Criticality Monitor Demo - Fisher & Curvature Singularities")
    print("=" * 70)

    monitor = CriticalityMonitor()

    print(f"\nExponents:")
    print(f"  γ (Fisher):    {monitor.config.gamma:.2f}  →  g(θ) ~ |E|^(-γ)")
    print(f"  β (Curvature): {monitor.config.beta:.2f}  →  R_eff(θ) ~ |E|^(-β)")
    print(f"  Note: β = γ + 2, so curvature diverges faster!")

    # Run simulation
    results = monitor.simulate_rnn_sweep(n_neurons=100, steps=50)

    print(f"\nSweep results:")
    print(f"  λ range: {results['lambda_values'][0]:.2f} to {results['lambda_values'][-1]:.2f}")
    print(f"  Critical λ: {results['critical_lambda']:.3f}")
    print(f"  Peak Fisher g: {max(results['fisher_infos']):.2f}")
    print(f"  Peak R_eff: {max(results['curvature_proxies']):.2f}")
    print(f"  R_eff/g ratio: {max(results['curvature_proxies'])/max(results['fisher_infos']):.1f}x")

    # Show regime transitions
    print(f"\nRegime transitions:")
    prev_regime = None
    for lam, edge in zip(results['lambda_values'], results['edge_distances']):
        regime = classify_regime(edge)
        if regime != prev_regime:
            print(f"  λ={lam:.3f}: {regime.value} (E={edge:.4f})")
            prev_regime = regime

    # Interactive update demo
    print(f"\nInteractive updates (both metrics):")
    print(f"  {'ρ':>5}  {'Regime':12}  {'E':>8}  {'g':>8}  {'R_eff':>10}  Phase")
    print(f"  {'-'*5}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*12}")
    for rho in [0.8, 0.9, 0.95, 1.0, 1.05]:
        state = monitor.update(spectral_radius=rho)
        print(f"  {rho:5.2f}  {state.regime.value:12s}  {state.edge_distance:+8.4f}  "
              f"{state.fisher_info:8.2f}  {state.curvature_proxy:10.2f}  "
              f"{state.recommended_phase.value}")

    print(f"\nFinal status: {monitor.status_string()}")

    # Demonstrate the geometric caveat
    print(f"\n{'='*70}")
    print("GEOMETRIC CAVEAT:")
    print("  R_eff is a 'curvature proxy' not true Riemann curvature (R=0 in 1D).")
    print("  It captures 'how sharply the Fisher metric changes' - the relevant")
    print("  control signal for MEIS to decide explore vs consolidate.")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
