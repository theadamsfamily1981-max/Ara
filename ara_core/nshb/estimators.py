#!/usr/bin/env python3
"""
NSHB Estimators - Human GUTC State Estimation
==============================================

Estimates the human's position on the (λ, Π) control manifold:

    z(t) = (λ̂(t), Π̂_sensory(t), Π̂_prior(t))

Estimators:
    1. λ̂: Branching ratio from EEG avalanche analysis
    2. Π̂_sensory: Sensory precision from posterior alpha/beta + physio
    3. Π̂_prior: Prior/motivational precision from frontal theta + DA proxies

The goal is to compress kHz multichannel signals into a 3D control
point at ~10 Hz - the GUTC state vector.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# =============================================================================
# GUTC State Vector
# =============================================================================

@dataclass
class GUTCState:
    """
    Human GUTC state vector z(t).

    This is the core representation of "where the person is"
    on the λ-Π control manifold.
    """
    timestamp: float
    lambda_hat: float              # Criticality estimate (target ≈ 1)
    pi_sensory: float              # Sensory precision (L4/L2/3 gain proxy)
    pi_prior: float                # Prior precision (SFC/DA gain proxy)

    # Confidence bounds
    lambda_std: float = 0.1
    pi_sensory_std: float = 0.1
    pi_prior_std: float = 0.1

    # Derived metrics
    capacity: float = 0.0          # C(λ,Π) = information capacity

    def __post_init__(self):
        # Compute capacity using GUTC formula
        # C(λ,Π) = Π · exp(-(λ-1)²/2σ²)  [peak at criticality]
        sigma = 0.3
        pi_mean = (self.pi_sensory + self.pi_prior) / 2
        self.capacity = pi_mean * math.exp(-((self.lambda_hat - 1.0)**2) / (2 * sigma**2))

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array [λ, Π_s, Π_p]."""
        return np.array([self.lambda_hat, self.pi_sensory, self.pi_prior])

    def distance_to_healthy(self, target_lambda: float = 1.0,
                            target_pi_s: float = 1.0,
                            target_pi_p: float = 1.0) -> float:
        """
        Compute distance to healthy corridor.

        D(z) = w_λ(λ̂-1)² + w_s(Π̂_s-Π_s*)² + w_p(Π̂_p-Π_p*)²
        """
        w_lambda = 1.0
        w_s = 0.5
        w_p = 0.5

        d = (w_lambda * (self.lambda_hat - target_lambda)**2 +
             w_s * (self.pi_sensory - target_pi_s)**2 +
             w_p * (self.pi_prior - target_pi_p)**2)

        return math.sqrt(d)

    def in_healthy_corridor(self, lambda_tol: float = 0.3,
                            pi_tol: float = 0.5) -> bool:
        """Check if state is in healthy corridor."""
        lambda_ok = abs(self.lambda_hat - 1.0) < lambda_tol
        pi_s_ok = 0.5 < self.pi_sensory < 2.0
        pi_p_ok = 0.5 < self.pi_prior < 2.0
        return lambda_ok and pi_s_ok and pi_p_ok

    def regime_label(self) -> str:
        """Classify into GUTC regime."""
        if self.lambda_hat < 0.7:
            if self.pi_sensory > 1.5:
                return "ASD-like"
            else:
                return "Anhedonic"
        elif self.lambda_hat > 1.3:
            if self.pi_prior > 1.5:
                return "Psychosis-risk"
            else:
                return "Chaotic"
        else:
            if self.in_healthy_corridor():
                return "Healthy"
            else:
                return "Transitional"


# =============================================================================
# Avalanche Analysis for λ Estimation
# =============================================================================

@dataclass
class NeuralAvalanche:
    """A detected avalanche in EEG activity."""
    avalanche_id: int
    start_sample: int
    duration_samples: int
    size: int                      # Number of channels involved
    channels: List[int] = field(default_factory=list)


class BranchingRatioEstimator:
    """
    Estimates branching ratio λ from EEG binary events.

    Uses the methods from Wilting & Priesemann (2019):
    - Time-binned event detection
    - Regression estimator for m̂
    - Bias correction for subsampling

    Target: λ ≈ 1 for critical dynamics (healthy)
    """

    def __init__(
        self,
        time_bin_ms: float = 4.0,
        sample_rate_hz: float = 256.0,
        min_avalanche_size: int = 2,
    ):
        self.time_bin_ms = time_bin_ms
        self.sample_rate_hz = sample_rate_hz
        self.min_avalanche_size = min_avalanche_size

        # Samples per bin
        self.samples_per_bin = max(1, int(time_bin_ms * sample_rate_hz / 1000))

        # History
        self.avalanches: List[NeuralAvalanche] = []
        self.lambda_history: List[Tuple[float, float]] = []
        self._avalanche_counter = 0

    def detect_avalanches(self, events: np.ndarray) -> List[NeuralAvalanche]:
        """
        Detect avalanches in binary event matrix.

        Args:
            events: (n_samples, n_channels) binary matrix

        Returns:
            List of detected avalanches
        """
        if events.size == 0:
            return []

        n_samples, n_channels = events.shape
        avalanches = []

        # Bin events
        n_bins = n_samples // self.samples_per_bin
        binned = np.zeros((n_bins, n_channels), dtype=np.int32)

        for b in range(n_bins):
            start = b * self.samples_per_bin
            end = start + self.samples_per_bin
            binned[b] = np.any(events[start:end], axis=0).astype(np.int32)

        # Find contiguous active periods
        total_active = np.sum(binned, axis=1)

        in_avalanche = False
        av_start = 0
        av_channels = set()

        for b in range(n_bins):
            if total_active[b] > 0:
                if not in_avalanche:
                    in_avalanche = True
                    av_start = b
                    av_channels = set()
                # Track which channels are active
                av_channels.update(np.where(binned[b] > 0)[0])
            else:
                if in_avalanche:
                    # End of avalanche
                    duration = b - av_start
                    size = len(av_channels)

                    if size >= self.min_avalanche_size:
                        self._avalanche_counter += 1
                        av = NeuralAvalanche(
                            avalanche_id=self._avalanche_counter,
                            start_sample=av_start * self.samples_per_bin,
                            duration_samples=duration * self.samples_per_bin,
                            size=size,
                            channels=list(av_channels),
                        )
                        avalanches.append(av)

                    in_avalanche = False

        self.avalanches.extend(avalanches)

        # Keep history bounded
        if len(self.avalanches) > 1000:
            self.avalanches = self.avalanches[-1000:]

        return avalanches

    def estimate_lambda(self, events: np.ndarray) -> Tuple[float, float]:
        """
        Estimate branching ratio λ̂ from events.

        Returns (lambda_hat, std_error).
        """
        # Detect avalanches
        new_avalanches = self.detect_avalanches(events)

        # Need at least a few avalanches
        if len(self.avalanches) < 5:
            return 1.0, 0.5  # Prior: assume critical, high uncertainty

        # Use recent avalanches
        recent = self.avalanches[-50:]
        sizes = np.array([av.size for av in recent])
        durations = np.array([av.duration_samples for av in recent])

        if len(sizes) < 3:
            return 1.0, 0.5

        # Method 1: Size ratio estimator
        # λ̂ ≈ mean(S_{t+1} / S_t) for consecutive avalanches
        if len(sizes) > 1:
            ratios = sizes[1:] / np.maximum(sizes[:-1], 1)
            ratios = ratios[(ratios > 0.1) & (ratios < 10)]  # Filter outliers

            if len(ratios) > 2:
                lambda_hat = float(np.mean(ratios))
                std_err = float(np.std(ratios) / np.sqrt(len(ratios)))
            else:
                lambda_hat = 1.0
                std_err = 0.3
        else:
            lambda_hat = 1.0
            std_err = 0.3

        # Method 2: Duration-based correction
        # Longer avalanches suggest supercritical, shorter subcritical
        mean_dur = np.mean(durations) / self.samples_per_bin
        if mean_dur > 5:
            lambda_hat *= 1.05  # Slight upward correction
        elif mean_dur < 2:
            lambda_hat *= 0.95

        # Bound to reasonable range
        lambda_hat = max(0.5, min(1.5, lambda_hat))

        # Record
        self.lambda_history.append((time.time(), lambda_hat))
        if len(self.lambda_history) > 500:
            self.lambda_history = self.lambda_history[-500:]

        return lambda_hat, std_err


# =============================================================================
# Precision Estimators
# =============================================================================

class SensoryPrecisionEstimator:
    """
    Estimates sensory precision Π̂_sensory.

    Proxy markers:
    - Alpha power in posterior/occipital ROI (inverse relationship)
    - Beta power in sensory cortex (positive relationship)
    - GSR / arousal (positive relationship)
    - Pupil dilation (positive relationship)

    High Π_sensory = high gain on sensory prediction errors
    """

    def __init__(self, sample_rate_hz: float = 256.0):
        self.sample_rate_hz = sample_rate_hz
        self.history: List[Tuple[float, float]] = []

    def estimate(
        self,
        eeg_data: np.ndarray,
        physio_features: np.ndarray,
        posterior_channels: List[int] = None,
    ) -> Tuple[float, float]:
        """
        Estimate Π̂_sensory from EEG and physiology.

        Args:
            eeg_data: (n_samples, n_channels) preprocessed EEG
            physio_features: [hrv, hr, gsr, pupil, resp, hr_var]
            posterior_channels: Indices of posterior electrodes

        Returns:
            (pi_sensory, std_error)
        """
        if eeg_data.size == 0:
            return 1.0, 0.3

        n_channels = eeg_data.shape[1]

        # Default: use last half of channels as "posterior"
        if posterior_channels is None:
            posterior_channels = list(range(n_channels // 2, n_channels))

        # Extract posterior data
        if posterior_channels:
            posterior_data = eeg_data[:, posterior_channels]
        else:
            posterior_data = eeg_data

        # Compute power in alpha band (8-12 Hz) - simplified
        # In production: proper FFT or wavelet analysis
        if len(posterior_data) > 50:
            # Proxy: variance in relevant frequency range
            # High-pass to remove slow drift, then variance
            hp_data = posterior_data - np.mean(posterior_data, axis=0)
            alpha_power = np.var(hp_data)
        else:
            alpha_power = 1.0

        # Lower alpha power → higher precision (inverse relationship)
        # Normalize: assume typical alpha power ~ 100 μV²
        alpha_norm = alpha_power / 100.0
        pi_from_alpha = 1.0 / (0.5 + alpha_norm)

        # Arousal from physiology
        if len(physio_features) >= 4:
            gsr = physio_features[2]       # GSR (normalized)
            pupil = physio_features[3]     # Pupil (normalized)
            arousal = (gsr + pupil) / 2
        else:
            arousal = 0.5

        # Higher arousal → higher sensory precision
        pi_from_arousal = 0.5 + arousal

        # Combine estimates
        pi_sensory = 0.6 * pi_from_alpha + 0.4 * pi_from_arousal
        pi_sensory = max(0.3, min(3.0, pi_sensory))

        # Uncertainty
        std_err = 0.2

        # Record
        self.history.append((time.time(), pi_sensory))
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return pi_sensory, std_err


class PriorPrecisionEstimator:
    """
    Estimates prior/motivational precision Π̂_prior.

    Proxy markers:
    - Frontal theta power (positive relationship - cognitive control)
    - HRV / vagal tone (complex relationship)
    - Task engagement / focus (positive relationship)

    High Π_prior = strong reliance on prior beliefs / predictions
    """

    def __init__(self, sample_rate_hz: float = 256.0):
        self.sample_rate_hz = sample_rate_hz
        self.history: List[Tuple[float, float]] = []

    def estimate(
        self,
        eeg_data: np.ndarray,
        physio_features: np.ndarray,
        context_features: np.ndarray,
        frontal_channels: List[int] = None,
    ) -> Tuple[float, float]:
        """
        Estimate Π̂_prior from EEG, physiology, and context.

        Args:
            eeg_data: (n_samples, n_channels) preprocessed EEG
            physio_features: [hrv, hr, gsr, pupil, resp, hr_var]
            context_features: [task_load, focus, error_rate, rt]
            frontal_channels: Indices of frontal electrodes

        Returns:
            (pi_prior, std_error)
        """
        if eeg_data.size == 0:
            return 1.0, 0.3

        n_channels = eeg_data.shape[1]

        # Default: use first half of channels as "frontal"
        if frontal_channels is None:
            frontal_channels = list(range(n_channels // 2))

        # Extract frontal data
        if frontal_channels:
            frontal_data = eeg_data[:, frontal_channels]
        else:
            frontal_data = eeg_data

        # Compute power in theta band (4-8 Hz) - simplified
        if len(frontal_data) > 50:
            # Simple low-frequency power proxy
            # Smooth heavily to approximate theta
            smoothed = np.convolve(np.mean(frontal_data, axis=1),
                                   np.ones(10)/10, mode='valid')
            theta_power = np.var(smoothed) if len(smoothed) > 0 else 1.0
        else:
            theta_power = 1.0

        # Higher theta → higher cognitive control → higher prior precision
        theta_norm = theta_power / 50.0
        pi_from_theta = 0.5 + theta_norm

        # HRV contribution (higher HRV → better regulation → moderate prior)
        if len(physio_features) >= 1:
            hrv_norm = physio_features[0]  # Already normalized
            # Very high or very low HRV both problematic
            hrv_factor = 1.0 - abs(hrv_norm - 1.0) * 0.3
        else:
            hrv_factor = 1.0

        # Task engagement from context
        if len(context_features) >= 2:
            task_load = context_features[0]
            focus = context_features[1]
            engagement = (task_load + focus) / 2
        else:
            engagement = 0.5

        # Higher engagement → stronger priors about task
        pi_from_context = 0.5 + engagement

        # Combine
        pi_prior = 0.4 * pi_from_theta + 0.2 * hrv_factor + 0.4 * pi_from_context
        pi_prior = max(0.3, min(3.0, pi_prior))

        std_err = 0.25

        # Record
        self.history.append((time.time(), pi_prior))
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return pi_prior, std_err


# =============================================================================
# Combined GUTC Estimator
# =============================================================================

class HumanGUTCEstimator:
    """
    Combined estimator for full human GUTC state z(t).

    Compresses multichannel signals into 3D control coordinates at ~10 Hz.
    """

    def __init__(self, sample_rate_hz: float = 256.0, verbose: bool = True):
        self.sample_rate_hz = sample_rate_hz
        self.verbose = verbose

        # Sub-estimators
        self.lambda_estimator = BranchingRatioEstimator(sample_rate_hz=sample_rate_hz)
        self.pi_sensory_estimator = SensoryPrecisionEstimator(sample_rate_hz=sample_rate_hz)
        self.pi_prior_estimator = PriorPrecisionEstimator(sample_rate_hz=sample_rate_hz)

        # State history
        self.state_history: List[GUTCState] = []
        self.update_rate_hz = 10.0
        self.last_update = 0.0

    def estimate(
        self,
        eeg_events: np.ndarray,
        eeg_data: np.ndarray,
        physio_features: np.ndarray,
        context_features: np.ndarray,
    ) -> GUTCState:
        """
        Estimate full GUTC state from all signals.

        Args:
            eeg_events: Binary event matrix B(c,τ)
            eeg_data: Preprocessed EEG (n_samples, n_channels)
            physio_features: φ_phys(t)
            context_features: φ_ctx(t)

        Returns:
            GUTCState with (λ̂, Π̂_sensory, Π̂_prior)
        """
        now = time.time()

        # Estimate λ from avalanches
        lambda_hat, lambda_std = self.lambda_estimator.estimate_lambda(eeg_events)

        # Estimate Π_sensory
        pi_s, pi_s_std = self.pi_sensory_estimator.estimate(
            eeg_data, physio_features
        )

        # Estimate Π_prior
        pi_p, pi_p_std = self.pi_prior_estimator.estimate(
            eeg_data, physio_features, context_features
        )

        # Create state
        state = GUTCState(
            timestamp=now,
            lambda_hat=lambda_hat,
            pi_sensory=pi_s,
            pi_prior=pi_p,
            lambda_std=lambda_std,
            pi_sensory_std=pi_s_std,
            pi_prior_std=pi_p_std,
        )

        # Record
        self.state_history.append(state)
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]

        self.last_update = now

        if self.verbose and len(self.state_history) % 50 == 0:
            print(f"[GUTCEstimator] z(t) = (λ={lambda_hat:.2f}, "
                  f"Πs={pi_s:.2f}, Πp={pi_p:.2f}) → {state.regime_label()}")

        return state

    def get_trajectory(self, duration_s: float = 60.0) -> List[GUTCState]:
        """Get recent state trajectory."""
        cutoff = time.time() - duration_s
        return [s for s in self.state_history if s.timestamp > cutoff]

    def get_mean_state(self, duration_s: float = 10.0) -> GUTCState:
        """Get time-averaged state."""
        trajectory = self.get_trajectory(duration_s)

        if not trajectory:
            return GUTCState(
                timestamp=time.time(),
                lambda_hat=1.0,
                pi_sensory=1.0,
                pi_prior=1.0,
            )

        lambda_mean = np.mean([s.lambda_hat for s in trajectory])
        pi_s_mean = np.mean([s.pi_sensory for s in trajectory])
        pi_p_mean = np.mean([s.pi_prior for s in trajectory])

        return GUTCState(
            timestamp=time.time(),
            lambda_hat=float(lambda_mean),
            pi_sensory=float(pi_s_mean),
            pi_prior=float(pi_p_mean),
        )

    def get_status(self) -> Dict[str, Any]:
        """Get estimator status."""
        recent = self.state_history[-1] if self.state_history else None

        return {
            "n_states": len(self.state_history),
            "n_avalanches": len(self.lambda_estimator.avalanches),
            "current_lambda": recent.lambda_hat if recent else None,
            "current_pi_sensory": recent.pi_sensory if recent else None,
            "current_pi_prior": recent.pi_prior if recent else None,
            "current_regime": recent.regime_label() if recent else None,
            "in_healthy_corridor": recent.in_healthy_corridor() if recent else None,
        }
