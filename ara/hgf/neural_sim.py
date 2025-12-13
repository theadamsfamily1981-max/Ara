"""
ara.hgf.neural_sim - Synthetic Neural Correlate Generation

Generates synthetic neural signals (EEG, fMRI) that correlate with
HGF computational variables. This allows:

1. Testing analysis pipelines before collecting real data
2. Validating that analysis methods can recover the expected relationships
3. Teaching about neural correlates of Bayesian inference

Key relationships modeled:
- δ₁ (sensory PE) → Feedback-Related Negativity (FRN) in EEG
- δ₂ (volatility PE) → Anterior Cingulate Cortex (ACC) in fMRI
- Π_prior (prior precision) → Frontal theta power
- Π_sensory (sensory precision) → Posterior alpha power

These are based on empirical findings in the literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal

from ara.hgf.agents import HGFTrajectory


@dataclass
class SyntheticEEG:
    """Synthetic EEG data with trial structure."""
    data: np.ndarray  # Shape: (n_trials, n_samples)
    times: np.ndarray  # Time vector in seconds
    sfreq: float  # Sampling frequency
    n_trials: int
    channel_names: list

    # Ground truth regressors
    delta_1: np.ndarray
    delta_2: np.ndarray
    pi_hat_2: np.ndarray

    def get_erp(self, baseline: Tuple[float, float] = (-0.2, 0)) -> np.ndarray:
        """Compute average ERP with baseline correction."""
        # Find baseline indices
        baseline_mask = (self.times >= baseline[0]) & (self.times < baseline[1])
        baseline_mean = np.mean(self.data[:, baseline_mask], axis=1, keepdims=True)

        # Baseline correct
        data_corrected = self.data - baseline_mean

        # Average
        return np.mean(data_corrected, axis=0)


@dataclass
class SyntheticFMRI:
    """Synthetic fMRI data with trial regressors."""
    data: np.ndarray  # Shape: (n_volumes, n_rois)
    tr: float  # Repetition time
    n_volumes: int
    roi_names: list

    # Trial-by-trial regressors (already convolved with HRF)
    delta_1_convolved: np.ndarray
    delta_2_convolved: np.ndarray
    volatility_convolved: np.ndarray

    # Ground truth weights used to generate data
    true_weights: dict


def generate_theta_alpha_signals(
    trajectory: HGFTrajectory,
    sfreq: float = 256.0,
    trial_duration: float = 2.0,
    theta_freq: Tuple[float, float] = (4.0, 8.0),
    alpha_freq: Tuple[float, float] = (8.0, 12.0),
    noise_level: float = 0.3,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate theta and alpha band signals that correlate with HGF precisions.

    Theta power (frontal) ∝ prior precision (Π_prior)
    Alpha power (posterior) ∝ sensory precision (Π_sensory)

    This models the precision-weighting signals hypothesized in Active Inference.

    Args:
        trajectory: HGF trajectory
        sfreq: Sampling frequency
        trial_duration: Duration of each trial
        theta_freq: Theta band frequency range
        alpha_freq: Alpha band frequency range
        noise_level: Noise level (0-1)
        seed: Random seed

    Returns:
        Dictionary with theta and alpha signals
    """
    rng = np.random.RandomState(seed)
    n_trials = trajectory.n_trials
    n_samples = int(trial_duration * sfreq)

    # Get precision trajectories
    precisions = trajectory.get_precisions()
    pi_hat_2 = precisions["pi_hat_2"]  # Prior precision
    pi_1 = precisions["pi_1"]  # Sensory precision

    # Normalize precisions to [0, 1]
    pi_hat_2_norm = (pi_hat_2 - pi_hat_2.min()) / (pi_hat_2.max() - pi_hat_2.min() + 1e-10)
    pi_1_norm = (pi_1 - pi_1.min()) / (pi_1.max() - pi_1.min() + 1e-10)

    # Generate time vector
    times = np.arange(n_samples) / sfreq

    # Generate oscillatory signals
    theta_signals = np.zeros((n_trials, n_samples))
    alpha_signals = np.zeros((n_trials, n_samples))

    for trial in range(n_trials):
        # Theta signal (amplitude modulated by prior precision)
        theta_freq_trial = rng.uniform(*theta_freq)
        theta_amp = 0.5 + 0.5 * pi_hat_2_norm[trial]  # Modulated by Π_prior
        theta = theta_amp * np.sin(2 * np.pi * theta_freq_trial * times)
        theta += noise_level * rng.randn(n_samples)
        theta_signals[trial] = theta

        # Alpha signal (amplitude modulated by sensory precision)
        alpha_freq_trial = rng.uniform(*alpha_freq)
        alpha_amp = 0.5 + 0.5 * pi_1_norm[trial]  # Modulated by Π_sensory
        alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq_trial * times)
        alpha += noise_level * rng.randn(n_samples)
        alpha_signals[trial] = alpha

    return {
        "theta": theta_signals,
        "alpha": alpha_signals,
        "times": times,
        "sfreq": sfreq,
        "pi_hat_2": pi_hat_2,
        "pi_1": pi_1,
        "n_trials": n_trials,
    }


def simulate_eeg_correlates(
    trajectory: HGFTrajectory,
    sfreq: float = 256.0,
    epoch_tmin: float = -0.5,
    epoch_tmax: float = 1.0,
    frn_peak: float = 0.25,  # FRN peaks ~250ms post-feedback
    frn_width: float = 0.05,
    delta_1_weight: float = 5.0,  # μV per unit PE
    delta_2_weight: float = 3.0,
    noise_level: float = 2.0,  # μV
    seed: Optional[int] = None,
) -> SyntheticEEG:
    """
    Simulate EEG signals with FRN correlating to prediction error.

    The Feedback-Related Negativity (FRN) is an ERP component that
    correlates with unsigned prediction error magnitude.

    Args:
        trajectory: HGF trajectory
        sfreq: Sampling frequency (Hz)
        epoch_tmin: Epoch start time (s)
        epoch_tmax: Epoch end time (s)
        frn_peak: FRN peak latency (s)
        frn_width: FRN temporal width (s)
        delta_1_weight: Amplitude scaling for δ₁
        delta_2_weight: Amplitude scaling for δ₂
        noise_level: Noise standard deviation (μV)
        seed: Random seed

    Returns:
        SyntheticEEG object
    """
    rng = np.random.RandomState(seed)
    n_trials = trajectory.n_trials

    # Time vector
    n_samples = int((epoch_tmax - epoch_tmin) * sfreq)
    times = np.linspace(epoch_tmin, epoch_tmax, n_samples)

    # Get prediction errors
    delta_1 = trajectory.get_prediction_errors(1)
    delta_2 = trajectory.get_prediction_errors(2)
    pi_hat_2 = trajectory.get_precisions()["pi_hat_2"]

    # FRN template (Gaussian-windowed negative deflection)
    frn_template = -np.exp(-0.5 * ((times - frn_peak) / frn_width) ** 2)

    # Generate EEG data
    data = np.zeros((n_trials, n_samples))

    for trial in range(n_trials):
        # FRN amplitude scales with |δ₁| and |δ₂|
        frn_amplitude = (
            delta_1_weight * np.abs(delta_1[trial]) +
            delta_2_weight * np.abs(delta_2[trial])
        )

        # Add FRN
        data[trial] = frn_amplitude * frn_template

        # Add noise
        data[trial] += noise_level * rng.randn(n_samples)

        # Add 1/f background
        pink_noise = _generate_pink_noise(n_samples, rng) * (noise_level * 0.5)
        data[trial] += pink_noise

    return SyntheticEEG(
        data=data,
        times=times,
        sfreq=sfreq,
        n_trials=n_trials,
        channel_names=["FCz"],  # Typical FRN electrode
        delta_1=delta_1,
        delta_2=delta_2,
        pi_hat_2=pi_hat_2,
    )


def simulate_fmri_correlates(
    trajectory: HGFTrajectory,
    tr: float = 2.0,
    n_volumes: Optional[int] = None,
    rois: list = ["ACC", "vmPFC", "striatum", "insula"],
    roi_weights: Optional[dict] = None,
    noise_level: float = 0.5,
    seed: Optional[int] = None,
) -> SyntheticFMRI:
    """
    Simulate fMRI signals with PE correlates in relevant ROIs.

    Based on empirical findings:
    - ACC: Volatility PE (δ₂)
    - vmPFC: Valence, reward prediction
    - Striatum: Reward PE (δ₁)
    - Insula: Uncertainty, interoception

    Args:
        trajectory: HGF trajectory
        tr: Repetition time (s)
        n_volumes: Number of volumes (default: ~3 per trial)
        rois: List of ROI names
        roi_weights: Dict of ROI -> {delta_1, delta_2, volatility} weights
        noise_level: Noise level
        seed: Random seed

    Returns:
        SyntheticFMRI object
    """
    rng = np.random.RandomState(seed)
    n_trials = trajectory.n_trials

    if n_volumes is None:
        n_volumes = n_trials * 3  # ~3 volumes per trial

    # Default ROI weights (based on literature)
    if roi_weights is None:
        roi_weights = {
            "ACC": {"delta_1": 0.3, "delta_2": 0.8, "volatility": 0.5},
            "vmPFC": {"delta_1": 0.6, "delta_2": 0.2, "volatility": 0.3},
            "striatum": {"delta_1": 0.9, "delta_2": 0.1, "volatility": 0.2},
            "insula": {"delta_1": 0.4, "delta_2": 0.5, "volatility": 0.6},
        }

    # Get HGF signals
    delta_1 = trajectory.get_prediction_errors(1)
    delta_2 = trajectory.get_prediction_errors(2)
    volatility = trajectory.get_beliefs(3)

    # Upsample to fMRI resolution
    trial_times = np.arange(n_trials) * 2.0  # Assume 2s per trial
    volume_times = np.arange(n_volumes) * tr

    # Interpolate to volume times
    delta_1_upsampled = np.interp(volume_times, trial_times, delta_1)
    delta_2_upsampled = np.interp(volume_times, trial_times, delta_2)
    volatility_upsampled = np.interp(volume_times, trial_times, volatility)

    # Convolve with HRF
    hrf = _generate_hrf(tr, duration=32.0)
    delta_1_conv = np.convolve(delta_1_upsampled, hrf)[:n_volumes]
    delta_2_conv = np.convolve(delta_2_upsampled, hrf)[:n_volumes]
    volatility_conv = np.convolve(volatility_upsampled, hrf)[:n_volumes]

    # Generate ROI data
    n_rois = len(rois)
    data = np.zeros((n_volumes, n_rois))

    true_weights = {}
    for i, roi in enumerate(rois):
        weights = roi_weights.get(roi, {"delta_1": 0.5, "delta_2": 0.5, "volatility": 0.0})
        true_weights[roi] = weights

        # Combine weighted signals
        data[:, i] = (
            weights["delta_1"] * delta_1_conv +
            weights["delta_2"] * delta_2_conv +
            weights["volatility"] * volatility_conv
        )

        # Add noise
        data[:, i] += noise_level * rng.randn(n_volumes)

        # Add slow drift
        data[:, i] += 0.5 * np.sin(2 * np.pi * np.arange(n_volumes) / (n_volumes / 2))

    return SyntheticFMRI(
        data=data,
        tr=tr,
        n_volumes=n_volumes,
        roi_names=rois,
        delta_1_convolved=delta_1_conv,
        delta_2_convolved=delta_2_conv,
        volatility_convolved=volatility_conv,
        true_weights=true_weights,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_pink_noise(n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate 1/f (pink) noise."""
    white = rng.randn(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1e-10  # Avoid division by zero
    fft_pink = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft_pink, n=n_samples)
    return pink / np.std(pink)


def _generate_hrf(
    tr: float,
    duration: float = 32.0,
    peak_delay: float = 6.0,
    undershoot_delay: float = 16.0,
    peak_disp: float = 1.0,
    undershoot_disp: float = 1.0,
    ratio: float = 0.35,
) -> np.ndarray:
    """
    Generate canonical double-gamma hemodynamic response function.

    Based on SPM's canonical HRF.
    """
    from scipy.stats import gamma as gamma_dist

    n_samples = int(duration / tr)
    times = np.arange(n_samples) * tr

    # Peak
    peak = gamma_dist.pdf(times, peak_delay / peak_disp, scale=peak_disp)

    # Undershoot
    undershoot = gamma_dist.pdf(times, undershoot_delay / undershoot_disp, scale=undershoot_disp)

    # Combine
    hrf = peak - ratio * undershoot

    # Normalize
    hrf = hrf / np.max(hrf)

    return hrf


def compute_erp_pe_correlation(eeg: SyntheticEEG, time_window: Tuple[float, float] = (0.2, 0.35)) -> dict:
    """
    Compute correlation between ERP amplitude and prediction errors.

    This is a key validation step: does FRN amplitude correlate with δ₁?

    Args:
        eeg: Synthetic EEG data
        time_window: Time window for FRN (seconds)

    Returns:
        Correlation results
    """
    # Get FRN amplitude (mean in window)
    mask = (eeg.times >= time_window[0]) & (eeg.times <= time_window[1])
    frn_amplitudes = np.mean(eeg.data[:, mask], axis=1)

    # Correlations
    from scipy.stats import pearsonr

    r_delta1, p_delta1 = pearsonr(np.abs(eeg.delta_1), -frn_amplitudes)
    r_delta2, p_delta2 = pearsonr(np.abs(eeg.delta_2), -frn_amplitudes)

    return {
        "delta_1": {"r": r_delta1, "p": p_delta1},
        "delta_2": {"r": r_delta2, "p": p_delta2},
        "frn_amplitudes": frn_amplitudes,
    }


def compute_fmri_roi_regression(fmri: SyntheticFMRI) -> dict:
    """
    Regress fMRI data against HGF regressors.

    This validates that we can recover the expected relationships.

    Args:
        fmri: Synthetic fMRI data

    Returns:
        Regression results for each ROI
    """
    from scipy.stats import pearsonr

    results = {}

    # Design matrix
    X = np.column_stack([
        fmri.delta_1_convolved,
        fmri.delta_2_convolved,
        fmri.volatility_convolved,
        np.ones(fmri.n_volumes),  # Intercept
    ])

    for i, roi in enumerate(fmri.roi_names):
        y = fmri.data[:, i]

        # OLS regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Correlations with individual regressors
        r_delta1, _ = pearsonr(fmri.delta_1_convolved, y)
        r_delta2, _ = pearsonr(fmri.delta_2_convolved, y)
        r_vol, _ = pearsonr(fmri.volatility_convolved, y)

        results[roi] = {
            "beta_delta1": beta[0],
            "beta_delta2": beta[1],
            "beta_volatility": beta[2],
            "r_delta1": r_delta1,
            "r_delta2": r_delta2,
            "r_volatility": r_vol,
            "true_weights": fmri.true_weights.get(roi, {}),
        }

    return results
