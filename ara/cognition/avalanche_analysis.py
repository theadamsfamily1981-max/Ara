#!/usr/bin/env python3
"""
Avalanche Analysis Pipeline for IG-Criticality Validation
==========================================================

Implements the experimental test of the Information-Geometric Criticality
hypothesis: that Fisher information for behaviorally relevant latent variables
is maximized when neuronal avalanche statistics are closest to their critical
power-law form (α_size ≈ 1.5, β_dur ≈ 2.0).

Pipeline Steps:
1. Preprocessing and event detection
2. Avalanche segmentation
3. Power-law fitting and criticality score
4. Fisher information estimation
5. FIM vs criticality correlation analysis

Theory:
    At criticality, avalanche distributions follow power laws:
    - P(S) ~ S^(-α)  with α* = 1.5 (mean-field branching process)
    - P(T) ~ T^(-β)  with β* = 2.0

    The criticality score K measures distance from ideal exponents:
    K = sqrt((α - 1.5)² + (β - 2.0)²)

    Prediction 1: Fisher information g_θθ is maximized when K is minimized.

Usage:
    from ara.cognition.avalanche_analysis import AvalancheAnalyzer

    analyzer = AvalancheAnalyzer()

    # Load neural data (T x N array: time x channels)
    X = load_neural_data()
    theta = load_stimulus_labels()  # Optional

    # Run full pipeline
    results = analyzer.analyze(X, theta)

    print(f"Criticality score: {results.criticality_score:.3f}")
    print(f"Fisher information: {results.fisher_info:.3f}")

References:
    - Beggs & Plenz (2003) - Neuronal Avalanches in Neocortical Circuits
    - Clauset et al. (2009) - Power-Law Distributions in Empirical Data
    - Shew et al. (2011) - Information Capacity and Transmission Optimized
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger("ara.cognition.avalanche")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Avalanche:
    """Single avalanche event."""
    start_time: int           # Start time bin
    end_time: int             # End time bin (exclusive)
    size: int                 # Total activity summed over duration
    duration: int             # Number of time bins
    peak_activity: int        # Maximum activity in any bin
    channel_spread: int       # Number of unique channels active


@dataclass
class PowerLawFit:
    """Result of power-law fitting."""
    exponent: float           # Estimated exponent α or β
    xmin: float               # Lower cutoff
    loglikelihood: float      # Log-likelihood of fit
    ks_statistic: float       # Kolmogorov-Smirnov statistic
    p_value: float            # Goodness-of-fit p-value
    n_tail: int               # Number of samples in tail


@dataclass
class CriticalityMetrics:
    """Avalanche criticality metrics for a session."""
    # Power-law exponents
    alpha_size: float         # Size exponent α
    beta_duration: float      # Duration exponent β

    # Criticality score (distance from ideal)
    criticality_score: float  # K = sqrt((α-1.5)² + (β-2.0)²)

    # Fit quality
    size_fit: PowerLawFit
    duration_fit: PowerLawFit

    # Summary statistics
    n_avalanches: int
    mean_size: float
    mean_duration: float
    branching_ratio: float    # σ = <S_t+1 / S_t>


@dataclass
class FisherEstimate:
    """Fisher information estimate for a latent variable."""
    fisher_info: float        # g_θθ estimate
    method: str               # 'linear_gaussian', 'poisson_glm', 'empirical'
    n_samples: int            # Number of samples used
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Full analysis result combining criticality and Fisher metrics."""
    # Criticality
    criticality: CriticalityMetrics
    criticality_score: float  # Shorthand

    # Fisher information (if stimulus/behavior available)
    fisher_info: Optional[float]
    fisher_estimate: Optional[FisherEstimate]

    # Avalanches
    avalanches: List[Avalanche]
    sizes: np.ndarray
    durations: np.ndarray

    # Metadata
    n_channels: int
    n_timepoints: int
    dt_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criticality_score": self.criticality_score,
            "alpha_size": self.criticality.alpha_size,
            "beta_duration": self.criticality.beta_duration,
            "branching_ratio": self.criticality.branching_ratio,
            "fisher_info": self.fisher_info,
            "n_avalanches": self.criticality.n_avalanches,
            "mean_size": self.criticality.mean_size,
            "mean_duration": self.criticality.mean_duration,
        }


# =============================================================================
# Preprocessing
# =============================================================================

def zscore_channels(X: np.ndarray) -> np.ndarray:
    """
    Z-score each channel across time.

    Args:
        X: Neural data array (T x N)

    Returns:
        Z-scored array
    """
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std < 1e-10] = 1.0  # Avoid division by zero
    return (X - mean) / std


def binarize_activity(
    X: np.ndarray,
    threshold: float = 0.0,
    method: str = 'threshold',
) -> np.ndarray:
    """
    Convert continuous activity to binary events.

    Args:
        X: Neural data array (T x N), z-scored
        threshold: Threshold for event detection
        method: 'threshold' or 'peak_detection'

    Returns:
        Binary array (T x N)
    """
    if method == 'threshold':
        return (X > threshold).astype(int)
    elif method == 'peak_detection':
        # Simple peak detection: above threshold AND local maximum
        above = X > threshold
        # Check if each point is a local max (greater than neighbors)
        padded = np.pad(X, ((1, 1), (0, 0)), mode='edge')
        local_max = (X >= padded[:-2]) & (X >= padded[2:])
        return (above & local_max).astype(int)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Avalanche Detection
# =============================================================================

def detect_avalanches(
    binary: np.ndarray,
    min_size: int = 1,
) -> List[Avalanche]:
    """
    Detect avalanches from binary activity matrix.

    An avalanche is a contiguous sequence of time bins with total
    activity > 0, bounded by bins with activity = 0.

    Args:
        binary: Binary activity array (T x N)
        min_size: Minimum avalanche size to include

    Returns:
        List of Avalanche objects
    """
    T, N = binary.shape
    activity = np.sum(binary, axis=1)  # Total activity per time bin

    avalanches = []
    in_avalanche = False
    start = 0

    for t in range(T):
        if activity[t] > 0 and not in_avalanche:
            # Start of new avalanche
            in_avalanche = True
            start = t
        elif activity[t] == 0 and in_avalanche:
            # End of avalanche
            in_avalanche = False
            end = t

            # Compute avalanche properties
            size = int(np.sum(activity[start:end]))
            duration = end - start
            peak = int(np.max(activity[start:end]))
            channels = int(np.sum(np.any(binary[start:end], axis=0)))

            if size >= min_size:
                avalanches.append(Avalanche(
                    start_time=start,
                    end_time=end,
                    size=size,
                    duration=duration,
                    peak_activity=peak,
                    channel_spread=channels,
                ))

    # Handle avalanche at end of recording
    if in_avalanche:
        end = T
        size = int(np.sum(activity[start:end]))
        duration = end - start
        peak = int(np.max(activity[start:end]))
        channels = int(np.sum(np.any(binary[start:end], axis=0)))

        if size >= min_size:
            avalanches.append(Avalanche(
                start_time=start,
                end_time=end,
                size=size,
                duration=duration,
                peak_activity=peak,
                channel_spread=channels,
            ))

    return avalanches


def compute_branching_ratio(activity: np.ndarray) -> float:
    """
    Compute branching ratio σ = <A_{t+1} / A_t>.

    At criticality, σ ≈ 1.

    Args:
        activity: Total activity per time bin (T,)

    Returns:
        Branching ratio estimate
    """
    # Find transitions where activity > 0 at time t
    valid = activity[:-1] > 0
    if np.sum(valid) == 0:
        return 0.0

    ratios = activity[1:][valid] / activity[:-1][valid]
    return float(np.mean(ratios))


# =============================================================================
# Power-Law Fitting
# =============================================================================

def fit_power_law_mle(
    data: np.ndarray,
    xmin: Optional[float] = None,
) -> PowerLawFit:
    """
    Fit power-law distribution using maximum likelihood.

    P(x) ~ x^(-α) for x >= xmin

    Uses the Hill estimator: α = 1 + n / Σ log(x_i / xmin)

    Args:
        data: Array of values (e.g., avalanche sizes)
        xmin: Lower cutoff (if None, estimate optimal)

    Returns:
        PowerLawFit object
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]  # Remove zeros

    if len(data) < 10:
        return PowerLawFit(
            exponent=1.5,
            xmin=1.0,
            loglikelihood=0.0,
            ks_statistic=1.0,
            p_value=0.0,
            n_tail=len(data),
        )

    # Estimate optimal xmin if not provided
    if xmin is None:
        xmin = _estimate_xmin(data)

    # Filter to tail
    tail = data[data >= xmin]
    n = len(tail)

    if n < 5:
        return PowerLawFit(
            exponent=1.5,
            xmin=xmin,
            loglikelihood=0.0,
            ks_statistic=1.0,
            p_value=0.0,
            n_tail=n,
        )

    # Hill estimator
    alpha = 1.0 + n / np.sum(np.log(tail / xmin))

    # Log-likelihood
    ll = n * np.log(alpha - 1) - n * (alpha - 1) * np.log(xmin) - alpha * np.sum(np.log(tail))

    # KS statistic
    ks_stat = _power_law_ks(tail, alpha, xmin)

    # Bootstrap p-value (simplified)
    p_value = _bootstrap_p_value(tail, alpha, xmin, n_boot=100)

    return PowerLawFit(
        exponent=alpha,
        xmin=xmin,
        loglikelihood=ll,
        ks_statistic=ks_stat,
        p_value=p_value,
        n_tail=n,
    )


def _estimate_xmin(data: np.ndarray) -> float:
    """Estimate optimal xmin by minimizing KS statistic."""
    unique_vals = np.unique(data)

    if len(unique_vals) < 5:
        return float(np.min(data))

    # Try different xmin values
    candidates = unique_vals[:-4]  # Need at least 5 points in tail
    best_ks = np.inf
    best_xmin = float(np.min(data))

    for xmin in candidates[:50]:  # Limit search
        tail = data[data >= xmin]
        if len(tail) < 5:
            continue

        n = len(tail)
        alpha = 1.0 + n / np.sum(np.log(tail / xmin))
        ks = _power_law_ks(tail, alpha, xmin)

        if ks < best_ks:
            best_ks = ks
            best_xmin = float(xmin)

    return best_xmin


def _power_law_ks(tail: np.ndarray, alpha: float, xmin: float) -> float:
    """Compute KS statistic for power-law fit."""
    n = len(tail)
    sorted_tail = np.sort(tail)

    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # Theoretical CDF: 1 - (x/xmin)^(-(alpha-1))
    tcdf = 1.0 - (sorted_tail / xmin) ** (-(alpha - 1))

    return float(np.max(np.abs(ecdf - tcdf)))


def _bootstrap_p_value(
    tail: np.ndarray,
    alpha: float,
    xmin: float,
    n_boot: int = 100,
) -> float:
    """Estimate p-value via bootstrap."""
    observed_ks = _power_law_ks(tail, alpha, xmin)
    n = len(tail)

    count_larger = 0
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        # Generate synthetic power-law samples
        u = rng.uniform(0, 1, n)
        synthetic = xmin * (1 - u) ** (-1 / (alpha - 1))

        # Fit and compute KS
        n_syn = len(synthetic)
        alpha_syn = 1.0 + n_syn / np.sum(np.log(synthetic / xmin))
        ks_syn = _power_law_ks(synthetic, alpha_syn, xmin)

        if ks_syn >= observed_ks:
            count_larger += 1

    return count_larger / n_boot


# =============================================================================
# Criticality Score
# =============================================================================

def compute_criticality_score(
    alpha: float,
    beta: float,
    alpha_target: float = 1.5,
    beta_target: float = 2.0,
) -> float:
    """
    Compute criticality score K (distance from ideal exponents).

    K = sqrt((α - α*)² + (β - β*)²)

    Smaller K = closer to criticality.
    """
    return np.sqrt((alpha - alpha_target) ** 2 + (beta - beta_target) ** 2)


def analyze_criticality(
    avalanches: List[Avalanche],
    activity: np.ndarray,
) -> CriticalityMetrics:
    """
    Analyze avalanche statistics for criticality.

    Args:
        avalanches: List of detected avalanches
        activity: Total activity per time bin

    Returns:
        CriticalityMetrics object
    """
    if len(avalanches) < 10:
        return CriticalityMetrics(
            alpha_size=1.5,
            beta_duration=2.0,
            criticality_score=0.0,
            size_fit=PowerLawFit(1.5, 1.0, 0.0, 1.0, 0.0, 0),
            duration_fit=PowerLawFit(2.0, 1.0, 0.0, 1.0, 0.0, 0),
            n_avalanches=len(avalanches),
            mean_size=0.0,
            mean_duration=0.0,
            branching_ratio=1.0,
        )

    # Extract sizes and durations
    sizes = np.array([a.size for a in avalanches])
    durations = np.array([a.duration for a in avalanches])

    # Fit power laws
    size_fit = fit_power_law_mle(sizes)
    duration_fit = fit_power_law_mle(durations)

    # Criticality score
    K = compute_criticality_score(size_fit.exponent, duration_fit.exponent)

    # Branching ratio
    sigma = compute_branching_ratio(activity)

    return CriticalityMetrics(
        alpha_size=size_fit.exponent,
        beta_duration=duration_fit.exponent,
        criticality_score=K,
        size_fit=size_fit,
        duration_fit=duration_fit,
        n_avalanches=len(avalanches),
        mean_size=float(np.mean(sizes)),
        mean_duration=float(np.mean(durations)),
        branching_ratio=sigma,
    )


# =============================================================================
# Fisher Information Estimation
# =============================================================================

def estimate_fisher_linear_gaussian(
    X: np.ndarray,
    theta: np.ndarray,
    feature_map: str = 'identity',
) -> FisherEstimate:
    """
    Estimate Fisher information for linear-Gaussian encoding model.

    X(t) = W @ f(theta(t)) + noise

    g_θθ = (∂μ/∂θ)^T Σ^(-1) (∂μ/∂θ)

    Args:
        X: Neural data (T x N)
        theta: Stimulus/behavior variable (T,) or (T x D)
        feature_map: 'identity', 'sincos' (for angles)

    Returns:
        FisherEstimate object
    """
    T, N = X.shape
    theta = np.atleast_1d(theta).flatten()

    if len(theta) != T:
        raise ValueError(f"theta length {len(theta)} != T {T}")

    # Feature map
    if feature_map == 'identity':
        f_theta = theta[:, None]  # (T, 1)
    elif feature_map == 'sincos':
        f_theta = np.column_stack([np.sin(theta), np.cos(theta)])  # (T, 2)
    else:
        raise ValueError(f"Unknown feature_map: {feature_map}")

    # Fit linear model: X = W @ f(theta).T + noise
    # Using least squares: W = X.T @ f(theta) @ (f(theta).T @ f(theta))^(-1)
    f_T = f_theta  # (T, D)
    try:
        W = np.linalg.lstsq(f_T, X, rcond=None)[0].T  # (N, D)
    except np.linalg.LinAlgError:
        return FisherEstimate(
            fisher_info=0.0,
            method='linear_gaussian',
            n_samples=T,
        )

    # Residuals and covariance
    X_pred = f_T @ W.T  # (T, N)
    residuals = X - X_pred
    Sigma = np.cov(residuals.T)

    # Regularize covariance
    Sigma = Sigma + np.eye(N) * 0.01 * np.trace(Sigma) / N

    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.eye(N)

    # Fisher information: g = (∂μ/∂θ)^T Σ^(-1) (∂μ/∂θ)
    # For identity map: ∂μ/∂θ = W[:, 0]
    # For sincos: need derivative w.r.t. original theta

    if feature_map == 'identity':
        dmu_dtheta = W[:, 0]  # (N,)
        g = float(dmu_dtheta @ Sigma_inv @ dmu_dtheta)
    elif feature_map == 'sincos':
        # ∂f/∂θ = [cos(θ), -sin(θ)]
        # Average over all theta values
        g_values = []
        for t in range(T):
            df_dtheta = np.array([np.cos(theta[t]), -np.sin(theta[t])])  # (2,)
            dmu_dtheta = W @ df_dtheta  # (N,)
            g_t = dmu_dtheta @ Sigma_inv @ dmu_dtheta
            g_values.append(g_t)
        g = float(np.mean(g_values))
    else:
        g = 0.0

    return FisherEstimate(
        fisher_info=g,
        method='linear_gaussian',
        n_samples=T,
        model_params={'W_shape': W.shape, 'feature_map': feature_map},
    )


def estimate_fisher_empirical(
    X: np.ndarray,
    theta: np.ndarray,
    decoder: str = 'linear',
) -> FisherEstimate:
    """
    Estimate empirical Fisher information from decoder gradients.

    g_θθ ≈ E[(∂L/∂θ)²]

    where L is the decoding loss.

    Args:
        X: Neural data (T x N)
        theta: Target variable (T,)
        decoder: 'linear' or 'ridge'

    Returns:
        FisherEstimate object
    """
    T, N = X.shape
    theta = np.atleast_1d(theta).flatten()

    if len(theta) != T:
        raise ValueError(f"theta length {len(theta)} != T {T}")

    # Fit decoder: theta = X @ w + b
    X_aug = np.column_stack([X, np.ones(T)])  # Add bias

    if decoder == 'ridge':
        # Ridge regression
        lambda_reg = 0.1
        w = np.linalg.lstsq(
            X_aug.T @ X_aug + lambda_reg * np.eye(N + 1),
            X_aug.T @ theta,
            rcond=None,
        )[0]
    else:
        w = np.linalg.lstsq(X_aug, theta, rcond=None)[0]

    # Predictions and residuals
    theta_pred = X_aug @ w
    residuals = theta - theta_pred

    # Empirical Fisher: variance of gradients
    # For linear decoder, ∂L/∂θ = 2 * residual * ∂θ_pred/∂θ
    # But θ is the target, so we use numerical differentiation

    # Simple approach: Fisher ≈ 1 / var(residuals)
    var_resid = np.var(residuals)
    if var_resid < 1e-10:
        var_resid = 1e-10

    g = 1.0 / var_resid

    return FisherEstimate(
        fisher_info=g,
        method='empirical',
        n_samples=T,
        model_params={'decoder': decoder, 'r_squared': 1 - var_resid / np.var(theta)},
    )


# =============================================================================
# Main Analyzer Class
# =============================================================================

class AvalancheAnalyzer:
    """
    Complete pipeline for avalanche-criticality-FIM analysis.

    Usage:
        analyzer = AvalancheAnalyzer()
        results = analyzer.analyze(X, theta)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        dt_ms: float = 1.0,
        min_avalanche_size: int = 1,
        fisher_method: str = 'linear_gaussian',
    ):
        """
        Initialize analyzer.

        Args:
            threshold: Z-score threshold for event detection
            dt_ms: Time bin width in milliseconds
            min_avalanche_size: Minimum avalanche size to include
            fisher_method: 'linear_gaussian' or 'empirical'
        """
        self.threshold = threshold
        self.dt_ms = dt_ms
        self.min_avalanche_size = min_avalanche_size
        self.fisher_method = fisher_method

    def analyze(
        self,
        X: np.ndarray,
        theta: Optional[np.ndarray] = None,
        zscore: bool = True,
    ) -> AnalysisResult:
        """
        Run full analysis pipeline.

        Args:
            X: Neural data array (T x N)
            theta: Optional stimulus/behavior variable (T,)
            zscore: Whether to z-score channels

        Returns:
            AnalysisResult object
        """
        T, N = X.shape
        logger.info(f"Analyzing {T} timepoints x {N} channels")

        # Step 1: Preprocessing
        if zscore:
            X_proc = zscore_channels(X)
        else:
            X_proc = X

        # Step 2: Binarize
        binary = binarize_activity(X_proc, self.threshold)
        activity = np.sum(binary, axis=1)

        # Step 3: Detect avalanches
        avalanches = detect_avalanches(binary, self.min_avalanche_size)
        logger.info(f"Detected {len(avalanches)} avalanches")

        # Step 4: Criticality analysis
        criticality = analyze_criticality(avalanches, activity)
        logger.info(f"Criticality score K = {criticality.criticality_score:.3f}")
        logger.info(f"  α_size = {criticality.alpha_size:.2f} (target: 1.5)")
        logger.info(f"  β_dur = {criticality.beta_duration:.2f} (target: 2.0)")
        logger.info(f"  σ (branching) = {criticality.branching_ratio:.3f}")

        # Step 5: Fisher information (if theta provided)
        fisher_estimate = None
        fisher_info = None

        if theta is not None:
            if self.fisher_method == 'linear_gaussian':
                fisher_estimate = estimate_fisher_linear_gaussian(X_proc, theta)
            else:
                fisher_estimate = estimate_fisher_empirical(X_proc, theta)

            fisher_info = fisher_estimate.fisher_info
            logger.info(f"Fisher information g = {fisher_info:.3f}")

        # Extract arrays
        sizes = np.array([a.size for a in avalanches])
        durations = np.array([a.duration for a in avalanches])

        return AnalysisResult(
            criticality=criticality,
            criticality_score=criticality.criticality_score,
            fisher_info=fisher_info,
            fisher_estimate=fisher_estimate,
            avalanches=avalanches,
            sizes=sizes,
            durations=durations,
            n_channels=N,
            n_timepoints=T,
            dt_ms=self.dt_ms,
        )


# =============================================================================
# Multi-Session Analysis
# =============================================================================

def analyze_fim_vs_criticality(
    sessions: List[AnalysisResult],
) -> Dict[str, Any]:
    """
    Analyze relationship between FIM and criticality across sessions.

    Tests Prediction 1: g_θθ is maximized when K is minimized.

    Args:
        sessions: List of AnalysisResult from multiple sessions

    Returns:
        Dictionary with correlation analysis
    """
    # Filter sessions with valid Fisher info
    valid = [s for s in sessions if s.fisher_info is not None]

    if len(valid) < 3:
        return {
            "n_sessions": len(valid),
            "correlation": None,
            "p_value": None,
            "message": "Not enough sessions with Fisher info",
        }

    K = np.array([s.criticality_score for s in valid])
    g = np.array([s.fisher_info for s in valid])

    # Test negative correlation (higher g when lower K)
    r, p = stats.pearsonr(K, g)
    rho, p_spearman = stats.spearmanr(K, g)

    # Log-space analysis
    log_g = np.log(g + 1e-10)
    r_log, p_log = stats.pearsonr(-np.log(K + 1e-6), log_g)

    # Quadratic fit (test for peak)
    try:
        coeffs = np.polyfit(K, log_g, 2)
        # coeffs[0] < 0 means concave (peaked)
        is_peaked = coeffs[0] < 0
    except Exception:
        coeffs = [0, 0, 0]
        is_peaked = False

    return {
        "n_sessions": len(valid),
        "K_values": K.tolist(),
        "g_values": g.tolist(),
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": p_spearman,
        "log_correlation_r": r_log,
        "log_correlation_p": p_log,
        "quadratic_coeffs": coeffs.tolist(),
        "is_peaked": is_peaked,
        "prediction_supported": r < 0 and p < 0.05,
    }


# =============================================================================
# Simulation for Testing
# =============================================================================

def simulate_critical_dynamics(
    n_neurons: int = 100,
    T: int = 10000,
    branching_ratio: float = 1.0,
    external_rate: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate branching process dynamics at given criticality.

    Args:
        n_neurons: Number of neurons
        T: Number of time steps
        branching_ratio: σ (1.0 = critical)
        external_rate: Rate of external input
        seed: Random seed

    Returns:
        (X, theta) where X is (T, N) activity and theta is (T,) stimulus
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((T, n_neurons))
    activity = np.zeros(n_neurons)

    # Stimulus (sinusoidal for orientation-like)
    theta = np.sin(2 * np.pi * np.arange(T) / 500) * np.pi

    for t in range(T):
        # External input (stimulus-modulated)
        ext_input = rng.poisson(external_rate * (1 + 0.5 * np.sin(theta[t])), n_neurons)

        # Propagate from previous activity
        if t > 0:
            n_active_prev = int(np.sum(X[t - 1] > 0))
            if n_active_prev > 0:
                # Each active neuron activates ~σ neurons on average
                n_propagate = rng.poisson(branching_ratio * n_active_prev)
                targets = rng.choice(n_neurons, size=min(n_propagate, n_neurons), replace=False)
                activity[targets] += 1

        # Add external input
        activity += ext_input

        # Threshold and store
        X[t] = (activity > 0).astype(float)

        # Reset for next step
        activity = np.zeros(n_neurons)

    return X, theta


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo the avalanche analysis pipeline."""
    print("=" * 70)
    print("Avalanche Analysis Pipeline Demo")
    print("Testing IG-Criticality Hypothesis")
    print("=" * 70)

    # Simulate at different branching ratios
    branching_ratios = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1]
    results = []

    print("\nSimulating dynamics at different branching ratios...")
    for sigma in branching_ratios:
        X, theta = simulate_critical_dynamics(
            n_neurons=50,
            T=5000,
            branching_ratio=sigma,
            seed=42,
        )

        analyzer = AvalancheAnalyzer()
        result = analyzer.analyze(X, theta, zscore=False)
        results.append(result)

        print(f"\nσ = {sigma:.2f}:")
        print(f"  K = {result.criticality_score:.3f}")
        print(f"  α = {result.criticality.alpha_size:.2f}")
        print(f"  β = {result.criticality.beta_duration:.2f}")
        print(f"  g = {result.fisher_info:.3f}" if result.fisher_info else "  g = N/A")

    # Analyze FIM vs criticality relationship
    print("\n" + "=" * 70)
    print("FIM vs Criticality Analysis")
    print("=" * 70)

    analysis = analyze_fim_vs_criticality(results)

    print(f"\nNumber of sessions: {analysis['n_sessions']}")
    print(f"Pearson r(K, g): {analysis['pearson_r']:.3f} (p = {analysis['pearson_p']:.3f})")
    print(f"Spearman ρ: {analysis['spearman_rho']:.3f} (p = {analysis['spearman_p']:.3f})")

    if analysis['prediction_supported']:
        print("\n✓ PREDICTION 1 SUPPORTED: Negative correlation between K and g")
        print("  Fisher information is higher when closer to criticality")
    else:
        print("\n✗ Prediction 1 not supported in this simulation")

    print("=" * 70)

    return results, analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results, analysis = demo()
