"""
ara.hgf.core - Core HGF Belief Update Equations

Implements the Hierarchical Gaussian Filter update equations for 2-level and
3-level models. This is the mathematical heart of the HGF.

The HGF is a hierarchical Bayesian model where beliefs at each level inform
the prior precision of the level below. This creates a natural mechanism for
learning about volatility (how quickly the environment is changing).

Key equations:
    Level 1 (Sensory): μ₁ update based on prediction error δ₁
    Level 2 (Hidden State): μ₂ update based on prediction error δ₂
    Level 3 (Volatility): μ₃ update based on prediction error δ₃

The precision-weighting of prediction errors implements optimal Bayesian
inference under hierarchical uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Utility Functions
# =============================================================================

def sigmoid(x: float) -> float:
    """Standard logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax function with temperature scaling."""
    x_scaled = x / temperature
    x_max = np.max(x_scaled)
    exp_x = np.exp(x_scaled - x_max)
    return exp_x / np.sum(exp_x)


def safe_exp(x: float, max_val: float = 700.0) -> float:
    """Exponential with overflow protection."""
    return math.exp(min(x, max_val))


def safe_log(x: float, min_val: float = 1e-10) -> float:
    """Logarithm with underflow protection."""
    return math.log(max(x, min_val))


# =============================================================================
# HGF State and Parameters
# =============================================================================

@dataclass
class HGFParams:
    """
    Parameters for the Hierarchical Gaussian Filter.

    Level-specific parameters:
        omega_1: Tonic log-volatility at level 1 (sensory noise)
        omega_2: Tonic log-volatility at level 2 (hidden state volatility)
        omega_3: Tonic log-volatility at level 3 (meta-volatility)

        kappa_1: Coupling strength from level 3 to level 2 variance
                 Controls how much volatility beliefs affect learning rate

        theta: Response model inverse temperature (action selection)

    Initial beliefs:
        mu_1_0: Initial belief about level 1 (observation)
        mu_2_0: Initial belief about level 2 (hidden state, logit space)
        mu_3_0: Initial belief about level 3 (log-volatility)

        sigma_1_0: Initial uncertainty at level 1
        sigma_2_0: Initial uncertainty at level 2
        sigma_3_0: Initial uncertainty at level 3
    """
    # Tonic volatilities (log-space)
    omega_1: float = -3.0  # Sensory noise
    omega_2: float = -4.0  # Hidden state volatility
    omega_3: float = -6.0  # Meta-volatility (for 3-level)

    # Coupling strength
    kappa_1: float = 1.0  # Level 3 → Level 2 coupling

    # Response model
    theta: float = 1.0  # Inverse temperature

    # Initial beliefs (means)
    mu_1_0: float = 0.5  # Initial observation belief
    mu_2_0: float = 0.0  # Initial hidden state (logit space, 0 = 50%)
    mu_3_0: float = 1.0  # Initial log-volatility

    # Initial uncertainties (variances)
    sigma_1_0: float = 0.1
    sigma_2_0: float = 1.0
    sigma_3_0: float = 1.0

    # Bounds for stability
    sigma_min: float = 1e-4
    sigma_max: float = 1e4
    mu_3_min: float = -10.0
    mu_3_max: float = 10.0

    def validate(self) -> None:
        """Check parameter validity."""
        assert self.sigma_1_0 > 0, "sigma_1_0 must be positive"
        assert self.sigma_2_0 > 0, "sigma_2_0 must be positive"
        assert self.sigma_3_0 > 0, "sigma_3_0 must be positive"
        assert self.theta > 0, "theta must be positive"


@dataclass
class HGFState:
    """
    Current state of the HGF (beliefs at all levels).

    Attributes:
        mu_1: Current belief about level 1 (observation space, [0,1])
        mu_2: Current belief about level 2 (logit space)
        mu_3: Current belief about level 3 (log-volatility)

        sigma_1: Current uncertainty at level 1
        sigma_2: Current uncertainty at level 2
        sigma_3: Current uncertainty at level 3

        delta_1: Prediction error at level 1 (sensory PE)
        delta_2: Prediction error at level 2 (volatility PE)
        delta_3: Prediction error at level 3 (meta-volatility PE)

        pi_1: Precision of sensory input
        pi_hat_2: Prior precision at level 2
        w_2: Precision-weighted PE (Kalman gain analog)

        t: Current trial number
    """
    # Beliefs (means)
    mu_1: float = 0.5
    mu_2: float = 0.0
    mu_3: float = 1.0

    # Uncertainties (variances)
    sigma_1: float = 0.1
    sigma_2: float = 1.0
    sigma_3: float = 1.0

    # Prediction errors
    delta_1: float = 0.0
    delta_2: float = 0.0
    delta_3: float = 0.0

    # Precisions and weights
    pi_1: float = 1.0        # Sensory precision
    pi_hat_2: float = 1.0    # Prior precision at level 2
    w_2: float = 0.5         # Precision weight (Kalman gain)

    # Trial counter
    t: int = 0

    @classmethod
    def from_params(cls, params: HGFParams) -> "HGFState":
        """Initialize state from parameters."""
        return cls(
            mu_1=params.mu_1_0,
            mu_2=params.mu_2_0,
            mu_3=params.mu_3_0,
            sigma_1=params.sigma_1_0,
            sigma_2=params.sigma_2_0,
            sigma_3=params.sigma_3_0,
            t=0,
        )

    def get_prediction(self) -> float:
        """Get current prediction in observation space [0,1]."""
        return sigmoid(self.mu_2)

    def get_uncertainty(self) -> float:
        """Get current predictive uncertainty."""
        return self.sigma_2

    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        return safe_exp(self.mu_3)

    def clone(self) -> "HGFState":
        """Create a deep copy of the state."""
        return HGFState(
            mu_1=self.mu_1,
            mu_2=self.mu_2,
            mu_3=self.mu_3,
            sigma_1=self.sigma_1,
            sigma_2=self.sigma_2,
            sigma_3=self.sigma_3,
            delta_1=self.delta_1,
            delta_2=self.delta_2,
            delta_3=self.delta_3,
            pi_1=self.pi_1,
            pi_hat_2=self.pi_hat_2,
            w_2=self.w_2,
            t=self.t,
        )


# =============================================================================
# 2-Level HGF Update
# =============================================================================

def hgf_update_2level(
    state: HGFState,
    observation: float,
    params: HGFParams,
) -> HGFState:
    """
    Perform one update step of the 2-level HGF.

    This is the standard HGF for binary outcomes (e.g., win/loss).

    Args:
        state: Current HGF state
        observation: New observation (0 or 1 for binary)
        params: HGF parameters

    Returns:
        Updated HGF state
    """
    # Unpack current state
    mu_2 = state.mu_2
    sigma_2 = state.sigma_2

    # === Level 1: Observation ===
    # Prediction in observation space
    mu_1_hat = sigmoid(mu_2)

    # Prediction error at level 1 (sensory PE)
    delta_1 = observation - mu_1_hat

    # === Level 2: Hidden State ===
    # Prior variance (increases with volatility)
    sigma_2_hat = sigma_2 + safe_exp(params.omega_2)

    # Sensory precision (inverse of Bernoulli variance)
    # For Bernoulli: var = p(1-p), so pi = 1/(p(1-p))
    pi_1 = 1.0 / max(mu_1_hat * (1.0 - mu_1_hat), params.sigma_min)

    # Prior precision at level 2
    pi_hat_2 = 1.0 / max(sigma_2_hat, params.sigma_min)

    # Precision-weighted prediction error weight (Kalman gain analog)
    # w_2 = pi_hat_2 / (pi_hat_2 + pi_1)
    # But for the HGF binary model, we use a different form
    w_2 = sigma_2_hat

    # Update mean at level 2
    # The key HGF update: precision-weighted PE
    mu_2_new = mu_2 + sigma_2_hat * delta_1

    # Update variance at level 2
    # Posterior precision = prior precision + information from observation
    sigma_2_new = 1.0 / (pi_hat_2 + pi_1)

    # Clamp for stability
    sigma_2_new = max(params.sigma_min, min(params.sigma_max, sigma_2_new))

    # Prediction error at level 2 (volatility PE)
    # This is the squared PE minus expected squared PE
    delta_2 = (
        (sigma_2_new + (mu_2_new - mu_2) ** 2) / sigma_2_hat
        - 1.0
    )

    # === Build new state ===
    new_state = HGFState(
        mu_1=observation,
        mu_2=mu_2_new,
        mu_3=state.mu_3,  # Fixed in 2-level
        sigma_1=mu_1_hat * (1.0 - mu_1_hat),  # Bernoulli variance
        sigma_2=sigma_2_new,
        sigma_3=state.sigma_3,
        delta_1=delta_1,
        delta_2=delta_2,
        delta_3=0.0,
        pi_1=pi_1,
        pi_hat_2=pi_hat_2,
        w_2=w_2,
        t=state.t + 1,
    )

    return new_state


# =============================================================================
# 3-Level HGF Update
# =============================================================================

def hgf_update_3level(
    state: HGFState,
    observation: float,
    params: HGFParams,
) -> HGFState:
    """
    Perform one update step of the 3-level HGF.

    The 3-level HGF adds a volatility learning level, allowing the agent
    to learn about how quickly the environment is changing.

    This is crucial for computational psychiatry applications where
    pathological parameter values (esp. kappa_1) model disorders.

    Args:
        state: Current HGF state
        observation: New observation (0 or 1 for binary)
        params: HGF parameters

    Returns:
        Updated HGF state
    """
    # Unpack current state
    mu_2 = state.mu_2
    mu_3 = state.mu_3
    sigma_2 = state.sigma_2
    sigma_3 = state.sigma_3

    # === Level 1: Observation ===
    # Prediction in observation space
    mu_1_hat = sigmoid(mu_2)

    # Prediction error at level 1 (sensory PE)
    delta_1 = observation - mu_1_hat

    # Sensory precision
    pi_1 = 1.0 / max(mu_1_hat * (1.0 - mu_1_hat), params.sigma_min)

    # === Level 2: Hidden State ===
    # Prior variance - THE KEY EQUATION
    # sigma_2_hat = sigma_2 + exp(kappa_1 * mu_3 + omega_2)
    # This is where level 3 (volatility) affects level 2 (hidden state)
    log_variance_increment = params.kappa_1 * mu_3 + params.omega_2
    sigma_2_hat = sigma_2 + safe_exp(log_variance_increment)

    # Prior precision at level 2
    pi_hat_2 = 1.0 / max(sigma_2_hat, params.sigma_min)

    # Precision weight
    w_2 = sigma_2_hat

    # Update mean at level 2
    mu_2_new = mu_2 + sigma_2_hat * delta_1

    # Update variance at level 2
    sigma_2_new = 1.0 / (pi_hat_2 + pi_1)
    sigma_2_new = max(params.sigma_min, min(params.sigma_max, sigma_2_new))

    # Prediction error at level 2 (volatility PE)
    delta_2 = (
        (sigma_2_new + (mu_2_new - mu_2) ** 2) / sigma_2_hat
        - 1.0
    )

    # === Level 3: Volatility (Meta-Learning) ===
    # Prior variance at level 3
    sigma_3_hat = sigma_3 + safe_exp(params.omega_3)

    # Prior precision at level 3
    pi_hat_3 = 1.0 / max(sigma_3_hat, params.sigma_min)

    # Precision weight for level 3 update
    # This involves the derivative of level 2's variance w.r.t. mu_3
    # d(sigma_2_hat)/d(mu_3) = kappa_1 * exp(kappa_1 * mu_3 + omega_2)
    variance_increment = safe_exp(log_variance_increment)
    d_sigma2_d_mu3 = params.kappa_1 * variance_increment

    # Effective precision gain from level 2
    w_3 = 0.5 * (params.kappa_1 ** 2) * variance_increment / sigma_2_hat

    # Update mean at level 3
    mu_3_new = mu_3 + 0.5 * sigma_3_hat * (params.kappa_1 * variance_increment / sigma_2_hat) * delta_2

    # Clamp mu_3 for stability
    mu_3_new = max(params.mu_3_min, min(params.mu_3_max, mu_3_new))

    # Update variance at level 3
    sigma_3_new = 1.0 / (pi_hat_3 + w_3)
    sigma_3_new = max(params.sigma_min, min(params.sigma_max, sigma_3_new))

    # Prediction error at level 3 (meta-volatility PE)
    delta_3 = (
        (sigma_3_new + (mu_3_new - mu_3) ** 2) / sigma_3_hat
        - 1.0
    )

    # === Build new state ===
    new_state = HGFState(
        mu_1=observation,
        mu_2=mu_2_new,
        mu_3=mu_3_new,
        sigma_1=mu_1_hat * (1.0 - mu_1_hat),
        sigma_2=sigma_2_new,
        sigma_3=sigma_3_new,
        delta_1=delta_1,
        delta_2=delta_2,
        delta_3=delta_3,
        pi_1=pi_1,
        pi_hat_2=pi_hat_2,
        w_2=w_2,
        t=state.t + 1,
    )

    return new_state


# =============================================================================
# Continuous HGF (for real-valued observations)
# =============================================================================

def hgf_update_continuous(
    state: HGFState,
    observation: float,
    params: HGFParams,
    observation_noise: float = 1.0,
) -> HGFState:
    """
    Perform one update step of the continuous HGF.

    For real-valued observations (e.g., stock prices, reaction times).

    Args:
        state: Current HGF state
        observation: New observation (real-valued)
        params: HGF parameters
        observation_noise: Known observation noise variance

    Returns:
        Updated HGF state
    """
    # Unpack current state
    mu_1 = state.mu_1
    mu_2 = state.mu_2
    sigma_1 = state.sigma_1
    sigma_2 = state.sigma_2

    # === Level 1 ===
    # Prior variance
    sigma_1_hat = sigma_1 + safe_exp(params.omega_1)

    # Precision
    pi_1 = 1.0 / observation_noise
    pi_hat_1 = 1.0 / max(sigma_1_hat, params.sigma_min)

    # Update
    mu_1_new = (pi_hat_1 * mu_1 + pi_1 * observation) / (pi_hat_1 + pi_1)
    sigma_1_new = 1.0 / (pi_hat_1 + pi_1)

    # Prediction error
    delta_1 = observation - mu_1

    # === Level 2 ===
    # Prior variance
    sigma_2_hat = sigma_2 + safe_exp(params.omega_2)
    pi_hat_2 = 1.0 / max(sigma_2_hat, params.sigma_min)

    # Prediction error at level 2
    delta_2 = ((sigma_1_new + (mu_1_new - mu_1) ** 2) / sigma_1_hat) - 1.0

    # Update
    mu_2_new = mu_2 + 0.5 * sigma_2_hat * delta_2
    sigma_2_new = 1.0 / (pi_hat_2 + 0.5 * safe_exp(params.omega_1) / sigma_1_hat)
    sigma_2_new = max(params.sigma_min, min(params.sigma_max, sigma_2_new))

    # === Build new state ===
    new_state = HGFState(
        mu_1=mu_1_new,
        mu_2=mu_2_new,
        mu_3=state.mu_3,
        sigma_1=sigma_1_new,
        sigma_2=sigma_2_new,
        sigma_3=state.sigma_3,
        delta_1=delta_1,
        delta_2=delta_2,
        delta_3=0.0,
        pi_1=pi_1,
        pi_hat_2=pi_hat_2,
        w_2=sigma_2_hat,
        t=state.t + 1,
    )

    return new_state


# =============================================================================
# Response Models
# =============================================================================

def response_model_softmax(
    prediction: float,
    theta: float = 1.0,
) -> float:
    """
    Softmax response model for binary choices.

    Returns probability of choosing option 1.

    Args:
        prediction: Model's prediction (probability of outcome 1)
        theta: Inverse temperature (higher = more deterministic)

    Returns:
        Action probability for option 1
    """
    # Transform to log-odds
    log_odds = safe_log(prediction) - safe_log(1.0 - prediction)
    # Apply temperature and transform back
    return sigmoid(theta * log_odds)


def sample_action(prediction: float, theta: float = 1.0) -> int:
    """
    Sample an action given prediction and temperature.

    Args:
        prediction: Model's prediction (probability of outcome 1)
        theta: Inverse temperature

    Returns:
        Sampled action (0 or 1)
    """
    p_action_1 = response_model_softmax(prediction, theta)
    return int(np.random.random() < p_action_1)
