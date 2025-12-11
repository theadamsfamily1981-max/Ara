"""
CTF-1: Critical Thought Field - Core Dynamics

The mathematical foundation:
    x_{t+1} = F_λ(x_t, u_t)

Where:
    - x ∈ ℝⁿ is the state vector
    - λ is the control parameter (spectral radius)
    - F is the nonlinear update rule

Edge-of-chaos function:
    E(λ) = ρ(W) - 1

    E < 0: ordered (perturbations decay)
    E = 0: critical (edge of chaos)
    E > 0: chaotic (perturbations grow)

Capacity functional:
    C(λ) = temporal autocorrelation proxy for I(x_past; x_future)

Thought measure:
    T(λ) = C(λ) · exp(-E(λ)² / σ²)

    Maximal when C is high AND E ≈ 0
"""

import numpy as np
from typing import Optional, Tuple, List


class CriticalCore:
    """
    Dynamical system at the edge of chaos.

    This is the "thought engine" - a recurrent network whose
    spectral radius λ determines its computational regime.
    """

    def __init__(
        self,
        n_dims: int = 100,
        lambda_init: float = 1.0,
        nonlinearity: str = "tanh",
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_dims: Dimension of state space
            lambda_init: Target spectral radius (λ)
            nonlinearity: Activation function ("tanh", "relu", "linear")
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.n = n_dims
        self.lambda_param = lambda_init
        self.nonlinearity = nonlinearity

        # State vector
        self.x = np.random.randn(self.n) * 0.1

        # Weight matrix (random, then scaled to target spectral radius)
        self.W = np.random.randn(self.n, self.n) / np.sqrt(self.n)
        self._set_spectral_radius(self.lambda_param)

        # Trajectory history for computing C(λ)
        self.history: List[np.ndarray] = []
        self.max_history = 1000

    def _set_spectral_radius(self, target_rho: float):
        """Scale W so that ρ(W) = target_rho."""
        eigvals = np.linalg.eigvals(self.W)
        rho = np.max(np.abs(eigvals))
        if rho > 1e-10:
            self.W *= (target_rho / rho)

    def _apply_nonlinearity(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.nonlinearity == "tanh":
            return np.tanh(z)
        elif self.nonlinearity == "relu":
            return np.maximum(0, z)
        elif self.nonlinearity == "linear":
            return z
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")

    def F(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        The update rule: x_{t+1} = F_λ(x_t, u_t)

        Args:
            x: Current state
            u: External input (optional)

        Returns:
            Next state
        """
        if u is None:
            u = np.zeros_like(x)
        elif len(u) < self.n:
            # Pad input if smaller than state
            u_full = np.zeros(self.n)
            u_full[:len(u)] = u
            u = u_full

        z = self.W @ x + u
        return self._apply_nonlinearity(z)

    def step(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance one timestep.

        Args:
            u: External input

        Returns:
            New state x_t
        """
        self.x = self.F(self.x, u)

        # Store history
        self.history.append(self.x.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return self.x

    def reset(self, x0: Optional[np.ndarray] = None):
        """Reset state and history."""
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.random.randn(self.n) * 0.1
        self.history = []

    # =========================================================================
    # Edge-of-chaos function: E(λ)
    # =========================================================================

    def E_spectral(self) -> float:
        """
        Edge function via spectral radius.

        E(λ) = ρ(W) - 1

        Returns:
            E value (0 = critical, <0 = ordered, >0 = chaotic)
        """
        eigvals = np.linalg.eigvals(self.W)
        rho = np.max(np.abs(eigvals))
        return float(rho - 1.0)

    def E_lyapunov(self, n_steps: int = 100) -> float:
        """
        Edge function via Lyapunov exponent estimate.

        Λ = lim_{t→∞} (1/t) ln(|δx_t| / |δx_0|)

        Args:
            n_steps: Number of steps to estimate over

        Returns:
            Estimated maximal Lyapunov exponent
        """
        # Save current state
        x_save = self.x.copy()

        # Initial perturbation
        delta = 1e-8
        x1 = self.x.copy()
        x2 = self.x + np.random.randn(self.n) * delta
        x2 /= np.linalg.norm(x2 - x1) / delta  # Normalize perturbation

        lyap_sum = 0.0
        for _ in range(n_steps):
            x1 = self.F(x1)
            x2 = self.F(x2)

            d = np.linalg.norm(x2 - x1)
            if d > 1e-15:
                lyap_sum += np.log(d / delta)
                # Renormalize
                x2 = x1 + (x2 - x1) * (delta / d)

        # Restore state
        self.x = x_save

        return lyap_sum / n_steps

    # =========================================================================
    # Capacity functional: C(λ)
    # =========================================================================

    def C_autocorr(self, lag: int = 1) -> float:
        """
        Capacity proxy via temporal autocorrelation.

        Higher autocorrelation = longer memory = higher capacity.

        Args:
            lag: Time lag for correlation

        Returns:
            Autocorrelation coefficient
        """
        if len(self.history) < lag + 50:
            return 0.0

        traj = np.array(self.history[-200:])

        # Use first component as proxy
        x0 = traj[:-lag, 0]
        x1 = traj[lag:, 0]

        corr = np.corrcoef(x0, x1)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(np.abs(corr))

    def C_capacity(self) -> float:
        """
        Aggregate capacity measure.

        Sum of autocorrelations at multiple lags.
        """
        if len(self.history) < 100:
            return 0.0

        total = 0.0
        for lag in [1, 2, 5, 10, 20]:
            total += self.C_autocorr(lag)
        return total / 5.0

    # =========================================================================
    # Thought measure: T(λ)
    # =========================================================================

    def T_thought(self, sigma: float = 0.1) -> float:
        """
        Thought measure: C(λ) weighted by proximity to criticality.

        T(λ) = C(λ) · exp(-E(λ)² / σ²)

        Maximal when:
        - C is high (good memory/computation)
        - E ≈ 0 (at edge of chaos)

        Args:
            sigma: Width of criticality window

        Returns:
            Thought measure
        """
        C = self.C_capacity()
        E = self.E_spectral()
        criticality_weight = np.exp(-(E ** 2) / (sigma ** 2))
        return float(C * criticality_weight)

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_spectral_radius(self) -> float:
        """Current spectral radius ρ(W)."""
        eigvals = np.linalg.eigvals(self.W)
        return float(np.max(np.abs(eigvals)))

    def set_lambda(self, new_lambda: float):
        """Change the spectral radius."""
        self.lambda_param = new_lambda
        self._set_spectral_radius(new_lambda)

    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return self.x.copy()

    def get_trajectory(self) -> np.ndarray:
        """Get trajectory history as array."""
        if not self.history:
            return np.array([])
        return np.array(self.history)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("CTF-1: Critical Core Test")
    print("=" * 50)

    for lam in [0.7, 1.0, 1.3]:
        core = CriticalCore(n_dims=100, lambda_init=lam, seed=42)

        # Run for a while
        for _ in range(500):
            u = np.random.randn(1) * 0.1
            core.step(u)

        E = core.E_spectral()
        C = core.C_capacity()
        T = core.T_thought()
        rho = core.get_spectral_radius()

        print(f"λ={lam:.1f}: ρ={rho:.3f}, E={E:+.3f}, C={C:.3f}, T={T:.3f}")

    print("\nExpected: T peaks near λ=1.0")
