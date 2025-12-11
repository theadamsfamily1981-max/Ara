"""
CTF-1: Self-Organized Criticality (SOC) Learner

Maintains the system at λ = 1 while learning a task.

Key insight: After each weight update, renormalize the reservoir
to maintain spectral radius = 1. This is the "SOC rule" that
keeps the system at the edge of chaos during learning.

Learning dynamics:
    dθ/dt = η · (α∇C - β∇|E|²)

In practice, we:
1. Train W_out via ridge regression (task objective)
2. Renormalize W_res to ρ = 1 (criticality constraint)
"""

import numpy as np
from typing import List, Tuple, Optional
from .critical_core import CriticalCore


class SOCLearner:
    """
    Reservoir computer with self-organized criticality.

    Architecture:
        u_t → [W_in] → x_t → [W_res (fixed, SOC)] → x_{t+1}
                              ↓
                         [W_out (learned)] → y_t
    """

    def __init__(
        self,
        core: CriticalCore,
        n_input: int,
        n_output: int,
        lambda_target: float = 1.0,
        input_scaling: float = 0.1,
    ):
        """
        Args:
            core: CriticalCore instance (the reservoir)
            n_input: Input dimension
            n_output: Output dimension
            lambda_target: Target spectral radius (default 1.0 = critical)
            input_scaling: Scale factor for input weights
        """
        self.core = core
        self.n_input = n_input
        self.n_output = n_output
        self.lambda_target = lambda_target

        # Input weights (fixed, random)
        self.W_in = np.random.randn(core.n, n_input) * input_scaling

        # Output weights (learned)
        self.W_out = np.random.randn(n_output, core.n) / np.sqrt(core.n)

        # Metrics history
        self.lambda_hist: List[float] = []
        self.E_hist: List[float] = []
        self.C_hist: List[float] = []
        self.loss_hist: List[float] = []

    def _renormalize_reservoir(self):
        """
        SOC rule: Renormalize reservoir to target spectral radius.

        This is the key constraint that maintains criticality.
        """
        eigvals = np.linalg.eigvals(self.core.W)
        rho = np.max(np.abs(eigvals))
        if rho > 1e-10:
            self.core.W *= (self.lambda_target / rho)

    def forward(self, u: np.ndarray) -> np.ndarray:
        """
        Forward pass: input → reservoir → output.

        Args:
            u: Input vector (n_input,)

        Returns:
            Output vector (n_output,)
        """
        # Project input into reservoir space
        u_res = self.W_in @ u

        # Update reservoir state
        x = self.core.step(u_res)

        # Read out
        y = self.W_out @ x
        return y

    def collect_states(
        self,
        inputs: np.ndarray,
        washout: int = 50,
    ) -> np.ndarray:
        """
        Drive reservoir with inputs, collect states.

        Args:
            inputs: Input sequence (T, n_input)
            washout: Initial steps to discard (transient)

        Returns:
            States array (T - washout, n_reservoir)
        """
        self.core.reset()

        states = []
        for t, u in enumerate(inputs):
            _ = self.forward(u)
            if t >= washout:
                states.append(self.core.x.copy())

        return np.array(states)

    def train_readout(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        reg: float = 1e-4,
        washout: int = 50,
    ) -> float:
        """
        Train output weights via ridge regression.

        Args:
            inputs: Input sequence (T, n_input)
            targets: Target sequence (T, n_output)
            reg: Regularization strength
            washout: Initial steps to discard

        Returns:
            Training MSE
        """
        # Collect reservoir states
        X = self.collect_states(inputs, washout)

        # Align targets
        Y = targets[washout:]

        # Ridge regression: W_out = Y^T X (X^T X + λI)^{-1}
        XtX = X.T @ X + reg * np.eye(X.shape[1])
        XtY = X.T @ Y
        self.W_out = np.linalg.solve(XtX, XtY).T

        # Compute training error
        Y_pred = X @ self.W_out.T
        mse = np.mean((Y - Y_pred) ** 2)

        return float(mse)

    def train_epoch_soc(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        reg: float = 1e-4,
        washout: int = 50,
    ) -> Tuple[float, dict]:
        """
        One training epoch with SOC constraint.

        1. Train W_out (ridge regression)
        2. Renormalize W_res to λ = 1 (SOC)
        3. Log metrics

        Args:
            inputs: Input sequence
            targets: Target sequence
            reg: Regularization
            washout: Washout period

        Returns:
            (mse, metrics_dict)
        """
        # Train readout
        mse = self.train_readout(inputs, targets, reg, washout)

        # SOC: Renormalize reservoir
        self._renormalize_reservoir()

        # Compute metrics
        E = self.core.E_spectral()
        C = self.core.C_capacity()
        rho = self.core.get_spectral_radius()

        # Store history
        self.E_hist.append(E)
        self.C_hist.append(C)
        self.lambda_hist.append(rho)
        self.loss_hist.append(mse)

        metrics = {
            'mse': mse,
            'E': E,
            'C': C,
            'lambda': rho,
        }

        return mse, metrics

    def evaluate(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        washout: int = 50,
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate on a sequence.

        Args:
            inputs: Input sequence
            targets: Target sequence
            washout: Washout period

        Returns:
            (mse, predictions)
        """
        X = self.collect_states(inputs, washout)
        Y = targets[washout:]
        Y_pred = X @ self.W_out.T

        mse = np.mean((Y - Y_pred) ** 2)
        return float(mse), Y_pred


class NoSOCLearner(SOCLearner):
    """
    Control: Same architecture but WITHOUT SOC constraint.

    Used to compare SOC vs no-SOC performance.
    """

    def train_epoch_soc(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        reg: float = 1e-4,
        washout: int = 50,
    ) -> Tuple[float, dict]:
        """
        Training WITHOUT SOC renormalization.
        """
        # Train readout (same as SOC)
        mse = self.train_readout(inputs, targets, reg, washout)

        # NO SOC: Skip renormalization
        # self._renormalize_reservoir()  # DISABLED

        # Compute metrics
        E = self.core.E_spectral()
        C = self.core.C_capacity()
        rho = self.core.get_spectral_radius()

        self.E_hist.append(E)
        self.C_hist.append(C)
        self.lambda_hist.append(rho)
        self.loss_hist.append(mse)

        metrics = {
            'mse': mse,
            'E': E,
            'C': C,
            'lambda': rho,
        }

        return mse, metrics


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("CTF-1: SOC Learner Test")
    print("=" * 50)

    # Create reservoir
    core = CriticalCore(n_dims=100, lambda_init=1.0, seed=42)

    # Create learner
    learner = SOCLearner(core, n_input=1, n_output=1)

    # Generate simple task: delayed copy
    T = 500
    delay = 5
    inputs = np.random.randn(T, 1)
    targets = np.zeros((T, 1))
    targets[delay:, 0] = inputs[:-delay, 0]

    # Train
    for epoch in range(5):
        mse, metrics = learner.train_epoch_soc(inputs, targets)
        print(f"Epoch {epoch+1}: MSE={mse:.4f}, λ={metrics['lambda']:.3f}, "
              f"E={metrics['E']:+.3f}, C={metrics['C']:.3f}")

    print("\nExpected: λ stays at 1.0, E stays near 0")
