# ara/cognition/imagination/world_model.py
"""
Latent World Model - Learning z' = f(z, u)
==========================================

The world model learns how the latent space evolves given actions.
This is what enables predictive thought: "if I do X from here, where do I go?"

Architecture Options:
1. Linear: z' = Az + Bu + c (fast, interpretable, good for v0)
2. MLP: z' = MLP([z, u]) (nonlinear, captures curved dynamics)
3. RNN/Transformer: z' = RNN(z_{t-k:t}, u) (captures temporal context)

Training:
- Input: logged (z_t, u_t, z_{t+1}) tuples
- Loss: ||z_{t+1} - f(z_t, u_t)||^2

Multi-step training:
- Roll out multiple steps, accumulate loss
- This reduces compounding error during planning

Usage:
    world = LatentWorldModel(latent_dim=10, action_dim=8)
    world.fit(z_log, u_log, z_next_log)

    # Single step
    z_next = world.predict(z_t, u_t)

    # Multi-step rollout
    trajectory = world.rollout(z_t, action_sequence)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatentWorldModelConfig:
    """Configuration for latent world model."""
    latent_dim: int = 10
    action_dim: int = 8

    # Architecture
    model_type: str = "mlp"  # "linear", "mlp", "residual"
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"

    # Training
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 10

    # Multi-step training
    multistep_horizon: int = 5
    multistep_weight_decay: float = 0.9  # Discount for future steps


class LatentWorldModel:
    """
    Learns latent space dynamics: z' = f(z, u)

    This is the core of predictive thought - given current state and action,
    predict the next state.
    """

    def __init__(self, config: Optional[LatentWorldModelConfig] = None):
        self.config = config or LatentWorldModelConfig()
        self.latent_dim = self.config.latent_dim
        self.action_dim = self.config.action_dim

        # Model parameters (initialized on fit)
        self._weights: Dict[str, np.ndarray] = {}
        self._is_fitted: bool = False

        # Training stats
        self._train_loss_history: List[float] = []
        self._val_loss_history: List[float] = []

        logger.info(f"LatentWorldModel: {self.config.model_type}, "
                    f"z_dim={self.latent_dim}, u_dim={self.action_dim}")

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _init_weights(self):
        """Initialize model weights."""
        rng = np.random.default_rng(42)
        input_dim = self.latent_dim + self.action_dim

        if self.config.model_type == "linear":
            # z' = Az + Bu + c
            self._weights = {
                "A": rng.randn(self.latent_dim, self.latent_dim) * 0.1,
                "B": rng.randn(self.latent_dim, self.action_dim) * 0.1,
                "c": np.zeros(self.latent_dim),
            }

        elif self.config.model_type == "mlp":
            # MLP: input -> hidden -> ... -> output
            dims = [input_dim] + self.config.hidden_dims + [self.latent_dim]
            self._weights = {}

            for i in range(len(dims) - 1):
                scale = np.sqrt(2.0 / dims[i])  # He init
                self._weights[f"W{i}"] = rng.randn(dims[i+1], dims[i]) * scale
                self._weights[f"b{i}"] = np.zeros(dims[i+1])

        elif self.config.model_type == "residual":
            # Residual: z' = z + MLP([z, u])
            dims = [input_dim] + self.config.hidden_dims + [self.latent_dim]
            self._weights = {}

            for i in range(len(dims) - 1):
                scale = np.sqrt(2.0 / dims[i])
                self._weights[f"W{i}"] = rng.randn(dims[i+1], dims[i]) * scale
                self._weights[f"b{i}"] = np.zeros(dims[i+1])

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == "tanh":
            return np.tanh(x)
        elif self.config.activation == "relu":
            return np.maximum(0, x)
        elif self.config.activation == "elu":
            return np.where(x > 0, x, np.exp(x) - 1)
        else:
            return x

    def _forward(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Forward pass through the model."""
        if self.config.model_type == "linear":
            # z' = Az + Bu + c
            return (self._weights["A"] @ z +
                    self._weights["B"] @ u +
                    self._weights["c"])

        elif self.config.model_type in ("mlp", "residual"):
            # Concatenate inputs
            x = np.concatenate([z, u])

            # Forward through layers
            n_layers = len([k for k in self._weights if k.startswith("W")])
            for i in range(n_layers):
                x = self._weights[f"W{i}"] @ x + self._weights[f"b{i}"]
                if i < n_layers - 1:  # Apply activation except last layer
                    x = self._activation(x)

            if self.config.model_type == "residual":
                return z + x  # Residual connection
            return x

        raise ValueError(f"Unknown model type: {self.config.model_type}")

    def predict(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Predict next latent state.

        Args:
            z: Current latent state (latent_dim,)
            u: Action (action_dim,)

        Returns:
            Predicted next state (latent_dim,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        z = np.asarray(z).flatten()
        u = np.asarray(u).flatten()

        # Pad action if needed
        if len(u) < self.action_dim:
            u = np.concatenate([u, np.zeros(self.action_dim - len(u))])

        return self._forward(z, u)

    def rollout(
        self,
        z_init: np.ndarray,
        actions: np.ndarray,
        include_start: bool = True,
    ) -> np.ndarray:
        """
        Roll out trajectory given action sequence.

        Args:
            z_init: Starting latent state
            actions: Action sequence (horizon, action_dim)
            include_start: Include z_init in output

        Returns:
            Trajectory (horizon+1, latent_dim) or (horizon, latent_dim)
        """
        z = np.asarray(z_init).flatten()
        actions = np.atleast_2d(actions)

        trajectory = [z.copy()] if include_start else []

        for u in actions:
            z = self.predict(z, u)
            trajectory.append(z.copy())

        return np.array(trajectory)

    def fit(
        self,
        z: np.ndarray,
        u: np.ndarray,
        z_next: np.ndarray,
        val_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train the world model on logged transitions.

        Args:
            z: Current states (n_samples, latent_dim)
            u: Actions (n_samples, action_dim)
            z_next: Next states (n_samples, latent_dim)
            val_split: Validation split fraction

        Returns:
            Training history dict
        """
        z = np.atleast_2d(z)
        u = np.atleast_2d(u)
        z_next = np.atleast_2d(z_next)

        n_samples = len(z)
        n_val = int(n_samples * val_split)

        # Split data
        indices = np.random.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        z_train, u_train, z_next_train = z[train_idx], u[train_idx], z_next[train_idx]
        z_val, u_val, z_next_val = z[val_idx], u[val_idx], z_next[val_idx]

        # Initialize weights
        self._init_weights()

        logger.info(f"Training world model: {len(train_idx)} train, {len(val_idx)} val")

        self._train_loss_history = []
        self._val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            # Shuffle training data
            perm = np.random.permutation(len(z_train))

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(z_train), self.config.batch_size):
                batch_idx = perm[i:i + self.config.batch_size]
                z_batch = z_train[batch_idx]
                u_batch = u_train[batch_idx]
                z_next_batch = z_next_train[batch_idx]

                # Forward + backward (simple gradient descent)
                loss, grads = self._compute_loss_and_grads(
                    z_batch, u_batch, z_next_batch
                )

                # Update weights
                for key in self._weights:
                    if key in grads:
                        self._weights[key] -= self.config.learning_rate * grads[key]

                epoch_loss += loss
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            self._train_loss_history.append(train_loss)

            # Validation
            val_loss = self._compute_loss(z_val, u_val, z_next_val)
            self._val_loss_history.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

        self._is_fitted = True
        logger.info(f"Training complete: final val_loss={val_loss:.6f}")

        return {
            "train_loss": self._train_loss_history,
            "val_loss": self._val_loss_history,
        }

    def _compute_loss(
        self,
        z: np.ndarray,
        u: np.ndarray,
        z_next: np.ndarray,
    ) -> float:
        """Compute MSE loss over a batch."""
        total_loss = 0.0
        for i in range(len(z)):
            pred = self._forward(z[i], u[i])
            total_loss += np.mean((pred - z_next[i]) ** 2)
        return total_loss / len(z)

    def _compute_loss_and_grads(
        self,
        z: np.ndarray,
        u: np.ndarray,
        z_next: np.ndarray,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Compute loss and gradients for a batch.

        Using numerical gradients for simplicity - in production you'd use
        autodiff (PyTorch/JAX).
        """
        eps = 1e-5
        loss = self._compute_loss(z, u, z_next)

        grads = {}
        for key in self._weights:
            grad = np.zeros_like(self._weights[key])

            # Flatten for numerical gradient
            flat = self._weights[key].flatten()
            for i in range(len(flat)):
                # Forward difference
                flat[i] += eps
                self._weights[key] = flat.reshape(self._weights[key].shape)
                loss_plus = self._compute_loss(z, u, z_next)

                flat[i] -= 2 * eps
                self._weights[key] = flat.reshape(self._weights[key].shape)
                loss_minus = self._compute_loss(z, u, z_next)

                grad.flat[i] = (loss_plus - loss_minus) / (2 * eps)

                # Restore
                flat[i] += eps
                self._weights[key] = flat.reshape(self._weights[key].shape)

            grads[key] = grad

        return loss, grads

    def prediction_error(self, z: np.ndarray, u: np.ndarray, z_next: np.ndarray) -> float:
        """Compute prediction error for a single transition."""
        pred = self.predict(z, u)
        return float(np.mean((pred - z_next) ** 2))

    def save(self, path: Union[str, Path]) -> None:
        """Save model to file."""
        path = Path(path)
        data = {
            "config": self.config,
            "weights": self._weights,
            "is_fitted": self._is_fitted,
            "train_loss": self._train_loss_history,
            "val_loss": self._val_loss_history,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"World model saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load model from file."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.config = data["config"]
        self._weights = data["weights"]
        self._is_fitted = data["is_fitted"]
        self._train_loss_history = data.get("train_loss", [])
        self._val_loss_history = data.get("val_loss", [])

        self.latent_dim = self.config.latent_dim
        self.action_dim = self.config.action_dim

        logger.info(f"World model loaded from {path}")


# =============================================================================
# Ensemble World Model (for uncertainty)
# =============================================================================

class EnsembleWorldModel:
    """
    Ensemble of world models for uncertainty quantification.

    Uses multiple models trained on different data subsets.
    Prediction uncertainty = disagreement between models.
    """

    def __init__(
        self,
        n_models: int = 5,
        config: Optional[LatentWorldModelConfig] = None,
    ):
        self.n_models = n_models
        self.config = config or LatentWorldModelConfig()
        self.models: List[LatentWorldModel] = []

    def fit(
        self,
        z: np.ndarray,
        u: np.ndarray,
        z_next: np.ndarray,
    ) -> None:
        """Train ensemble with bootstrap sampling."""
        n_samples = len(z)
        self.models = []

        for i in range(self.n_models):
            logger.info(f"Training ensemble model {i+1}/{self.n_models}")

            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            z_boot = z[indices]
            u_boot = u[indices]
            z_next_boot = z_next[indices]

            model = LatentWorldModel(self.config)
            model.fit(z_boot, u_boot, z_next_boot)
            self.models.append(model)

    def predict_with_uncertainty(
        self,
        z: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimate.

        Returns:
            (mean_prediction, std_prediction)
        """
        predictions = np.array([m.predict(z, u) for m in self.models])
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std

    def rollout_with_uncertainty(
        self,
        z_init: np.ndarray,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rollout with uncertainty propagation.

        Returns:
            (mean_trajectory, std_trajectory)
        """
        trajectories = np.array([
            m.rollout(z_init, actions, include_start=True)
            for m in self.models
        ])
        mean = trajectories.mean(axis=0)
        std = trajectories.std(axis=0)
        return mean, std


# =============================================================================
# Testing
# =============================================================================

def _test_world_model():
    """Test world model."""
    print("=" * 60)
    print("Latent World Model Test")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    latent_dim = 10
    action_dim = 4

    # Simple linear dynamics: z' = 0.9*z + 0.3*u + noise
    A = np.eye(latent_dim) * 0.9
    B = np.random.randn(latent_dim, action_dim) * 0.3

    z = np.random.randn(n_samples, latent_dim)
    u = np.random.randn(n_samples, action_dim)
    z_next = z @ A.T + u @ B.T + np.random.randn(n_samples, latent_dim) * 0.1

    # Train model
    config = LatentWorldModelConfig(
        latent_dim=latent_dim,
        action_dim=action_dim,
        model_type="linear",
        max_epochs=50,
    )
    model = LatentWorldModel(config)
    history = model.fit(z, u, z_next)

    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")

    # Test prediction
    z_test = np.random.randn(latent_dim)
    u_test = np.random.randn(action_dim)
    z_pred = model.predict(z_test, u_test)
    z_true = A @ z_test + B @ u_test

    print(f"\nPrediction test:")
    print(f"  True next (noiseless): {z_true[:3]}...")
    print(f"  Predicted: {z_pred[:3]}...")

    # Test rollout
    actions = np.random.randn(5, action_dim)
    trajectory = model.rollout(z_test, actions)
    print(f"\nRollout shape: {trajectory.shape}")


if __name__ == "__main__":
    _test_world_model()
