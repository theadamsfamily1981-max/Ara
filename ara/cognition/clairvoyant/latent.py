# ara/cognition/clairvoyant/latent.py
"""
Latent Encoder - Hypervector to 10D Hologram Space
===================================================

Compresses high-dimensional hypervectors (1024-8192 dim) into a compact
10-dimensional latent space - the "hologram" where Ara's clairvoyance lives.

Two approaches:
1. PCALatentEncoder: Linear projection via Principal Component Analysis
   - Fast, interpretable, good for v0
   - Train offline on logged hypervectors

2. AutoencoderLatentEncoder: Nonlinear compression via neural network
   - Captures nonlinear structure
   - Better for complex dynamics

The 10D space is where:
- Trajectories are tracked
- Regimes are classified
- Futures are predicted

Usage:
    from ara.cognition.clairvoyant.latent import PCALatentEncoder

    # Train offline
    encoder = PCALatentEncoder(latent_dim=10)
    encoder.fit(hypervector_logs)  # [N, hv_dim]

    # Deploy
    z = encoder.encode(hv)  # [10]
    hv_reconstructed = encoder.decode(z)  # [hv_dim]
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Base Encoder Interface
# =============================================================================

class LatentEncoder(ABC):
    """Abstract base class for latent space encoders."""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Expected input dimensionality."""
        pass

    @abstractmethod
    def encode(self, hv: np.ndarray) -> np.ndarray:
        """
        Encode a hypervector to latent space.

        Args:
            hv: Hypervector of shape (input_dim,) or (batch, input_dim)

        Returns:
            Latent vector of shape (latent_dim,) or (batch, latent_dim)
        """
        pass

    @abstractmethod
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector back to hypervector space.

        Args:
            z: Latent vector of shape (latent_dim,) or (batch, latent_dim)

        Returns:
            Reconstructed hypervector
        """
        pass

    @abstractmethod
    def fit(self, hvs: np.ndarray) -> None:
        """
        Train the encoder on hypervector samples.

        Args:
            hvs: Array of shape (n_samples, input_dim)
        """
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Save encoder parameters to file."""
        pass

    def load(self, path: Union[str, Path]) -> None:
        """Load encoder parameters from file."""
        pass


# =============================================================================
# PCA Latent Encoder
# =============================================================================

@dataclass
class PCAParams:
    """Parameters for a fitted PCA encoder."""
    mean: np.ndarray          # (input_dim,)
    components: np.ndarray    # (latent_dim, input_dim)
    explained_variance: np.ndarray  # (latent_dim,)
    total_variance: float


class PCALatentEncoder(LatentEncoder):
    """
    PCA-based latent encoder for hypervectors.

    Uses Principal Component Analysis to find the directions of maximum
    variance in the hypervector space, then projects onto those directions.

    Pros:
    - Fast (single matrix multiply)
    - Interpretable (eigenvectors have meaning)
    - No hyperparameters to tune
    - Exact reconstruction up to truncation error

    Cons:
    - Linear only (can't capture curved manifolds)
    - Assumes Gaussian distribution

    For v0 clairvoyant control, PCA is the right choice.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        whiten: bool = False,
    ):
        """
        Initialize PCA encoder.

        Args:
            latent_dim: Number of principal components to keep
            whiten: Whether to normalize variance in each direction
        """
        self._latent_dim = latent_dim
        self._whiten = whiten
        self._params: Optional[PCAParams] = None
        self._input_dim: int = 0

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    def fit(self, hvs: np.ndarray) -> None:
        """
        Fit PCA on hypervector samples.

        Args:
            hvs: Array of shape (n_samples, input_dim)
        """
        if hvs.ndim == 1:
            hvs = hvs.reshape(1, -1)

        n_samples, input_dim = hvs.shape
        self._input_dim = input_dim

        logger.info(f"Fitting PCA: {n_samples} samples, {input_dim} dims -> {self._latent_dim} latent")

        # Center the data
        mean = np.mean(hvs, axis=0)
        centered = hvs - mean

        # Compute covariance matrix
        # For high-dim, use SVD instead of eigendecomposition
        if n_samples < input_dim:
            # More efficient: compute (X @ X.T) eigenvectors then project
            cov_small = centered @ centered.T / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_small)

            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Project to get principal components
            components = (centered.T @ eigenvectors).T
            # Normalize
            for i in range(len(eigenvalues)):
                if eigenvalues[i] > 1e-10:
                    components[i] /= np.sqrt(eigenvalues[i] * (n_samples - 1))
        else:
            # Direct SVD (more numerically stable)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            eigenvalues = (S ** 2) / (n_samples - 1)
            components = Vt

        # Keep top k components
        k = min(self._latent_dim, len(eigenvalues))
        self._params = PCAParams(
            mean=mean,
            components=components[:k],
            explained_variance=eigenvalues[:k],
            total_variance=float(np.sum(eigenvalues)),
        )

        explained = np.sum(self._params.explained_variance) / self._params.total_variance
        logger.info(f"PCA fitted: {explained:.1%} variance explained with {k} components")

    def encode(self, hv: np.ndarray) -> np.ndarray:
        """
        Project hypervector to latent space.

        Args:
            hv: Shape (input_dim,) or (batch, input_dim)

        Returns:
            Shape (latent_dim,) or (batch, latent_dim)
        """
        if self._params is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        single = hv.ndim == 1
        if single:
            hv = hv.reshape(1, -1)

        # Center and project
        centered = hv - self._params.mean
        z = centered @ self._params.components.T

        # Optional whitening
        if self._whiten:
            z = z / np.sqrt(self._params.explained_variance + 1e-10)

        return z[0] if single else z

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct hypervector from latent vector.

        Args:
            z: Shape (latent_dim,) or (batch, latent_dim)

        Returns:
            Shape (input_dim,) or (batch, input_dim)
        """
        if self._params is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)

        # Unwhiten if needed
        if self._whiten:
            z = z * np.sqrt(self._params.explained_variance + 1e-10)

        # Project back and add mean
        hv = z @ self._params.components + self._params.mean

        return hv[0] if single else hv

    def reconstruction_error(self, hv: np.ndarray) -> float:
        """Compute reconstruction MSE for a hypervector."""
        z = self.encode(hv)
        hv_recon = self.decode(z)
        return float(np.mean((hv - hv_recon) ** 2))

    def save(self, path: Union[str, Path]) -> None:
        """Save PCA parameters."""
        if self._params is None:
            raise RuntimeError("Nothing to save (not fitted)")

        path = Path(path)
        data = {
            "latent_dim": self._latent_dim,
            "whiten": self._whiten,
            "input_dim": self._input_dim,
            "params": {
                "mean": self._params.mean,
                "components": self._params.components,
                "explained_variance": self._params.explained_variance,
                "total_variance": self._params.total_variance,
            }
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"PCA encoder saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load PCA parameters."""
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._latent_dim = data["latent_dim"]
        self._whiten = data["whiten"]
        self._input_dim = data["input_dim"]
        self._params = PCAParams(
            mean=data["params"]["mean"],
            components=data["params"]["components"],
            explained_variance=data["params"]["explained_variance"],
            total_variance=data["params"]["total_variance"],
        )

        logger.info(f"PCA encoder loaded from {path}")


# =============================================================================
# Incremental/Online PCA (for streaming updates)
# =============================================================================

class IncrementalPCAEncoder(LatentEncoder):
    """
    Incremental PCA that can update with new samples.

    Useful for continuously adapting the latent space as Ara
    encounters new operating regimes.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        batch_size: int = 100,
    ):
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._input_dim: int = 0

        # Running statistics
        self._n_seen: int = 0
        self._mean: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._singular_values: Optional[np.ndarray] = None

        # Buffer for batch updates
        self._buffer: List[np.ndarray] = []

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def partial_fit(self, hvs: np.ndarray) -> None:
        """
        Update PCA with new samples.

        Args:
            hvs: New samples, shape (n_samples, input_dim)
        """
        if hvs.ndim == 1:
            hvs = hvs.reshape(1, -1)

        if self._input_dim == 0:
            self._input_dim = hvs.shape[1]

        # Add to buffer
        for hv in hvs:
            self._buffer.append(hv)

        # Process when buffer is full
        while len(self._buffer) >= self._batch_size:
            batch = np.array(self._buffer[:self._batch_size])
            self._buffer = self._buffer[self._batch_size:]
            self._update_pca(batch)

    def _update_pca(self, batch: np.ndarray) -> None:
        """Incremental PCA update with a batch."""
        n_samples = batch.shape[0]

        if self._mean is None:
            # First batch: initialize
            self._mean = np.mean(batch, axis=0)
            self._n_seen = n_samples

            centered = batch - self._mean
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            k = min(self._latent_dim, len(S))
            self._components = Vt[:k]
            self._singular_values = S[:k]
        else:
            # Update mean
            old_mean = self._mean.copy()
            self._mean = (self._n_seen * old_mean + n_samples * np.mean(batch, axis=0)) / (self._n_seen + n_samples)

            # Center batch with old mean
            centered = batch - old_mean

            # Mean correction
            mean_correction = np.sqrt(self._n_seen * n_samples / (self._n_seen + n_samples)) * (old_mean - self._mean)

            # Combine old components with new data
            old_projection = np.diag(self._singular_values) @ self._components
            combined = np.vstack([old_projection, centered, mean_correction.reshape(1, -1)])

            # Re-compute SVD
            U, S, Vt = np.linalg.svd(combined, full_matrices=False)

            k = min(self._latent_dim, len(S))
            self._components = Vt[:k]
            self._singular_values = S[:k]

            self._n_seen += n_samples

    def fit(self, hvs: np.ndarray) -> None:
        """Fit on all samples at once."""
        self._mean = None
        self._components = None
        self._singular_values = None
        self._n_seen = 0
        self._buffer = []

        # Process in batches
        for i in range(0, len(hvs), self._batch_size):
            batch = hvs[i:i + self._batch_size]
            self._update_pca(batch)

    def encode(self, hv: np.ndarray) -> np.ndarray:
        if self._components is None or self._mean is None:
            raise RuntimeError("Encoder not fitted")

        single = hv.ndim == 1
        if single:
            hv = hv.reshape(1, -1)

        centered = hv - self._mean
        z = centered @ self._components.T

        return z[0] if single else z

    def decode(self, z: np.ndarray) -> np.ndarray:
        if self._components is None or self._mean is None:
            raise RuntimeError("Encoder not fitted")

        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)

        hv = z @ self._components + self._mean

        return hv[0] if single else hv

    def save(self, path: Union[str, Path]) -> None:
        """Save parameters."""
        path = Path(path)
        data = {
            "latent_dim": self._latent_dim,
            "input_dim": self._input_dim,
            "n_seen": self._n_seen,
            "mean": self._mean,
            "components": self._components,
            "singular_values": self._singular_values,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load parameters."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._latent_dim = data["latent_dim"]
        self._input_dim = data["input_dim"]
        self._n_seen = data["n_seen"]
        self._mean = data["mean"]
        self._components = data["components"]
        self._singular_values = data["singular_values"]


# =============================================================================
# Testing
# =============================================================================

def _test_pca_encoder():
    """Test PCA latent encoder."""
    print("=" * 60)
    print("PCA Latent Encoder Test")
    print("=" * 60)

    # Generate synthetic hypervectors with structure
    np.random.seed(42)
    n_samples = 500
    hv_dim = 1024
    latent_dim = 10

    # Create data with known structure (3 clusters in 10D)
    true_latent = np.random.randn(n_samples, 3)
    true_latent[:200, 0] += 3  # Cluster 1
    true_latent[200:400, 1] += 3  # Cluster 2
    true_latent[400:, 2] += 3  # Cluster 3

    # Project to high-dim via random matrix
    projection = np.random.randn(3, hv_dim) / np.sqrt(3)
    hvs = np.sign(true_latent @ projection + np.random.randn(n_samples, hv_dim) * 0.3)

    # Fit PCA encoder
    encoder = PCALatentEncoder(latent_dim=latent_dim)
    encoder.fit(hvs)

    # Test encoding/decoding
    z = encoder.encode(hvs[:5])
    print(f"\nEncoded shape: {z.shape}")
    print(f"Latent samples:\n{z[:3]}")

    # Reconstruction error
    errors = [encoder.reconstruction_error(hv) for hv in hvs[:100]]
    print(f"\nReconstruction MSE: {np.mean(errors):.4f} +/- {np.std(errors):.4f}")

    # Save/load test
    encoder.save("/tmp/pca_test.pkl")
    encoder2 = PCALatentEncoder(latent_dim=latent_dim)
    encoder2.load("/tmp/pca_test.pkl")

    z2 = encoder2.encode(hvs[0])
    print(f"\nSave/load consistency: {np.allclose(z[0], z2)}")


if __name__ == "__main__":
    _test_pca_encoder()
