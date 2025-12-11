# ara/cognition/clairvoyant/hypervector.py
"""
Cathedral Hypervector Encoder
=============================

Encodes state features from StateSampler into high-dimensional hypervectors
for the clairvoyant control loop.

Uses VSA/HD Computing operations:
- Binding (XOR/multiply): Creates dissimilar HV from two inputs
- Bundling (sum + threshold): Creates similar HV from multiple inputs
- Temporal binding: Encodes recent history via permutation

The hypervector represents "now + a bit of recent past" and gets compressed
to 10D by the LatentEncoder for trajectory tracking.

Performance Notes:
- Uses numpy for AVX2-friendly vectorized operations
- 1024-dim HVs fit in L2 cache for fast inference
- Bipolar {-1, +1} representation for efficient binding
"""

from __future__ import annotations

import hashlib
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class HypervectorConfig:
    """Configuration for hypervector encoding."""
    dim: int = 1024                  # Hypervector dimensionality
    temporal_steps: int = 3          # History steps to encode
    temporal_decay: float = 0.7      # Weight decay for older states
    seed: int = 42                   # Random seed for reproducibility
    use_value_encoding: bool = True  # Encode continuous values (not just bins)


class CathedralHypervectorEncoder:
    """
    Encodes cathedral state features into hypervectors.

    Each tick:
    1. Receives feature dict from StateSampler
    2. Encodes each feature via: key_vector * scaled_value
    3. Bundles all features into single HV
    4. Optionally binds with permuted history for temporal context

    The result is a high-dimensional representation of "now" that:
    - Is nearly orthogonal to unrelated states
    - Has graded similarity to similar states
    - Can be compressed to low-D latent space
    """

    def __init__(self, config: Optional[HypervectorConfig] = None):
        """
        Initialize the encoder.

        Args:
            config: Encoder configuration
        """
        self.config = config or HypervectorConfig()
        self.dim = self.config.dim
        self.rng = np.random.default_rng(self.config.seed)

        # Key vectors (one per feature name)
        self._key_vectors: Dict[str, np.ndarray] = {}

        # Temporal history
        self._history: deque = deque(maxlen=self.config.temporal_steps)

        # Stats for normalization
        self._feature_stats: Dict[str, Tuple[float, float]] = {}  # (mean, std)
        self._seen_features: set = set()

        logger.info(f"CathedralHypervectorEncoder: dim={self.dim}")

    def _get_key_vector(self, name: str) -> np.ndarray:
        """
        Get or create a deterministic key vector for a feature name.

        Key vectors are random bipolar {-1, +1} vectors that represent
        "what slot" a feature occupies in the hypervector.
        """
        if name not in self._key_vectors:
            # Deterministic seed from feature name
            seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            local_rng = np.random.default_rng(seed)
            self._key_vectors[name] = local_rng.choice([-1.0, 1.0], size=self.dim)
        return self._key_vectors[name]

    def _scale_value(self, value: float) -> float:
        """
        Scale a normalized [0,1] value for binding.

        We want values near 0 to have low impact and values near 1
        to have high impact. Using tanh for smooth saturation.
        """
        # Map [0,1] to [-1, 1] then scale
        centered = (value - 0.5) * 2.0
        return np.tanh(centered * 1.5)  # Soft saturation

    def encode_state(
        self,
        features: Dict[str, float],
        include_temporal: bool = True,
    ) -> np.ndarray:
        """
        Encode a feature dict into a hypervector.

        Args:
            features: Dict of feature_name -> normalized_value [0,1]
            include_temporal: Include permuted history binding

        Returns:
            Bipolar hypervector of shape (dim,)
        """
        if not features:
            return np.zeros(self.dim)

        # Track seen features
        self._seen_features.update(features.keys())

        # Accumulator for bundling
        hv_sum = np.zeros(self.dim)

        # Encode each feature: key * scaled_value
        for name, value in features.items():
            if np.isnan(value) or value is None:
                continue

            key = self._get_key_vector(name)

            if self.config.use_value_encoding:
                # Continuous value encoding
                scaled = self._scale_value(float(value))
                hv_sum += key * scaled
            else:
                # Binary presence encoding (just add key if value > 0.5)
                if value > 0.5:
                    hv_sum += key

        # Normalize to bipolar
        current_hv = np.sign(hv_sum)
        current_hv[current_hv == 0] = 1  # Break ties

        # Temporal binding (optional)
        if include_temporal and self._history:
            temporal_hv = self._encode_temporal(current_hv)
            # Blend current and temporal
            blended = current_hv + 0.5 * temporal_hv
            current_hv = np.sign(blended)
            current_hv[current_hv == 0] = 1

        # Store in history
        self._history.append(current_hv.copy())

        return current_hv

    def _encode_temporal(self, current: np.ndarray) -> np.ndarray:
        """
        Encode temporal context via permutation binding.

        Creates HV that represents "current given recent past".
        Uses circular shift (permutation) to mark temporal position.
        """
        if not self._history:
            return np.zeros(self.dim)

        temporal_sum = np.zeros(self.dim)
        decay = 1.0

        for i, past_hv in enumerate(reversed(list(self._history))):
            # Permute by shift (marks temporal position)
            shift = (i + 1) * 7  # Different shift per step
            shifted = np.roll(past_hv, shift)

            # Decay older states
            decay *= self.config.temporal_decay
            temporal_sum += shifted * decay

        return np.sign(temporal_sum)

    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """
        Compute cosine similarity between two hypervectors.

        Returns value in [-1, 1] where:
        - 1.0 = identical
        - 0.0 = orthogonal (unrelated)
        - -1.0 = opposite
        """
        return float(np.dot(hv1, hv2) / self.dim)

    def get_feature_count(self) -> int:
        """Get number of unique features seen."""
        return len(self._seen_features)

    def reset_history(self):
        """Clear temporal history."""
        self._history.clear()


# =============================================================================
# Fast Batch Encoder (for training)
# =============================================================================

class BatchHypervectorEncoder:
    """
    Batch encoder for training the latent space.

    Encodes many state samples efficiently using matrix operations.
    Used for offline PCA/autoencoder training.
    """

    def __init__(self, dim: int = 1024, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self._key_matrix: Optional[np.ndarray] = None
        self._feature_names: List[str] = []

    def fit(self, feature_names: List[str]):
        """
        Initialize key vectors for a fixed feature set.

        Args:
            feature_names: List of feature names in canonical order
        """
        self._feature_names = feature_names
        n_features = len(feature_names)

        # Create key matrix: (n_features, dim)
        rng = np.random.default_rng(self.seed)
        self._key_matrix = rng.choice([-1.0, 1.0], size=(n_features, self.dim))

    def encode_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Encode a batch of feature vectors.

        Args:
            features: Array of shape (batch_size, n_features) with values in [0,1]

        Returns:
            Array of shape (batch_size, dim) with bipolar hypervectors
        """
        if self._key_matrix is None:
            raise ValueError("Must call fit() first")

        # Scale values: [0,1] -> [-1,1] with tanh saturation
        scaled = np.tanh((features - 0.5) * 3.0)

        # Matrix multiply: (batch, features) @ (features, dim) -> (batch, dim)
        hv_sums = scaled @ self._key_matrix

        # Bipolar sign
        hvs = np.sign(hv_sums)
        hvs[hvs == 0] = 1

        return hvs


# =============================================================================
# Singleton Access
# =============================================================================

_encoder: Optional[CathedralHypervectorEncoder] = None


def get_hypervector_encoder(
    dim: int = 1024,
    **kwargs,
) -> CathedralHypervectorEncoder:
    """Get the default hypervector encoder."""
    global _encoder
    if _encoder is None or _encoder.dim != dim:
        config = HypervectorConfig(dim=dim, **kwargs)
        _encoder = CathedralHypervectorEncoder(config)
    return _encoder


# =============================================================================
# Testing
# =============================================================================

def _test_encoder():
    """Test the hypervector encoder."""
    print("=" * 60)
    print("Cathedral Hypervector Encoder Test")
    print("=" * 60)

    encoder = CathedralHypervectorEncoder()

    # Simulate state samples
    states = [
        {"system.cpu": 0.3, "system.memory": 0.4, "user.stress": 0.2},
        {"system.cpu": 0.35, "system.memory": 0.42, "user.stress": 0.22},
        {"system.cpu": 0.8, "system.memory": 0.7, "user.stress": 0.6},
        {"system.cpu": 0.85, "system.memory": 0.75, "user.stress": 0.65},
    ]

    hvs = []
    for i, state in enumerate(states):
        hv = encoder.encode_state(state)
        hvs.append(hv)
        print(f"\n[{i}] State: {state}")
        print(f"    HV shape: {hv.shape}, sum: {hv.sum():.0f}")

    # Check similarities
    print("\nSimilarities:")
    for i in range(len(hvs)):
        for j in range(i + 1, len(hvs)):
            sim = encoder.similarity(hvs[i], hvs[j])
            print(f"  [{i}] <-> [{j}]: {sim:.3f}")

    # Similar states should have high similarity
    print("\nExpected: [0]<->[1] high, [2]<->[3] high, [0]<->[2] low")


if __name__ == "__main__":
    _test_encoder()
