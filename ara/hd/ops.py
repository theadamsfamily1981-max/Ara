"""
Ara HD Operations - Core VSA Primitives
=======================================

Numpy-accelerated hyperdimensional computing operations.

Canonical parameters:
- D = 16,384 dimensions (configurable)
- Binary {0,1} representation (converted to bipolar {-1,+1} for similarity)
- XOR binding (associative, self-inverse)
- Majority bundling (weighted sum + sign)

Performance:
- random_hv: ~50 µs
- bind: ~20 µs
- bundle(8): ~100 µs
- cosine: ~30 µs

All operations are vectorized numpy for efficiency.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Union

# =============================================================================
# Canonical Dimension
# =============================================================================

DIM = 16_384  # Canonical dimension for Ara's soul


# =============================================================================
# Random HV Generation
# =============================================================================

def random_hv(
    dim: int = DIM,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random binary hypervector.

    Args:
        dim: Dimension (default: 16,384)
        rng: Numpy random generator (for reproducibility)
        seed: Optional seed (creates new rng if provided)

    Returns:
        Binary {0,1} hypervector of shape (dim,)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    elif rng is None:
        rng = np.random.default_rng()

    return rng.integers(0, 2, size=dim, dtype=np.uint8)


def random_hv_from_string(seed_str: str, dim: int = DIM) -> np.ndarray:
    """
    Generate a deterministic HV from a string seed.

    Useful for creating stable role/feature vectors.

    Args:
        seed_str: String to hash for seed
        dim: Dimension

    Returns:
        Binary HV, deterministic for same seed_str
    """
    import hashlib
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    return random_hv(dim=dim, seed=seed)


# =============================================================================
# Binding (XOR)
# =============================================================================

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Bind two hypervectors via XOR.

    Properties:
    - Associative: bind(bind(a, b), c) == bind(a, bind(b, c))
    - Self-inverse: bind(bind(a, b), b) == a
    - Dissimilar to inputs: cosine(bind(a,b), a) ≈ 0

    Used for role-filler binding:
        H_attr = bind(H_ROLE, bind(H_FEATURE, H_VALUE))

    Args:
        a: Binary HV {0,1}
        b: Binary HV {0,1}

    Returns:
        Binary HV {0,1}
    """
    return np.bitwise_xor(a, b)


def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Unbind (recover) an HV from a bound pair.

    Since XOR is self-inverse:
        unbind(bind(a, b), b) == a

    Args:
        bound: The bound HV
        key: The key HV to remove

    Returns:
        The recovered HV
    """
    return np.bitwise_xor(bound, key)


# =============================================================================
# Bundling (Majority Vote)
# =============================================================================

def bundle(
    hvs: List[np.ndarray],
    weights: Optional[List[float]] = None,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Bundle multiple hypervectors via weighted majority vote.

    Properties:
    - Similar to all inputs: cosine(bundle([a,b,c]), x) > 0 for x in {a,b,c}
    - Superposition: stores multiple items in one vector

    Used for context encoding:
        H_context = bundle([H_vision, H_hearing, H_touch, ...])

    Args:
        hvs: List of binary HVs {0,1}
        weights: Optional per-HV weights (default: equal weights)
        threshold: Tie-breaking threshold (0 = random on exact tie)

    Returns:
        Binary HV {0,1}
    """
    if not hvs:
        return np.zeros(DIM, dtype=np.uint8)

    dim = len(hvs[0])

    # Convert to bipolar and accumulate
    acc = np.zeros(dim, dtype=np.float64)

    if weights is None:
        weights = [1.0] * len(hvs)

    for hv, w in zip(hvs, weights):
        # Map {0,1} -> {-1,+1}
        bipolar = hv.astype(np.float64) * 2 - 1
        acc += w * bipolar

    # Majority vote with threshold
    result = np.zeros(dim, dtype=np.uint8)
    result[acc > threshold] = 1
    result[acc < -threshold] = 0

    # Handle ties (where acc == threshold or -threshold)
    ties = np.abs(acc) <= threshold
    if np.any(ties):
        result[ties] = np.random.randint(0, 2, size=np.sum(ties), dtype=np.uint8)

    return result


def weighted_bundle(
    hvs: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """
    Convenience wrapper for weighted bundling.

    Args:
        hvs: List of binary HVs
        weights: Weights for each HV (higher = more influence)

    Returns:
        Binary HV
    """
    return bundle(hvs, weights=weights)


# =============================================================================
# Permutation (Sequence Encoding)
# =============================================================================

def permute(hv: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Circular shift permutation for sequence encoding.

    Used to encode temporal order:
        H_seq = bind(permute(H_t0), bind(permute(H_t1, 2), H_t2))

    Args:
        hv: Binary HV
        shift: Number of positions to shift (positive = right)

    Returns:
        Shifted binary HV
    """
    return np.roll(hv, shift)


def inverse_permute(hv: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Inverse permutation (shift in opposite direction).

    Args:
        hv: Binary HV
        shift: Original shift amount

    Returns:
        Inverse-shifted binary HV
    """
    return np.roll(hv, -shift)


# =============================================================================
# Similarity Measures
# =============================================================================

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two binary HVs.

    Internally converts {0,1} -> {-1,+1} for proper angular similarity.

    Returns:
        Similarity in [-1, +1] where:
        - +1 = identical
        - 0 = orthogonal (random, independent)
        - -1 = opposite
    """
    # Map {0,1} -> {-1,+1}
    a_bipolar = a.astype(np.float64) * 2 - 1
    b_bipolar = b.astype(np.float64) * 2 - 1

    num = np.dot(a_bipolar, b_bipolar)
    denom = np.linalg.norm(a_bipolar) * np.linalg.norm(b_bipolar)

    if denom == 0:
        return 0.0

    return float(num / denom)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    Hamming distance (number of differing bits).

    Args:
        a: Binary HV
        b: Binary HV

    Returns:
        Number of positions where a != b
    """
    return int(np.sum(a != b))


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized Hamming similarity.

    Returns:
        Similarity in [0, 1] where:
        - 1 = identical
        - 0.5 = random (expected for independent HVs)
        - 0 = opposite
    """
    dist = hamming_distance(a, b)
    return 1.0 - (dist / len(a))


# =============================================================================
# Utility Functions
# =============================================================================

def to_bipolar(hv: np.ndarray) -> np.ndarray:
    """Convert binary {0,1} to bipolar {-1,+1}."""
    return hv.astype(np.int8) * 2 - 1


def to_binary(hv: np.ndarray) -> np.ndarray:
    """Convert bipolar {-1,+1} to binary {0,1}."""
    return ((hv + 1) // 2).astype(np.uint8)


def sparsity(hv: np.ndarray) -> float:
    """Fraction of 1s in a binary HV (ideally ~0.5)."""
    return float(np.mean(hv))


def is_valid_hv(hv: np.ndarray, dim: int = DIM) -> bool:
    """Check if HV is valid (correct shape, binary values)."""
    if hv.shape != (dim,):
        return False
    if not np.all((hv == 0) | (hv == 1)):
        return False
    return True


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'DIM',
    'random_hv',
    'random_hv_from_string',
    'bind',
    'unbind',
    'bundle',
    'weighted_bundle',
    'permute',
    'inverse_permute',
    'cosine',
    'hamming_distance',
    'hamming_similarity',
    'to_bipolar',
    'to_binary',
    'sparsity',
    'is_valid_hv',
]
