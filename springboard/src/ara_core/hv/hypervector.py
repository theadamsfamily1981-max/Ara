"""
Hypervector primitives for Hyperdimensional Computing.

This is a lightweight implementation suitable for small-data pattern finding.
For production at scale, consider optimized libraries like torchhd.
"""

import numpy as np
from typing import Optional, List

# Default dimension - 8192 is a good balance of accuracy and speed
DIM = 8192


def random_hv(seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random bipolar hypervector.

    Args:
        seed: Optional seed for reproducibility

    Returns:
        Bipolar vector of shape (DIM,) with values in {-1, +1}
    """
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=DIM).astype(np.int8)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Bind two hypervectors (creates association).

    In bipolar space, this is element-wise multiplication.
    bind(a, a) = identity, bind(a, b) is dissimilar to both a and b.

    Args:
        a: First hypervector
        b: Second hypervector

    Returns:
        Bound hypervector
    """
    return (a * b).astype(np.int8)


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Bundle multiple hypervectors (creates superposition).

    This is element-wise sum followed by sign normalization.
    The result is similar to all inputs.

    Args:
        vectors: List of hypervectors to bundle

    Returns:
        Bundled hypervector
    """
    if not vectors:
        return np.zeros(DIM, dtype=np.int8)

    summed = np.sum(vectors, axis=0)
    return normalize(summed)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two hypervectors.

    For bipolar vectors, this is the normalized dot product.

    Args:
        a: First hypervector
        b: Second hypervector

    Returns:
        Similarity in range [-1, 1]
    """
    return float(np.dot(a.astype(np.float32), b.astype(np.float32))) / DIM


def normalize(hv: np.ndarray) -> np.ndarray:
    """
    Normalize a hypervector to bipolar {-1, +1}.

    Args:
        hv: Input hypervector (can be any numeric type)

    Returns:
        Bipolar hypervector
    """
    # Handle ties by random assignment
    result = np.sign(hv)
    zeros = result == 0
    if np.any(zeros):
        result[zeros] = np.random.choice([-1, 1], size=np.sum(zeros))
    return result.astype(np.int8)


def permute(hv: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Permute a hypervector (for sequence encoding).

    Args:
        hv: Input hypervector
        n: Number of positions to shift

    Returns:
        Permuted hypervector
    """
    return np.roll(hv, n)
