"""
Ara HD Vector Types - Dense and Sparse Representations
======================================================

Provides unified abstractions for hypervector storage:
- DenseHV: Full bipolar {-1,+1} or binary {0,1} vectors
- SparseHV: Index+sign representation for mostly-zero vectors
- Conversions between representations

Mythic: The soul is still a 16k field; most of the field is quiescent
Physical: Sparse ops can be 5-30x faster for low-activity regions
Safety: Lossless conversions preserve all information
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np


# =============================================================================
# Dense Hypervector
# =============================================================================

@dataclass
class DenseHV:
    """
    Dense hypervector representation.

    Stores full vector as numpy array. Supports both:
    - Binary {0, 1} encoding
    - Bipolar {-1, +1} encoding

    The HTC and most core operations use dense representation.
    """
    bits: np.ndarray  # shape (D,), values in {0,1} or {-1,+1}

    def __post_init__(self):
        if not isinstance(self.bits, np.ndarray):
            self.bits = np.array(self.bits)

    @property
    def D(self) -> int:
        """Dimension of the hypervector."""
        return self.bits.shape[0]

    @property
    def is_bipolar(self) -> bool:
        """Check if vector is in bipolar {-1,+1} format."""
        unique = np.unique(self.bits)
        return set(unique).issubset({-1, 1})

    @property
    def is_binary(self) -> bool:
        """Check if vector is in binary {0,1} format."""
        unique = np.unique(self.bits)
        return set(unique).issubset({0, 1})

    @property
    def sparsity(self) -> float:
        """Fraction of zero elements (for sparse detection)."""
        return np.mean(self.bits == 0)

    def to_bipolar(self) -> "DenseHV":
        """Convert to bipolar {-1,+1} representation."""
        if self.is_bipolar:
            return self
        # {0,1} -> {-1,+1}
        return DenseHV(2 * self.bits.astype(np.int8) - 1)

    def to_binary(self) -> "DenseHV":
        """Convert to binary {0,1} representation."""
        if self.is_binary:
            return self
        # {-1,+1} -> {0,1}
        return DenseHV(((self.bits + 1) // 2).astype(np.uint8))

    def popcount(self) -> int:
        """Count of 1s (for binary) or +1s (for bipolar)."""
        if self.is_binary:
            return int(np.sum(self.bits))
        return int(np.sum(self.bits == 1))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DenseHV):
            return False
        return np.array_equal(self.bits, other.bits)

    def __hash__(self) -> int:
        return hash(self.bits.tobytes())


# =============================================================================
# Sparse Hypervector
# =============================================================================

@dataclass
class SparseHV:
    """
    Sparse hypervector representation.

    Stores only non-zero indices and their signs. Efficient for:
    - Vectors with many zeros
    - Fast similarity computation when both vectors are sparse
    - Memory-efficient storage of large vocabularies

    Physical: Uses ~10-30% memory when sparsity > 70%
    """
    idx: np.ndarray   # indices of non-zero dims (int32)
    sign: np.ndarray  # +/-1 for those indices (int8)
    D: int            # original dimension

    def __post_init__(self):
        if not isinstance(self.idx, np.ndarray):
            self.idx = np.array(self.idx, dtype=np.int32)
        if not isinstance(self.sign, np.ndarray):
            self.sign = np.array(self.sign, dtype=np.int8)

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.idx)

    @property
    def sparsity(self) -> float:
        """Fraction of zero elements."""
        return 1.0 - (self.nnz / self.D)

    @property
    def density(self) -> float:
        """Fraction of non-zero elements."""
        return self.nnz / self.D

    def popcount_positive(self) -> int:
        """Count of +1 values."""
        return int(np.sum(self.sign == 1))

    def popcount_negative(self) -> int:
        """Count of -1 values."""
        return int(np.sum(self.sign == -1))


# =============================================================================
# Conversions
# =============================================================================

def dense_to_sparse(h: DenseHV, thresh: float = 0.0) -> SparseHV:
    """
    Convert dense HV to sparse representation.

    Args:
        h: Dense hypervector (bipolar {-1,+1} assumed)
        thresh: Threshold for "zero" (default 0.0 = exact zeros only)

    Returns:
        SparseHV with indices and signs of non-zero elements
    """
    bits = h.bits
    if not h.is_bipolar:
        bits = h.to_bipolar().bits

    # Find non-zero indices
    if thresh == 0.0:
        mask = bits != 0
    else:
        mask = np.abs(bits) > thresh

    idx = np.where(mask)[0].astype(np.int32)
    sign = bits[idx].astype(np.int8)

    return SparseHV(idx=idx, sign=sign, D=h.D)


def sparse_to_dense(hs: SparseHV, dtype: np.dtype = np.int8) -> DenseHV:
    """
    Convert sparse HV to dense representation.

    Args:
        hs: Sparse hypervector
        dtype: Output dtype (default int8 for bipolar)

    Returns:
        DenseHV with full vector (zeros filled in)
    """
    arr = np.zeros(hs.D, dtype=dtype)
    arr[hs.idx] = hs.sign
    return DenseHV(arr)


def sparsify(h: DenseHV, target_sparsity: float = 0.9, seed: int = 42) -> SparseHV:
    """
    Artificially sparsify a dense HV by zeroing random elements.

    This is useful for testing sparse operations or when you want
    to reduce computation at the cost of some information.

    Args:
        h: Dense hypervector
        target_sparsity: Desired fraction of zeros (0.9 = 90% zeros)
        seed: Random seed for reproducibility

    Returns:
        SparseHV with random subset of original non-zeros
    """
    rng = np.random.default_rng(seed)

    bits = h.bits if h.is_bipolar else h.to_bipolar().bits
    nonzero_idx = np.where(bits != 0)[0]

    # How many to keep?
    target_nnz = int(h.D * (1 - target_sparsity))
    keep_count = min(target_nnz, len(nonzero_idx))

    if keep_count >= len(nonzero_idx):
        # Already sparse enough
        return dense_to_sparse(h)

    # Randomly select which to keep
    keep_idx = rng.choice(nonzero_idx, size=keep_count, replace=False)
    keep_idx = np.sort(keep_idx)

    return SparseHV(
        idx=keep_idx.astype(np.int32),
        sign=bits[keep_idx].astype(np.int8),
        D=h.D,
    )


# =============================================================================
# Sparse Operations
# =============================================================================

def sparse_cosine(a: SparseHV, b: SparseHV) -> float:
    """
    Compute cosine similarity between two sparse HVs.

    Only considers overlapping non-zero indices.
    Much faster than dense when both are sparse.

    Time complexity: O(nnz_a + nnz_b) instead of O(D)
    """
    if a.D != b.D:
        raise ValueError(f"Dimension mismatch: {a.D} vs {b.D}")

    # Find overlapping indices
    # Use set intersection for speed
    idx_a = set(a.idx.tolist())
    idx_b = set(b.idx.tolist())
    overlap = idx_a & idx_b

    if not overlap:
        return 0.0

    # Build lookup for b
    b_lookup = dict(zip(b.idx.tolist(), b.sign.tolist()))

    # Compute dot product over overlap
    dot = 0
    a_signs = dict(zip(a.idx.tolist(), a.sign.tolist()))
    for idx in overlap:
        dot += a_signs[idx] * b_lookup[idx]

    # Norms
    norm_a = np.sqrt(a.nnz)  # All non-zeros have magnitude 1
    norm_b = np.sqrt(b.nnz)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def sparse_hamming(a: SparseHV, b: SparseHV) -> int:
    """
    Compute Hamming distance between two sparse HVs.

    Counts positions where they differ (including zeros vs non-zeros).
    """
    if a.D != b.D:
        raise ValueError(f"Dimension mismatch: {a.D} vs {b.D}")

    # Positions where only a is non-zero
    idx_a = set(a.idx.tolist())
    idx_b = set(b.idx.tolist())

    only_a = idx_a - idx_b
    only_b = idx_b - idx_a
    both = idx_a & idx_b

    # Disagreements in "both" set
    a_lookup = dict(zip(a.idx.tolist(), a.sign.tolist()))
    b_lookup = dict(zip(b.idx.tolist(), b.sign.tolist()))

    disagree_both = sum(1 for idx in both if a_lookup[idx] != b_lookup[idx])

    # Total Hamming = only_a + only_b + disagree_both
    return len(only_a) + len(only_b) + disagree_both


def sparse_bind(a: SparseHV, b: SparseHV) -> SparseHV:
    """
    Bind two sparse HVs (element-wise multiplication for bipolar).

    For bipolar: bind(a,b)[i] = a[i] * b[i]
    - (+1) * (+1) = +1
    - (+1) * (-1) = -1
    - (-1) * (-1) = +1
    - 0 * anything = 0

    Result is sparse: only non-zero where BOTH are non-zero.
    """
    if a.D != b.D:
        raise ValueError(f"Dimension mismatch: {a.D} vs {b.D}")

    # Find overlapping indices
    idx_a = set(a.idx.tolist())
    idx_b = set(b.idx.tolist())
    overlap = sorted(idx_a & idx_b)

    if not overlap:
        return SparseHV(
            idx=np.array([], dtype=np.int32),
            sign=np.array([], dtype=np.int8),
            D=a.D,
        )

    a_lookup = dict(zip(a.idx.tolist(), a.sign.tolist()))
    b_lookup = dict(zip(b.idx.tolist(), b.sign.tolist()))

    result_idx = np.array(overlap, dtype=np.int32)
    result_sign = np.array([a_lookup[i] * b_lookup[i] for i in overlap], dtype=np.int8)

    return SparseHV(idx=result_idx, sign=result_sign, D=a.D)


def sparse_bundle(hvs: list[SparseHV], weights: Optional[list[float]] = None) -> DenseHV:
    """
    Bundle multiple sparse HVs using majority vote.

    Returns dense result (bundling typically produces dense output).
    """
    if not hvs:
        raise ValueError("Cannot bundle empty list")

    D = hvs[0].D
    if weights is None:
        weights = [1.0] * len(hvs)

    # Accumulate weighted votes
    accum = np.zeros(D, dtype=np.float32)

    for hv, w in zip(hvs, weights):
        if hv.D != D:
            raise ValueError(f"Dimension mismatch: {hv.D} vs {D}")
        accum[hv.idx] += w * hv.sign

    # Majority vote
    result = np.sign(accum).astype(np.int8)
    # Handle ties (accum == 0) by random assignment
    ties = accum == 0
    if np.any(ties):
        result[ties] = np.random.choice([-1, 1], size=np.sum(ties))

    return DenseHV(result)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'DenseHV',
    'SparseHV',
    'dense_to_sparse',
    'sparse_to_dense',
    'sparsify',
    'sparse_cosine',
    'sparse_hamming',
    'sparse_bind',
    'sparse_bundle',
]
