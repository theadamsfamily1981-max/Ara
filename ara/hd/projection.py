"""
Ara HD Projection - Dimension Mapping Between Organs
====================================================

Provides MicroHD projections for different organ dimensions:
- HTC global: D=16k
- GPU shard: D=4k
- NIC shard: D=2k
- Node shards: D=4k-8k

Each organ operates in its native dimension; projections
translate between them losslessly (within capacity limits).

Mythic: Organs discover their minimal sufficient resolution
Physical: Random projection preserves distances (JL lemma)
Safety: Projection matrices are fixed per-seed for reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from .hv_types import DenseHV, SparseHV, dense_to_sparse, sparse_to_dense


# =============================================================================
# Projection Matrix
# =============================================================================

@dataclass
class HDProjection:
    """
    Random projection between HD dimensions.

    Uses Johnson-Lindenstrauss style projection to map between
    different dimensionalities while approximately preserving
    pairwise distances.

    Properties:
    - down(h): Map from D_src -> D_tgt (lossy but distance-preserving)
    - up(h): Approximate inverse from D_tgt -> D_src
    """
    D_src: int
    D_tgt: int
    seed: int = 0
    _P: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Generate random projection matrix."""
        rng = np.random.default_rng(self.seed)

        # Bipolar random matrix {-1, +1}
        self._P = rng.choice([-1, 1], size=(self.D_src, self.D_tgt)).astype(np.int8)

    @property
    def compression_ratio(self) -> float:
        """How much smaller is target vs source."""
        return self.D_tgt / self.D_src

    def down(self, h: DenseHV) -> DenseHV:
        """
        Project from D_src to D_tgt (dimension reduction).

        Args:
            h: Hypervector in source dimension

        Returns:
            Projected hypervector in target dimension
        """
        if h.D != self.D_src:
            raise ValueError(f"Expected D={self.D_src}, got {h.D}")

        # Ensure bipolar
        bits = h.bits if h.is_bipolar else h.to_bipolar().bits
        vec = bits.astype(np.int16)  # int16 to avoid overflow in matmul

        # Project and sign
        proj = vec @ self._P
        result = np.sign(proj).astype(np.int8)

        # Handle zeros (ties) by random assignment
        zeros = result == 0
        if np.any(zeros):
            rng = np.random.default_rng(hash(h.bits.tobytes()) % (2**31))
            result[zeros] = rng.choice([-1, 1], size=np.sum(zeros))

        return DenseHV(result)

    def up(self, h: DenseHV) -> DenseHV:
        """
        Approximate inverse projection from D_tgt to D_src.

        This is NOT a true inverse - it's an approximation.
        Useful for "lifting" shard results back to global space.

        Args:
            h: Hypervector in target dimension

        Returns:
            Approximated hypervector in source dimension
        """
        if h.D != self.D_tgt:
            raise ValueError(f"Expected D={self.D_tgt}, got {h.D}")

        bits = h.bits if h.is_bipolar else h.to_bipolar().bits
        vec = bits.astype(np.int16)

        # Transpose projection (approximate inverse)
        proj = self._P @ vec
        result = np.sign(proj).astype(np.int8)

        # Handle zeros
        zeros = result == 0
        if np.any(zeros):
            rng = np.random.default_rng(hash(h.bits.tobytes()) % (2**31))
            result[zeros] = rng.choice([-1, 1], size=np.sum(zeros))

        return DenseHV(result)

    def down_sparse(self, h: SparseHV) -> DenseHV:
        """
        Project sparse HV to lower dimension.

        More efficient than dense for very sparse inputs.
        """
        if h.D != self.D_src:
            raise ValueError(f"Expected D={self.D_src}, got {h.D}")

        # Accumulate contributions from non-zero elements
        accum = np.zeros(self.D_tgt, dtype=np.int32)

        for idx, sign in zip(h.idx, h.sign):
            accum += sign * self._P[idx, :]

        result = np.sign(accum).astype(np.int8)

        # Handle zeros
        zeros = result == 0
        if np.any(zeros):
            rng = np.random.default_rng(int(np.sum(h.idx)) % (2**31))
            result[zeros] = rng.choice([-1, 1], size=np.sum(zeros))

        return DenseHV(result)


# =============================================================================
# Projection Registry
# =============================================================================

# Standard organ dimensions
ORGAN_DIMENSIONS = {
    "global": 16384,      # HTC global soul
    "node_default": 8192, # Node shards
    "graphics": 4096,     # GPU/visual cortex
    "lan": 2048,          # NIC/spinal reflexes
}


@dataclass
class ProjectionRegistry:
    """
    Registry of projections between organ dimensions.

    Caches projection matrices for efficiency.
    """
    base_dim: int = 16384
    seed: int = 42
    _projections: Dict[Tuple[int, int], HDProjection] = field(default_factory=dict)

    def get(self, D_src: int, D_tgt: int) -> HDProjection:
        """Get or create projection between dimensions."""
        key = (D_src, D_tgt)

        if key not in self._projections:
            # Deterministic seed based on dimensions
            proj_seed = self.seed + D_src * 1000 + D_tgt
            self._projections[key] = HDProjection(D_src, D_tgt, seed=proj_seed)

        return self._projections[key]

    def down_to(self, h: DenseHV, organ: str) -> DenseHV:
        """Project to a named organ's dimension."""
        D_tgt = ORGAN_DIMENSIONS.get(organ, self.base_dim)
        if h.D == D_tgt:
            return h
        proj = self.get(h.D, D_tgt)
        return proj.down(h)

    def up_from(self, h: DenseHV, organ: str) -> DenseHV:
        """Project from organ dimension back to base."""
        D_src = ORGAN_DIMENSIONS.get(organ, self.base_dim)
        if h.D == self.base_dim:
            return h
        proj = self.get(self.base_dim, D_src)
        return proj.up(h)


# Global registry
_projection_registry: Optional[ProjectionRegistry] = None


def get_projection_registry() -> ProjectionRegistry:
    """Get the global projection registry."""
    global _projection_registry
    if _projection_registry is None:
        _projection_registry = ProjectionRegistry()
    return _projection_registry


def project_down(h: DenseHV, D_tgt: int, seed: int = 42) -> DenseHV:
    """Convenience function to project down to target dimension."""
    registry = get_projection_registry()
    return registry.get(h.D, D_tgt).down(h)


def project_up(h: DenseHV, D_src: int, seed: int = 42) -> DenseHV:
    """Convenience function to project up from source dimension."""
    registry = get_projection_registry()
    return registry.get(D_src, h.D).up(h)


# =============================================================================
# Distance Preservation Tests
# =============================================================================

def test_projection_preserves_distances(
    D_src: int = 16384,
    D_tgt: int = 4096,
    n_pairs: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Test that projection approximately preserves pairwise distances.

    Returns statistics on distance distortion.
    """
    from .ops import cosine, random_hv

    rng = np.random.default_rng(seed)
    proj = HDProjection(D_src, D_tgt, seed=seed)

    # Generate random pairs
    distortions = []

    for _ in range(n_pairs):
        h1 = DenseHV(random_hv(D_src))
        h2 = DenseHV(random_hv(D_src))

        # Original distance
        orig_cos = cosine(h1.bits, h2.bits)

        # Projected distance
        p1 = proj.down(h1)
        p2 = proj.down(h2)
        proj_cos = cosine(p1.bits, p2.bits)

        # Distortion
        if abs(orig_cos) > 1e-6:
            distortion = abs(proj_cos - orig_cos) / abs(orig_cos)
        else:
            distortion = abs(proj_cos - orig_cos)

        distortions.append(distortion)

    return {
        "mean_distortion": float(np.mean(distortions)),
        "max_distortion": float(np.max(distortions)),
        "std_distortion": float(np.std(distortions)),
        "compression_ratio": D_tgt / D_src,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HDProjection',
    'ProjectionRegistry',
    'ORGAN_DIMENSIONS',
    'get_projection_registry',
    'project_down',
    'project_up',
    'test_projection_preserves_distances',
]
