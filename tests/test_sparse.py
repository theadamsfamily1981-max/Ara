"""
Sparse HD Tests - Geometry and Operations for SparseHV
======================================================

Tests that sparse representations maintain HD properties:
- Sparse HVs should be quasi-orthogonal (mean |cos| < 0.02 for equivalent density)
- Sparse operations (bind, bundle, similarity) should match dense equivalents
- Sparsification should preserve information above threshold

These tests gate SparseHD optimizations for graphics and LAN organs.
"""

import pytest
import numpy as np
from typing import List, Tuple

from ara.hd.ops import DIM, random_hv, cosine, bind, bundle
from ara.hd.hv_types import (
    DenseHV,
    SparseHV,
    dense_to_sparse,
    sparse_to_dense,
    sparsify,
    sparse_cosine,
    sparse_bind,
    sparse_bundle,
)


# =============================================================================
# Test Thresholds
# =============================================================================

# Sparse geometry thresholds (adjusted for effective dimension)
SPARSE_MEAN_COS_MAX = 0.05  # More relaxed than dense due to lower effective D
SPARSE_TAIL_THRESHOLD = 0.15
SPARSE_TAIL_FRACTION_MAX = 0.05


# =============================================================================
# Helper Functions
# =============================================================================

def generate_sparse_hvs(n: int, D: int, sparsity: float, seed: int = 42) -> List[SparseHV]:
    """Generate n sparse HVs with target sparsity."""
    rng = np.random.default_rng(seed)
    hvs = []

    for i in range(n):
        # Number of non-zero elements
        nnz = int(D * (1 - sparsity))

        # Random indices and signs
        idx = rng.choice(D, size=nnz, replace=False)
        idx = np.sort(idx).astype(np.int32)
        sign = rng.choice([-1, 1], size=nnz).astype(np.int8)

        hvs.append(SparseHV(idx=idx, sign=sign, D=D))

    return hvs


def compute_sparse_pairwise_cosines(hvs: List[SparseHV]) -> np.ndarray:
    """Compute all pairwise sparse cosine similarities."""
    n = len(hvs)
    cosines = []
    for i in range(n):
        for j in range(i + 1, n):
            cosines.append(abs(sparse_cosine(hvs[i], hvs[j])))
    return np.array(cosines)


# =============================================================================
# Tests: Sparse Geometry
# =============================================================================

class TestSparseGeometry:
    """Test that sparse HVs maintain geometric properties."""

    @pytest.mark.parametrize("sparsity", [0.7, 0.8, 0.9])
    def test_sparse_quasi_orthogonal(self, sparsity: float):
        """Sparse HVs should be quasi-orthogonal."""
        hvs = generate_sparse_hvs(50, DIM, sparsity)
        cosines = compute_sparse_pairwise_cosines(hvs)

        mean_cos = np.mean(cosines)

        # Effective dimension is lower, so threshold is relaxed
        effective_D = DIM * (1 - sparsity)
        adjusted_threshold = SPARSE_MEAN_COS_MAX * np.sqrt(DIM / effective_D)

        assert mean_cos < adjusted_threshold, (
            f"Sparse HVs (sparsity={sparsity}) mean |cos| = {mean_cos:.4f} "
            f"exceeds threshold {adjusted_threshold:.4f}"
        )

    @pytest.mark.parametrize("sparsity", [0.7, 0.8, 0.9])
    def test_sparse_tail_distribution(self, sparsity: float):
        """Less than 5% of sparse pairs should have high correlation."""
        hvs = generate_sparse_hvs(50, DIM, sparsity)
        cosines = compute_sparse_pairwise_cosines(hvs)

        tail_fraction = np.mean(cosines > SPARSE_TAIL_THRESHOLD)

        assert tail_fraction < SPARSE_TAIL_FRACTION_MAX, (
            f"Sparse HVs (sparsity={sparsity}) tail fraction = {tail_fraction:.4f} "
            f"exceeds {SPARSE_TAIL_FRACTION_MAX}"
        )


# =============================================================================
# Tests: Conversion Correctness
# =============================================================================

class TestConversions:
    """Test dense <-> sparse conversions."""

    def test_dense_to_sparse_roundtrip(self):
        """Converting dense->sparse->dense should preserve values."""
        dense = DenseHV(random_hv())

        sparse = dense_to_sparse(dense)
        recovered = sparse_to_dense(sparse)

        assert np.array_equal(dense.bits, recovered.bits), (
            "Dense->sparse->dense roundtrip should be lossless"
        )

    def test_sparsify_preserves_sign(self):
        """Sparsify should preserve signs of kept elements."""
        dense = DenseHV(random_hv())
        sparse = sparsify(dense, target_sparsity=0.9)

        # Check that signs match
        dense_sparse = dense_to_sparse(dense)
        for idx, sign in zip(sparse.idx, sparse.sign):
            orig_idx = np.where(dense_sparse.idx == idx)[0]
            if len(orig_idx) > 0:
                assert dense_sparse.sign[orig_idx[0]] == sign

    def test_sparsify_achieves_target(self):
        """Sparsify should achieve approximately target sparsity."""
        dense = DenseHV(random_hv())

        for target in [0.7, 0.8, 0.9]:
            sparse = sparsify(dense, target_sparsity=target)
            actual_sparsity = sparse.sparsity

            assert abs(actual_sparsity - target) < 0.05, (
                f"Sparsify target={target}, actual={actual_sparsity:.3f}"
            )


# =============================================================================
# Tests: Operation Equivalence
# =============================================================================

class TestOperationEquivalence:
    """Test that sparse ops match dense equivalents."""

    def test_sparse_cosine_matches_dense(self):
        """Sparse cosine should match dense cosine."""
        # Generate two dense HVs
        d1 = DenseHV(random_hv())
        d2 = DenseHV(random_hv())

        # Convert to sparse
        s1 = dense_to_sparse(d1)
        s2 = dense_to_sparse(d2)

        # Compare cosines
        dense_cos = cosine(d1.bits, d2.bits)
        sparse_cos = sparse_cosine(s1, s2)

        assert abs(dense_cos - sparse_cos) < 0.01, (
            f"Sparse cosine {sparse_cos:.4f} doesn't match dense {dense_cos:.4f}"
        )

    def test_sparse_bind_matches_dense(self):
        """Sparse bind should match dense bind."""
        d1 = DenseHV(random_hv())
        d2 = DenseHV(random_hv())

        # Dense bind
        dense_bound = DenseHV(bind(d1.bits, d2.bits))

        # Sparse bind
        s1 = dense_to_sparse(d1)
        s2 = dense_to_sparse(d2)
        sparse_bound = sparse_bind(s1, s2)

        # Convert to dense for comparison
        sparse_bound_dense = sparse_to_dense(sparse_bound)

        # Sparse bind only keeps overlap, so check those positions
        for idx, sign in zip(sparse_bound.idx, sparse_bound.sign):
            assert dense_bound.bits[idx] == sign, (
                f"Sparse bind mismatch at index {idx}"
            )

    def test_sparse_bundle_similar_to_dense(self):
        """Sparse bundle should produce similar result to dense bundle."""
        hvs = [DenseHV(random_hv()) for _ in range(5)]

        # Dense bundle
        dense_bundled = DenseHV(bundle([h.bits for h in hvs]))

        # Sparse bundle
        sparse_hvs = [dense_to_sparse(h) for h in hvs]
        sparse_bundled = sparse_bundle(sparse_hvs)

        # Should be highly similar
        sim = cosine(dense_bundled.bits, sparse_bundled.bits)
        assert sim > 0.8, f"Sparse bundle similarity = {sim:.4f}, expected > 0.8"


# =============================================================================
# Tests: Sparse Operations Performance
# =============================================================================

class TestSparsePerformance:
    """Test that sparse operations are actually faster."""

    @pytest.mark.slow
    def test_sparse_cosine_faster_for_sparse_inputs(self):
        """Sparse cosine should be faster when inputs are sparse."""
        import time

        n_trials = 100

        # Generate very sparse HVs (90% zeros)
        sparse_hvs = generate_sparse_hvs(2, DIM, 0.9)

        # Dense equivalents
        dense_hvs = [sparse_to_dense(s) for s in sparse_hvs]

        # Time sparse
        start = time.perf_counter()
        for _ in range(n_trials):
            sparse_cosine(sparse_hvs[0], sparse_hvs[1])
        sparse_time = time.perf_counter() - start

        # Time dense
        start = time.perf_counter()
        for _ in range(n_trials):
            cosine(dense_hvs[0].bits, dense_hvs[1].bits)
        dense_time = time.perf_counter() - start

        # Sparse should be faster (at 90% sparsity, expect ~10x)
        # Allow some slack for overhead
        speedup = dense_time / sparse_time
        print(f"Sparse cosine speedup: {speedup:.1f}x (at 90% sparsity)")

        assert speedup > 2, f"Sparse cosine speedup = {speedup:.1f}x, expected > 2x"


# =============================================================================
# Tests: Information Preservation
# =============================================================================

class TestInformationPreservation:
    """Test that sparsification preserves retrievable information."""

    def test_sparsified_retrieves_from_bundle(self):
        """Sparsified HVs should still be retrievable from bundles."""
        # Create 5 dense HVs and bundle them
        hvs = [DenseHV(random_hv()) for _ in range(5)]
        bundled = DenseHV(bundle([h.bits for h in hvs]))

        # Sparsify to 80% sparsity
        sparse_hvs = [sparsify(h, target_sparsity=0.8) for h in hvs]

        # Check that each sparse HV can still be retrieved
        for i, (dense, sparse) in enumerate(zip(hvs, sparse_hvs)):
            # Similarity to bundle
            dense_sim = cosine(bundled.bits, dense.bits)
            sparse_dense = sparse_to_dense(sparse)
            sparse_sim = cosine(bundled.bits, sparse_dense.bits)

            # Sparse should still have positive similarity
            assert sparse_sim > 0.1, (
                f"Sparsified HV {i} has sim={sparse_sim:.4f} to bundle, expected > 0.1"
            )


# =============================================================================
# Comprehensive Sparse Report
# =============================================================================

@pytest.mark.slow
def test_sparse_geometry_report():
    """Generate comprehensive sparse geometry report."""
    print("\n" + "=" * 70)
    print("SPARSE GEOMETRY REPORT")
    print("=" * 70)

    sparsities = [0.5, 0.7, 0.8, 0.9, 0.95]
    results = []

    for sparsity in sparsities:
        hvs = generate_sparse_hvs(100, DIM, sparsity)
        cosines = compute_sparse_pairwise_cosines(hvs)

        effective_D = DIM * (1 - sparsity)

        result = {
            "sparsity": sparsity,
            "effective_D": effective_D,
            "mean_cos": np.mean(cosines),
            "std_cos": np.std(cosines),
            "max_cos": np.max(cosines),
            "tail_fraction": np.mean(cosines > SPARSE_TAIL_THRESHOLD),
        }
        results.append(result)

    print(f"\n{'Sparsity':>10} | {'Eff D':>8} | {'Mean |cos|':>10} | {'Max |cos|':>10} | {'Tail':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r['sparsity']:>10.0%} | {r['effective_D']:>8.0f} | "
              f"{r['mean_cos']:>10.4f} | {r['max_cos']:>10.4f} | "
              f"{r['tail_fraction']:>8.2%}")

    print("=" * 70)

    # All should pass relaxed thresholds
    for r in results:
        if r["sparsity"] <= 0.9:
            assert r["mean_cos"] < 0.1, f"Failed at sparsity={r['sparsity']}"
