"""
Codebook Geometry Tests - Hard Gates for HD Health
==================================================

These tests verify that the HD vocabulary maintains proper geometry:
- Random HVs are quasi-orthogonal (mean |cos| < 0.02)
- No systematic correlations (tail > 0.1 in < 1% of pairs)
- Codebook diversity preserved across categories

Failing these tests means the soul's representational capacity is compromised.
"""

import pytest
import numpy as np
from typing import List, Tuple

from ara.hd.ops import DIM, random_hv, random_hv_from_string, cosine, bind, bundle
from ara.hd.vocab import HDVocab, get_vocab


# =============================================================================
# Test Thresholds (Hard Gates)
# =============================================================================

MEAN_COS_MAX = 0.02          # Mean |cosine| must be below this
TAIL_THRESHOLD = 0.1         # High correlation threshold
TAIL_FRACTION_MAX = 0.01     # Max fraction of pairs with |cos| > threshold
MIN_SAMPLE_SIZE = 500        # Minimum pairs to test


# =============================================================================
# Helper Functions
# =============================================================================

def compute_pairwise_cosines(hvs: List[np.ndarray]) -> np.ndarray:
    """Compute all pairwise cosine similarities."""
    n = len(hvs)
    cosines = []
    for i in range(n):
        for j in range(i + 1, n):
            cosines.append(abs(cosine(hvs[i], hvs[j])))
    return np.array(cosines)


def check_geometry(
    hvs: List[np.ndarray],
    name: str,
    mean_max: float = MEAN_COS_MAX,
    tail_thresh: float = TAIL_THRESHOLD,
    tail_frac_max: float = TAIL_FRACTION_MAX,
) -> Tuple[bool, dict]:
    """Check geometry constraints on a set of HVs."""
    cosines = compute_pairwise_cosines(hvs)

    mean_cos = np.mean(cosines)
    std_cos = np.std(cosines)
    max_cos = np.max(cosines)
    tail_fraction = np.mean(cosines > tail_thresh)

    passed = mean_cos < mean_max and tail_fraction < tail_frac_max

    stats = {
        "name": name,
        "n_vectors": len(hvs),
        "n_pairs": len(cosines),
        "mean_cos": float(mean_cos),
        "std_cos": float(std_cos),
        "max_cos": float(max_cos),
        "tail_fraction": float(tail_fraction),
        "passed": passed,
    }

    return passed, stats


# =============================================================================
# Tests: Random HV Geometry
# =============================================================================

class TestRandomHVGeometry:
    """Test that random HVs are quasi-orthogonal."""

    def test_random_hv_mean_cosine(self):
        """Random HVs should have mean |cosine| < 0.02."""
        hvs = [random_hv() for _ in range(100)]
        passed, stats = check_geometry(hvs, "random_hv")

        assert stats["mean_cos"] < MEAN_COS_MAX, (
            f"Random HV mean |cos| = {stats['mean_cos']:.4f} exceeds {MEAN_COS_MAX}"
        )

    def test_random_hv_tail_distribution(self):
        """Less than 1% of random pairs should have |cos| > 0.1."""
        hvs = [random_hv() for _ in range(100)]
        passed, stats = check_geometry(hvs, "random_hv_tail")

        assert stats["tail_fraction"] < TAIL_FRACTION_MAX, (
            f"Random HV tail fraction = {stats['tail_fraction']:.4f} exceeds {TAIL_FRACTION_MAX}"
        )

    def test_seeded_hv_determinism(self):
        """Same seed should produce same HV."""
        hv1 = random_hv_from_string("test_seed")
        hv2 = random_hv_from_string("test_seed")

        assert np.array_equal(hv1, hv2), "Seeded HVs should be deterministic"

    def test_seeded_hv_diversity(self):
        """Different seeds should produce quasi-orthogonal HVs."""
        seeds = [f"seed_{i}" for i in range(100)]
        hvs = [random_hv_from_string(s) for s in seeds]
        passed, stats = check_geometry(hvs, "seeded_hv")

        assert stats["mean_cos"] < MEAN_COS_MAX, (
            f"Seeded HV mean |cos| = {stats['mean_cos']:.4f} exceeds {MEAN_COS_MAX}"
        )


# =============================================================================
# Tests: Vocabulary Geometry
# =============================================================================

class TestVocabGeometry:
    """Test that vocabulary HVs maintain proper geometry."""

    @pytest.fixture
    def vocab(self):
        """Fresh vocabulary for each test."""
        return HDVocab()

    def test_role_vectors_orthogonal(self, vocab):
        """Role vectors should be quasi-orthogonal."""
        roles = ["VISION", "HEARING", "TOUCH", "SMELL", "TASTE",
                 "VESTIBULAR", "PROPRIOCEPTION", "INTEROCEPTION",
                 "TIME", "TASK", "TELEOLOGY", "NETWORK", "UI"]
        hvs = [vocab.role(r) for r in roles]
        passed, stats = check_geometry(hvs, "roles")

        assert stats["mean_cos"] < MEAN_COS_MAX * 2, (  # Slightly relaxed for small set
            f"Role vectors mean |cos| = {stats['mean_cos']:.4f} too high"
        )

    def test_feature_vectors_orthogonal(self, vocab):
        """Feature vectors should be quasi-orthogonal."""
        features = [f"FEATURE_{i}" for i in range(50)]
        hvs = [vocab.feature(f) for f in features]
        passed, stats = check_geometry(hvs, "features")

        assert stats["mean_cos"] < MEAN_COS_MAX, (
            f"Feature vectors mean |cos| = {stats['mean_cos']:.4f} exceeds {MEAN_COS_MAX}"
        )

    def test_bin_vectors_orthogonal(self, vocab):
        """Bin vectors should be quasi-orthogonal."""
        bins = ["MINIMAL", "LOW", "MED", "HIGH", "CRITICAL", "EXTREME", "ZERO"]
        hvs = [vocab.bin(b) for b in bins]
        passed, stats = check_geometry(hvs, "bins")

        # Small set, use relaxed threshold
        assert stats["mean_cos"] < 0.1, (
            f"Bin vectors mean |cos| = {stats['mean_cos']:.4f} too high"
        )

    def test_custom_vectors_orthogonal(self, vocab):
        """Custom vectors in same namespace should be quasi-orthogonal."""
        nodes = [f"node_{i}" for i in range(50)]
        hvs = [vocab.custom("node", n) for n in nodes]
        passed, stats = check_geometry(hvs, "custom_nodes")

        assert stats["mean_cos"] < MEAN_COS_MAX, (
            f"Custom vectors mean |cos| = {stats['mean_cos']:.4f} exceeds {MEAN_COS_MAX}"
        )

    def test_cross_category_orthogonal(self, vocab):
        """Vectors from different categories should be quasi-orthogonal."""
        hvs = [
            vocab.role("VISION"),
            vocab.role("HEARING"),
            vocab.feature("SRC_NODE"),
            vocab.feature("DST_NODE"),
            vocab.bin("HIGH"),
            vocab.bin("LOW"),
            vocab.custom("node", "alpha"),
            vocab.custom("service", "http"),
        ]
        passed, stats = check_geometry(hvs, "cross_category")

        assert stats["mean_cos"] < 0.1, (
            f"Cross-category mean |cos| = {stats['mean_cos']:.4f} too high"
        )


# =============================================================================
# Tests: Binding Preserves Orthogonality
# =============================================================================

class TestBindingGeometry:
    """Test that binding preserves geometric properties."""

    def test_bind_preserves_orthogonality(self):
        """Binding with same key should preserve relative distances."""
        key = random_hv()
        originals = [random_hv() for _ in range(50)]
        bound = [bind(key, hv) for hv in originals]

        # Original geometry
        _, orig_stats = check_geometry(originals, "originals")
        _, bound_stats = check_geometry(bound, "bound")

        # Bound vectors should maintain similar geometry
        assert abs(orig_stats["mean_cos"] - bound_stats["mean_cos"]) < 0.02, (
            "Binding should preserve relative geometry"
        )

    def test_bind_different_keys_orthogonal(self):
        """Same value bound with different keys should be quasi-orthogonal."""
        value = random_hv()
        keys = [random_hv() for _ in range(50)]
        bound = [bind(k, value) for k in keys]

        passed, stats = check_geometry(bound, "different_keys")

        assert stats["mean_cos"] < MEAN_COS_MAX, (
            f"Different-key bindings mean |cos| = {stats['mean_cos']:.4f} exceeds {MEAN_COS_MAX}"
        )

    def test_bind_is_invertible(self):
        """Binding should be self-inverse (XOR property)."""
        key = random_hv()
        value = random_hv()

        bound = bind(key, value)
        recovered = bind(key, bound)

        assert np.array_equal(value, recovered), "Binding should be self-inverse"


# =============================================================================
# Tests: Attractor Geometry (Placeholder for HTC)
# =============================================================================

class TestAttractorGeometry:
    """Test attractor diversity constraints."""

    def test_random_attractors_diverse(self):
        """Random attractors should be quasi-orthogonal."""
        # Simulate attractor HVs
        attractors = [random_hv() for _ in range(64)]
        passed, stats = check_geometry(
            attractors,
            "attractors",
            mean_max=0.15,  # Attractor threshold is more relaxed
        )

        assert stats["mean_cos"] < 0.15, (
            f"Attractor mean |cos| = {stats['mean_cos']:.4f} exceeds 0.15"
        )

    def test_attractor_cluster_fraction(self):
        """Less than 10% of attractors should be highly similar."""
        attractors = [random_hv() for _ in range(64)]
        cosines = compute_pairwise_cosines(attractors)

        cluster_threshold = 0.5  # High similarity = potential cluster
        cluster_fraction = np.mean(cosines > cluster_threshold)

        assert cluster_fraction < 0.10, (
            f"Cluster fraction = {cluster_fraction:.4f} exceeds 0.10"
        )


# =============================================================================
# Tests: Dimension Sanity
# =============================================================================

class TestDimensionSanity:
    """Test that HV dimensions are correct."""

    def test_dimension_constant(self):
        """DIM should be 16384."""
        assert DIM == 16384, f"DIM should be 16384, got {DIM}"

    def test_random_hv_shape(self):
        """Random HVs should have correct shape."""
        hv = random_hv()
        assert hv.shape == (DIM,), f"Shape should be ({DIM},), got {hv.shape}"

    def test_random_hv_dtype(self):
        """Random HVs should be uint8."""
        hv = random_hv()
        assert hv.dtype == np.uint8, f"dtype should be uint8, got {hv.dtype}"

    def test_random_hv_binary(self):
        """Random HVs should only contain 0 and 1."""
        hv = random_hv()
        unique = np.unique(hv)
        assert set(unique).issubset({0, 1}), f"HV should be binary, got values {unique}"


# =============================================================================
# Comprehensive Report
# =============================================================================

@pytest.mark.slow
def test_full_geometry_report():
    """Generate a full geometry report (slow)."""
    vocab = HDVocab()

    reports = []

    # Random HVs
    hvs = [random_hv() for _ in range(200)]
    _, stats = check_geometry(hvs, "random_200")
    reports.append(stats)

    # Seeded HVs
    hvs = [random_hv_from_string(f"seed_{i}") for i in range(200)]
    _, stats = check_geometry(hvs, "seeded_200")
    reports.append(stats)

    # Mixed vocabulary
    hvs = []
    for i in range(50):
        hvs.append(vocab.role(f"ROLE_{i}"))
        hvs.append(vocab.feature(f"FEAT_{i}"))
        hvs.append(vocab.custom("entity", f"entity_{i}"))
    _, stats = check_geometry(hvs, "mixed_vocab_150")
    reports.append(stats)

    # Print report
    print("\n" + "=" * 60)
    print("GEOMETRY REPORT")
    print("=" * 60)
    for r in reports:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{status}] {r['name']}")
        print(f"  n_vectors: {r['n_vectors']}")
        print(f"  n_pairs: {r['n_pairs']}")
        print(f"  mean |cos|: {r['mean_cos']:.4f} (max: {MEAN_COS_MAX})")
        print(f"  std |cos|: {r['std_cos']:.4f}")
        print(f"  max |cos|: {r['max_cos']:.4f}")
        print(f"  tail(>{TAIL_THRESHOLD}): {r['tail_fraction']:.4f} (max: {TAIL_FRACTION_MAX})")
    print("=" * 60)

    # All should pass
    assert all(r["passed"] for r in reports), "Some geometry checks failed"
