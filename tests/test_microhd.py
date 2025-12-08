"""
MicroHD Tuning Tests - Dimension Selection for Organs
=====================================================

Tests for selecting optimal dimension per organ:
- Projection preserves pairwise distances (Johnson-Lindenstrauss)
- Retrieval accuracy degrades gracefully with lower D
- Attractor diversity maintained at target dimensions

These tests determine minimum D per organ without soul collapse.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple

from ara.hd.ops import DIM, random_hv, cosine
from ara.hd.hv_types import DenseHV
from ara.hd.projection import (
    HDProjection,
    ProjectionRegistry,
    ORGAN_DIMENSIONS,
    test_projection_preserves_distances,
)
from ara.hd.shards import (
    SoftwareHTCShard,
    ShardConfig,
    ShardRole,
)


# =============================================================================
# Test Thresholds
# =============================================================================

# Maximum acceptable distance distortion per compression ratio
MAX_DISTORTION = {
    0.5: 0.15,    # 50% compression: max 15% distortion
    0.25: 0.25,   # 75% compression: max 25% distortion
    0.125: 0.35,  # 87.5% compression: max 35% distortion
}

# Minimum retrieval accuracy per dimension
MIN_RETRIEVAL_ACCURACY = {
    16384: 0.95,
    8192: 0.90,
    4096: 0.80,
    2048: 0.70,
}

# Maximum mean cosine for attractor diversity
MAX_ATTRACTOR_COS = {
    16384: 0.02,
    8192: 0.03,
    4096: 0.05,
    2048: 0.08,
}


# =============================================================================
# Tests: Projection Quality
# =============================================================================

class TestProjectionQuality:
    """Test that projections preserve geometric properties."""

    @pytest.mark.parametrize("D_tgt,max_dist", [
        (8192, 0.15),
        (4096, 0.25),
        (2048, 0.35),
    ])
    def test_distance_preservation(self, D_tgt: int, max_dist: float):
        """Projection should preserve pairwise distances."""
        stats = test_projection_preserves_distances(
            D_src=DIM,
            D_tgt=D_tgt,
            n_pairs=100,
        )

        assert stats["mean_distortion"] < max_dist, (
            f"Projection to D={D_tgt} has mean distortion {stats['mean_distortion']:.3f}, "
            f"expected < {max_dist}"
        )

    def test_projection_deterministic(self):
        """Same seed should produce same projection."""
        proj1 = HDProjection(DIM, 4096, seed=42)
        proj2 = HDProjection(DIM, 4096, seed=42)

        h = DenseHV(random_hv())

        p1 = proj1.down(h)
        p2 = proj2.down(h)

        assert np.array_equal(p1.bits, p2.bits), (
            "Projections with same seed should be identical"
        )

    def test_projection_preserves_similarity_order(self):
        """Similar HVs should remain similar after projection."""
        proj = HDProjection(DIM, 4096)

        # Create a reference and variations
        ref = DenseHV(random_hv())

        # Create variants by flipping different numbers of bits
        variants = []
        for flip_frac in [0.1, 0.2, 0.3, 0.4]:
            variant = ref.bits.copy()
            n_flip = int(DIM * flip_frac)
            flip_idx = np.random.choice(DIM, size=n_flip, replace=False)
            variant[flip_idx] *= -1
            variants.append((flip_frac, DenseHV(variant)))

        # Project all
        ref_proj = proj.down(ref)
        variant_projs = [(f, proj.down(v)) for f, v in variants]

        # Check that similarity ordering is preserved
        orig_sims = [cosine(ref.bits, v.bits) for _, v in variants]
        proj_sims = [cosine(ref_proj.bits, p.bits) for _, p in variant_projs]

        # Correlation between original and projected similarities
        corr = np.corrcoef(orig_sims, proj_sims)[0, 1]

        assert corr > 0.9, (
            f"Similarity order correlation = {corr:.3f}, expected > 0.9"
        )


# =============================================================================
# Tests: Retrieval Accuracy by Dimension
# =============================================================================

class TestRetrievalAccuracy:
    """Test retrieval accuracy at different dimensions."""

    def _create_shard_with_patterns(
        self,
        D: int,
        n_patterns: int = 50,
        seed: int = 42,
    ) -> Tuple[SoftwareHTCShard, List[DenseHV]]:
        """Create shard and store patterns."""
        config = ShardConfig(
            role=ShardRole.CUSTOM,
            D=D,
            R=n_patterns,
            plasticity_enabled=False,
        )
        shard = SoftwareHTCShard(config)
        shard.initialize(seed=seed)

        # Generate "stored" patterns (these are now the attractors)
        rng = np.random.default_rng(seed + 1)
        patterns = [
            DenseHV(rng.choice([-1, 1], size=D).astype(np.int8))
            for _ in range(n_patterns)
        ]

        # Copy patterns into attractor matrix
        for i, p in enumerate(patterns):
            shard._attractors[i] = p.bits.copy()

        return shard, patterns

    @pytest.mark.parametrize("D,min_acc", [
        (16384, 0.95),
        (8192, 0.90),
        (4096, 0.80),
        (2048, 0.70),
    ])
    def test_exact_retrieval(self, D: int, min_acc: float):
        """Stored patterns should be retrievable with high accuracy."""
        shard, patterns = self._create_shard_with_patterns(D)

        correct = 0
        for i, p in enumerate(patterns):
            row, sim = shard.query(p)
            if row == i:
                correct += 1

        accuracy = correct / len(patterns)

        assert accuracy >= min_acc, (
            f"Retrieval accuracy at D={D} is {accuracy:.1%}, expected >= {min_acc:.0%}"
        )

    @pytest.mark.parametrize("D", [8192, 4096, 2048])
    def test_noisy_retrieval(self, D: int):
        """Noisy queries should still retrieve correct pattern."""
        shard, patterns = self._create_shard_with_patterns(D)

        noise_levels = [0.1, 0.2, 0.3]
        results = {}

        for noise in noise_levels:
            correct = 0
            for i, p in enumerate(patterns):
                # Add noise
                noisy = p.bits.copy()
                n_flip = int(D * noise)
                flip_idx = np.random.choice(D, size=n_flip, replace=False)
                noisy[flip_idx] *= -1

                row, _ = shard.query(DenseHV(noisy))
                if row == i:
                    correct += 1

            results[noise] = correct / len(patterns)

        # Higher noise should mean lower accuracy
        assert results[0.1] > results[0.3], (
            "Higher noise should reduce accuracy"
        )


# =============================================================================
# Tests: Attractor Diversity by Dimension
# =============================================================================

class TestAttractorDiversity:
    """Test attractor diversity at different dimensions."""

    @pytest.mark.parametrize("D,max_cos", [
        (16384, 0.02),
        (8192, 0.03),
        (4096, 0.05),
        (2048, 0.08),
    ])
    def test_random_attractor_diversity(self, D: int, max_cos: float):
        """Random attractors should be quasi-orthogonal."""
        rng = np.random.default_rng(42)

        # Generate random attractors
        n_attractors = 64
        attractors = [
            DenseHV(rng.choice([-1, 1], size=D).astype(np.int8))
            for _ in range(n_attractors)
        ]

        # Compute pairwise cosines
        cosines = []
        for i in range(n_attractors):
            for j in range(i + 1, n_attractors):
                sim = cosine(attractors[i].bits, attractors[j].bits)
                cosines.append(abs(sim))

        mean_cos = np.mean(cosines)

        assert mean_cos < max_cos, (
            f"Mean |cos| at D={D} is {mean_cos:.4f}, expected < {max_cos}"
        )


# =============================================================================
# Tests: Organ-Specific Dimension Validation
# =============================================================================

class TestOrganDimensions:
    """Test that organ dimensions are appropriate for their tasks."""

    def test_graphics_dimension_sufficient(self):
        """Graphics dimension should support attractor visualization."""
        D = ORGAN_DIMENSIONS["graphics"]  # 4096

        # Can we distinguish 256 attractors?
        rng = np.random.default_rng(42)
        attractors = [
            DenseHV(rng.choice([-1, 1], size=D).astype(np.int8))
            for _ in range(256)
        ]

        cosines = []
        for i in range(256):
            for j in range(i + 1, 256):
                cosines.append(abs(cosine(attractors[i].bits, attractors[j].bits)))

        mean_cos = np.mean(cosines)
        max_cos = np.max(cosines)

        assert mean_cos < 0.1, f"Graphics D={D}: mean |cos| = {mean_cos:.3f}"
        assert max_cos < 0.3, f"Graphics D={D}: max |cos| = {max_cos:.3f}"

    def test_lan_dimension_sufficient(self):
        """LAN dimension should discriminate good vs bad flows."""
        D = ORGAN_DIMENSIONS["lan"]  # 2048

        rng = np.random.default_rng(42)

        # Create "good" and "bad" templates
        h_good = DenseHV(rng.choice([-1, 1], size=D).astype(np.int8))
        h_bad = DenseHV(rng.choice([-1, 1], size=D).astype(np.int8))

        # Templates should be distinguishable
        template_sim = abs(cosine(h_good.bits, h_bad.bits))
        assert template_sim < 0.2, (
            f"LAN templates too similar: {template_sim:.3f}"
        )

        # Create flows that are "good-like" and "bad-like"
        good_flows = []
        bad_flows = []

        for _ in range(20):
            # Good flow: similar to h_good
            gf = h_good.bits.copy()
            flip_idx = rng.choice(D, size=int(D * 0.2), replace=False)
            gf[flip_idx] *= -1
            good_flows.append(DenseHV(gf))

            # Bad flow: similar to h_bad
            bf = h_bad.bits.copy()
            flip_idx = rng.choice(D, size=int(D * 0.2), replace=False)
            bf[flip_idx] *= -1
            bad_flows.append(DenseHV(bf))

        # Good flows should be more similar to h_good than h_bad
        for gf in good_flows:
            sim_good = cosine(gf.bits, h_good.bits)
            sim_bad = cosine(gf.bits, h_bad.bits)
            assert sim_good > sim_bad, "Good flow should be more similar to good template"

        # Bad flows should be more similar to h_bad than h_good
        for bf in bad_flows:
            sim_good = cosine(bf.bits, h_good.bits)
            sim_bad = cosine(bf.bits, h_bad.bits)
            assert sim_bad > sim_good, "Bad flow should be more similar to bad template"


# =============================================================================
# MicroHD Tuning Harness
# =============================================================================

@pytest.mark.slow
def test_microhd_tuning_harness():
    """
    MicroHD tuning harness - determines minimum D per organ.

    This is the actual tuning script referenced in the spec.
    """
    print("\n" + "=" * 70)
    print("MICROHD TUNING HARNESS")
    print("=" * 70)

    dimensions = [16384, 8192, 4096, 2048, 1024]
    results = []

    for D in dimensions:
        rng = np.random.default_rng(42)

        # 1. Retrieval accuracy test
        n_patterns = 64
        config = ShardConfig(
            role=ShardRole.CUSTOM,
            D=D,
            R=n_patterns,
            plasticity_enabled=False,
        )
        shard = SoftwareHTCShard(config)
        shard.initialize()

        # Store patterns
        patterns = []
        for i in range(n_patterns):
            p = DenseHV(rng.choice([-1, 1], size=D).astype(np.int8))
            shard._attractors[i] = p.bits.copy()
            patterns.append(p)

        # Test retrieval
        correct = 0
        for i, p in enumerate(patterns):
            row, _ = shard.query(p)
            if row == i:
                correct += 1
        retrieval_acc = correct / n_patterns

        # 2. Attractor diversity test
        cosines = []
        for i in range(n_patterns):
            for j in range(i + 1, n_patterns):
                sim = cosine(patterns[i].bits, patterns[j].bits)
                cosines.append(abs(sim))
        mean_cos = np.mean(cosines)

        # 3. Projection distortion test
        if D < 16384:
            stats = test_projection_preserves_distances(
                D_src=16384,
                D_tgt=D,
                n_pairs=50,
            )
            proj_distortion = stats["mean_distortion"]
        else:
            proj_distortion = 0.0

        results.append({
            "D": D,
            "retrieval_accuracy": retrieval_acc,
            "mean_cos": mean_cos,
            "projection_distortion": proj_distortion,
        })

    # Print results
    print(f"\n{'D':>8} | {'Retrieval':>10} | {'Mean |cos|':>10} | {'Proj Dist':>10}")
    print("-" * 50)

    for r in results:
        print(f"{r['D']:>8} | {r['retrieval_accuracy']:>10.1%} | "
              f"{r['mean_cos']:>10.4f} | {r['projection_distortion']:>10.3f}")

    # Recommendations
    print("\n" + "-" * 50)
    print("RECOMMENDATIONS:")

    for organ, target_D in ORGAN_DIMENSIONS.items():
        matching = [r for r in results if r["D"] == target_D]
        if matching:
            r = matching[0]
            status = "OK" if r["retrieval_accuracy"] >= 0.70 and r["mean_cos"] < 0.1 else "WARN"
            print(f"  {organ}: D={target_D} [{status}]")

    print("=" * 70)

    # All standard dimensions should pass basic thresholds
    for r in results:
        if r["D"] >= 2048:
            assert r["retrieval_accuracy"] >= 0.60, (
                f"D={r['D']}: retrieval accuracy too low"
            )
            assert r["mean_cos"] < 0.15, (
                f"D={r['D']}: mean |cos| too high"
            )
