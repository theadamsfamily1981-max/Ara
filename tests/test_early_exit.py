"""
Early Exit Accuracy Tests - Chunked Inference Quality
=====================================================

Tests that early exit from chunked inference maintains acceptable accuracy:
- Partial similarity correlates with full similarity
- Early exit achieves target accuracy vs speedup tradeoff
- Threshold tuning produces expected behavior

These tests gate early-exit optimizations for graphics and LAN organs.
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple

from ara.hd.ops import DIM, random_hv, cosine
from ara.hd.hv_types import DenseHV
from ara.hd.shards import (
    SoftwareHTCShard,
    ShardConfig,
    ShardRole,
)


# =============================================================================
# Test Thresholds
# =============================================================================

# Minimum agreement rate with full query at various chunk counts
EARLY_EXIT_ACCURACY_TARGETS = {
    8: 0.90,   # 8 chunks (~25%) should agree 90%+ with full
    16: 0.95,  # 16 chunks (~50%) should agree 95%+ with full
    24: 0.98,  # 24 chunks (~75%) should agree 98%+ with full
}


# =============================================================================
# Helper Functions
# =============================================================================

def create_test_shard(
    D: int = 16384,
    R: int = 256,
    early_threshold: float = 0.3,
) -> SoftwareHTCShard:
    """Create a test shard with given parameters."""
    config = ShardConfig(
        role=ShardRole.CUSTOM,
        D=D,
        R=R,
        early_exit_threshold=early_threshold,
        plasticity_enabled=False,
    )
    shard = SoftwareHTCShard(config)
    shard.initialize()
    return shard


def generate_queries(n: int, D: int, seed: int = 42) -> List[DenseHV]:
    """Generate n random query HVs."""
    rng = np.random.default_rng(seed)
    return [DenseHV(rng.choice([-1, 1], size=D).astype(np.int8)) for _ in range(n)]


# =============================================================================
# Tests: Early Exit Accuracy
# =============================================================================

class TestEarlyExitAccuracy:
    """Test that early exit maintains acceptable accuracy."""

    @pytest.fixture
    def shard(self):
        """Create test shard."""
        return create_test_shard()

    def test_partial_full_correlation(self, shard):
        """Partial similarity should correlate with full similarity."""
        queries = generate_queries(50, shard.D)

        partial_results = []
        full_results = []

        for q in queries:
            # Full query
            full_row, full_sim = shard.query(q)

            # Partial query (8 chunks = 25%)
            partial_row, partial_sim, _ = shard.query_partial(q, n_chunks=8)

            partial_results.append((partial_row, partial_sim))
            full_results.append((full_row, full_sim))

        # Check correlation of similarities
        partial_sims = [p[1] for p in partial_results]
        full_sims = [f[1] for f in full_results]

        correlation = np.corrcoef(partial_sims, full_sims)[0, 1]

        assert correlation > 0.7, (
            f"Partial/full similarity correlation = {correlation:.3f}, expected > 0.7"
        )

    @pytest.mark.parametrize("n_chunks,min_accuracy", [
        (8, 0.80),
        (16, 0.90),
        (24, 0.95),
    ])
    def test_agreement_rate(self, shard, n_chunks: int, min_accuracy: float):
        """Early exit should agree with full query at target rate."""
        queries = generate_queries(100, shard.D)

        agreements = 0
        for q in queries:
            full_row, _ = shard.query(q)
            partial_row, _, _ = shard.query_partial(q, n_chunks=n_chunks)

            if full_row == partial_row:
                agreements += 1

        agreement_rate = agreements / len(queries)

        assert agreement_rate >= min_accuracy, (
            f"Agreement rate at {n_chunks} chunks = {agreement_rate:.1%}, "
            f"expected >= {min_accuracy:.0%}"
        )


# =============================================================================
# Tests: Threshold Behavior
# =============================================================================

class TestThresholdBehavior:
    """Test that early exit threshold works correctly."""

    def test_higher_threshold_fewer_exits(self):
        """Higher threshold should result in fewer early exits."""
        queries = generate_queries(50, DIM)

        exit_counts = {}
        for threshold in [0.2, 0.3, 0.4, 0.5]:
            shard = create_test_shard(early_threshold=threshold)

            exits = 0
            for q in queries:
                _, _, is_early = shard.query_partial(q, n_chunks=8)
                if is_early:
                    exits += 1

            exit_counts[threshold] = exits

        # Higher threshold should mean fewer exits
        thresholds = sorted(exit_counts.keys())
        for i in range(len(thresholds) - 1):
            assert exit_counts[thresholds[i]] >= exit_counts[thresholds[i+1]], (
                f"Expected fewer exits at higher threshold: "
                f"t={thresholds[i]}→{exit_counts[thresholds[i]]}, "
                f"t={thresholds[i+1]}→{exit_counts[thresholds[i+1]]}"
            )

    def test_zero_threshold_always_exits(self):
        """Zero threshold should always trigger early exit."""
        shard = create_test_shard(early_threshold=0.0)
        queries = generate_queries(20, shard.D)

        for q in queries:
            _, _, is_early = shard.query_partial(q, n_chunks=8)
            assert is_early, "Zero threshold should always trigger early exit"

    def test_very_high_threshold_never_exits(self):
        """Very high threshold should never trigger early exit."""
        shard = create_test_shard(early_threshold=0.99)
        queries = generate_queries(20, shard.D)

        for q in queries:
            _, _, is_early = shard.query_partial(q, n_chunks=8)
            assert not is_early, "Very high threshold should never trigger early exit"


# =============================================================================
# Tests: Speedup vs Accuracy Tradeoff
# =============================================================================

class TestSpeedupAccuracyTradeoff:
    """Test the tradeoff between speedup and accuracy."""

    @pytest.mark.slow
    def test_tradeoff_curve(self):
        """Generate speedup vs accuracy tradeoff curve."""
        shard = create_test_shard()
        queries = generate_queries(100, shard.D)

        # Compute full results as ground truth
        full_results = []
        for q in queries:
            row, sim = shard.query(q)
            full_results.append(row)

        # Test various chunk counts
        chunk_counts = [2, 4, 8, 12, 16, 24, 32]
        results = []

        for n_chunks in chunk_counts:
            agreements = 0
            for i, q in enumerate(queries):
                partial_row, _, _ = shard.query_partial(q, n_chunks=n_chunks)
                if partial_row == full_results[i]:
                    agreements += 1

            accuracy = agreements / len(queries)
            speedup = 32 / n_chunks  # Assuming 32 total chunks

            results.append({
                "n_chunks": n_chunks,
                "accuracy": accuracy,
                "speedup": speedup,
            })

        print("\n" + "=" * 50)
        print("EARLY EXIT TRADEOFF CURVE")
        print("=" * 50)
        print(f"{'Chunks':>8} | {'Speedup':>8} | {'Accuracy':>10}")
        print("-" * 50)

        for r in results:
            print(f"{r['n_chunks']:>8} | {r['speedup']:>7.1f}x | {r['accuracy']:>10.1%}")

        print("=" * 50)

        # Verify expected tradeoff characteristics
        # Early chunks should have lower accuracy
        assert results[0]["accuracy"] < results[-1]["accuracy"], (
            "Accuracy should increase with more chunks"
        )

        # Later chunks should approach 100%
        assert results[-1]["accuracy"] >= 0.95, (
            f"Full chunks accuracy = {results[-1]['accuracy']:.1%}, expected >= 95%"
        )


# =============================================================================
# Tests: Organ-Specific Configurations
# =============================================================================

class TestOrganConfigurations:
    """Test early exit for different organ configurations."""

    @pytest.mark.parametrize("organ,D,R,threshold,min_rate", [
        ("graphics", 4096, 256, 0.4, 0.30),  # Graphics: fast, moderate accuracy
        ("lan", 2048, 128, 0.5, 0.40),       # LAN: very fast, lower accuracy ok
        ("node", 8192, 512, 0.3, 0.20),      # Node: balanced
    ])
    def test_organ_early_exit_rate(
        self,
        organ: str,
        D: int,
        R: int,
        threshold: float,
        min_rate: float,
    ):
        """Each organ should achieve its target early exit rate."""
        config = ShardConfig(
            role=ShardRole.CUSTOM,
            D=D,
            R=R,
            early_exit_threshold=threshold,
            early_exit_chunks=8,
            plasticity_enabled=False,
        )
        shard = SoftwareHTCShard(config)
        shard.initialize()

        queries = generate_queries(50, D)

        exits = 0
        for q in queries:
            _, _, is_early = shard.query_partial(q, n_chunks=8)
            if is_early:
                exits += 1

        exit_rate = exits / len(queries)

        assert exit_rate >= min_rate, (
            f"{organ} early exit rate = {exit_rate:.1%}, expected >= {min_rate:.0%}"
        )


# =============================================================================
# Comprehensive Early Exit Report
# =============================================================================

@pytest.mark.slow
def test_early_exit_comprehensive_report():
    """Generate comprehensive early exit report."""
    print("\n" + "=" * 70)
    print("EARLY EXIT COMPREHENSIVE REPORT")
    print("=" * 70)

    # Test multiple configurations
    configs = [
        ("Global (16k)", 16384, 256),
        ("Node (8k)", 8192, 256),
        ("Graphics (4k)", 4096, 128),
        ("LAN (2k)", 2048, 64),
    ]

    for name, D, R in configs:
        shard = create_test_shard(D=D, R=R, early_threshold=0.3)
        queries = generate_queries(100, D)

        # Full results
        full_results = [shard.query(q)[0] for q in queries]

        print(f"\n{name}:")
        print(f"  D={D}, R={R}")
        print(f"  {'Chunks':>8} | {'Agreement':>10} | {'Exit Rate':>10} | {'Speedup':>8}")
        print(f"  {'-'*50}")

        total_chunks = 32
        for n_chunks in [4, 8, 16, 24, 32]:
            agreements = 0
            exits = 0

            for i, q in enumerate(queries):
                row, _, is_early = shard.query_partial(q, n_chunks=n_chunks)
                if row == full_results[i]:
                    agreements += 1
                if is_early:
                    exits += 1

            agreement_rate = agreements / len(queries)
            exit_rate = exits / len(queries)
            speedup = total_chunks / n_chunks

            print(f"  {n_chunks:>8} | {agreement_rate:>10.1%} | {exit_rate:>10.1%} | {speedup:>7.1f}x")

    print("\n" + "=" * 70)
