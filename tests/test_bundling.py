"""
Bundling Capacity Tests - Hard Gates for HD Superposition
=========================================================

These tests verify that bundling maintains signal integrity:
- Bundled components can be recovered with high accuracy
- Signal remains above noise floor at target feature counts
- Graceful degradation beyond capacity limits

Failing these tests means the soul cannot reliably compose moments.
"""

import pytest
import numpy as np
from typing import List, Tuple, Dict

from ara.hd.ops import DIM, random_hv, bind, bundle, cosine
from ara.hd.vocab import HDVocab


# =============================================================================
# Test Thresholds (Hard Gates)
# =============================================================================

MAX_FEATURES_PER_MOMENT = 50      # Target capacity
MIN_SIGNAL_SIGMA = 3.0            # Signal must be 3σ above noise
RECOVERY_THRESHOLD = 0.3          # Min cosine to consider "recovered"
TARGET_RECOVERY_RATE = 0.95       # 95% of components should be recoverable


# =============================================================================
# Helper Functions
# =============================================================================

def measure_signal_noise(
    bundled: np.ndarray,
    components: List[np.ndarray],
    n_noise_samples: int = 100,
) -> Tuple[float, float, float]:
    """
    Measure signal strength vs noise floor.

    Returns:
        (mean_signal, noise_mean, noise_std)
    """
    # Signal: similarity to actual components
    signals = [cosine(bundled, c) for c in components]
    mean_signal = np.mean(signals)

    # Noise: similarity to random HVs
    noise_samples = [cosine(bundled, random_hv()) for _ in range(n_noise_samples)]
    noise_mean = np.mean(noise_samples)
    noise_std = np.std(noise_samples)

    return mean_signal, noise_mean, noise_std


def compute_sigma(signal: float, noise_mean: float, noise_std: float) -> float:
    """Compute how many sigmas the signal is above noise."""
    if noise_std < 1e-6:
        return float('inf') if signal > noise_mean else 0.0
    return (signal - noise_mean) / noise_std


def test_recovery_rate(
    bundled: np.ndarray,
    components: List[np.ndarray],
    threshold: float = RECOVERY_THRESHOLD,
) -> float:
    """Compute fraction of components recoverable above threshold."""
    recovered = sum(1 for c in components if cosine(bundled, c) > threshold)
    return recovered / len(components) if components else 0.0


# =============================================================================
# Tests: Basic Bundling
# =============================================================================

class TestBasicBundling:
    """Test fundamental bundling properties."""

    def test_bundle_single_is_identity(self):
        """Bundling a single HV should return it (or equivalent)."""
        hv = random_hv()
        bundled = bundle([hv])

        # Should be highly similar (exact for single)
        sim = cosine(bundled, hv)
        assert sim > 0.99, f"Single bundle similarity = {sim}, expected ~1.0"

    def test_bundle_two_recoverable(self):
        """Both components should be recoverable from 2-bundle."""
        hv1 = random_hv()
        hv2 = random_hv()
        bundled = bundle([hv1, hv2])

        sim1 = cosine(bundled, hv1)
        sim2 = cosine(bundled, hv2)

        assert sim1 > RECOVERY_THRESHOLD, f"Component 1 similarity = {sim1}"
        assert sim2 > RECOVERY_THRESHOLD, f"Component 2 similarity = {sim2}"

    def test_bundle_commutative(self):
        """Bundle order shouldn't matter."""
        hvs = [random_hv() for _ in range(5)]

        bundled1 = bundle(hvs)
        bundled2 = bundle(hvs[::-1])

        sim = cosine(bundled1, bundled2)
        assert sim > 0.99, f"Bundle should be order-independent, sim = {sim}"

    def test_bundle_excludes_non_members(self):
        """Non-members should have low similarity to bundle."""
        members = [random_hv() for _ in range(10)]
        non_member = random_hv()
        bundled = bundle(members)

        member_sims = [cosine(bundled, m) for m in members]
        non_member_sim = cosine(bundled, non_member)

        mean_member = np.mean(member_sims)
        assert non_member_sim < mean_member * 0.5, (
            f"Non-member sim = {non_member_sim}, member mean = {mean_member}"
        )


# =============================================================================
# Tests: Capacity Scaling
# =============================================================================

class TestCapacityScaling:
    """Test bundling capacity at various scales."""

    @pytest.mark.parametrize("n_components", [4, 8, 16, 32, 50])
    def test_recovery_rate_by_size(self, n_components: int):
        """Recovery rate should stay high up to MAX_FEATURES."""
        components = [random_hv() for _ in range(n_components)]
        bundled = bundle(components)

        rate = test_recovery_rate(bundled, components)

        if n_components <= MAX_FEATURES_PER_MOMENT:
            assert rate >= TARGET_RECOVERY_RATE, (
                f"Recovery rate at n={n_components} is {rate:.2%}, "
                f"expected >= {TARGET_RECOVERY_RATE:.0%}"
            )

    @pytest.mark.parametrize("n_components", [4, 8, 16, 32, 50])
    def test_signal_above_noise(self, n_components: int):
        """Signal should remain above noise floor."""
        components = [random_hv() for _ in range(n_components)]
        bundled = bundle(components)

        signal, noise_mean, noise_std = measure_signal_noise(bundled, components)
        sigma = compute_sigma(signal, noise_mean, noise_std)

        if n_components <= MAX_FEATURES_PER_MOMENT:
            assert sigma >= MIN_SIGNAL_SIGMA, (
                f"Signal at n={n_components} is {sigma:.1f}σ above noise, "
                f"expected >= {MIN_SIGNAL_SIGMA}σ"
            )

    def test_graceful_degradation_beyond_capacity(self):
        """Performance should degrade gracefully beyond capacity."""
        sizes = [50, 64, 80, 100, 128]
        recovery_rates = []

        for n in sizes:
            components = [random_hv() for _ in range(n)]
            bundled = bundle(components)
            rate = test_recovery_rate(bundled, components)
            recovery_rates.append(rate)

        # Should monotonically decrease (with some tolerance)
        for i in range(1, len(recovery_rates)):
            # Allow small increases due to randomness
            assert recovery_rates[i] <= recovery_rates[i-1] + 0.05, (
                f"Recovery should decrease: {recovery_rates}"
            )


# =============================================================================
# Tests: Weighted Bundling
# =============================================================================

class TestWeightedBundling:
    """Test weighted bundling behavior."""

    def test_higher_weight_higher_similarity(self):
        """Higher weighted components should have higher similarity."""
        components = [random_hv() for _ in range(5)]
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]

        bundled = bundle(components, weights=weights)

        sims = [cosine(bundled, c) for c in components]

        # Higher weight should correlate with higher similarity
        # Check that highest weight has highest sim
        assert sims[4] > sims[0], (
            f"Highest weight sim = {sims[4]}, lowest = {sims[0]}"
        )

    def test_zero_weight_excluded(self):
        """Zero-weighted components should be effectively excluded."""
        important = random_hv()
        ignored = random_hv()

        bundled = bundle([important, ignored], weights=[1.0, 0.0])

        sim_important = cosine(bundled, important)
        sim_ignored = cosine(bundled, ignored)

        assert sim_important > 0.9, f"Important sim = {sim_important}"
        # Ignored should be at noise level
        assert sim_ignored < 0.3, f"Ignored sim = {sim_ignored}"


# =============================================================================
# Tests: Role-Filler Pattern
# =============================================================================

class TestRoleFillerPattern:
    """Test the role-filler binding pattern used in encoding."""

    @pytest.fixture
    def vocab(self):
        return HDVocab()

    def test_role_filler_recovery(self, vocab):
        """Role-filler pairs should be recoverable from bundle."""
        # Create role-filler bindings
        pairs = [
            (vocab.feature("SRC_NODE"), vocab.custom("node", "alpha")),
            (vocab.feature("DST_NODE"), vocab.custom("node", "beta")),
            (vocab.feature("SERVICE"), vocab.custom("service", "http")),
            (vocab.feature("LATENCY"), vocab.bin("LOW")),
        ]

        bound_pairs = [bind(role, filler) for role, filler in pairs]
        bundled = bundle(bound_pairs)

        # Each bound pair should be recoverable
        for i, (role, filler) in enumerate(pairs):
            bound = bound_pairs[i]
            sim = cosine(bundled, bound)
            assert sim > RECOVERY_THRESHOLD, (
                f"Pair {i} similarity = {sim}, expected > {RECOVERY_THRESHOLD}"
            )

    def test_unbind_recovers_filler(self, vocab):
        """Unbinding with role should recover filler."""
        role = vocab.feature("SERVICE")
        filler = vocab.custom("service", "http")

        bound = bind(role, filler)

        # Create bundle with other stuff
        noise = [bind(vocab.feature(f"F{i}"), random_hv()) for i in range(10)]
        bundled = bundle([bound] + noise)

        # Unbind with role
        recovered = bind(role, bundled)  # XOR is self-inverse

        # Should have high similarity to original filler
        sim = cosine(recovered, filler)
        assert sim > 0.2, f"Recovered filler similarity = {sim}"


# =============================================================================
# Tests: Moment Encoding Simulation
# =============================================================================

class TestMomentEncoding:
    """Test realistic moment encoding scenarios."""

    @pytest.fixture
    def vocab(self):
        return HDVocab()

    def test_typical_moment_encoding(self, vocab):
        """Simulate a typical moment with ~20 features."""
        # Simulate encoding from multiple senses
        components = []

        # Vision sense (3 features)
        for feat in ["BRIGHTNESS", "COLOR", "MOTION"]:
            components.append(bind(vocab.feature(feat), vocab.bin("MED")))

        # Network sense (5 features)
        components.append(bind(vocab.feature("SRC_NODE"), vocab.custom("node", "fpga1")))
        components.append(bind(vocab.feature("DST_NODE"), vocab.custom("node", "gpu2")))
        components.append(bind(vocab.feature("SERVICE"), vocab.custom("service", "ml_train")))
        components.append(bind(vocab.feature("LATENCY"), vocab.bin("LOW")))
        components.append(bind(vocab.feature("RATE"), vocab.bin("HIGH")))

        # Proprioception (4 features)
        components.append(bind(vocab.feature("CPU_LOAD"), vocab.bin("MED")))
        components.append(bind(vocab.feature("GPU_UTIL"), vocab.bin("HIGH")))
        components.append(bind(vocab.feature("TEMP"), vocab.bin("MED")))
        components.append(bind(vocab.feature("MEMORY"), vocab.bin("HIGH")))

        # Teleology anchor (1 feature)
        components.append(bind(vocab.role("TELEOLOGY"), vocab.tag("MAINTAIN_HOMESTEAD")))

        # Time encoding (2 features)
        components.append(bind(vocab.feature("HOUR"), vocab.custom("time", "14")))
        components.append(bind(vocab.feature("DAY"), vocab.custom("time", "WEEKDAY")))

        # Bundle all
        moment = bundle(components)

        # All components should be recoverable
        rate = test_recovery_rate(moment, components)
        assert rate >= 0.9, f"Moment recovery rate = {rate:.0%}"

        # Signal should be well above noise
        signal, noise_mean, noise_std = measure_signal_noise(moment, components)
        sigma = compute_sigma(signal, noise_mean, noise_std)
        assert sigma >= MIN_SIGNAL_SIGMA, f"Moment signal = {sigma:.1f}σ"

    def test_max_capacity_moment(self, vocab):
        """Test moment at maximum capacity (50 features)."""
        components = []
        for i in range(MAX_FEATURES_PER_MOMENT):
            feat = vocab.feature(f"FEAT_{i}")
            val = vocab.custom("val", f"val_{i}")
            components.append(bind(feat, val))

        moment = bundle(components)

        rate = test_recovery_rate(moment, components)
        signal, noise_mean, noise_std = measure_signal_noise(moment, components)
        sigma = compute_sigma(signal, noise_mean, noise_std)

        assert rate >= TARGET_RECOVERY_RATE, (
            f"Max capacity recovery = {rate:.0%}, expected >= {TARGET_RECOVERY_RATE:.0%}"
        )
        assert sigma >= MIN_SIGNAL_SIGMA, (
            f"Max capacity signal = {sigma:.1f}σ, expected >= {MIN_SIGNAL_SIGMA}σ"
        )


# =============================================================================
# Stress Test Report
# =============================================================================

@pytest.mark.slow
def test_bundling_stress_report():
    """Generate comprehensive bundling stress report."""
    print("\n" + "=" * 70)
    print("BUNDLING STRESS TEST REPORT")
    print("=" * 70)

    sizes = [4, 8, 16, 24, 32, 40, 50, 64, 80, 100]
    results = []

    for n in sizes:
        # Run multiple trials
        rates = []
        sigmas = []

        for _ in range(10):
            components = [random_hv() for _ in range(n)]
            bundled = bundle(components)

            rate = test_recovery_rate(bundled, components)
            signal, noise_mean, noise_std = measure_signal_noise(bundled, components)
            sigma = compute_sigma(signal, noise_mean, noise_std)

            rates.append(rate)
            sigmas.append(sigma)

        result = {
            "n": n,
            "recovery_mean": np.mean(rates),
            "recovery_std": np.std(rates),
            "sigma_mean": np.mean(sigmas),
            "sigma_std": np.std(sigmas),
            "pass_recovery": np.mean(rates) >= TARGET_RECOVERY_RATE if n <= MAX_FEATURES_PER_MOMENT else True,
            "pass_sigma": np.mean(sigmas) >= MIN_SIGNAL_SIGMA if n <= MAX_FEATURES_PER_MOMENT else True,
        }
        results.append(result)

    # Print results
    print(f"\nTarget: {MAX_FEATURES_PER_MOMENT} features, {TARGET_RECOVERY_RATE:.0%} recovery, {MIN_SIGNAL_SIGMA}σ signal")
    print("-" * 70)
    print(f"{'N':>4} | {'Recovery':>12} | {'Signal (σ)':>12} | {'Status':>10}")
    print("-" * 70)

    for r in results:
        status = "PASS" if r["pass_recovery"] and r["pass_sigma"] else "FAIL"
        if r["n"] > MAX_FEATURES_PER_MOMENT:
            status = "OVER CAP"

        print(f"{r['n']:>4} | {r['recovery_mean']:>5.1%} ± {r['recovery_std']:.1%} | "
              f"{r['sigma_mean']:>5.1f}σ ± {r['sigma_std']:.1f} | {status:>10}")

    print("=" * 70)

    # Verify hard gates pass
    for r in results:
        if r["n"] <= MAX_FEATURES_PER_MOMENT:
            assert r["pass_recovery"], f"Recovery failed at n={r['n']}"
            assert r["pass_sigma"], f"Signal failed at n={r['n']}"
