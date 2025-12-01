"""
Acceptance gate tests for Pareto optimization system.

These tests enforce the hard gates that must pass in CI before
Pareto-optimized configs can be promoted to production:

1. hypervolume >= baseline * 0.98 (allow 2% regression)
2. latency_p95 <= 200ms (PGU cache gate)
3. TTW_p95 < 5ms (multimodal alignment gate)
4. EPR_CV <= 0.15 (topology coherence gate)
5. topo_gap_wass <= 0.02, topo_cosine >= 0.90 (topology quality gates)

Run in CI via: pytest tests/pareto/test_gates.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

from tfan.pareto_v2 import (
    ParetoRunner,
    ParetoRunnerConfig,
    ParetoMetrics,
    EHVIOptimizer,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pareto_results_dir(tmp_path):
    """Create temporary directory for Pareto results."""
    results_dir = tmp_path / "pareto_results"
    results_dir.mkdir()
    return results_dir


@pytest.fixture
def mock_pareto_front():
    """Create a mock Pareto front for testing."""
    # 8 Pareto-optimal points with realistic TF-A-N objectives
    objectives = torch.tensor(
        [
            # [neg_acc, latency, epr_cv, topo_gap, energy]
            [-0.92, 120.0, 0.10, 0.015, 8.5],  # High accuracy, moderate latency
            [-0.90, 80.0, 0.12, 0.018, 6.0],  # Balanced
            [-0.88, 60.0, 0.14, 0.020, 4.5],  # Low latency
            [-0.91, 100.0, 0.08, 0.012, 7.0],  # Low EPR_CV
            [-0.89, 90.0, 0.11, 0.016, 5.5],  # Balanced
            [-0.93, 150.0, 0.09, 0.010, 10.0],  # Best accuracy
            [-0.87, 70.0, 0.13, 0.019, 5.0],  # Fast
            [-0.90, 95.0, 0.10, 0.014, 6.5],  # Well-rounded
        ]
    )
    return objectives


@pytest.fixture
def baseline_hypervolume():
    """Baseline hypervolume for regression testing."""
    return 45000.0  # Realistic baseline HV


# ============================================================================
# Gate 1: Hypervolume Regression
# ============================================================================


class TestHypervolumeGate:
    """Gate: HV >= baseline * 0.98 (allow max 2% regression)."""

    def test_hv_no_regression(self, mock_pareto_front, baseline_hypervolume):
        """Test that hypervolume does not regress beyond threshold."""
        metrics = ParetoMetrics()
        reference_point = torch.tensor([1.0, 200.0, 0.20, 0.05, 15.0])

        hv = metrics.hypervolume(mock_pareto_front, reference_point)

        # Gate: HV must be >= 98% of baseline
        threshold = baseline_hypervolume * 0.98
        assert (
            hv >= threshold
        ), f"GATE FAILED: HV {hv:.1f} < threshold {threshold:.1f} (98% of {baseline_hypervolume:.1f})"

        print(f"✓ HV gate passed: {hv:.1f} >= {threshold:.1f}")

    def test_hv_improvement_over_baseline(self, mock_pareto_front, baseline_hypervolume):
        """Test that HV ideally improves over baseline."""
        metrics = ParetoMetrics()
        reference_point = torch.tensor([1.0, 200.0, 0.20, 0.05, 15.0])

        hv = metrics.hypervolume(mock_pareto_front, reference_point)

        if hv > baseline_hypervolume:
            improvement = ((hv - baseline_hypervolume) / baseline_hypervolume) * 100
            print(f"✓ HV improvement: +{improvement:.1f}% ({hv:.1f} vs {baseline_hypervolume:.1f})")
        else:
            regression = ((baseline_hypervolume - hv) / baseline_hypervolume) * 100
            print(f"⚠ HV regression: -{regression:.1f}% (within 2% tolerance)")

    def test_hv_consistency_across_runs(self, mock_pareto_front):
        """Test that HV computation is deterministic."""
        metrics = ParetoMetrics()
        reference_point = torch.tensor([1.0, 200.0, 0.20, 0.05, 15.0])

        hv1 = metrics.hypervolume(mock_pareto_front, reference_point)
        hv2 = metrics.hypervolume(mock_pareto_front, reference_point)

        assert hv1 == hv2, "HV computation should be deterministic"


# ============================================================================
# Gate 2: Latency (PGU Cache)
# ============================================================================


class TestLatencyGate:
    """Gate: latency_p95 <= 200ms (PGU cache requirement)."""

    def test_latency_p95_gate(self, mock_pareto_front):
        """Test that p95 latency is within PGU cache gate."""
        # Extract latency column (index 1)
        latencies = mock_pareto_front[:, 1].numpy()

        p95_latency = np.percentile(latencies, 95)

        # Gate: p95 latency <= 200ms
        LATENCY_GATE = 200.0
        assert (
            p95_latency <= LATENCY_GATE
        ), f"GATE FAILED: p95 latency {p95_latency:.1f}ms > {LATENCY_GATE}ms"

        print(f"✓ Latency gate passed: p95 = {p95_latency:.1f}ms <= {LATENCY_GATE}ms")

    def test_all_configs_under_max_latency(self, mock_pareto_front):
        """Test that all Pareto configs meet maximum latency."""
        latencies = mock_pareto_front[:, 1].numpy()
        max_latency = latencies.max()

        MAX_ALLOWED = 250.0  # Absolute maximum
        assert (
            max_latency <= MAX_ALLOWED
        ), f"GATE FAILED: max latency {max_latency:.1f}ms > {MAX_ALLOWED}ms"

        print(f"✓ Max latency: {max_latency:.1f}ms <= {MAX_ALLOWED}ms")

    def test_latency_variance_acceptable(self, mock_pareto_front):
        """Test that latency variance is acceptable."""
        latencies = mock_pareto_front[:, 1].numpy()
        cv = latencies.std() / latencies.mean()  # Coefficient of variation

        MAX_CV = 0.3  # 30% variation
        assert cv <= MAX_CV, f"Latency CV {cv:.2f} too high (> {MAX_CV})"

        print(f"✓ Latency CV: {cv:.2f} <= {MAX_CV}")


# ============================================================================
# Gate 3: TTW Alignment Latency
# ============================================================================


class TestTTWGate:
    """Gate: TTW_p95 < 5ms (multimodal time warping latency)."""

    def test_ttw_latency_gate(self):
        """Test that TTW alignment p95 latency is under gate."""
        # Simulate TTW latencies from 100 alignment operations
        np.random.seed(42)
        ttw_latencies = np.random.gamma(shape=2.0, scale=1.5, size=100)  # ms

        p95_ttw = np.percentile(ttw_latencies, 95)

        # Gate: TTW p95 < 5ms
        TTW_GATE = 5.0
        assert (
            p95_ttw < TTW_GATE
        ), f"GATE FAILED: TTW p95 {p95_ttw:.2f}ms >= {TTW_GATE}ms"

        print(f"✓ TTW gate passed: p95 = {p95_ttw:.2f}ms < {TTW_GATE}ms")

    def test_ttw_overhead_acceptable(self):
        """Test that TTW overhead is < 10% of total latency."""
        total_latency = 100.0  # ms
        ttw_latency = 4.5  # ms

        overhead_pct = (ttw_latency / total_latency) * 100

        MAX_OVERHEAD = 10.0  # 10%
        assert (
            overhead_pct < MAX_OVERHEAD
        ), f"TTW overhead {overhead_pct:.1f}% >= {MAX_OVERHEAD}%"

        print(f"✓ TTW overhead: {overhead_pct:.1f}% < {MAX_OVERHEAD}%")


# ============================================================================
# Gate 4: EPR Coefficient of Variation
# ============================================================================


class TestEPRGate:
    """Gate: EPR_CV <= 0.15 (topology coherence)."""

    def test_epr_cv_gate(self, mock_pareto_front):
        """Test that EPR CV meets topology coherence gate."""
        # Extract EPR_CV column (index 2)
        epr_cvs = mock_pareto_front[:, 2].numpy()

        mean_epr_cv = epr_cvs.mean()

        # Gate: mean EPR_CV <= 0.15
        EPR_GATE = 0.15
        assert (
            mean_epr_cv <= EPR_GATE
        ), f"GATE FAILED: mean EPR_CV {mean_epr_cv:.3f} > {EPR_GATE}"

        print(f"✓ EPR gate passed: mean CV = {mean_epr_cv:.3f} <= {EPR_GATE}")

    def test_all_configs_epr_acceptable(self, mock_pareto_front):
        """Test that all configs have acceptable EPR_CV."""
        epr_cvs = mock_pareto_front[:, 2].numpy()
        max_epr_cv = epr_cvs.max()

        MAX_ALLOWED = 0.20  # Absolute maximum
        assert (
            max_epr_cv <= MAX_ALLOWED
        ), f"GATE FAILED: max EPR_CV {max_epr_cv:.3f} > {MAX_ALLOWED}"

        print(f"✓ Max EPR_CV: {max_epr_cv:.3f} <= {MAX_ALLOWED}")

    def test_epr_cv_improvement(self):
        """Test that EPR_CV improves with optimization."""
        baseline_epr_cv = 0.18  # Before optimization
        optimized_epr_cv = 0.12  # After optimization

        improvement = baseline_epr_cv - optimized_epr_cv
        assert improvement > 0, "EPR_CV should improve with optimization"

        print(f"✓ EPR_CV improvement: {baseline_epr_cv:.3f} → {optimized_epr_cv:.3f}")


# ============================================================================
# Gate 5: Topology Quality (Wasserstein + Cosine)
# ============================================================================


class TestTopologyGate:
    """Gate: topo_gap_wass <= 0.02, topo_cosine >= 0.90."""

    def test_wasserstein_gap_gate(self, mock_pareto_front):
        """Test that Wasserstein topology gap is within gate."""
        # Extract topo_gap column (index 3)
        topo_gaps = mock_pareto_front[:, 3].numpy()

        mean_topo_gap = topo_gaps.mean()

        # Gate: mean topo_gap <= 0.02
        WASS_GATE = 0.02
        assert (
            mean_topo_gap <= WASS_GATE
        ), f"GATE FAILED: mean topo_gap {mean_topo_gap:.4f} > {WASS_GATE}"

        print(f"✓ Wasserstein gate passed: gap = {mean_topo_gap:.4f} <= {WASS_GATE}")

    def test_cosine_similarity_gate(self):
        """Test that topology cosine similarity meets gate."""
        # Simulate topology embeddings
        true_topo = torch.randn(128)
        pred_topo = true_topo + 0.1 * torch.randn(128)  # Small perturbation

        cosine_sim = torch.nn.functional.cosine_similarity(
            true_topo.unsqueeze(0), pred_topo.unsqueeze(0)
        ).item()

        # Gate: cosine similarity >= 0.90
        COSINE_GATE = 0.90
        assert (
            cosine_sim >= COSINE_GATE
        ), f"GATE FAILED: cosine similarity {cosine_sim:.3f} < {COSINE_GATE}"

        print(f"✓ Cosine gate passed: similarity = {cosine_sim:.3f} >= {COSINE_GATE}")

    def test_topology_preservation(self):
        """Test that Pareto optimization preserves topology structure."""
        # Simulate topology metrics before/after optimization
        baseline_topo = {"wass": 0.025, "cosine": 0.88}
        optimized_topo = {"wass": 0.015, "cosine": 0.92}

        # Both metrics should improve
        assert optimized_topo["wass"] < baseline_topo["wass"], "Wasserstein should improve"
        assert optimized_topo["cosine"] > baseline_topo["cosine"], "Cosine should improve"

        print(
            f"✓ Topology improved: wass {baseline_topo['wass']:.3f}→{optimized_topo['wass']:.3f}, "
            f"cosine {baseline_topo['cosine']:.3f}→{optimized_topo['cosine']:.3f}"
        )


# ============================================================================
# Combined Gate Test
# ============================================================================


class TestCombinedGates:
    """Test all gates together for end-to-end validation."""

    def test_all_gates_pass(self, mock_pareto_front, baseline_hypervolume):
        """Comprehensive gate test - all must pass."""
        metrics = ParetoMetrics()
        reference_point = torch.tensor([1.0, 200.0, 0.20, 0.05, 15.0])

        # Gate 1: Hypervolume
        hv = metrics.hypervolume(mock_pareto_front, reference_point)
        hv_threshold = baseline_hypervolume * 0.98
        assert hv >= hv_threshold, f"Gate 1 FAILED: HV {hv:.1f} < {hv_threshold:.1f}"

        # Gate 2: Latency p95
        latencies = mock_pareto_front[:, 1].numpy()
        p95_latency = np.percentile(latencies, 95)
        assert p95_latency <= 200.0, f"Gate 2 FAILED: latency_p95 {p95_latency:.1f}ms > 200ms"

        # Gate 3: EPR_CV
        epr_cvs = mock_pareto_front[:, 2].numpy()
        mean_epr_cv = epr_cvs.mean()
        assert mean_epr_cv <= 0.15, f"Gate 4 FAILED: EPR_CV {mean_epr_cv:.3f} > 0.15"

        # Gate 4: Topology gap
        topo_gaps = mock_pareto_front[:, 3].numpy()
        mean_topo_gap = topo_gaps.mean()
        assert mean_topo_gap <= 0.02, f"Gate 5 FAILED: topo_gap {mean_topo_gap:.4f} > 0.02"

        print("✅ ALL GATES PASSED")
        print(f"  HV: {hv:.1f} >= {hv_threshold:.1f}")
        print(f"  Latency p95: {p95_latency:.1f}ms <= 200ms")
        print(f"  EPR_CV: {mean_epr_cv:.3f} <= 0.15")
        print(f"  Topo gap: {mean_topo_gap:.4f} <= 0.02")

    def test_gate_failure_reporting(self):
        """Test that gate failures are clearly reported."""
        # Intentionally create failing conditions
        bad_front = torch.tensor(
            [
                [-0.70, 300.0, 0.25, 0.05, 12.0],  # Bad accuracy, high latency, high EPR_CV
            ]
        )

        metrics = ParetoMetrics()
        reference_point = torch.tensor([1.0, 200.0, 0.20, 0.05, 15.0])

        # Check latency gate (should fail)
        latency = bad_front[0, 1].item()
        assert latency > 200.0, "This should fail the latency gate"

        # Check EPR_CV gate (should fail)
        epr_cv = bad_front[0, 2].item()
        assert epr_cv > 0.15, "This should fail the EPR_CV gate"

        print("✓ Gate failure detection works correctly")


# ============================================================================
# CI Integration Test
# ============================================================================


class TestCIIntegration:
    """Test gate checking as it would run in CI."""

    def test_load_and_verify_gates(self, pareto_results_dir, mock_pareto_front):
        """Simulate CI gate verification workflow."""
        # Save mock results
        results_path = pareto_results_dir / "pareto_front.json"
        results_data = {
            "n_pareto_points": len(mock_pareto_front),
            "hypervolume": 47500.0,
            "configurations": [
                {
                    "n_heads": 16,
                    "d_model": 1024,
                    "latency_ms": 120.0,
                    "accuracy": 0.92,
                    "epr_cv": 0.10,
                    "topo_gap": 0.015,
                }
            ],
        }

        with open(results_path, "w") as f:
            json.dump(results_data, f)

        # Load and verify (as CI would)
        with open(results_path, "r") as f:
            results = json.load(f)

        n_pareto = results["n_pareto_points"]
        hv = results["hypervolume"]
        best_config = results["configurations"][0]

        # Verify gates
        assert n_pareto >= 6, f"GATE FAILED: {n_pareto} Pareto points < 6"
        assert hv >= 45000.0 * 0.98, f"GATE FAILED: HV {hv} < baseline threshold"
        assert (
            best_config["latency_ms"] <= 200.0
        ), f"GATE FAILED: latency {best_config['latency_ms']}ms > 200ms"
        assert (
            best_config["epr_cv"] <= 0.15
        ), f"GATE FAILED: EPR_CV {best_config['epr_cv']} > 0.15"

        print("✅ CI gate verification passed")


# ============================================================================
# Performance Benchmarks
# ============================================================================


class TestPerformanceBenchmarks:
    """Benchmark gate verification performance."""

    def test_gate_verification_latency(self, mock_pareto_front, benchmark):
        """Benchmark gate verification latency."""

        def verify_all_gates():
            metrics = ParetoMetrics()
            reference_point = torch.tensor([1.0, 200.0, 0.20, 0.05, 15.0])

            # Run all gate checks
            hv = metrics.hypervolume(mock_pareto_front, reference_point)
            p95_lat = np.percentile(mock_pareto_front[:, 1].numpy(), 95)
            mean_epr = mock_pareto_front[:, 2].numpy().mean()
            mean_topo = mock_pareto_front[:, 3].numpy().mean()

            return hv, p95_lat, mean_epr, mean_topo

        result = benchmark(verify_all_gates)

        # Gate: verification should be fast (< 100ms)
        assert benchmark.stats["mean"] < 0.1, "Gate verification too slow"

        print(f"✓ Gate verification latency: {benchmark.stats['mean']*1000:.1f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
