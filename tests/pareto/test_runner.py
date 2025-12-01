#!/usr/bin/env python
"""
Tests for Pareto v2 Optimization System.

Tests:
- EHVI optimizer
- Pareto runner
- Pareto metrics
- Hard gate verification
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from tfan.pareto_v2 import (
    EHVIOptimizer,
    EHVIConfig,
    ParetoFront,
    ParetoRunner,
    ParetoRunnerConfig,
    TFANConfigEvaluator,
    ParetoMetrics,
    compare_pareto_fronts,
)


class TestEHVIOptimizer:
    """Test EHVI optimizer."""

    def test_ehvi_initialization(self):
        """Test EHVI optimizer initialization."""
        config = EHVIConfig(
            n_initial_points=5,
            n_iterations=10,
            n_objectives=3,
        )

        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        objective_names = ["obj1", "obj2", "obj3"]

        optimizer = EHVIOptimizer(config, bounds, objective_names)

        assert optimizer.config.n_initial_points == 5
        assert optimizer.config.n_iterations == 10
        assert optimizer.n_vars == 3

    def test_ehvi_initialization_step(self):
        """Test EHVI initialization with random samples."""
        config = EHVIConfig(n_initial_points=3, n_objectives=2)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        optimizer = EHVIOptimizer(config, bounds, ["obj1", "obj2"])

        # Simple evaluation function
        def evaluate_fn(x):
            return torch.tensor([x[0] ** 2, x[1] ** 2])

        optimizer.initialize(evaluate_fn)

        assert len(optimizer.X_observed) == 3
        assert len(optimizer.Y_observed) == 3

    def test_compute_pareto_front(self):
        """Test Pareto front computation."""
        config = EHVIConfig(n_objectives=2)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        optimizer = EHVIOptimizer(config, bounds, ["obj1", "obj2"])

        # Add some observations
        optimizer.X_observed = [
            torch.tensor([0.2, 0.8]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.8, 0.2]),
            torch.tensor([0.6, 0.6]),  # Dominated by (0.5, 0.5)
        ]

        optimizer.Y_observed = [
            torch.tensor([0.2, 0.8]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.8, 0.2]),
            torch.tensor([0.7, 0.7]),  # Dominated
        ]

        front = optimizer.compute_pareto_front()

        assert front.n_dominated == 3  # First 3 are non-dominated
        assert len(front.configs) == 3

    def test_hypervolume_computation(self):
        """Test hypervolume computation."""
        config = EHVIConfig(n_objectives=2)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        optimizer = EHVIOptimizer(config, bounds, ["obj1", "obj2"])

        optimizer.reference_point = torch.tensor([1.0, 1.0])

        pareto_objectives = torch.tensor([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2],
        ])

        hv = optimizer.compute_hypervolume(pareto_objectives)

        assert hv > 0  # Should have positive hypervolume
        assert hv < 1.0  # Should be less than total area

    def test_verify_gates(self):
        """Test gate verification."""
        config = EHVIConfig(n_initial_points=3, n_iterations=5, n_objectives=2)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        optimizer = EHVIOptimizer(config, bounds, ["obj1", "obj2"])

        # Simulate observations
        for i in range(8):  # 3 initial + 5 iterations
            x = torch.rand(2)
            y = torch.tensor([x[0] ** 2, x[1] ** 2])
            optimizer.X_observed.append(x)
            optimizer.Y_observed.append(y)

        optimizer.best_front = optimizer.compute_pareto_front()

        gates = optimizer.verify_gates()

        assert "min_pareto_points" in gates
        assert "converged" in gates
        assert "all_pass" in gates


class TestParetoRunner:
    """Test Pareto runner."""

    def test_runner_initialization(self):
        """Test runner initialization."""
        config = ParetoRunnerConfig(
            n_initial_points=3,
            n_iterations=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            runner = ParetoRunner(config)

            assert runner.config.n_initial_points == 3
            assert runner.evaluator is not None
            assert runner.optimizer is not None

    def test_config_evaluator(self):
        """Test TF-A-N config evaluator."""
        config = ParetoRunnerConfig()
        evaluator = TFANConfigEvaluator(config)

        # Test evaluation
        x = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        objectives = evaluator(x)

        assert objectives.shape == (5,)  # 5 objectives
        assert objectives[0] < 0  # neg_accuracy should be negative

    def test_decode_params(self):
        """Test parameter decoding."""
        config = ParetoRunnerConfig()
        evaluator = TFANConfigEvaluator(config)

        x = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        params = evaluator.decode_params(x)

        assert "n_heads" in params
        assert "d_model" in params
        assert "keep_ratio" in params
        assert "alpha" in params
        assert "lr" in params

        # Check ranges
        assert 4 <= params["n_heads"] <= 16
        assert 256 <= params["d_model"] <= 1024
        assert 0.1 <= params["keep_ratio"] <= 0.5

    def test_runner_short_run(self):
        """Test short optimization run."""
        config = ParetoRunnerConfig(
            n_initial_points=3,
            n_iterations=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            runner = ParetoRunner(config)

            front = runner.run(verbose=False)

            assert front is not None
            assert front.n_dominated > 0
            assert len(runner.evaluator.eval_count) > 0 or runner.evaluator.eval_count > 0

    def test_save_and_load_results(self):
        """Test saving and loading results."""
        config = ParetoRunnerConfig(
            n_initial_points=3,
            n_iterations=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            runner = ParetoRunner(config)

            front = runner.run(verbose=False)
            runner.save_results(front)

            # Load results
            results = runner.load_results()

            assert results is not None
            assert "n_pareto_points" in results
            assert "hypervolume" in results
            assert "configurations" in results

    def test_get_best_config(self):
        """Test getting best configuration."""
        config = ParetoRunnerConfig(
            n_initial_points=3,
            n_iterations=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            runner = ParetoRunner(config)

            front = runner.run(verbose=False)
            runner.save_results(front)

            # Get best config with equal weights
            best_config = runner.get_best_config()

            assert best_config is not None
            assert "n_heads" in best_config
            assert "objectives" in best_config


class TestParetoMetrics:
    """Test Pareto metrics."""

    def test_hypervolume_2d(self):
        """Test 2D hypervolume computation."""
        metrics = ParetoMetrics()

        pareto_objectives = torch.tensor([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2],
        ])

        reference_point = torch.tensor([1.0, 1.0])

        hv = metrics.hypervolume(pareto_objectives, reference_point)

        assert hv > 0
        assert hv < 1.0  # Less than total area

    def test_spread(self):
        """Test spread metric."""
        metrics = ParetoMetrics()

        pareto_objectives = torch.tensor([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2],
        ])

        spread = metrics.spread(pareto_objectives)

        assert spread >= 0  # Non-negative

    def test_spacing(self):
        """Test spacing metric."""
        metrics = ParetoMetrics()

        pareto_objectives = torch.tensor([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2],
        ])

        spacing = metrics.spacing(pareto_objectives)

        assert spacing >= 0  # Non-negative

    def test_coverage(self):
        """Test coverage metric."""
        metrics = ParetoMetrics()

        front_a = torch.tensor([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2],
        ])

        front_b = torch.tensor([
            [0.3, 0.9],
            [0.6, 0.6],
            [0.9, 0.3],
        ])

        coverage = metrics.coverage(front_a, front_b)

        assert 0 <= coverage <= 1  # Should be a fraction

    def test_compute_all_metrics(self):
        """Test computing all metrics."""
        metrics = ParetoMetrics()

        pareto_objectives = torch.tensor([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2],
        ])

        reference_point = torch.tensor([1.0, 1.0])

        all_metrics = metrics.compute_all_metrics(pareto_objectives, reference_point)

        assert "hypervolume" in all_metrics
        assert "spread" in all_metrics
        assert "spacing" in all_metrics
        assert "n_points" in all_metrics

        assert all_metrics["n_points"] == 3

    def test_compare_fronts(self):
        """Test comparing two Pareto fronts."""
        front_a = torch.tensor([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2],
        ])

        front_b = torch.tensor([
            [0.3, 0.9],
            [0.6, 0.6],
            [0.9, 0.3],
        ])

        reference_point = torch.tensor([1.0, 1.0])

        comparison = compare_pareto_fronts(front_a, front_b, reference_point)

        assert "front_a" in comparison
        assert "front_b" in comparison
        assert "coverage_a_over_b" in comparison
        assert "coverage_b_over_a" in comparison


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
