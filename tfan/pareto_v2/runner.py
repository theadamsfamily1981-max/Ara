#!/usr/bin/env python
"""
Pareto Optimization Runner

Orchestrates multi-objective optimization for TF-A-N configurations.

Optimizes trade-offs between:
- Accuracy (maximize)
- Latency (minimize)
- EPR-CV stability (minimize)
- Topology gap (minimize)
- Energy consumption (minimize)

Usage:
    runner = ParetoRunner(config)
    front = runner.run()
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import time

from .ehvi import EHVIOptimizer, EHVIConfig, ParetoFront


@dataclass
class ParetoRunnerConfig:
    """Configuration for Pareto runner."""
    # EHVI settings
    n_initial_points: int = 10
    n_iterations: int = 100
    n_objectives: int = 5
    batch_size: int = 1

    # TF-A-N parameter ranges
    n_heads_range: Tuple[int, int] = (4, 16)
    d_model_range: Tuple[int, int] = (256, 1024)
    keep_ratio_range: Tuple[float, float] = (0.1, 0.5)
    alpha_range: Tuple[float, float] = (0.3, 0.9)
    lr_range: Tuple[float, float] = (1e-5, 1e-3)

    # Evaluation settings
    eval_steps: int = 100  # Steps to evaluate each config
    eval_batch_size: int = 32

    # Output
    output_dir: str = "artifacts/pareto"
    save_checkpoints: bool = True

    # Random seed
    seed: int = 42


class TFANConfigEvaluator:
    """
    Evaluates TF-A-N configurations.

    In production, this would train/evaluate the actual model.
    For testing, uses synthetic objectives based on parameter values.
    """

    def __init__(self, config: ParetoRunnerConfig):
        self.config = config
        self.eval_count = 0

    def decode_params(self, x: torch.Tensor) -> Dict:
        """
        Decode normalized parameters [0, 1] to actual config.

        Args:
            x: Normalized parameters (n_vars,)

        Returns:
            config: Dictionary of parameters
        """
        # Decode each parameter from [0, 1] to its range
        config = {
            "n_heads": int(x[0] * (self.config.n_heads_range[1] - self.config.n_heads_range[0]) + self.config.n_heads_range[0]),
            "d_model": int(x[1] * (self.config.d_model_range[1] - self.config.d_model_range[0]) + self.config.d_model_range[0]),
            "keep_ratio": float(x[2] * (self.config.keep_ratio_range[1] - self.config.keep_ratio_range[0]) + self.config.keep_ratio_range[0]),
            "alpha": float(x[3] * (self.config.alpha_range[1] - self.config.alpha_range[0]) + self.config.alpha_range[0]),
            "lr": float(x[4] * (self.config.lr_range[1] - self.config.lr_range[0]) + self.config.lr_range[0]),
        }

        return config

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a configuration.

        Args:
            x: Normalized parameters

        Returns:
            objectives: (n_objectives,) tensor of objective values
        """
        config = self.decode_params(x)

        # Simulate evaluation (in production, would train/evaluate model)
        # Objectives are synthetic functions of parameters

        # Accuracy (negate for minimization): higher d_model, more heads = better
        accuracy = 0.7 + 0.2 * (config["d_model"] / 1024) + 0.1 * (config["n_heads"] / 16)
        accuracy += np.random.randn() * 0.02  # Add noise
        neg_accuracy = -accuracy  # Negate for minimization

        # Latency: larger model = higher latency
        latency = 10 + 50 * (config["d_model"] / 1024) * (config["n_heads"] / 16)
        latency += np.random.randn() * 2

        # EPR-CV: lower learning rate and higher keep_ratio = more stable
        epr_cv = 0.05 + 0.1 * (config["lr"] / 1e-3) + 0.05 * (1 - config["keep_ratio"])
        epr_cv += np.random.randn() * 0.01

        # Topology gap: alpha controls this
        topo_gap = 0.02 * (1 - config["alpha"]) + 0.01
        topo_gap += np.random.randn() * 0.005

        # Energy: proportional to model size
        energy = 5 + 15 * (config["d_model"] / 1024) * (config["n_heads"] / 16)
        energy += np.random.randn() * 1

        objectives = torch.tensor([neg_accuracy, latency, epr_cv, topo_gap, energy], dtype=torch.float32)

        self.eval_count += 1

        return objectives


class ParetoRunner:
    """
    Pareto optimization runner for TF-A-N.

    Runs EHVI optimization to find Pareto-optimal configurations.
    """

    def __init__(self, config: Optional[ParetoRunnerConfig] = None):
        """
        Initialize Pareto runner.

        Args:
            config: Runner configuration
        """
        self.config = config or ParetoRunnerConfig()

        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluator
        self.evaluator = TFANConfigEvaluator(self.config)

        # Parameter bounds (all normalized to [0, 1])
        self.bounds = torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.float32)

        # Objective names
        self.objective_names = ["neg_accuracy", "latency", "epr_cv", "topo_gap", "energy"]

        # EHVI optimizer
        ehvi_config = EHVIConfig(
            n_initial_points=self.config.n_initial_points,
            n_iterations=self.config.n_iterations,
            n_objectives=self.config.n_objectives,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
        )

        self.optimizer = EHVIOptimizer(
            config=ehvi_config,
            bounds=self.bounds,
            objective_names=self.objective_names,
        )

        # Metrics tracking
        self.metrics_history = []

    def run(self, verbose: bool = True) -> ParetoFront:
        """
        Run Pareto optimization.

        Args:
            verbose: Print progress

        Returns:
            front: Final Pareto front
        """
        if verbose:
            print("=" * 70)
            print("PARETO OPTIMIZATION FOR TF-A-N")
            print("=" * 70)
            print(f"Initial samples: {self.config.n_initial_points}")
            print(f"Optimization iterations: {self.config.n_iterations}")
            print(f"Output directory: {self.output_dir}")
            print()

        start_time = time.time()

        # Run optimization
        front = self.optimizer.optimize(
            evaluate_fn=self.evaluator,
            verbose=verbose,
        )

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\nOptimization completed in {elapsed_time:.1f}s")
            print(f"Total evaluations: {self.evaluator.eval_count}")
            print()

        # Save results
        self.save_results(front)

        # Verify gates
        gates = self.optimizer.verify_gates()

        if verbose:
            print("HARD GATE VERIFICATION")
            print("-" * 70)
            print(f"Minimum Pareto points (≥6): {gates['min_pareto_points']} {'✓' if gates['min_pareto_points'] else '✗'}")
            print(f"Converged: {gates['converged']} {'✓' if gates['converged'] else '✗'}")
            print()
            print(f"Overall: {'ALL GATES PASSED ✓✓✓' if gates['all_pass'] else 'SOME GATES FAILED ✗✗✗'}")

        return front

    def save_results(self, front: ParetoFront):
        """
        Save Pareto front results.

        Args:
            front: Pareto front to save
        """
        # Convert configs to dicts
        configs_list = []
        for i, config_tensor in enumerate(front.configs):
            config_dict = self.evaluator.decode_params(config_tensor)
            config_dict["objectives"] = front.objectives[i].tolist()
            configs_list.append(config_dict)

        # Save as JSON
        results = {
            "n_pareto_points": front.n_dominated,
            "hypervolume": front.hypervolume,
            "configurations": configs_list,
        }

        output_path = self.output_dir / "pareto_front.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved Pareto front to: {output_path}")

        # Save objectives as numpy for easy loading
        np.save(self.output_dir / "pareto_objectives.npy", front.objectives.numpy())

    def load_results(self) -> Optional[Dict]:
        """
        Load previously saved results.

        Returns:
            results: Loaded results or None
        """
        output_path = self.output_dir / "pareto_front.json"

        if not output_path.exists():
            return None

        with open(output_path, "r") as f:
            results = json.load(f)

        return results

    def get_best_config(self, objective_weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Get best configuration from Pareto front based on weighted objectives.

        Args:
            objective_weights: Weights for each objective (default: equal)

        Returns:
            best_config: Configuration with lowest weighted sum
        """
        results = self.load_results()

        if results is None:
            raise ValueError("No results found. Run optimization first.")

        if objective_weights is None:
            # Default: equal weights
            objective_weights = {name: 1.0 for name in self.objective_names}

        # Compute weighted scores
        configs = results["configurations"]
        best_score = float("inf")
        best_config = None

        for config in configs:
            score = sum(
                config["objectives"][i] * objective_weights.get(name, 1.0)
                for i, name in enumerate(self.objective_names)
            )

            if score < best_score:
                best_score = score
                best_config = config

        return best_config
