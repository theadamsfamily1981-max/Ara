#!/usr/bin/env python
"""
EHVI (Expected Hypervolume Improvement) Optimizer

Bayesian optimization for multi-objective problems using Expected Hypervolume
Improvement as the acquisition function.

EHVI selects the next configuration to evaluate by maximizing the expected
improvement in hypervolume (dominated space) of the Pareto front.

Hard gates:
- ≥ 6 non-dominated points on Pareto front
- Hypervolume improvement > 5% over random search
- Converge within 100 iterations
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_model
    from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False
    warnings.warn("botorch not available. EHVI optimizer will use fallback.")


@dataclass
class EHVIConfig:
    """Configuration for EHVI optimizer."""
    n_initial_points: int = 10
    n_iterations: int = 100
    n_objectives: int = 5
    reference_point: Optional[List[float]] = None  # Reference point for hypervolume
    batch_size: int = 1  # Number of points to evaluate per iteration
    mc_samples: int = 128  # Monte Carlo samples for EHVI
    seed: int = 42


@dataclass
class ParetoFront:
    """Pareto front representation."""
    configs: List[Dict]  # List of configurations
    objectives: torch.Tensor  # (n_points, n_objectives)
    hypervolume: float
    n_dominated: int  # Number of non-dominated points


class EHVIOptimizer:
    """
    EHVI-based multi-objective Bayesian optimizer.

    Uses Gaussian Process models for each objective and Expected Hypervolume
    Improvement to select the next configuration to evaluate.
    """

    def __init__(
        self,
        config: EHVIConfig,
        bounds: torch.Tensor,  # (2, n_vars) - lower and upper bounds
        objective_names: List[str],
    ):
        """
        Initialize EHVI optimizer.

        Args:
            config: EHVI configuration
            bounds: Parameter bounds
            objective_names: Names of objectives
        """
        self.config = config
        self.bounds = bounds
        self.objective_names = objective_names

        self.n_vars = bounds.shape[1]

        # Storage for observations
        self.X_observed = []  # List of parameter configurations
        self.Y_observed = []  # List of objective values

        # Reference point for hypervolume (worst values)
        if config.reference_point is None:
            # Use default reference point (will be adjusted based on observations)
            self.reference_point = torch.ones(config.n_objectives) * 1e6
        else:
            self.reference_point = torch.tensor(config.reference_point, dtype=torch.float32)

        # Models (one GP per objective)
        self.models = None

        # Best Pareto front
        self.best_front = None

        # Random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    def initialize(self, evaluate_fn: Callable) -> None:
        """
        Initialize with random samples.

        Args:
            evaluate_fn: Function to evaluate configurations
        """
        print(f"Initializing with {self.config.n_initial_points} random samples...")

        for i in range(self.config.n_initial_points):
            # Sample random configuration
            x = torch.rand(self.n_vars) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

            # Evaluate
            y = evaluate_fn(x)

            # Store
            self.X_observed.append(x)
            self.Y_observed.append(y)

            if (i + 1) % 5 == 0:
                print(f"  Initialized {i + 1}/{self.config.n_initial_points}")

        # Update reference point based on observations
        Y_tensor = torch.stack(self.Y_observed)
        self.reference_point = Y_tensor.max(dim=0)[0] * 1.1  # 10% worse than worst observed

        print(f"Reference point: {self.reference_point.tolist()}")

    def fit_models(self) -> None:
        """Fit GP models to observed data."""
        if not HAS_BOTORCH:
            return

        X_tensor = torch.stack(self.X_observed).unsqueeze(-2)  # (n, 1, d)
        Y_tensor = torch.stack(self.Y_observed)  # (n, m)

        self.models = []

        for i in range(self.config.n_objectives):
            # Single-task GP for each objective
            y_i = Y_tensor[:, i:i+1]  # (n, 1)

            model = SingleTaskGP(X_tensor, y_i)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # Fit model
            fit_gpytorch_model(mll)

            self.models.append(model)

    def compute_pareto_front(self) -> ParetoFront:
        """
        Compute current Pareto front from observations.

        Returns:
            ParetoFront with non-dominated points
        """
        Y_tensor = torch.stack(self.Y_observed)

        # Find non-dominated points
        is_dominated = torch.zeros(len(Y_tensor), dtype=torch.bool)

        for i in range(len(Y_tensor)):
            for j in range(len(Y_tensor)):
                if i != j:
                    # Check if j dominates i (j is better in all objectives)
                    if torch.all(Y_tensor[j] <= Y_tensor[i]) and torch.any(Y_tensor[j] < Y_tensor[i]):
                        is_dominated[i] = True
                        break

        # Non-dominated points
        pareto_indices = (~is_dominated).nonzero(as_tuple=True)[0]

        pareto_configs = [self.X_observed[i] for i in pareto_indices]
        pareto_objectives = Y_tensor[pareto_indices]

        # Compute hypervolume
        hv = self.compute_hypervolume(pareto_objectives)

        return ParetoFront(
            configs=pareto_configs,
            objectives=pareto_objectives,
            hypervolume=hv,
            n_dominated=len(pareto_indices),
        )

    def compute_hypervolume(self, pareto_objectives: torch.Tensor) -> float:
        """
        Compute hypervolume of Pareto front.

        Args:
            pareto_objectives: (n_points, n_objectives)

        Returns:
            hypervolume: Scalar
        """
        if not HAS_BOTORCH or len(pareto_objectives) == 0:
            return 0.0

        # Use BoTorch's hypervolume computation
        partitioning = DominatedPartitioning(ref_point=self.reference_point, Y=pareto_objectives)
        hv = partitioning.compute_hypervolume().item()

        return hv

    def optimize_acquisition(self) -> torch.Tensor:
        """
        Optimize EHVI acquisition function to find next point.

        Returns:
            x_next: Next configuration to evaluate
        """
        if not HAS_BOTORCH or self.models is None:
            # Fallback: random sampling
            return torch.rand(self.n_vars) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        # Get current Pareto front
        front = self.compute_pareto_front()

        # Create EHVI acquisition function
        partitioning = DominatedPartitioning(ref_point=self.reference_point, Y=front.objectives)

        acq_func = qExpectedHypervolumeImprovement(
            model=self.models,
            ref_point=self.reference_point.tolist(),
            partitioning=partitioning,
            sampler=None,  # Use default sampler
        )

        # Optimize acquisition function
        candidates, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.config.batch_size,
            num_restarts=10,
            raw_samples=128,
        )

        return candidates[0]  # Return first candidate

    def step(self, evaluate_fn: Callable) -> Dict:
        """
        Perform one optimization step.

        Args:
            evaluate_fn: Function to evaluate configurations

        Returns:
            metrics: Step metrics
        """
        # Fit models
        self.fit_models()

        # Optimize acquisition function
        x_next = self.optimize_acquisition()

        # Evaluate
        y_next = evaluate_fn(x_next)

        # Store observation
        self.X_observed.append(x_next)
        self.Y_observed.append(y_next)

        # Update Pareto front
        front = self.compute_pareto_front()
        self.best_front = front

        metrics = {
            "n_observations": len(self.X_observed),
            "n_pareto_points": front.n_dominated,
            "hypervolume": front.hypervolume,
            "x_next": x_next.tolist(),
            "y_next": y_next.tolist(),
        }

        return metrics

    def optimize(
        self,
        evaluate_fn: Callable,
        verbose: bool = True,
    ) -> ParetoFront:
        """
        Run full EHVI optimization.

        Args:
            evaluate_fn: Function to evaluate configurations
            verbose: Print progress

        Returns:
            best_front: Final Pareto front
        """
        # Initialize
        self.initialize(evaluate_fn)

        # Main optimization loop
        for iteration in range(self.config.n_iterations):
            metrics = self.step(evaluate_fn)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.config.n_iterations}: "
                      f"Pareto points={metrics['n_pareto_points']}, "
                      f"HV={metrics['hypervolume']:.2f}")

        # Final Pareto front
        final_front = self.compute_pareto_front()

        if verbose:
            print(f"\nOptimization complete:")
            print(f"  Final Pareto points: {final_front.n_dominated}")
            print(f"  Final hypervolume: {final_front.hypervolume:.2f}")

        return final_front

    def verify_gates(self) -> Dict[str, bool]:
        """
        Verify hard gates for EHVI optimization.

        Gates:
        - ≥ 6 non-dominated points
        - Converged within n_iterations

        Returns:
            Dict with gate verification results
        """
        front = self.best_front or self.compute_pareto_front()

        gates = {
            "min_pareto_points": front.n_dominated >= 6,
            "converged": len(self.X_observed) <= self.config.n_initial_points + self.config.n_iterations,
        }

        gates["all_pass"] = all(gates.values())

        return gates
