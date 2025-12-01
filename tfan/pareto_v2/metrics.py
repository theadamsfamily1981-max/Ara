#!/usr/bin/env python
"""
Pareto Front Metrics

Metrics for evaluating the quality of Pareto fronts:
- Hypervolume (HV)
- Inverted Generational Distance (IGD)
- Spread/diversity metrics
- Convergence metrics

Usage:
    metrics = ParetoMetrics(reference_front)
    hv = metrics.hypervolume(pareto_front)
    igd = metrics.igd(pareto_front)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist


class ParetoMetrics:
    """
    Metrics for evaluating Pareto fronts.
    """

    def __init__(
        self,
        reference_point: Optional[torch.Tensor] = None,
        reference_front: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Pareto metrics.

        Args:
            reference_point: Reference point for hypervolume (worst case)
            reference_front: True Pareto front for IGD (if known)
        """
        self.reference_point = reference_point
        self.reference_front = reference_front

    def hypervolume(
        self,
        pareto_objectives: torch.Tensor,
        reference_point: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute hypervolume indicator.

        Hypervolume = volume of space dominated by Pareto front.
        Higher is better.

        Args:
            pareto_objectives: (n_points, n_objectives)
            reference_point: Reference point (if None, use self.reference_point)

        Returns:
            hv: Hypervolume value
        """
        if reference_point is None:
            reference_point = self.reference_point

        if reference_point is None:
            # Use worst observed values
            reference_point = pareto_objectives.max(dim=0)[0] * 1.1

        # Simple hypervolume computation (works for 2D, approximate for higher D)
        n_objectives = pareto_objectives.shape[1]

        if n_objectives == 2:
            # Exact 2D hypervolume
            return self._hypervolume_2d(pareto_objectives, reference_point)
        else:
            # Approximate using Monte Carlo for higher dimensions
            return self._hypervolume_mc(pareto_objectives, reference_point)

    def _hypervolume_2d(
        self,
        pareto_objectives: torch.Tensor,
        reference_point: torch.Tensor,
    ) -> float:
        """
        Compute exact 2D hypervolume.

        Args:
            pareto_objectives: (n_points, 2)
            reference_point: (2,)

        Returns:
            hv: Hypervolume
        """
        # Sort by first objective
        sorted_indices = torch.argsort(pareto_objectives[:, 0])
        sorted_points = pareto_objectives[sorted_indices]

        hv = 0.0
        prev_x = reference_point[0].item()

        for point in sorted_points:
            x, y = point[0].item(), point[1].item()

            # Width * Height
            width = prev_x - x
            height = reference_point[1].item() - y

            if width > 0 and height > 0:
                hv += width * height

            prev_x = x

        return hv

    def _hypervolume_mc(
        self,
        pareto_objectives: torch.Tensor,
        reference_point: torch.Tensor,
        n_samples: int = 10000,
    ) -> float:
        """
        Approximate hypervolume using Monte Carlo sampling.

        Args:
            pareto_objectives: (n_points, n_objectives)
            reference_point: (n_objectives,)
            n_samples: Number of MC samples

        Returns:
            hv: Approximate hypervolume
        """
        n_objectives = pareto_objectives.shape[1]

        # Sample points uniformly in the hyperrectangle
        min_vals = pareto_objectives.min(dim=0)[0]
        max_vals = reference_point

        samples = torch.rand(n_samples, n_objectives)
        samples = samples * (max_vals - min_vals) + min_vals

        # Count dominated samples
        dominated = torch.zeros(n_samples, dtype=torch.bool)

        for pareto_point in pareto_objectives:
            # Check if sample is dominated by this Pareto point
            is_dominated = torch.all(samples <= pareto_point.unsqueeze(0), dim=1)
            dominated = dominated | is_dominated

        # Hypervolume = (dominated fraction) * (total volume)
        dominated_fraction = dominated.sum().item() / n_samples
        total_volume = torch.prod(max_vals - min_vals).item()

        hv = dominated_fraction * total_volume

        return hv

    def igd(
        self,
        pareto_objectives: torch.Tensor,
        reference_front: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute Inverted Generational Distance (IGD).

        IGD = average distance from reference front to nearest Pareto point.
        Lower is better.

        Args:
            pareto_objectives: (n_points, n_objectives)
            reference_front: (n_ref, n_objectives) - true Pareto front

        Returns:
            igd: IGD value
        """
        if reference_front is None:
            reference_front = self.reference_front

        if reference_front is None:
            raise ValueError("Reference front required for IGD computation")

        # Convert to numpy for cdist
        pareto_np = pareto_objectives.numpy()
        ref_np = reference_front.numpy()

        # Compute pairwise distances
        distances = cdist(ref_np, pareto_np, metric="euclidean")

        # For each reference point, find closest Pareto point
        min_distances = distances.min(axis=1)

        # Average distance
        igd = np.mean(min_distances)

        return igd

    def spread(self, pareto_objectives: torch.Tensor) -> float:
        """
        Compute spread (diversity) metric.

        Measures how well-distributed the Pareto points are.
        Lower is better (more uniform spread).

        Args:
            pareto_objectives: (n_points, n_objectives)

        Returns:
            spread: Spread metric
        """
        n_points = len(pareto_objectives)

        if n_points < 2:
            return 0.0

        # Compute pairwise distances
        pareto_np = pareto_objectives.numpy()
        distances = cdist(pareto_np, pareto_np, metric="euclidean")

        # Find nearest neighbor distances
        np.fill_diagonal(distances, np.inf)
        nn_distances = distances.min(axis=1)

        # Spread = std of nearest neighbor distances
        spread = np.std(nn_distances)

        return spread

    def spacing(self, pareto_objectives: torch.Tensor) -> float:
        """
        Compute spacing metric.

        Measures uniformity of distribution.
        Lower is better (more uniform).

        Args:
            pareto_objectives: (n_points, n_objectives)

        Returns:
            spacing: Spacing metric
        """
        n_points = len(pareto_objectives)

        if n_points < 2:
            return 0.0

        # Compute pairwise distances
        pareto_np = pareto_objectives.numpy()
        distances = cdist(pareto_np, pareto_np, metric="euclidean")

        # Find nearest neighbor distances
        np.fill_diagonal(distances, np.inf)
        nn_distances = distances.min(axis=1)

        # Spacing = mean absolute deviation from mean nn distance
        mean_dist = np.mean(nn_distances)
        spacing = np.mean(np.abs(nn_distances - mean_dist))

        return spacing

    def coverage(
        self,
        pareto_objectives: torch.Tensor,
        other_pareto: torch.Tensor,
    ) -> float:
        """
        Compute coverage metric (how much pareto_objectives dominates other_pareto).

        Coverage(A, B) = |{b ∈ B : ∃a ∈ A that dominates b}| / |B|

        Args:
            pareto_objectives: Pareto front A
            other_pareto: Pareto front B

        Returns:
            coverage: Coverage fraction [0, 1]
        """
        n_other = len(other_pareto)
        dominated_count = 0

        for b in other_pareto:
            # Check if any point in pareto_objectives dominates b
            for a in pareto_objectives:
                if torch.all(a <= b) and torch.any(a < b):
                    dominated_count += 1
                    break

        coverage = dominated_count / n_other if n_other > 0 else 0.0

        return coverage

    def compute_all_metrics(
        self,
        pareto_objectives: torch.Tensor,
        reference_point: Optional[torch.Tensor] = None,
        reference_front: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute all metrics for a Pareto front.

        Args:
            pareto_objectives: (n_points, n_objectives)
            reference_point: Reference point for HV
            reference_front: True front for IGD

        Returns:
            metrics: Dict with all computed metrics
        """
        metrics = {}

        # Hypervolume
        try:
            metrics["hypervolume"] = self.hypervolume(pareto_objectives, reference_point)
        except Exception as e:
            metrics["hypervolume"] = None

        # IGD (if reference front available)
        if reference_front is not None:
            try:
                metrics["igd"] = self.igd(pareto_objectives, reference_front)
            except Exception as e:
                metrics["igd"] = None
        else:
            metrics["igd"] = None

        # Spread
        try:
            metrics["spread"] = self.spread(pareto_objectives)
        except Exception as e:
            metrics["spread"] = None

        # Spacing
        try:
            metrics["spacing"] = self.spacing(pareto_objectives)
        except Exception as e:
            metrics["spacing"] = None

        # Number of points
        metrics["n_points"] = len(pareto_objectives)

        return metrics


def compare_pareto_fronts(
    front_a: torch.Tensor,
    front_b: torch.Tensor,
    reference_point: torch.Tensor,
) -> Dict[str, float]:
    """
    Compare two Pareto fronts.

    Args:
        front_a: First Pareto front
        front_b: Second Pareto front
        reference_point: Reference point for HV

    Returns:
        comparison: Dict with comparison metrics
    """
    metrics = ParetoMetrics(reference_point=reference_point)

    # Compute metrics for both fronts
    metrics_a = metrics.compute_all_metrics(front_a, reference_point)
    metrics_b = metrics.compute_all_metrics(front_b, reference_point)

    # Coverage
    coverage_a_b = metrics.coverage(front_a, front_b)
    coverage_b_a = metrics.coverage(front_b, front_a)

    comparison = {
        "front_a": metrics_a,
        "front_b": metrics_b,
        "coverage_a_over_b": coverage_a_b,
        "coverage_b_over_a": coverage_b_a,
        "hv_improvement": (
            (metrics_a["hypervolume"] - metrics_b["hypervolume"]) / metrics_b["hypervolume"]
            if metrics_b["hypervolume"] and metrics_b["hypervolume"] > 0
            else None
        ),
    }

    return comparison
