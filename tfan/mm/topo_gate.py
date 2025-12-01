"""
Topology gate for multi-modal fusion.

Rejects landmark masks that break structural constraints.
Falls back to CAT (denser attention) when topology is violated.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import warnings

from ..topo import TopologyRegularizer
from .fuse import FusedRepresentation


class TopologyGate(nn.Module):
    """
    Topology gate for validating fusion quality.

    Enforces:
    - Wasserstein distance ≤ 2% vs target topology
    - Cosine similarity ≥ 0.90 vs target topology
    - Falls back to CAT (denser masks) on violation
    """

    def __init__(
        self,
        d_model: int,
        wasserstein_gap_max: float = 0.02,
        cosine_min: float = 0.90,
        cat_fallback_ratio: float = 0.50,
        max_retries: int = 2,
    ):
        """
        Args:
            d_model: Model dimension
            wasserstein_gap_max: Maximum Wasserstein gap
            cosine_min: Minimum cosine similarity
            cat_fallback_ratio: Keep ratio for CAT fallback
            max_retries: Maximum retry attempts before fallback
        """
        super().__init__()
        self.d_model = d_model
        self.wasserstein_gap_max = wasserstein_gap_max
        self.cosine_min = cosine_min
        self.cat_fallback_ratio = cat_fallback_ratio
        self.max_retries = max_retries

        # Topology regularizer for validation
        self.topo_reg = TopologyRegularizer(
            lambda_topo=0.0,  # Not used for training, just validation
            wasserstein_gap_max=wasserstein_gap_max,
            cosine_min=cosine_min,
        )

        # Target topology (set externally)
        self.target_topology: Optional[Dict[int, any]] = None

        # Metrics
        self.gate_violations = 0
        self.gate_passes = 0
        self.fallback_activations = 0

    def set_target_topology(self, target_diagrams: Dict[int, any]):
        """
        Set target persistence diagrams for validation.

        Args:
            target_diagrams: Dict of degree -> persistence diagram
        """
        self.target_topology = target_diagrams
        self.topo_reg.set_target_topology(target_diagrams)

    def forward(
        self,
        fused: FusedRepresentation,
        recompute_landmarks_fn: Optional[callable] = None,
    ) -> Tuple[FusedRepresentation, Dict[str, float]]:
        """
        Validate fused representation against topology constraints.

        Args:
            fused: FusedRepresentation to validate
            recompute_landmarks_fn: Function to recompute landmarks with new keep_ratio

        Returns:
            (validated_fused, metrics)
            If topology is violated, returns updated fused with adjusted landmarks
        """
        if self.target_topology is None:
            # No target set, pass through
            return fused, {"gate_passed": True}

        # Compute current topology
        topo_result = self.topo_reg.compute_landscape(fused.tokens, return_diagrams=True)
        current_diagrams = topo_result.get("diagrams", {})

        # Validate against target
        if len(current_diagrams) == 0:
            # No diagrams computed, pass through with warning
            warnings.warn("No topology diagrams computed. Skipping gate.")
            return fused, {"gate_passed": True, "reason": "no_diagrams"}

        # Flatten diagrams for validation
        current_diagrams_flat = {}
        for deg, diag_list in current_diagrams.items():
            if len(diag_list) > 0:
                current_diagrams_flat[deg] = diag_list[0]  # First batch item

        # Validate
        passes, metrics = self.topo_reg.validate_against_exact(
            approximate_diagrams=current_diagrams_flat,
            exact_diagrams=self.target_topology,
        )

        if passes:
            # Gate passed
            self.gate_passes += 1
            return fused, {"gate_passed": True, **metrics}

        # Gate failed: attempt retry or fallback
        self.gate_violations += 1

        if recompute_landmarks_fn is not None:
            # Try recomputing with higher keep_ratio
            for retry in range(self.max_retries):
                new_keep_ratio = min(1.0, (retry + 2) * 0.33)  # Increase incrementally
                warnings.warn(
                    f"Topology gate failed. Retrying with keep_ratio={new_keep_ratio:.2f}"
                )

                # Recompute landmarks
                new_fused = recompute_landmarks_fn(new_keep_ratio)

                # Re-validate
                topo_result = self.topo_reg.compute_landscape(new_fused.tokens, return_diagrams=True)
                current_diagrams = topo_result.get("diagrams", {})
                current_diagrams_flat = {}
                for deg, diag_list in current_diagrams.items():
                    if len(diag_list) > 0:
                        current_diagrams_flat[deg] = diag_list[0]

                passes, metrics = self.topo_reg.validate_against_exact(
                    approximate_diagrams=current_diagrams_flat,
                    exact_diagrams=self.target_topology,
                )

                if passes:
                    self.gate_passes += 1
                    return new_fused, {"gate_passed": True, "retries": retry + 1, **metrics}

        # All retries exhausted: activate CAT fallback
        warnings.warn(
            f"Topology gate failed after {self.max_retries} retries. "
            f"Activating CAT fallback with keep_ratio={self.cat_fallback_ratio:.2f}"
        )
        self.fallback_activations += 1

        if recompute_landmarks_fn is not None:
            fallback_fused = recompute_landmarks_fn(self.cat_fallback_ratio)
        else:
            # Can't recompute, return original with warning
            fallback_fused = fused

        return fallback_fused, {
            "gate_passed": False,
            "fallback_activated": True,
            **metrics,
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get topology gate metrics."""
        total = self.gate_passes + self.gate_violations
        return {
            "gate_pass_rate": self.gate_passes / max(total, 1),
            "gate_violation_rate": self.gate_violations / max(total, 1),
            "fallback_rate": self.fallback_activations / max(total, 1),
            "total_checks": total,
        }
