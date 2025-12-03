"""Phase 4: Reality Check - Topology-Based Hallucination Detection.

The RealityMonitor validates model outputs by comparing the topological
"shape" of the output against the input. If they diverge too much,
the output is flagged as a potential hallucination.

Key Metrics:
    Wasserstein Distance: Measures how different two distributions are
        - ≤ 2% gap: Output is consistent with input
        - > 2% gap: Potential hallucination

    Cosine Similarity: Measures angular alignment
        - ≥ 0.90: Output aligns with input topology
        - < 0.90: Output deviates significantly

Fallback Mechanism:
    CAT (Concentrated Attention Trigger): When hallucination detected,
    retry with denser attention (higher keep_ratio) to force the model
    to look more closely at the input.

This implements the TopologyGate from TFAN mm.topo_gate.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum, auto
import warnings
import sys
from pathlib import Path

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Try to import TFAN topology gate
_TFAN_TOPO_AVAILABLE = False
try:
    from tfan.mm.topo_gate import TopologyGate as TFANTopologyGate
    from tfan.topo import TopologyRegularizer
    _TFAN_TOPO_AVAILABLE = True
except ImportError:
    pass


class VerificationStatus(Enum):
    """Output verification status."""
    VERIFIED = auto()           # Topology matches, output is valid
    SUSPICIOUS = auto()         # Minor deviation, proceed with caution
    HALLUCINATION = auto()      # Major deviation, likely hallucinating
    CAT_ACTIVATED = auto()      # Fallback triggered
    VALIDATION_FAILED = auto()  # Could not validate


@dataclass
class VerificationResult:
    """Result of reality check verification."""
    status: VerificationStatus
    is_valid: bool
    message: str
    wasserstein_distance: float
    cosine_similarity: float
    gate_passed: bool
    cat_activated: bool
    retry_count: int
    metrics: Dict[str, float]


class RealityMonitor:
    """
    The Reality Check - Topology-Based Hallucination Detection.

    Validates model outputs by comparing their topological structure
    against the input. Prevents the model from "making things up"
    that don't match the input context.

    Args:
        d_model: Model dimension
        wasserstein_gap_max: Maximum allowed Wasserstein gap (default 0.02 = 2%)
        cosine_min: Minimum cosine similarity (default 0.90)
        cat_fallback_ratio: Keep ratio for CAT fallback (default 0.50)
        max_retries: Maximum retry attempts before giving up
        device: Compute device
    """

    def __init__(
        self,
        d_model: int = 4096,
        wasserstein_gap_max: float = 0.02,
        cosine_min: float = 0.90,
        cat_fallback_ratio: float = 0.50,
        max_retries: int = 2,
        device: str = "cpu",
    ):
        self.d_model = d_model
        self.wasserstein_gap_max = wasserstein_gap_max
        self.cosine_min = cosine_min
        self.cat_fallback_ratio = cat_fallback_ratio
        self.max_retries = max_retries
        self.device = device

        # TFAN topology gate if available
        self.tfan_gate = None
        if _TFAN_TOPO_AVAILABLE:
            try:
                self.tfan_gate = TFANTopologyGate(
                    d_model=d_model,
                    wasserstein_gap_max=wasserstein_gap_max,
                    cosine_min=cosine_min,
                    cat_fallback_ratio=cat_fallback_ratio,
                    max_retries=max_retries,
                )
            except Exception as e:
                warnings.warn(f"Failed to init TFAN topology gate: {e}")

        # Statistics
        self.total_checks = 0
        self.gate_passes = 0
        self.gate_failures = 0
        self.cat_activations = 0

        # Input topology cache (for comparison)
        self._input_topology_cache: Optional[torch.Tensor] = None

    def set_input_topology(self, input_representation: torch.Tensor):
        """
        Cache the input topology for comparison.

        Call this after processing input through the thalamus,
        before generating output.

        Args:
            input_representation: The conscious input tokens
        """
        self._input_topology_cache = self._compute_topology_signature(
            input_representation
        )

    def verify(
        self,
        model_output: torch.Tensor,
        input_topology: Optional[torch.Tensor] = None,
        recompute_fn: Optional[Callable[[float], torch.Tensor]] = None,
    ) -> VerificationResult:
        """
        Verify model output against input topology.

        This is the main hallucination detection interface.

        Args:
            model_output: Raw output from the model (before decoding)
            input_topology: Input topology to compare against (uses cache if None)
            recompute_fn: Function to regenerate with different keep_ratio

        Returns:
            VerificationResult with validation status
        """
        self.total_checks += 1

        # Get input topology
        if input_topology is None:
            input_topology = self._input_topology_cache

        if input_topology is None:
            return VerificationResult(
                status=VerificationStatus.VALIDATION_FAILED,
                is_valid=True,  # Pass through if no baseline
                message="No input topology baseline. Passing through.",
                wasserstein_distance=0.0,
                cosine_similarity=1.0,
                gate_passed=True,
                cat_activated=False,
                retry_count=0,
                metrics={},
            )

        # Compute output topology
        output_topology = self._compute_topology_signature(model_output)

        # Compare topologies
        wass_dist, cosine_sim = self._compare_topologies(
            output_topology, input_topology
        )

        # Check gates
        wass_passed = wass_dist <= self.wasserstein_gap_max
        cosine_passed = cosine_sim >= self.cosine_min
        gate_passed = wass_passed and cosine_passed

        if gate_passed:
            self.gate_passes += 1
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                is_valid=True,
                message="Output topology verified. Consistent with input.",
                wasserstein_distance=wass_dist,
                cosine_similarity=cosine_sim,
                gate_passed=True,
                cat_activated=False,
                retry_count=0,
                metrics={
                    "wasserstein_gap": wass_dist,
                    "cosine_similarity": cosine_sim,
                },
            )

        # Gate failed - check if CAT fallback available
        self.gate_failures += 1

        if recompute_fn is None:
            # No fallback available
            status = (
                VerificationStatus.SUSPICIOUS
                if cosine_sim > 0.8
                else VerificationStatus.HALLUCINATION
            )
            return VerificationResult(
                status=status,
                is_valid=False,
                message=(
                    f"Hallucination detected. Wasserstein gap ({wass_dist:.4f}) "
                    f"or cosine sim ({cosine_sim:.4f}) out of bounds. "
                    "No fallback available."
                ),
                wasserstein_distance=wass_dist,
                cosine_similarity=cosine_sim,
                gate_passed=False,
                cat_activated=False,
                retry_count=0,
                metrics={
                    "wasserstein_gap": wass_dist,
                    "cosine_similarity": cosine_sim,
                },
            )

        # Attempt CAT fallback
        return self._attempt_cat_fallback(
            input_topology, recompute_fn, wass_dist, cosine_sim
        )

    def _attempt_cat_fallback(
        self,
        input_topology: torch.Tensor,
        recompute_fn: Callable[[float], torch.Tensor],
        initial_wass: float,
        initial_cosine: float,
    ) -> VerificationResult:
        """Attempt CAT (Concentrated Attention Trigger) fallback."""
        self.cat_activations += 1

        for retry in range(self.max_retries):
            # Increase keep_ratio incrementally
            new_keep_ratio = min(1.0, self.cat_fallback_ratio + retry * 0.15)

            warnings.warn(
                f"Topology gate failed. CAT retry {retry + 1}/{self.max_retries} "
                f"with keep_ratio={new_keep_ratio:.2f}"
            )

            try:
                # Regenerate with denser attention
                new_output = recompute_fn(new_keep_ratio)
                new_topology = self._compute_topology_signature(new_output)

                # Re-check
                wass_dist, cosine_sim = self._compare_topologies(
                    new_topology, input_topology
                )

                if wass_dist <= self.wasserstein_gap_max and cosine_sim >= self.cosine_min:
                    self.gate_passes += 1
                    return VerificationResult(
                        status=VerificationStatus.CAT_ACTIVATED,
                        is_valid=True,
                        message=(
                            f"CAT fallback succeeded on retry {retry + 1}. "
                            f"Denser attention restored consistency."
                        ),
                        wasserstein_distance=wass_dist,
                        cosine_similarity=cosine_sim,
                        gate_passed=True,
                        cat_activated=True,
                        retry_count=retry + 1,
                        metrics={
                            "wasserstein_gap": wass_dist,
                            "cosine_similarity": cosine_sim,
                            "keep_ratio_used": new_keep_ratio,
                        },
                    )

            except Exception as e:
                warnings.warn(f"CAT retry {retry + 1} failed: {e}")
                continue

        # All retries exhausted
        return VerificationResult(
            status=VerificationStatus.HALLUCINATION,
            is_valid=False,
            message=(
                f"Hallucination detected. CAT fallback failed after "
                f"{self.max_retries} retries. Initial wasserstein: {initial_wass:.4f}, "
                f"cosine: {initial_cosine:.4f}"
            ),
            wasserstein_distance=initial_wass,
            cosine_similarity=initial_cosine,
            gate_passed=False,
            cat_activated=True,
            retry_count=self.max_retries,
            metrics={
                "wasserstein_gap": initial_wass,
                "cosine_similarity": initial_cosine,
                "cat_retries": self.max_retries,
            },
        )

    def _compute_topology_signature(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute topological signature of a tensor.

        This is a simplified approximation of persistence landscapes.
        For full topological analysis, use TFAN's TopologyRegularizer.

        Args:
            tensor: Input tensor (batch, seq_len, d_model)

        Returns:
            Topology signature tensor
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        batch, seq_len, d_model = tensor.shape

        # Compute pairwise distances (simplified topology approximation)
        # In full implementation, this would use persistent homology
        flat = tensor.view(batch, -1)

        # Centroid-based signature
        centroid = tensor.mean(dim=1)  # (batch, d_model)

        # Distance distribution from centroid
        distances = torch.norm(tensor - centroid.unsqueeze(1), dim=2)  # (batch, seq_len)

        # Compute distribution statistics as signature
        signature = torch.stack([
            distances.mean(dim=1),
            distances.std(dim=1),
            distances.max(dim=1)[0],
            distances.min(dim=1)[0],
            torch.quantile(distances, 0.25, dim=1),
            torch.quantile(distances, 0.50, dim=1),
            torch.quantile(distances, 0.75, dim=1),
        ], dim=1)  # (batch, 7)

        return signature

    def _compare_topologies(
        self,
        topology1: torch.Tensor,
        topology2: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compare two topology signatures.

        Returns:
            (wasserstein_distance, cosine_similarity)
        """
        # Flatten if needed
        t1 = topology1.flatten().float()
        t2 = topology2.flatten().float()

        # Normalize
        t1_norm = t1 / (t1.norm() + 1e-8)
        t2_norm = t2 / (t2.norm() + 1e-8)

        # Cosine similarity
        cosine_sim = torch.dot(t1_norm, t2_norm).item()

        # Wasserstein-like distance (simplified: L1 of sorted values)
        t1_sorted = torch.sort(t1_norm)[0]
        t2_sorted = torch.sort(t2_norm)[0]

        # Handle different lengths
        min_len = min(len(t1_sorted), len(t2_sorted))
        wass_dist = torch.abs(t1_sorted[:min_len] - t2_sorted[:min_len]).mean().item()

        return wass_dist, cosine_sim

    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        total = max(self.total_checks, 1)
        return {
            "total_checks": self.total_checks,
            "gate_passes": self.gate_passes,
            "gate_failures": self.gate_failures,
            "cat_activations": self.cat_activations,
            "pass_rate": self.gate_passes / total,
            "failure_rate": self.gate_failures / total,
            "cat_rate": self.cat_activations / total,
        }

    def reset_statistics(self):
        """Reset verification statistics."""
        self.total_checks = 0
        self.gate_passes = 0
        self.gate_failures = 0
        self.cat_activations = 0


# Convenience factory
def create_reality_monitor(
    wasserstein_max: float = 0.02,
    cosine_min: float = 0.90,
    d_model: int = 4096,
) -> RealityMonitor:
    """Create a RealityMonitor instance."""
    return RealityMonitor(
        d_model=d_model,
        wasserstein_gap_max=wasserstein_max,
        cosine_min=cosine_min,
    )


__all__ = [
    "RealityMonitor",
    "VerificationResult",
    "VerificationStatus",
    "create_reality_monitor",
]
