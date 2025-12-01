"""
Hyperbolic Sparsity Regularizer

Exploits hyperbolic geometry for control signals in SNN training.

NOTE: Requires PyTorch. Functions are no-ops if torch not available.
Penalizes embeddings that move too close to the Poincaré ball boundary,
especially when the L2 stability gap is high.

Key insight: Points near the boundary of the Poincaré ball represent
extreme/uncertain states. High instability + boundary proximity = danger.

Control Law:
    penalty = ||z||² / (1 - ||z||²)² * stability_gap

This creates a soft boundary that becomes harder as instability increases.

Usage:
    from tfan.losses.hyperbolic_reg import HyperbolicSparsityRegularizer

    reg = HyperbolicSparsityRegularizer(c=1.0, stability_weight=2.0)

    # During training
    loss = base_loss + reg(embeddings, stability_gap)
"""

import math
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger("tfan.losses.hyperbolic")

# Optional torch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# Small epsilon for numerical stability
EPS = 1e-7

# Use Any for type hints when torch not available
if not TORCH_AVAILABLE:
    from typing import Any as TensorType
else:
    TensorType = torch.Tensor


def project_to_poincare(
    x: torch.Tensor,
    c: float = 1.0,
    max_norm: float = 0.99,
) -> torch.Tensor:
    """
    Project points to the interior of the Poincaré ball.

    Ensures ||x|| < 1/sqrt(c) (ball radius).

    Args:
        x: Points in R^d [..., d]
        c: Curvature parameter (c > 0)
        max_norm: Maximum allowed norm as fraction of radius

    Returns:
        Projected points inside ball
    """
    radius = 1.0 / math.sqrt(c)
    max_allowed = radius * max_norm

    norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(max_allowed / (norm + EPS), max=1.0)

    return x * scale


def hyperbolic_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Compute hyperbolic distance in the Poincaré ball.

    d_c(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||−x ⊕_c y||)

    where ⊕_c is the Möbius addition.

    Args:
        x, y: Points in Poincaré ball [..., d]
        c: Curvature parameter

    Returns:
        Hyperbolic distances [...]
    """
    sqrt_c = math.sqrt(c)

    # Möbius addition: -x ⊕_c y
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)

    # Numerator: (1 + 2c<x,y> + c||y||²)(-x) + (1 - c||x||²)y
    num = ((1 - 2 * c * xy_inner + c * y_norm_sq) * (-x) +
           (1 - c * x_norm_sq) * y)

    # Denominator: 1 - 2c<x,y> + c²||x||²||y||²
    denom = 1 - 2 * c * xy_inner + c * c * x_norm_sq * y_norm_sq

    mobius = num / (denom + EPS)
    mobius_norm = torch.norm(mobius, dim=-1)

    # Distance formula
    dist = (2.0 / sqrt_c) * torch.arctanh(
        torch.clamp(sqrt_c * mobius_norm, max=1.0 - EPS)
    )

    return dist


def conformal_factor(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Compute conformal factor λ_c(x) = 2 / (1 - c||x||²).

    This measures the local "stretching" of the hyperbolic metric.
    Higher values near boundary = more sensitive region.

    Args:
        x: Points in Poincaré ball [..., d]
        c: Curvature parameter

    Returns:
        Conformal factors [...]
    """
    x_norm_sq = torch.sum(x * x, dim=-1)
    return 2.0 / (1.0 - c * x_norm_sq + EPS)


def poincare_boundary_penalty(
    x: torch.Tensor,
    c: float = 1.0,
    stability_gap: float = 0.0,
    stability_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute boundary penalty for Poincaré ball embeddings.

    Penalty increases as points approach boundary, scaled by stability gap.

    penalty = λ(x)² * (1 + stability_weight * stability_gap)

    Args:
        x: Points in Poincaré ball [batch, d]
        c: Curvature parameter
        stability_gap: L2 topology gap (0 = stable, higher = unstable)
        stability_weight: How much to amplify penalty based on instability

    Returns:
        Scalar penalty
    """
    # Conformal factor (increases near boundary)
    lambda_x = conformal_factor(x, c)

    # Base penalty: mean of squared conformal factors
    base_penalty = (lambda_x ** 2).mean()

    # Stability amplification
    stability_mult = 1.0 + stability_weight * stability_gap

    return base_penalty * stability_mult


def curvature_stability_loss(
    x: torch.Tensor,
    c: float = 1.0,
    target_curvature: float = 1.0,
    curvature_weight: float = 0.1,
) -> torch.Tensor:
    """
    Loss for maintaining target curvature.

    Penalizes deviation from optimal curvature for the data distribution.

    Args:
        x: Points in manifold [batch, d]
        c: Current curvature
        target_curvature: Desired curvature
        curvature_weight: Loss weight

    Returns:
        Curvature deviation loss
    """
    # Estimate optimal curvature from point distribution
    # Using negative curvature formula: c_opt ≈ var(||x||) / mean(||x||)²
    norms = torch.norm(x, dim=-1)
    estimated_c = norms.var() / (norms.mean() ** 2 + EPS)

    # Penalize deviation from target
    curvature_loss = curvature_weight * (c - target_curvature) ** 2

    return curvature_loss


class HyperbolicSparsityRegularizer(nn.Module):
    """
    Hyperbolic sparsity regularizer for SNN training.

    Combines:
    1. Boundary penalty - prevents extreme embeddings
    2. Stability-aware scaling - tighter constraints when unstable
    3. Curvature regularization - maintains geometric stability

    Usage:
        reg = HyperbolicSparsityRegularizer(c=1.0, stability_weight=2.0)

        # In training loop
        embeddings = model.get_embeddings()
        stability_gap = compute_topology_gap()

        reg_loss = reg(embeddings, stability_gap)
        total_loss = task_loss + reg_loss
    """

    def __init__(
        self,
        c: float = 1.0,
        stability_weight: float = 2.0,
        boundary_weight: float = 0.1,
        curvature_weight: float = 0.01,
        target_curvature: float = 1.0,
        adaptive_c: bool = False,
    ):
        """
        Initialize regularizer.

        Args:
            c: Curvature parameter (c > 0)
            stability_weight: How much instability amplifies boundary penalty
            boundary_weight: Weight for boundary penalty term
            curvature_weight: Weight for curvature regularization
            target_curvature: Target curvature for regularization
            adaptive_c: If True, treat c as learnable parameter
        """
        super().__init__()

        if adaptive_c:
            self.c = nn.Parameter(torch.tensor(c))
        else:
            self.register_buffer('c', torch.tensor(c))

        self.stability_weight = stability_weight
        self.boundary_weight = boundary_weight
        self.curvature_weight = curvature_weight
        self.target_curvature = target_curvature
        self.adaptive_c = adaptive_c

    def forward(
        self,
        x: torch.Tensor,
        stability_gap: Union[float, torch.Tensor] = 0.0,
    ) -> torch.Tensor:
        """
        Compute hyperbolic regularization loss.

        Args:
            x: Embeddings [batch, d] (assumed in Poincaré ball)
            stability_gap: L2 topology stability gap

        Returns:
            Regularization loss (scalar)
        """
        c = self.c.item() if isinstance(self.c, torch.Tensor) else self.c

        # Project to ensure inside ball
        x_proj = project_to_poincare(x, c=c)

        # Boundary penalty
        boundary_loss = poincare_boundary_penalty(
            x_proj,
            c=c,
            stability_gap=float(stability_gap),
            stability_weight=self.stability_weight,
        )

        # Curvature regularization (if adaptive)
        if self.adaptive_c:
            curv_loss = curvature_stability_loss(
                x_proj,
                c=c,
                target_curvature=self.target_curvature,
                curvature_weight=self.curvature_weight,
            )
        else:
            curv_loss = torch.tensor(0.0, device=x.device)

        total = self.boundary_weight * boundary_loss + curv_loss

        return total

    def get_curvature(self) -> float:
        """Get current curvature value."""
        return self.c.item() if isinstance(self.c, torch.Tensor) else self.c

    def set_curvature(self, c: float):
        """Set curvature value."""
        if self.adaptive_c:
            self.c.data.fill_(c)
        else:
            self.c = torch.tensor(c)


def compute_l3_valence_from_geometry(
    x: torch.Tensor,
    c: float = 1.0,
    stability_gap: float = 0.0,
) -> float:
    """
    Compute L3 valence signal from hyperbolic geometry.

    Valence ∝ 1 / (1 - c²) when near boundary with high instability.
    Low valence (negative) → conservative policy in L3.

    Args:
        x: Embeddings in Poincaré ball
        c: Curvature parameter
        stability_gap: Topology stability gap

    Returns:
        Valence signal in [-1, 1]
    """
    # Mean norm of embeddings
    mean_norm = torch.norm(x, dim=-1).mean().item()

    # Distance from boundary (1/sqrt(c) is radius)
    radius = 1.0 / math.sqrt(c) if c > 0 else 1.0
    boundary_dist = radius - mean_norm

    # Normalized boundary distance [0, 1]
    norm_dist = max(0, min(1, boundary_dist / radius))

    # Base valence from geometry (farther from boundary = higher valence)
    geo_valence = 2 * norm_dist - 1  # [-1, 1]

    # Stability adjustment (high instability pushes valence negative)
    stability_penalty = min(1.0, stability_gap * 2.0)
    adjusted_valence = geo_valence - stability_penalty

    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, adjusted_valence))


# Exports
__all__ = [
    "HyperbolicSparsityRegularizer",
    "poincare_boundary_penalty",
    "curvature_stability_loss",
    "hyperbolic_distance",
    "project_to_poincare",
    "conformal_factor",
    "compute_l3_valence_from_geometry",
]
