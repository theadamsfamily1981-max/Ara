"""
RMSNorm (Root Mean Square Layer Normalization) for TF-A-N 7B.

More efficient than LayerNorm, used in LLaMA and similar models.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes using RMS instead of mean/variance:
        y = x / RMS(x) * weight
        RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dim: Hidden dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RMSNormFused(nn.Module):
    """
    Fused RMSNorm implementation for better performance.

    Uses torch.compile or custom CUDA kernels when available.
    Falls back to standard implementation otherwise.

    Args:
        dim: Hidden dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        # Try to compile the norm function for better performance
        try:
            self._norm = torch.compile(self._norm_impl)
        except Exception:
            self._norm = self._norm_impl

    def _norm_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization (implementation)."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm with optional fusion.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# Default to standard RMSNorm unless fused is explicitly requested
__all__ = ["RMSNorm", "RMSNormFused"]
