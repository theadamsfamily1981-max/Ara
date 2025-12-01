# tfan/snn/lowrank_synapse.py
"""
Low-rank masked synaptic weight matrices for 97-99% parameter reduction.

Implements W ≈ M ⊙ (U V^T) where:
- M is a sparse topological mask from TLS (Topological Landmark Selection)
- U, V are low-rank factors (rank r << N)

This achieves massive parameter reduction:
- Dense: N² parameters
- Low-rank masked: 2Nr parameters (typically r ≈ 0.01-0.02N)
- Reduction: 97-99% for typical configurations

Example:
    N=4096, r=32, density=1.56% (k=64 per row)
    Dense: 16,777,216 params
    Low-rank: 262,144 params
    Reduction: 98.4%
"""

import torch
from torch import nn
from typing import Optional, Dict


def _coo_bool_mask_from_csr(indptr, indices, N, device):
    """
    Build dense boolean mask from CSR representation.

    Args:
        indptr: CSR row pointers [N+1]
        indices: CSR column indices [nnz]
        N: Matrix dimension
        device: Tensor device

    Returns:
        Boolean mask [N, N] with True where edges exist
    """
    # Build row indices from indptr
    rows = torch.repeat_interleave(
        torch.arange(N, device=device),
        torch.diff(indptr)
    )
    cols = indices
    mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    mask[rows, cols] = True
    return mask


class LowRankMaskedSynapse(nn.Module):
    """
    Synaptic weight matrix as W ≈ M ⊙ (U V^T).

    Trainable parameters: U, V (rank-r factors)
    Non-trainable: Sparse mask M (from TLS)

    This achieves 97-99% parameter reduction vs dense while preserving
    topologically salient connectivity patterns.

    Args:
        N: Number of neurons
        r: Low-rank dimension (typically 0.01-0.02 × N)
        mask_csr: Sparse mask in CSR format {'indptr': Tensor, 'indices': Tensor}
        dtype: Parameter dtype
        device: Parameter device
        dense_fallback: Use dense masking (simpler, faster for low density)

    Example:
        >>> # Build TLS mask with avg degree 64
        >>> mask = build_tls_mask_from_scores(scores, k_per_row=64)
        >>> # Create low-rank synapse
        >>> syn = LowRankMaskedSynapse(N=4096, r=32, mask_csr=mask)
        >>> # Forward pass
        >>> x = torch.randn(2, 4096)  # [batch, N]
        >>> y = syn(x)                 # [batch, N]
        >>> # Verify reduction
        >>> print(f"Params: {syn.param_count(4096, 32):,}")  # 262,144
        >>> print(f"Reduction: {100 * (1 - 262144/16777216):.1f}%")  # 98.4%
    """

    def __init__(
        self,
        N: int,
        r: int = 32,
        mask_csr: Optional[Dict[str, torch.Tensor]] = None,
        dtype: torch.dtype = torch.float16,
        device = None,
        dense_fallback: bool = False
    ):
        super().__init__()
        self.N = N
        self.r = r

        # Low-rank factors U, V ∈ R^{N×r}
        self.U = nn.Parameter(
            torch.empty(N, r, dtype=dtype, device=device).uniform_(-0.02, 0.02)
        )
        self.V = nn.Parameter(
            torch.empty(N, r, dtype=dtype, device=device).uniform_(-0.02, 0.02)
        )

        # Sparse mask (non-trainable)
        self.mask_csr = mask_csr  # dict: {'indptr': Tensor[int64], 'indices': Tensor[int64]}
        self.dense_fallback = dense_fallback or (mask_csr is None)

        # Precompute dense boolean mask if using fallback
        self.register_buffer('bool_mask', None)
        if self.mask_csr is not None and self.dense_fallback:
            indptr = self.mask_csr['indptr']
            indices = self.mask_csr['indices']
            self.bool_mask = _coo_bool_mask_from_csr(
                indptr, indices, N, device or self.U.device
            )

    def forward(self, x):
        """
        Compute y = x @ W where W = M ⊙ (U V^T).

        Args:
            x: Presynaptic activity [batch, N]

        Returns:
            y: Postsynaptic drive [batch, N]
        """
        # Compute y = x @ (U V^T) = (x @ U) @ V^T
        pre = x @ self.U              # [batch, r]
        y = pre @ self.V.T            # [batch, N] (dense candidate)

        # Apply sparse mask
        if self.dense_fallback:
            if self.bool_mask is None:
                # No mask provided, return full low-rank approximation
                return y
            # Zero entries not allowed by mask
            return torch.where(self.bool_mask, y, torch.zeros_like(y))
        else:
            # Sparse masking using CSR indices
            # For production: replace with fused CUDA kernel for maximum throughput
            indptr = self.mask_csr['indptr']
            indices = self.mask_csr['indices']
            B, N = y.shape[0], self.N
            out = torch.zeros_like(y)

            # Apply mask row-by-row
            for i in range(N):
                j0, j1 = indptr[i].item(), indptr[i+1].item()
                if j1 > j0:  # Row has outgoing edges
                    cols = indices[j0:j1]
                    # Keep only allowed columns
                    out[:, cols] = y[:, cols]

            return out

    @staticmethod
    def param_count(N: int, r: int) -> int:
        """
        Count trainable parameters.

        Args:
            N: Number of neurons
            r: Rank

        Returns:
            Total parameters: 2Nr (for U and V)
        """
        return 2 * N * r

    @staticmethod
    def reduction_percent(N: int, r: int) -> float:
        """
        Compute parameter reduction vs dense.

        Args:
            N: Number of neurons
            r: Rank

        Returns:
            Reduction percentage
        """
        dense = N * N
        lowrank = 2 * N * r
        return 100.0 * (1.0 - lowrank / dense)

    def summary(self):
        """
        Print parameter summary.
        """
        dense = self.N * self.N
        lowrank = self.param_count(self.N, self.r)
        reduction = self.reduction_percent(self.N, self.r)

        avg_degree = 0
        if self.mask_csr is not None:
            indptr = self.mask_csr['indptr']
            avg_degree = float((indptr[1:] - indptr[:-1]).float().mean().item())

        return {
            'N': self.N,
            'rank': self.r,
            'dense_params': dense,
            'lowrank_params': lowrank,
            'reduction_pct': reduction,
            'avg_degree': avg_degree,
            'degree_frac': avg_degree / self.N if self.N > 0 else 0,
        }


__all__ = ['LowRankMaskedSynapse']
