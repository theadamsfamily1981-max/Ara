# tfan/snn/mask_tls.py
"""
TLS (Topological Landmark Selection) mask builder for sparse synaptic connectivity.

Builds sparse connectivity masks that preserve topologically salient connections
while achieving target sparsity levels (typically 98-99% zero weights).

The mask enforces:
1. Bounded out-degree (k edges per neuron)
2. Preservation of topological structure (Betti numbers, persistence landscapes)
3. Maximum information flow through landmark neurons
"""

import torch
from typing import Dict, Optional


def build_tls_mask_from_scores(
    scores: torch.Tensor,
    k_per_row: int,
    device = None
) -> Dict[str, torch.Tensor]:
    """
    Build sparse CSR mask by keeping top-k scoring connections per row.

    Args:
        scores: Connection saliency scores [N, N]
                Higher scores = more important connections
                Typically from TLS: α·persistence + (1-α)·diversity
        k_per_row: Number of outgoing connections to keep per neuron
        device: Target device (defaults to scores.device)

    Returns:
        CSR mask dict with:
            'indptr': Row pointers [N+1] (int64)
            'indices': Column indices [N*k] (int64)

    Example:
        >>> # TLS scoring (from tfan/models/tfan7b/mask_builder.py)
        >>> scores = compute_tls_scores(hidden_states, alpha=0.7)
        >>> # Build mask with avg degree 64
        >>> mask = build_tls_mask_from_scores(scores, k_per_row=64)
        >>> # Verify sparsity
        >>> density = 64 / scores.shape[0]
        >>> print(f"Density: {density:.2%}")  # 1.56% for N=4096
    """
    N = scores.shape[0]
    device = device or scores.device

    # Select top-k per row
    k = min(k_per_row, N)  # Don't exceed matrix dimension
    topk = torch.topk(scores, k=k, dim=-1, largest=True, sorted=False)

    # Extract column indices
    indices = topk.indices.cpu().to(torch.int64).flatten()

    # Build CSR row pointers
    # Each row has exactly k outgoing edges
    indptr = torch.zeros(N + 1, dtype=torch.int64, device='cpu')
    indptr[1:] = k
    indptr = torch.cumsum(indptr, dim=0)

    return {
        'indptr': indptr,
        'indices': indices,
    }


def build_uniform_random_mask(
    N: int,
    k_per_row: int,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Build uniform random sparse mask (baseline for comparison).

    Args:
        N: Matrix dimension
        k_per_row: Outgoing degree per row
        seed: Random seed for reproducibility

    Returns:
        CSR mask dict
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Random scores for each connection
    scores = torch.rand(N, N)

    return build_tls_mask_from_scores(scores, k_per_row=k_per_row)


def build_local_plus_random_mask(
    N: int,
    k_local: int,
    k_random: int
) -> Dict[str, torch.Tensor]:
    """
    Build mask with local + random long-range connections.

    Combines:
    - k_local nearest neighbors (spatial locality)
    - k_random distant connections (small-world shortcuts)

    Args:
        N: Matrix dimension
        k_local: Local connections per neuron
        k_random: Random long-range connections per neuron

    Returns:
        CSR mask dict

    Example:
        >>> # Small-world connectivity: 32 local + 16 random
        >>> mask = build_local_plus_random_mask(N=4096, k_local=32, k_random=16)
    """
    # Build scores favoring local connections
    scores = torch.zeros(N, N)

    for i in range(N):
        # Local connections (high score)
        local_range = range(max(0, i - k_local // 2),
                           min(N, i + k_local // 2 + 1))
        for j in local_range:
            if i != j:  # No self-loops
                scores[i, j] = 2.0 + torch.rand(1).item()

        # Random long-range (medium score)
        random_targets = torch.randint(0, N, (k_random,))
        for j in random_targets:
            if j != i:
                scores[i, j] = max(scores[i, j].item(), 1.0 + torch.rand(1).item())

    return build_tls_mask_from_scores(scores, k_per_row=k_local + k_random)


def degree_from_csr(indptr: torch.Tensor) -> torch.Tensor:
    """
    Compute out-degree for each row from CSR indptr.

    Args:
        indptr: CSR row pointers [N+1]

    Returns:
        Out-degree per row [N]

    Example:
        >>> mask = build_tls_mask_from_scores(scores, k_per_row=64)
        >>> degrees = degree_from_csr(mask['indptr'])
        >>> print(f"Avg degree: {degrees.float().mean():.1f}")  # 64.0
    """
    deg = (indptr[1:] - indptr[:-1]).to(torch.int32)
    return deg


def mask_density(indptr: torch.Tensor, N: int) -> float:
    """
    Compute mask density (fraction of non-zero entries).

    Args:
        indptr: CSR row pointers
        N: Matrix dimension

    Returns:
        Density in [0, 1]

    Example:
        >>> density = mask_density(mask['indptr'], N=4096)
        >>> print(f"Density: {density:.4f}")  # 0.0156 for k=64
        >>> print(f"Sparsity: {1-density:.4f}")  # 0.9844 (98.44% sparse)
    """
    nnz = indptr[-1].item()  # Total non-zero entries
    dense_size = N * N
    return nnz / dense_size


def verify_mask_properties(
    mask: Dict[str, torch.Tensor],
    N: int,
    expected_degree: int,
    tolerance: float = 0.01
) -> Dict[str, bool]:
    """
    Verify mask satisfies expected properties.

    Args:
        mask: CSR mask dict
        N: Matrix dimension
        expected_degree: Target out-degree
        tolerance: Allowed deviation fraction

    Returns:
        Dict of verification results

    Example:
        >>> mask = build_tls_mask_from_scores(scores, k_per_row=64)
        >>> results = verify_mask_properties(mask, N=4096, expected_degree=64)
        >>> assert all(results.values()), f"Mask verification failed: {results}"
    """
    indptr = mask['indptr']
    indices = mask['indices']

    # Check dimensions
    valid_dims = (len(indptr) == N + 1) and (len(indices) == indptr[-1].item())

    # Check degree consistency
    degrees = degree_from_csr(indptr)
    avg_degree = degrees.float().mean().item()
    degree_ok = abs(avg_degree - expected_degree) < expected_degree * tolerance

    # Check no self-loops
    has_self_loops = False
    for i in range(N):
        j0, j1 = indptr[i].item(), indptr[i + 1].item()
        cols = indices[j0:j1]
        if i in cols:
            has_self_loops = True
            break

    # Check indices in bounds
    indices_valid = (indices >= 0).all() and (indices < N).all()

    return {
        'valid_dimensions': valid_dims,
        'degree_consistent': degree_ok,
        'no_self_loops': not has_self_loops,
        'indices_valid': indices_valid.item(),
    }


__all__ = [
    'build_tls_mask_from_scores',
    'build_uniform_random_mask',
    'build_local_plus_random_mask',
    'degree_from_csr',
    'mask_density',
    'verify_mask_properties',
]
