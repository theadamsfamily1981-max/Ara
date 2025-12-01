# tfan/snn/auditors.py
"""
Parameter auditing and gate validation for SNN emulation.

Enforces hard gates:
- Parameter reduction ≥ 97% vs dense baseline
- Average degree ≤ 2% of N (sparsity ≥ 98%)
- Rank r ≤ 0.02N (low-rank constraint)
"""

import torch
from typing import Dict, Any


def dense_params(N: int) -> int:
    """
    Count parameters in dense N×N matrix.

    Args:
        N: Matrix dimension

    Returns:
        N²
    """
    return N * N


def lowrank_params(N: int, r: int) -> int:
    """
    Count parameters in low-rank factorization U V^T.

    Args:
        N: Matrix dimension
        r: Rank

    Returns:
        2Nr (for U ∈ R^{N×r} and V ∈ R^{N×r})
    """
    return 2 * N * r


def param_reduction_pct(N: int, r: int) -> float:
    """
    Compute parameter reduction percentage.

    Args:
        N: Matrix dimension
        r: Rank

    Returns:
        Reduction percentage vs dense

    Example:
        >>> param_reduction_pct(N=4096, r=32)
        98.4375  # 98.4% reduction
    """
    base = dense_params(N)
    lowr = lowrank_params(N, r)
    red = 1.0 - (lowr / base)
    return 100.0 * red


def assert_param_gate(N: int, r: int, pct_required: float = 97.0):
    """
    Assert parameter reduction meets requirement.

    Args:
        N: Matrix dimension
        r: Rank
        pct_required: Minimum reduction percentage

    Raises:
        AssertionError if reduction < pct_required

    Example:
        >>> assert_param_gate(N=4096, r=32, pct_required=97.0)  # PASS
        >>> assert_param_gate(N=4096, r=100, pct_required=97.0)  # FAIL
    """
    pct = param_reduction_pct(N, r)
    assert pct >= pct_required, (
        f"Parameter reduction {pct:.2f}% < {pct_required}% "
        f"(N={N}, r={r}, params={lowrank_params(N, r):,} vs dense={dense_params(N):,})"
    )


def assert_degree_gate(
    indptr: torch.Tensor,
    N: int,
    max_frac: float = 0.02
):
    """
    Assert average degree meets sparsity requirement.

    Args:
        indptr: CSR row pointers
        N: Matrix dimension
        max_frac: Maximum allowed degree as fraction of N

    Raises:
        AssertionError if avg_degree > max_frac * N

    Example:
        >>> # k=64, N=4096 -> 1.56% density -> PASS
        >>> indptr = torch.arange(0, 4096*64+1, 64, dtype=torch.int64)
        >>> assert_degree_gate(indptr, N=4096, max_frac=0.02)
    """
    avg_d = float((indptr[1:] - indptr[:-1]).float().mean().item())
    max_allowed = max_frac * N

    assert avg_d <= max_allowed, (
        f"Average degree {avg_d:.1f} exceeds {max_allowed:.1f} "
        f"(max_frac={max_frac:.2%}, N={N})"
    )


def assert_rank_gate(N: int, r: int, max_frac: float = 0.02):
    """
    Assert rank meets low-rank constraint.

    Args:
        N: Matrix dimension
        r: Rank
        max_frac: Maximum rank as fraction of N

    Raises:
        AssertionError if r > max_frac * N

    Example:
        >>> assert_rank_gate(N=4096, r=32, max_frac=0.02)  # PASS (32 < 81.92)
        >>> assert_rank_gate(N=4096, r=100, max_frac=0.02)  # FAIL (100 > 81.92)
    """
    max_allowed = max_frac * N
    assert r <= max_allowed, (
        f"Rank {r} exceeds {max_allowed:.1f} (max_frac={max_frac:.2%}, N={N})"
    )


def report(N: int, r: int, indptr: torch.Tensor) -> Dict[str, Any]:
    """
    Generate comprehensive parameter audit report.

    Args:
        N: Matrix dimension
        r: Rank
        indptr: CSR row pointers

    Returns:
        Dict with:
            - N, rank
            - dense_params, lowrank_params
            - param_reduction_pct
            - avg_degree, degree_frac
            - nnz (non-zero entries)
            - density, sparsity

    Example:
        >>> mask = build_tls_mask_from_scores(scores, k_per_row=64)
        >>> stats = report(N=4096, r=32, indptr=mask['indptr'])
        >>> print(f"Reduction: {stats['param_reduction_pct']:.2f}%")
        >>> print(f"Sparsity: {stats['sparsity']:.2%}")
    """
    # Parameter counts
    dense = dense_params(N)
    lowr = lowrank_params(N, r)
    reduction = param_reduction_pct(N, r)

    # Degree statistics
    degrees = indptr[1:] - indptr[:-1]
    avg_d = float(degrees.float().mean().item())
    min_d = int(degrees.min().item())
    max_d = int(degrees.max().item())

    # Sparsity statistics
    nnz = int(indptr[-1].item())
    dense_size = N * N
    density = nnz / dense_size
    sparsity = 1.0 - density

    return {
        'N': N,
        'rank': r,
        'dense_params': dense,
        'lowrank_params': lowr,
        'param_reduction_pct': reduction,
        'avg_degree': avg_d,
        'min_degree': min_d,
        'max_degree': max_d,
        'degree_frac': avg_d / N,
        'nnz': nnz,
        'density': density,
        'sparsity': sparsity,
    }


def verify_all_gates(
    N: int,
    r: int,
    indptr: torch.Tensor,
    param_reduction_min: float = 97.0,
    degree_frac_max: float = 0.02,
    rank_frac_max: float = 0.02
) -> Dict[str, bool]:
    """
    Verify all acceptance gates.

    Args:
        N: Matrix dimension
        r: Rank
        indptr: CSR row pointers
        param_reduction_min: Minimum param reduction %
        degree_frac_max: Maximum avg degree as fraction of N
        rank_frac_max: Maximum rank as fraction of N

    Returns:
        Dict with pass/fail for each gate

    Example:
        >>> gates = verify_all_gates(N=4096, r=32, indptr=mask['indptr'])
        >>> if not all(gates.values()):
        ...     print(f"GATE FAILURE: {gates}")
        ...     raise RuntimeError("Acceptance gates failed")
    """
    results = {}

    # Param reduction gate
    try:
        assert_param_gate(N, r, pct_required=param_reduction_min)
        results['param_reduction'] = True
    except AssertionError:
        results['param_reduction'] = False

    # Degree gate
    try:
        assert_degree_gate(indptr, N, max_frac=degree_frac_max)
        results['degree'] = True
    except AssertionError:
        results['degree'] = False

    # Rank gate
    try:
        assert_rank_gate(N, r, max_frac=rank_frac_max)
        results['rank'] = True
    except AssertionError:
        results['rank'] = False

    return results


def print_report(stats: Dict[str, Any]):
    """
    Pretty-print audit report.

    Args:
        stats: Report dict from report()
    """
    print("=" * 60)
    print("SNN PARAMETER AUDIT REPORT")
    print("=" * 60)
    print(f"Matrix dimension:        N = {stats['N']:,}")
    print(f"Low-rank dimension:      r = {stats['rank']}")
    print()
    print("PARAMETERS:")
    print(f"  Dense baseline:        {stats['dense_params']:,}")
    print(f"  Low-rank (U, V):       {stats['lowrank_params']:,}")
    print(f"  Reduction:             {stats['param_reduction_pct']:.2f}%")
    print()
    print("SPARSITY:")
    print(f"  Non-zero entries:      {stats['nnz']:,}")
    print(f"  Density:               {stats['density']:.4f} ({stats['density']*100:.2f}%)")
    print(f"  Sparsity:              {stats['sparsity']:.4f} ({stats['sparsity']*100:.2f}%)")
    print()
    print("DEGREE:")
    print(f"  Average:               {stats['avg_degree']:.1f}")
    print(f"  Min:                   {stats['min_degree']}")
    print(f"  Max:                   {stats['max_degree']}")
    print(f"  As fraction of N:      {stats['degree_frac']:.4f} ({stats['degree_frac']*100:.2f}%)")
    print("=" * 60)


__all__ = [
    'dense_params',
    'lowrank_params',
    'param_reduction_pct',
    'assert_param_gate',
    'assert_degree_gate',
    'assert_rank_gate',
    'report',
    'verify_all_gates',
    'print_report',
]
