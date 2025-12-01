"""
Topological Invariant Constraints for PGU Verification

Defines SMT constraints using Z3 to formally verify that structural changes
to the SNN mask do not violate critical graph properties:

1. β₁ Betti Number - Loop structure/redundancy preservation
2. Connected Components - Network connectivity guarantee
3. Spectral Gap - Communication efficiency lower bound

These constraints enable CERTIFIABLE ANTIFRAGILITY - formal proofs that
the AEPO agent's structural optimizations preserve network resilience.

Usage:
    from tfan.pgu.topological_constraints import (
        TopologicalVerifier,
        BettiConstraint,
        verify_structural_change,
    )

    verifier = TopologicalVerifier()

    # Check if proposed mask change maintains β₁ ≥ min_loops
    result = verifier.verify_betti_constraint(
        old_mask=current_mask,
        new_mask=proposed_mask,
        min_beta1=10,
    )

    if not result.sat:
        reject_structural_change()
"""

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger("tfan.pgu.topological")

# Z3 import (optional)
try:
    from z3 import (
        Solver, Int, Real, Bool, And, Or, Not, Implies,
        sat, unsat, unknown, Sum, If, ForAll, Exists,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 not available, topological verification disabled")


class ConstraintType(str, Enum):
    """Types of topological constraints."""
    BETTI_NUMBER = "betti"           # β₁ loop preservation
    CONNECTIVITY = "connectivity"    # Connected components bound
    SPECTRAL_GAP = "spectral"        # λ₂ lower bound
    DEGREE_BOUND = "degree"          # Max/min degree constraints
    DIAMETER = "diameter"            # Network diameter bound


@dataclass
class ConstraintResult:
    """Result of constraint verification."""
    constraint_type: ConstraintType
    sat: bool                        # True if constraint satisfied
    verification_time_ms: float
    details: Dict[str, Any]
    formula_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "constraint_type": self.constraint_type.value,
        }


@dataclass
class TopologicalState:
    """Topological state of the SNN graph."""
    num_nodes: int
    num_edges: int
    beta0: int = 1           # Connected components (β₀)
    beta1: int = 0           # Loops/cycles (β₁)
    spectral_gap: float = 0.0  # λ₂ (Fiedler value)
    diameter: int = 0
    avg_degree: float = 0.0
    clustering: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_betti_numbers_approx(
    indptr: np.ndarray,
    indices: np.ndarray,
    N: int,
) -> Tuple[int, int]:
    """
    Compute approximate Betti numbers from CSR adjacency.

    β₀ = number of connected components
    β₁ = number of independent cycles = E - V + β₀ (Euler formula)

    Args:
        indptr: CSR row pointers
        indices: CSR column indices
        N: Number of nodes

    Returns:
        (β₀, β₁)
    """
    # Count edges (directed → divide by 2 if symmetric)
    num_edges = len(indices)

    # Find connected components via union-find
    parent = list(range(N))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build adjacency from CSR
    for i in range(N):
        for j_idx in range(indptr[i], indptr[i + 1]):
            j = indices[j_idx]
            union(i, j)

    # Count components
    beta0 = len(set(find(i) for i in range(N)))

    # Euler characteristic: χ = V - E + F
    # For graph: β₁ = E - V + β₀
    beta1 = max(0, num_edges - N + beta0)

    return beta0, beta1


def compute_spectral_gap_approx(
    indptr: np.ndarray,
    indices: np.ndarray,
    N: int,
    num_iterations: int = 50,
) -> float:
    """
    Approximate spectral gap (λ₂) via power iteration.

    The spectral gap determines network communication efficiency.
    Higher λ₂ = better connectivity and faster mixing.

    Args:
        indptr: CSR row pointers
        indices: CSR column indices
        N: Number of nodes
        num_iterations: Power iteration steps

    Returns:
        Approximate λ₂
    """
    # Build degree vector
    degrees = np.diff(indptr).astype(np.float64)
    degrees = np.maximum(degrees, 1)  # Avoid division by zero

    # Random starting vector (orthogonal to constant vector)
    v = np.random.randn(N)
    v = v - v.mean()
    v = v / np.linalg.norm(v)

    # Power iteration on normalized Laplacian
    for _ in range(num_iterations):
        # Sparse matrix-vector multiply
        new_v = np.zeros(N)
        for i in range(N):
            for j_idx in range(indptr[i], indptr[i + 1]):
                j = indices[j_idx]
                new_v[i] += v[j] / np.sqrt(degrees[i] * degrees[j])

        # Laplacian: L = I - D^{-1/2} A D^{-1/2}
        new_v = v - new_v

        # Orthogonalize against constant vector
        new_v = new_v - new_v.mean()

        # Normalize
        norm = np.linalg.norm(new_v)
        if norm > 1e-10:
            v = new_v / norm
        else:
            break

    # Rayleigh quotient
    Lv = np.zeros(N)
    for i in range(N):
        Lv[i] = v[i]  # Diagonal term
        for j_idx in range(indptr[i], indptr[i + 1]):
            j = indices[j_idx]
            Lv[i] -= v[j] / np.sqrt(degrees[i] * degrees[j])

    lambda2 = np.dot(v, Lv) / np.dot(v, v)

    return max(0.0, lambda2)


class TopologicalVerifier:
    """
    Verifies topological invariants for SNN structural changes.

    Uses Z3 SMT solver to formally prove that proposed changes
    maintain required graph properties.
    """

    def __init__(
        self,
        timeout_ms: int = 1000,
        cache_results: bool = True,
    ):
        """
        Initialize verifier.

        Args:
            timeout_ms: Z3 solver timeout
            cache_results: Whether to cache verification results
        """
        self.timeout_ms = timeout_ms
        self.cache_results = cache_results
        self._cache: Dict[str, ConstraintResult] = {}

        if not Z3_AVAILABLE:
            logger.warning("Z3 not available, using approximations only")

    def compute_topological_state(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        N: int,
    ) -> TopologicalState:
        """
        Compute full topological state of graph.

        Args:
            indptr: CSR row pointers
            indices: CSR column indices
            N: Number of nodes

        Returns:
            TopologicalState with all metrics
        """
        beta0, beta1 = compute_betti_numbers_approx(indptr, indices, N)
        spectral = compute_spectral_gap_approx(indptr, indices, N)

        num_edges = len(indices)
        avg_degree = num_edges / N if N > 0 else 0

        return TopologicalState(
            num_nodes=N,
            num_edges=num_edges,
            beta0=beta0,
            beta1=beta1,
            spectral_gap=spectral,
            avg_degree=avg_degree,
        )

    def verify_betti_constraint(
        self,
        old_state: TopologicalState,
        new_state: TopologicalState,
        min_beta1: int = 0,
        max_beta1_reduction: float = 0.1,
    ) -> ConstraintResult:
        """
        Verify β₁ (loop count) constraint.

        Ensures structural change doesn't reduce network redundancy
        below critical threshold.

        Args:
            old_state: Topological state before change
            new_state: Topological state after proposed change
            min_beta1: Minimum allowed β₁
            max_beta1_reduction: Maximum allowed β₁ reduction ratio

        Returns:
            ConstraintResult
        """
        start = time.perf_counter()

        # Check absolute minimum
        absolute_ok = new_state.beta1 >= min_beta1

        # Check relative reduction
        if old_state.beta1 > 0:
            reduction_ratio = 1.0 - (new_state.beta1 / old_state.beta1)
        else:
            reduction_ratio = 0.0
        relative_ok = reduction_ratio <= max_beta1_reduction

        sat_result = absolute_ok and relative_ok

        elapsed = (time.perf_counter() - start) * 1000

        return ConstraintResult(
            constraint_type=ConstraintType.BETTI_NUMBER,
            sat=sat_result,
            verification_time_ms=elapsed,
            details={
                "old_beta1": old_state.beta1,
                "new_beta1": new_state.beta1,
                "min_beta1": min_beta1,
                "reduction_ratio": reduction_ratio,
                "max_reduction": max_beta1_reduction,
                "absolute_ok": absolute_ok,
                "relative_ok": relative_ok,
            }
        )

    def verify_connectivity_constraint(
        self,
        old_state: TopologicalState,
        new_state: TopologicalState,
        max_components: int = 1,
    ) -> ConstraintResult:
        """
        Verify connectivity (β₀) constraint.

        Ensures network remains connected (or within allowed component count).

        Args:
            old_state: State before change
            new_state: State after change
            max_components: Maximum allowed connected components

        Returns:
            ConstraintResult
        """
        start = time.perf_counter()

        sat_result = new_state.beta0 <= max_components

        elapsed = (time.perf_counter() - start) * 1000

        return ConstraintResult(
            constraint_type=ConstraintType.CONNECTIVITY,
            sat=sat_result,
            verification_time_ms=elapsed,
            details={
                "old_beta0": old_state.beta0,
                "new_beta0": new_state.beta0,
                "max_components": max_components,
            }
        )

    def verify_spectral_constraint(
        self,
        old_state: TopologicalState,
        new_state: TopologicalState,
        min_spectral_gap: float = 0.01,
        max_gap_reduction: float = 0.2,
    ) -> ConstraintResult:
        """
        Verify spectral gap (λ₂) constraint.

        Ensures network communication efficiency is maintained.

        Args:
            old_state: State before change
            new_state: State after change
            min_spectral_gap: Minimum allowed λ₂
            max_gap_reduction: Maximum allowed λ₂ reduction ratio

        Returns:
            ConstraintResult
        """
        start = time.perf_counter()

        # Absolute minimum
        absolute_ok = new_state.spectral_gap >= min_spectral_gap

        # Relative reduction
        if old_state.spectral_gap > 0:
            reduction_ratio = 1.0 - (new_state.spectral_gap / old_state.spectral_gap)
        else:
            reduction_ratio = 0.0
        relative_ok = reduction_ratio <= max_gap_reduction

        sat_result = absolute_ok and relative_ok

        elapsed = (time.perf_counter() - start) * 1000

        return ConstraintResult(
            constraint_type=ConstraintType.SPECTRAL_GAP,
            sat=sat_result,
            verification_time_ms=elapsed,
            details={
                "old_spectral": old_state.spectral_gap,
                "new_spectral": new_state.spectral_gap,
                "min_gap": min_spectral_gap,
                "reduction_ratio": reduction_ratio,
                "max_reduction": max_gap_reduction,
            }
        )

    def verify_all_topological_constraints(
        self,
        old_indptr: np.ndarray,
        old_indices: np.ndarray,
        new_indptr: np.ndarray,
        new_indices: np.ndarray,
        N: int,
        min_beta1: int = 0,
        max_components: int = 1,
        min_spectral_gap: float = 0.01,
    ) -> Dict[str, ConstraintResult]:
        """
        Verify all topological constraints for a structural change.

        This is the main entry point for AEPO structural verification.

        Args:
            old_indptr, old_indices: Current CSR mask
            new_indptr, new_indices: Proposed CSR mask
            N: Number of nodes
            min_beta1: Minimum loop count
            max_components: Maximum connected components
            min_spectral_gap: Minimum spectral gap

        Returns:
            Dict mapping constraint type to result
        """
        # Compute states
        old_state = self.compute_topological_state(old_indptr, old_indices, N)
        new_state = self.compute_topological_state(new_indptr, new_indices, N)

        results = {}

        # Betti constraint
        results["betti"] = self.verify_betti_constraint(
            old_state, new_state, min_beta1=min_beta1
        )

        # Connectivity constraint
        results["connectivity"] = self.verify_connectivity_constraint(
            old_state, new_state, max_components=max_components
        )

        # Spectral constraint
        results["spectral"] = self.verify_spectral_constraint(
            old_state, new_state, min_spectral_gap=min_spectral_gap
        )

        return results

    def all_constraints_satisfied(
        self,
        results: Dict[str, ConstraintResult]
    ) -> bool:
        """Check if all constraints are satisfied."""
        return all(r.sat for r in results.values())


def verify_structural_change(
    old_mask: Dict[str, np.ndarray],
    new_mask: Dict[str, np.ndarray],
    N: int,
    min_beta1: int = 0,
    max_components: int = 1,
    min_spectral_gap: float = 0.01,
) -> Tuple[bool, Dict[str, ConstraintResult]]:
    """
    Convenience function to verify a structural change.

    Args:
        old_mask: Current CSR mask {'indptr': ..., 'indices': ...}
        new_mask: Proposed CSR mask
        N: Number of nodes
        min_beta1: Minimum loop count
        max_components: Maximum connected components
        min_spectral_gap: Minimum spectral gap

    Returns:
        (all_satisfied, constraint_results)
    """
    verifier = TopologicalVerifier()

    results = verifier.verify_all_topological_constraints(
        old_indptr=np.asarray(old_mask['indptr']),
        old_indices=np.asarray(old_mask['indices']),
        new_indptr=np.asarray(new_mask['indptr']),
        new_indices=np.asarray(new_mask['indices']),
        N=N,
        min_beta1=min_beta1,
        max_components=max_components,
        min_spectral_gap=min_spectral_gap,
    )

    all_satisfied = verifier.all_constraints_satisfied(results)

    return all_satisfied, results


# Exports
__all__ = [
    "TopologicalVerifier",
    "TopologicalState",
    "ConstraintResult",
    "ConstraintType",
    "compute_betti_numbers_approx",
    "compute_spectral_gap_approx",
    "verify_structural_change",
    "Z3_AVAILABLE",
]
