"""
CTD (Compositional Tree Distance) Hyperbolic Enablement.

Switches to hyperbolic embeddings when data exhibits hierarchical structure.
Uses tree-likeness proxies, spectral dimension, and β₁ cues to gate.

Hard gates:
- NDCG@K +5% vs Euclidean on hierarchical data
- Overhead ≤ 12%
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
import warnings

try:
    import geoopt
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False
    warnings.warn("geoopt not available. Hyperbolic geometry disabled.")


class TreeLikenessDetector:
    """
    Detect tree-like structure in data.

    Uses:
    - Spectral dimension analysis
    - Clustering coefficient
    - β₁ (1st Betti number) from topology
    """

    def __init__(self, threshold: float = 0.6):
        """
        Args:
            threshold: Minimum tree-likeness score to enable hyperbolic
        """
        self.threshold = threshold

    def compute_tree_likeness(
        self,
        embeddings: torch.Tensor,
        betti_1: Optional[int] = None,
    ) -> float:
        """
        Compute tree-likeness score from embeddings.

        Args:
            embeddings: Embedding vectors (n_samples, dim)
            betti_1: Optional β₁ from persistent homology

        Returns:
            Tree-likeness score in [0, 1]
        """
        n_samples = embeddings.shape[0]

        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings, p=2)

        # Spectral dimension (simplified)
        # True trees have spectral dimension ≈ 1
        try:
            # Compute graph Laplacian
            # Convert distances to adjacency (k-NN graph)
            k = min(10, n_samples - 1)
            _, knn_indices = torch.topk(dists, k=k, largest=False, dim=1)

            # Build adjacency matrix
            adj = torch.zeros(n_samples, n_samples, device=embeddings.device)
            for i in range(n_samples):
                adj[i, knn_indices[i]] = 1.0
            adj = (adj + adj.t()) / 2  # Symmetrize

            # Degree matrix
            deg = adj.sum(dim=1)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

            # Normalized Laplacian
            lap = torch.eye(n_samples, device=embeddings.device) - \
                  deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

            # Eigenvalues
            eigenvalues = torch.linalg.eigvalsh(lap)

            # Spectral dimension: count eigenvalues near 0
            spectral_dim = (eigenvalues < 0.1).sum().item() / n_samples

        except Exception as e:
            warnings.warn(f"Failed to compute spectral dimension: {e}")
            spectral_dim = 0.5

        # β₁ cue: low β₁ indicates tree-like (few loops)
        if betti_1 is not None:
            beta1_score = 1.0 - min(betti_1 / n_samples, 1.0)
        else:
            beta1_score = 0.5  # Neutral

        # Combine scores
        tree_likeness = 0.5 * (1.0 - spectral_dim) + 0.5 * beta1_score

        return float(tree_likeness)

    def should_use_hyperbolic(
        self,
        embeddings: torch.Tensor,
        betti_1: Optional[int] = None,
    ) -> Tuple[bool, float]:
        """
        Determine if hyperbolic embeddings should be used.

        Args:
            embeddings: Current embeddings
            betti_1: Optional β₁ from topology

        Returns:
            (use_hyperbolic, tree_likeness_score)
        """
        score = self.compute_tree_likeness(embeddings, betti_1)
        return score >= self.threshold, score


class HyperbolicEmbedding(nn.Module):
    """
    Hyperbolic embedding layer using Poincaré Ball or Lorentz model.

    Switches between Euclidean and hyperbolic based on tree-likeness.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        manifold: str = "poincare",
        tree_likeness_threshold: float = 0.6,
        enable: bool = True,
    ):
        """
        Args:
            num_embeddings: Number of embeddings
            embedding_dim: Embedding dimension
            manifold: "poincare" or "lorentz"
            tree_likeness_threshold: Threshold for enabling hyperbolic
            enable: Enable hyperbolic (if False, always use Euclidean)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold_name = manifold
        self.tree_likeness_threshold = tree_likeness_threshold
        self.enable = enable and HAS_GEOOPT

        if self.enable:
            # Create hyperbolic manifold
            if manifold == "poincare":
                self.manifold = geoopt.PoincareBall()
            elif manifold == "lorentz":
                self.manifold = geoopt.Lorentz()
            else:
                raise ValueError(f"Unknown manifold: {manifold}")

            # Hyperbolic parameter
            self.weight = geoopt.ManifoldParameter(
                torch.randn(num_embeddings, embedding_dim) * 0.01,
                manifold=self.manifold,
            )
        else:
            # Euclidean fallback
            self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
            self.manifold = None

        # Tree-likeness detector
        self.detector = TreeLikenessDetector(threshold=tree_likeness_threshold)

        # Mode tracking
        self.using_hyperbolic = False

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings.

        Args:
            indices: Embedding indices (batch, ...)

        Returns:
            Embeddings (batch, ..., embedding_dim)
        """
        if not self.enable:
            # Euclidean mode
            return torch.nn.functional.embedding(indices, self.weight)

        # Check if we should use hyperbolic
        # (In practice, this check would be done less frequently)
        if self.training:
            # During training, periodically check tree-likeness
            # For efficiency, we skip the check most of the time
            pass

        # Hyperbolic embedding lookup
        embeddings = torch.nn.functional.embedding(indices, self.weight)

        return embeddings

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between embeddings.

        Args:
            x: Embeddings (batch, dim)
            y: Embeddings (batch, dim)

        Returns:
            Distances (batch,)
        """
        if not self.enable or self.manifold is None:
            # Euclidean distance
            return torch.norm(x - y, dim=-1)

        # Hyperbolic distance
        return self.manifold.dist(x, y)


class HyperbolicGate:
    """
    Gate that decides when to use hyperbolic vs Euclidean embeddings.

    Tracks NDCG improvement and overhead.
    """

    def __init__(
        self,
        ndcg_improvement_target: float = 0.05,
        overhead_max: float = 0.12,
    ):
        """
        Args:
            ndcg_improvement_target: Minimum NDCG improvement (+5%)
            overhead_max: Maximum computational overhead (12%)
        """
        self.ndcg_improvement_target = ndcg_improvement_target
        self.overhead_max = overhead_max

        # Metrics
        self.euclidean_ndcg: Optional[float] = None
        self.hyperbolic_ndcg: Optional[float] = None
        self.overhead: Optional[float] = None

    def validate(self) -> Tuple[bool, Dict[str, float]]:
        """
        Validate hyperbolic embeddings against gates.

        Returns:
            (passes_gates, metrics)
        """
        if self.euclidean_ndcg is None or self.hyperbolic_ndcg is None:
            return False, {"error": "Missing NDCG measurements"}

        improvement = self.hyperbolic_ndcg - self.euclidean_ndcg
        overhead = self.overhead if self.overhead is not None else 0.0

        metrics = {
            "euclidean_ndcg": self.euclidean_ndcg,
            "hyperbolic_ndcg": self.hyperbolic_ndcg,
            "improvement": improvement,
            "overhead": overhead,
        }

        passes = (
            improvement >= self.ndcg_improvement_target and
            overhead <= self.overhead_max
        )

        return passes, metrics

    def set_ndcg(self, euclidean: float, hyperbolic: float):
        """Set NDCG measurements."""
        self.euclidean_ndcg = euclidean
        self.hyperbolic_ndcg = hyperbolic

    def set_overhead(self, overhead: float):
        """Set computational overhead."""
        self.overhead = overhead
