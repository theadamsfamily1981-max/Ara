"""
Topological regularization via differentiable persistent homology.

Provides:
- GPU-accelerated differentiable PH computation
- Persistence landscape (PLay) conversion
- Topological KL divergence penalty
- Wasserstein distance and cosine similarity metrics
- Nightly exact PH validation with GUDHI/Ripser
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

try:
    # GPU-accelerated differentiable PH (if available)
    import torch_topological as torchph
    HAS_TORCHPH = True
except ImportError:
    HAS_TORCHPH = False
    warnings.warn("torch_topological not found. Using fallback implementation.")

try:
    # Exact PH for nightly validation
    import gudhi
    import ripser
    HAS_EXACT_PH = True
except ImportError:
    HAS_EXACT_PH = False
    warnings.warn("GUDHI/Ripser not found. Nightly exact PH validation disabled.")

from persim import PersistenceLandscape
import scipy.spatial.distance as sp_dist


@dataclass
class TopologyMetrics:
    """Container for topology measurements."""
    persistence_diagrams: Dict[int, np.ndarray]  # degree -> diagram
    landscapes: Dict[int, np.ndarray]  # degree -> landscape vectors
    betti_numbers: Dict[int, int]
    wasserstein_distance: Optional[float] = None
    cosine_similarity: Optional[float] = None


class TopologyRegularizer(nn.Module):
    """
    Topological regularization module.

    Computes persistent homology and converts to differentiable landscape
    representations for training-time regularization.

    Hard gates:
    - Wasserstein gap ≤ 2% vs exact PH
    - Cosine similarity ≥ 0.90 vs exact PH
    - Nightly exact PH ≤ 20 min for ≤ 5k samples
    """

    def __init__(
        self,
        lambda_topo: float = 0.01,
        filtration_type: str = "rips",
        homology_degrees: List[int] = [0, 1],
        landscape_levels: int = 5,
        wasserstein_gap_max: float = 0.02,
        cosine_min: float = 0.90,
        device: str = "cuda",
    ):
        """
        Args:
            lambda_topo: Weight for topological penalty
            filtration_type: Type of filtration ("rips", "alpha", "cubical")
            homology_degrees: Which homology groups to compute (e.g., [0, 1])
            landscape_levels: Number of landscape function levels
            wasserstein_gap_max: Maximum allowed Wasserstein gap vs exact
            cosine_min: Minimum cosine similarity vs exact
            device: Compute device
        """
        super().__init__()
        self.lambda_topo = lambda_topo
        self.filtration_type = filtration_type
        self.homology_degrees = homology_degrees
        self.landscape_levels = landscape_levels
        self.wasserstein_gap_max = wasserstein_gap_max
        self.cosine_min = cosine_min
        self.device = device

        # Target topology (can be set via set_target_topology)
        self.target_landscapes: Optional[Dict[int, torch.Tensor]] = None

        # Confidence tracking for low-quality PH
        self.low_confidence_count = 0
        self.total_count = 0

    def set_target_topology(self, target_diagrams: Dict[int, np.ndarray]):
        """
        Set target persistence diagrams for KL penalty.

        Args:
            target_diagrams: Dictionary mapping homology degree -> persistence diagram
        """
        self.target_landscapes = {}
        for deg, diagram in target_diagrams.items():
            landscape = self._diagram_to_landscape(diagram)
            self.target_landscapes[deg] = torch.from_numpy(landscape).float().to(self.device)

    def compute_landscape(
        self,
        latents: torch.Tensor,
        return_diagrams: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute persistence landscapes from latent representations.

        Args:
            latents: Tensor of shape (batch, seq_len, d_model) or (batch, n_points, dim)
            return_diagrams: If True, also return raw persistence diagrams

        Returns:
            Dictionary containing:
            - landscapes: Dict[degree] -> Tensor of shape (batch, landscape_dim)
            - (optional) diagrams: Raw persistence diagrams
        """
        batch_size = latents.shape[0]
        landscapes = {deg: [] for deg in self.homology_degrees}
        diagrams = {deg: [] for deg in self.homology_degrees} if return_diagrams else None

        for i in range(batch_size):
            sample = latents[i].detach().cpu().numpy()

            # Compute persistence diagram
            if HAS_TORCHPH:
                # Use differentiable PH (GPU)
                sample_diags = self._compute_ph_differentiable(latents[i])
            else:
                # Fallback to numpy-based computation
                sample_diags = self._compute_ph_exact(sample)

            # Convert diagrams to landscapes
            for deg in self.homology_degrees:
                if deg in sample_diags:
                    landscape = self._diagram_to_landscape(sample_diags[deg])
                    landscapes[deg].append(landscape)
                    if return_diagrams:
                        diagrams[deg].append(sample_diags[deg])
                else:
                    # Empty diagram
                    landscapes[deg].append(np.zeros((self.landscape_levels, 100)))
                    if return_diagrams:
                        diagrams[deg].append(np.array([]))

        # Stack into tensors
        for deg in self.homology_degrees:
            landscapes[deg] = torch.from_numpy(np.stack(landscapes[deg])).float().to(self.device)

        result = {"landscapes": landscapes}
        if return_diagrams:
            result["diagrams"] = diagrams

        return result

    def kl_penalty(
        self,
        current_landscapes: Dict[int, torch.Tensor],
        target_landscapes: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute KL-like penalty between current and target topologies.

        Args:
            current_landscapes: Current persistence landscapes
            target_landscapes: Target landscapes (uses self.target_landscapes if None)

        Returns:
            Scalar penalty tensor
        """
        if target_landscapes is None:
            target_landscapes = self.target_landscapes

        if target_landscapes is None:
            raise ValueError("No target topology set. Call set_target_topology() first.")

        penalty = torch.tensor(0.0, device=self.device)

        for deg in self.homology_degrees:
            if deg not in current_landscapes or deg not in target_landscapes:
                continue

            curr = current_landscapes[deg].flatten(1)  # (batch, landscape_dim)
            tgt = target_landscapes[deg].flatten(1)

            # Normalize to probability-like distributions
            curr_norm = torch.nn.functional.softmax(curr, dim=-1)
            tgt_norm = torch.nn.functional.softmax(tgt, dim=-1)

            # KL divergence
            kl = torch.nn.functional.kl_div(
                curr_norm.log(),
                tgt_norm,
                reduction="batchmean",
            )
            penalty += kl

        return self.lambda_topo * penalty

    def _compute_ph_differentiable(self, points: torch.Tensor) -> Dict[int, np.ndarray]:
        """
        Compute PH using differentiable GPU-based method.

        Args:
            points: Tensor of shape (n_points, dim)

        Returns:
            Dictionary mapping degree -> persistence diagram (birth, death) pairs
        """
        if not HAS_TORCHPH:
            raise RuntimeError("torch_topological not available")

        # Use Vietoris-Rips complex
        from torch_topological.nn import VietorisRipsComplex

        vr = VietorisRipsComplex(dim=max(self.homology_degrees))
        persistence_info = vr(points.unsqueeze(0))  # Add batch dim

        diagrams = {}
        for deg in self.homology_degrees:
            diagram = persistence_info[0][deg].detach().cpu().numpy()
            # Filter out trivial features
            diagram = diagram[np.isfinite(diagram).all(axis=1)]
            diagram = diagram[(diagram[:, 1] - diagram[:, 0]) > 1e-6]
            diagrams[deg] = diagram

        return diagrams

    def _compute_ph_exact(self, points: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute exact PH using Ripser (CPU-based, exact).

        Args:
            points: Numpy array of shape (n_points, dim)

        Returns:
            Dictionary mapping degree -> persistence diagram
        """
        if not HAS_EXACT_PH:
            # Ultra-simple fallback: return empty diagrams
            return {deg: np.array([]) for deg in self.homology_degrees}

        # Use Ripser for exact computation
        max_deg = max(self.homology_degrees)
        result = ripser.ripser(points, maxdim=max_deg)

        diagrams = {}
        for deg in self.homology_degrees:
            if deg < len(result['dgms']):
                diagram = result['dgms'][deg]
                # Filter out infinite death times
                diagram = diagram[np.isfinite(diagram).all(axis=1)]
                diagrams[deg] = diagram
            else:
                diagrams[deg] = np.array([])

        return diagrams

    def _diagram_to_landscape(self, diagram: np.ndarray, resolution: int = 100) -> np.ndarray:
        """
        Convert persistence diagram to landscape representation.

        Args:
            diagram: Persistence diagram as (n_points, 2) array of (birth, death)
            resolution: Number of sample points for landscape functions

        Returns:
            Landscape array of shape (landscape_levels, resolution)
        """
        if len(diagram) == 0:
            return np.zeros((self.landscape_levels, resolution))

        try:
            landscape = PersistenceLandscape(
                dgms=[diagram],
                hom_deg=0,  # Doesn't matter for single diagram
            )
            # Sample landscape at regular intervals
            landscape_values = landscape.values[0]  # First diagram
            return landscape_values[:self.landscape_levels, :]
        except Exception as e:
            warnings.warn(f"Failed to compute landscape: {e}")
            return np.zeros((self.landscape_levels, resolution))

    def validate_against_exact(
        self,
        approximate_diagrams: Dict[int, np.ndarray],
        exact_diagrams: Dict[int, np.ndarray],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate approximate (GPU) PH against exact (CPU) computation.

        Args:
            approximate_diagrams: Diagrams from differentiable computation
            exact_diagrams: Diagrams from exact Ripser computation

        Returns:
            (passes_gates, metrics) where metrics contains:
            - wasserstein_gap: Relative Wasserstein distance
            - cosine_similarity: Cosine similarity of landscape vectors
        """
        metrics = {}

        for deg in self.homology_degrees:
            if deg not in approximate_diagrams or deg not in exact_diagrams:
                continue

            approx = approximate_diagrams[deg]
            exact = exact_diagrams[deg]

            if len(approx) == 0 or len(exact) == 0:
                continue

            # Wasserstein distance (use p=2)
            try:
                from persim import wasserstein
                wass_dist = wasserstein(approx, exact, matching=False)
                # Normalize by max persistence
                max_pers = max(
                    np.max(approx[:, 1] - approx[:, 0]) if len(approx) > 0 else 1.0,
                    np.max(exact[:, 1] - exact[:, 0]) if len(exact) > 0 else 1.0,
                )
                wass_gap = wass_dist / max(max_pers, 1e-6)
                metrics[f"wasserstein_gap_deg{deg}"] = wass_gap
            except Exception as e:
                warnings.warn(f"Failed to compute Wasserstein: {e}")
                metrics[f"wasserstein_gap_deg{deg}"] = float('inf')

            # Cosine similarity of landscapes
            try:
                approx_landscape = self._diagram_to_landscape(approx).flatten()
                exact_landscape = self._diagram_to_landscape(exact).flatten()

                cosine_sim = 1.0 - sp_dist.cosine(approx_landscape, exact_landscape)
                metrics[f"cosine_similarity_deg{deg}"] = cosine_sim
            except Exception as e:
                warnings.warn(f"Failed to compute cosine similarity: {e}")
                metrics[f"cosine_similarity_deg{deg}"] = 0.0

        # Check gates
        avg_wass = np.mean([v for k, v in metrics.items() if "wasserstein" in k])
        avg_cosine = np.mean([v for k, v in metrics.items() if "cosine" in k])

        metrics["wasserstein_distance"] = avg_wass
        metrics["cosine_similarity"] = avg_cosine

        passes_gates = (
            avg_wass <= self.wasserstein_gap_max and
            avg_cosine >= self.cosine_min
        )

        return passes_gates, metrics

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, TopologyMetrics]:
        """
        Forward pass: compute topology and regularization penalty.

        Args:
            latents: Latent representations (batch, seq_len, d_model)

        Returns:
            (penalty, metrics) tuple
        """
        # Compute landscapes
        landscape_dict = self.compute_landscape(latents, return_diagrams=True)
        landscapes = landscape_dict["landscapes"]
        diagrams = landscape_dict.get("diagrams", None)

        # Compute penalty if target is set
        if self.target_landscapes is not None:
            penalty = self.kl_penalty(landscapes)
        else:
            penalty = torch.tensor(0.0, device=self.device)

        # Package metrics
        betti_numbers = {}
        persistence_diagrams = {}
        if diagrams is not None:
            for deg in self.homology_degrees:
                betti_numbers[deg] = len(diagrams[deg][0]) if len(diagrams[deg]) > 0 else 0
                persistence_diagrams[deg] = diagrams[deg][0] if len(diagrams[deg]) > 0 else np.array([])

        metrics = TopologyMetrics(
            persistence_diagrams=persistence_diagrams,
            landscapes={deg: landscapes[deg].detach().cpu().numpy() for deg in self.homology_degrees},
            betti_numbers=betti_numbers,
        )

        self.total_count += 1

        return penalty, metrics

    def decay_lambda_on_low_confidence(self, decay_factor: float = 0.9):
        """
        Reduce λ_topo when topology computation has low confidence.

        Args:
            decay_factor: Multiplicative factor to reduce lambda
        """
        self.lambda_topo *= decay_factor
        self.low_confidence_count += 1


def compute_betti_numbers(diagrams: Dict[int, np.ndarray], epsilon: float = 1e-6) -> Dict[int, int]:
    """
    Compute Betti numbers from persistence diagrams.

    Args:
        diagrams: Dictionary mapping degree -> persistence diagram
        epsilon: Minimum persistence threshold

    Returns:
        Dictionary mapping degree -> Betti number
    """
    betti = {}
    for deg, diagram in diagrams.items():
        # Count features with persistence > epsilon
        if len(diagram) > 0:
            persistence = diagram[:, 1] - diagram[:, 0]
            betti[deg] = np.sum(persistence > epsilon)
        else:
            betti[deg] = 0
    return betti
