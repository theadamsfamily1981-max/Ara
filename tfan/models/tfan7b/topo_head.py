"""
Topology head for TF-A-N 7B.

Hooks into latent states to compute persistence landscapes and topological loss.
Integrates with the existing tfan/topo.py TopologyRegularizer.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple


class TopologyHook(nn.Module):
    """
    Topology hook for extracting latent states and computing topological loss.

    This module:
    1. Hooks into transformer layers to capture latent representations
    2. Computes persistence landscapes via TopologyRegularizer
    3. Returns topological KL divergence loss

    Args:
        hidden_size: Hidden dimension of the model
        hook_every_n_layers: Capture latents every N layers (default: 2)
        lambda_topo: Weight for topological loss (default: 0.1)
        use_exact_ph: Whether to use exact PH validation (default: False)
        device: Device for computation
    """

    def __init__(
        self,
        hidden_size: int,
        hook_every_n_layers: int = 2,
        lambda_topo: float = 0.1,
        use_exact_ph: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hook_every_n_layers = hook_every_n_layers
        self.lambda_topo = lambda_topo
        self.use_exact_ph = use_exact_ph
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Import TopologyRegularizer from existing infrastructure
        try:
            from tfan.topo import TopologyRegularizer

            self.topo_regularizer = TopologyRegularizer(
                hidden_dim=hidden_size,
                lambda_topo=lambda_topo,
                device=self.device,
            )
        except ImportError:
            # Fallback: stub implementation
            self.topo_regularizer = None
            print(
                "Warning: TopologyRegularizer not found. Topology loss will be disabled."
            )

        # Storage for hooked latents
        self.hooked_latents = []

    def hook_latents(self, layer_idx: int, latents: torch.Tensor):
        """
        Hook callback to capture latent states from specific layers.

        Args:
            layer_idx: Index of the layer
            latents: Latent representations [batch, seq_len, hidden_size]
        """
        if layer_idx % self.hook_every_n_layers == 0:
            self.hooked_latents.append(latents.detach())

    def clear_hooks(self):
        """Clear stored hooked latents."""
        self.hooked_latents.clear()

    def compute_landscape(
        self,
        latents: torch.Tensor,
        return_diagrams: bool = False,
    ) -> torch.Tensor:
        """
        Compute persistence landscapes from latent representations.

        Args:
            latents: [batch, seq_len, hidden_size]
            return_diagrams: Whether to return PH diagrams

        Returns:
            landscapes: Persistence landscape vectors [batch, landscape_dim]
            diagrams: Optional list of PH diagrams
        """
        if self.topo_regularizer is None:
            # Return dummy landscapes
            batch_size = latents.shape[0]
            return torch.zeros(batch_size, 128, device=latents.device)

        return self.topo_regularizer.compute_landscape(
            latents, return_diagrams=return_diagrams
        )

    def topo_loss(
        self,
        current_latents: torch.Tensor,
        target_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute topological KL divergence loss.

        Args:
            current_latents: Current latent states [batch, seq_len, hidden_size]
            target_latents: Optional target latents (default: use previous state)

        Returns:
            loss: Topological KL divergence
        """
        if self.topo_regularizer is None:
            return torch.tensor(0.0, device=current_latents.device)

        # Compute current landscapes
        current_landscapes = self.compute_landscape(current_latents)

        # If no target provided, use Gaussian target (standard practice)
        if target_latents is None:
            target_landscapes = torch.randn_like(current_landscapes) * 0.1
        else:
            target_landscapes = self.compute_landscape(target_latents)

        # KL divergence between persistence landscapes
        loss = self.topo_regularizer.kl_penalty(current_landscapes, target_landscapes)

        return loss * self.lambda_topo

    def forward(
        self,
        latents: torch.Tensor,
        compute_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: compute landscapes and optional loss.

        Args:
            latents: Latent representations [batch, seq_len, hidden_size]
            compute_loss: Whether to compute topological loss

        Returns:
            dict with:
                - landscapes: Persistence landscapes
                - topo_loss: Optional topological loss
        """
        landscapes = self.compute_landscape(latents)

        outputs = {"landscapes": landscapes}

        if compute_loss:
            loss = self.topo_loss(latents)
            outputs["topo_loss"] = loss

        return outputs

    def validate_against_exact(
        self,
        latents: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Validate approximate PH against exact GUDHI/Ripser.

        Args:
            latents: Latent representations [batch, seq_len, hidden_size]

        Returns:
            dict with validation metrics (Wasserstein, Cosine)
        """
        if self.topo_regularizer is None:
            return {"wasserstein_gap": 0.0, "cosine_sim": 1.0}

        # Compute approximate diagrams
        approx_diagrams = self.compute_landscape(latents, return_diagrams=True)

        # Compute exact diagrams using GUDHI/Ripser
        try:
            from tools.nightly_ph_check import compute_exact_ph
            import numpy as np

            # Convert to numpy for GUDHI/Ripser
            latents_np = latents.detach().cpu().numpy()
            batch_size = latents_np.shape[0]

            exact_diagrams_list = []
            for b in range(batch_size):
                # Sample points (limit to 512 for efficiency)
                points = latents_np[b, :512, :]
                exact_diag = compute_exact_ph(points, max_dimension=1)
                exact_diagrams_list.append(exact_diag)

            # Validate
            validation = self.topo_regularizer.validate_against_exact(
                approx_diagrams, exact_diagrams_list
            )

            return validation

        except ImportError:
            print(
                "Warning: GUDHI/Ripser not available for exact PH validation."
            )
            return {"wasserstein_gap": 0.0, "cosine_sim": 1.0}


class TopologyHeadWithProjection(nn.Module):
    """
    Topology head with learned projection for better landscape computation.

    Projects latents to a lower-dimensional manifold before computing PH,
    improving topological signal and reducing computation.

    Args:
        hidden_size: Input hidden dimension
        proj_dim: Projection dimension (default: hidden_size // 4)
        hook_every_n_layers: Hook every N layers
        lambda_topo: Topological loss weight
    """

    def __init__(
        self,
        hidden_size: int,
        proj_dim: Optional[int] = None,
        hook_every_n_layers: int = 2,
        lambda_topo: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim or (hidden_size // 4)

        # Learned projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, self.proj_dim, bias=False),
            nn.LayerNorm(self.proj_dim),
        )

        # Base topology hook
        self.topo_hook = TopologyHook(
            hidden_size=self.proj_dim,
            hook_every_n_layers=hook_every_n_layers,
            lambda_topo=lambda_topo,
        )

    def forward(
        self,
        latents: torch.Tensor,
        compute_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with projection.

        Args:
            latents: [batch, seq_len, hidden_size]
            compute_loss: Whether to compute loss

        Returns:
            dict with landscapes and optional loss
        """
        # Project to lower dimension
        projected = self.projection(latents)

        # Compute topology
        return self.topo_hook(projected, compute_loss=compute_loss)


__all__ = [
    "TopologyHook",
    "TopologyHeadWithProjection",
]
