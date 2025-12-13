"""
ara.vae.core - Core VAE Architecture for HGF Trajectories

Implements a Variational Autoencoder specialized for encoding HGF belief
trajectories into disentangled latent representations.

The encoder maps time-series of HGF states (μ₂, σ₂, μ₃, δ₁, δ₂, ...) to
a low-dimensional latent space. The decoder reconstructs trajectories,
enabling generative modeling of computational phenotypes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = Any


@dataclass
class VAEConfig:
    """Configuration for TrajectoryVAE."""

    # Input dimensions
    n_trials: int = 200  # Trajectory length
    n_features: int = 8  # Features per trial: [μ₂, σ₂, μ₃, σ₃, δ₁, δ₂, π₁, π̂₂]

    # Latent space
    latent_dim: int = 8  # Dimensionality of z

    # Encoder architecture
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    encoder_type: str = "lstm"  # "lstm", "transformer", or "mlp"

    # Decoder architecture
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128])
    decoder_type: str = "lstm"

    # Training
    beta: float = 1.0  # β-VAE weight on KL term
    learning_rate: float = 1e-3

    # Regularization
    dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.beta < 0:
            raise ValueError("beta must be >= 0")


@dataclass
class VAELoss:
    """Container for VAE loss components."""
    total: float
    reconstruction: float
    kl_divergence: float
    beta: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "reconstruction": self.reconstruction,
            "kl_divergence": self.kl_divergence,
            "beta": self.beta,
        }


if HAS_TORCH:

    class TrajectoryEncoder(nn.Module):
        """
        Encodes HGF trajectories to latent distribution parameters.

        Maps [batch, n_trials, n_features] → (μ, log_σ²) for latent z.
        """

        def __init__(self, config: VAEConfig):
            super().__init__()
            self.config = config

            if config.encoder_type == "lstm":
                self.encoder = self._build_lstm_encoder()
            elif config.encoder_type == "transformer":
                self.encoder = self._build_transformer_encoder()
            else:
                self.encoder = self._build_mlp_encoder()

            # Final projection to latent parameters
            final_dim = config.encoder_hidden_dims[-1]
            self.fc_mu = nn.Linear(final_dim, config.latent_dim)
            self.fc_logvar = nn.Linear(final_dim, config.latent_dim)

        def _build_lstm_encoder(self) -> nn.Module:
            """Build LSTM-based encoder for sequential data."""
            return nn.LSTM(
                input_size=self.config.n_features,
                hidden_size=self.config.encoder_hidden_dims[-1],
                num_layers=len(self.config.encoder_hidden_dims),
                batch_first=True,
                dropout=self.config.dropout if len(self.config.encoder_hidden_dims) > 1 else 0,
            )

        def _build_transformer_encoder(self) -> nn.Module:
            """Build Transformer encoder."""
            d_model = self.config.encoder_hidden_dims[-1]

            # Project input to model dimension
            self.input_proj = nn.Linear(self.config.n_features, d_model)

            # Positional encoding
            self.pos_encoding = self._create_positional_encoding(
                self.config.n_trials, d_model
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 4,
                dropout=self.config.dropout,
                batch_first=True,
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=2)

        def _build_mlp_encoder(self) -> nn.Module:
            """Build MLP encoder (flattens trajectory)."""
            input_dim = self.config.n_trials * self.config.n_features
            layers = []

            dims = [input_dim] + self.config.encoder_hidden_dims
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
                if self.config.dropout > 0:
                    layers.append(nn.Dropout(self.config.dropout))

            return nn.Sequential(*layers)

        def _create_positional_encoding(
            self, max_len: int, d_model: int
        ) -> Tensor:
            """Create sinusoidal positional encoding."""
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)

        def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
            """
            Encode trajectory to latent distribution.

            Args:
                x: [batch, n_trials, n_features]

            Returns:
                mu: [batch, latent_dim]
                logvar: [batch, latent_dim]
            """
            if self.config.encoder_type == "lstm":
                # LSTM encoding
                _, (h_n, _) = self.encoder(x)
                h = h_n[-1]  # Last layer hidden state

            elif self.config.encoder_type == "transformer":
                # Transformer encoding
                x = self.input_proj(x)
                x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
                x = self.encoder(x)
                h = x.mean(dim=1)  # Pool across time

            else:
                # MLP encoding
                x = x.view(x.size(0), -1)
                h = self.encoder(x)

            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)

            return mu, logvar


    class TrajectoryDecoder(nn.Module):
        """
        Decodes latent z back to HGF trajectories.

        Maps [batch, latent_dim] → [batch, n_trials, n_features].
        """

        def __init__(self, config: VAEConfig):
            super().__init__()
            self.config = config

            # Project latent to decoder input
            first_dim = config.decoder_hidden_dims[0]
            self.fc_in = nn.Linear(config.latent_dim, first_dim)

            if config.decoder_type == "lstm":
                self.decoder = self._build_lstm_decoder()
            else:
                self.decoder = self._build_mlp_decoder()

            # Final projection to output features
            last_dim = config.decoder_hidden_dims[-1]
            self.fc_out = nn.Linear(last_dim, config.n_features)

        def _build_lstm_decoder(self) -> nn.Module:
            """Build LSTM decoder."""
            return nn.LSTM(
                input_size=self.config.decoder_hidden_dims[0],
                hidden_size=self.config.decoder_hidden_dims[-1],
                num_layers=len(self.config.decoder_hidden_dims),
                batch_first=True,
                dropout=self.config.dropout if len(self.config.decoder_hidden_dims) > 1 else 0,
            )

        def _build_mlp_decoder(self) -> nn.Module:
            """Build MLP decoder."""
            layers = []
            dims = self.config.decoder_hidden_dims

            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
                if self.config.dropout > 0:
                    layers.append(nn.Dropout(self.config.dropout))

            return nn.Sequential(*layers)

        def forward(self, z: Tensor) -> Tensor:
            """
            Decode latent to trajectory.

            Args:
                z: [batch, latent_dim]

            Returns:
                x_recon: [batch, n_trials, n_features]
            """
            h = self.fc_in(z)
            h = F.relu(h)

            if self.config.decoder_type == "lstm":
                # Repeat latent across time steps
                h = h.unsqueeze(1).repeat(1, self.config.n_trials, 1)
                h, _ = self.decoder(h)
            else:
                # MLP: expand to full trajectory
                h = h.unsqueeze(1).repeat(1, self.config.n_trials, 1)
                h = self.decoder(h)

            x_recon = self.fc_out(h)
            return x_recon


    class TrajectoryVAE(nn.Module):
        """
        Variational Autoencoder for HGF Trajectories.

        Learns disentangled latent representations of computational phenotypes
        by encoding HGF belief dynamics into a low-dimensional space.

        The latent dimensions should ideally correspond to:
        - z₁ ≈ ω₂ (tonic volatility)
        - z₂ ≈ κ₁ (coupling strength)
        - z₃ ≈ θ (response temperature)
        - z₄+ = additional phenotype-specific features

        Usage:
            config = VAEConfig(latent_dim=8, n_trials=200)
            vae = TrajectoryVAE(config)

            # Forward pass
            x_recon, mu, logvar = vae(trajectories)
            loss = vae.compute_loss(trajectories, x_recon, mu, logvar)

            # Encode only
            z = vae.encode(trajectories)

            # Generate new trajectories
            x_new = vae.generate(n_samples=10)
        """

        def __init__(self, config: Optional[VAEConfig] = None):
            super().__init__()
            self.config = config or VAEConfig()

            self.encoder = TrajectoryEncoder(self.config)
            self.decoder = TrajectoryDecoder(self.config)

        def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
            """
            Reparameterization trick: z = μ + σ * ε

            Enables backpropagation through the sampling operation.
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(
            self, x: Tensor
        ) -> Tuple[Tensor, Tensor, Tensor]:
            """
            Full forward pass.

            Args:
                x: [batch, n_trials, n_features]

            Returns:
                x_recon: Reconstructed trajectory
                mu: Latent mean
                logvar: Latent log-variance
            """
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z)
            return x_recon, mu, logvar

        def encode(self, x: Tensor, return_distribution: bool = False):
            """
            Encode trajectories to latent space.

            Args:
                x: Input trajectories
                return_distribution: If True, return (mu, logvar)

            Returns:
                z or (mu, logvar)
            """
            mu, logvar = self.encoder(x)
            if return_distribution:
                return mu, logvar
            return mu  # Use mean as point estimate

        def decode(self, z: Tensor) -> Tensor:
            """Decode latent codes to trajectories."""
            return self.decoder(z)

        def generate(self, n_samples: int = 1) -> Tensor:
            """
            Generate new trajectories by sampling from prior.

            Args:
                n_samples: Number of samples to generate

            Returns:
                Generated trajectories
            """
            device = next(self.parameters()).device
            z = torch.randn(n_samples, self.config.latent_dim, device=device)
            return self.decode(z)

        def compute_loss(
            self,
            x: Tensor,
            x_recon: Tensor,
            mu: Tensor,
            logvar: Tensor,
        ) -> VAELoss:
            """
            Compute VAE loss: reconstruction + β * KL divergence.

            ELBO = E[log p(x|z)] - β * KL(q(z|x) || p(z))

            Args:
                x: Original trajectories
                x_recon: Reconstructed trajectories
                mu: Latent means
                logvar: Latent log-variances

            Returns:
                VAELoss with total, reconstruction, and KL terms
            """
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(x_recon, x, reduction="mean")

            # KL divergence: KL(N(μ,σ²) || N(0,1))
            # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )

            # Total loss with β weighting
            total_loss = recon_loss + self.config.beta * kl_loss

            return VAELoss(
                total=total_loss.item(),
                reconstruction=recon_loss.item(),
                kl_divergence=kl_loss.item(),
                beta=self.config.beta,
            )

        def interpolate(
            self, x1: Tensor, x2: Tensor, n_steps: int = 10
        ) -> Tensor:
            """
            Interpolate between two trajectories in latent space.

            Useful for visualizing the latent manifold structure.
            """
            z1 = self.encode(x1.unsqueeze(0))
            z2 = self.encode(x2.unsqueeze(0))

            alphas = torch.linspace(0, 1, n_steps, device=z1.device)
            interpolations = []

            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)

            return torch.cat(interpolations, dim=0)

        def traverse_latent(
            self,
            x: Tensor,
            dim: int,
            range_vals: Tuple[float, float] = (-3.0, 3.0),
            n_steps: int = 10,
        ) -> Tensor:
            """
            Traverse a single latent dimension while holding others fixed.

            Useful for understanding what each latent dim encodes.
            """
            z = self.encode(x.unsqueeze(0))
            z = z.repeat(n_steps, 1)

            values = torch.linspace(
                range_vals[0], range_vals[1], n_steps, device=z.device
            )
            z[:, dim] = values

            return self.decode(z)


else:
    # Fallback classes when PyTorch not available
    class TrajectoryEncoder:
        def __init__(self, config):
            raise ImportError("PyTorch required for TrajectoryEncoder")

    class TrajectoryDecoder:
        def __init__(self, config):
            raise ImportError("PyTorch required for TrajectoryDecoder")

    class TrajectoryVAE:
        def __init__(self, config=None):
            raise ImportError("PyTorch required for TrajectoryVAE")


# =============================================================================
# NumPy-based Simple VAE (for environments without PyTorch)
# =============================================================================

class SimpleVAE:
    """
    Minimal VAE implementation using only NumPy.

    For demonstration and environments where PyTorch isn't available.
    Uses basic MLP architecture with gradient descent.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization."""
        scale = lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out))

        # Encoder
        self.W_enc1 = np.random.randn(self.input_dim, self.hidden_dim) * scale(self.input_dim, self.hidden_dim)
        self.b_enc1 = np.zeros(self.hidden_dim)
        self.W_mu = np.random.randn(self.hidden_dim, self.latent_dim) * scale(self.hidden_dim, self.latent_dim)
        self.b_mu = np.zeros(self.latent_dim)
        self.W_logvar = np.random.randn(self.hidden_dim, self.latent_dim) * scale(self.hidden_dim, self.latent_dim)
        self.b_logvar = np.zeros(self.latent_dim)

        # Decoder
        self.W_dec1 = np.random.randn(self.latent_dim, self.hidden_dim) * scale(self.latent_dim, self.hidden_dim)
        self.b_dec1 = np.zeros(self.hidden_dim)
        self.W_dec2 = np.random.randn(self.hidden_dim, self.input_dim) * scale(self.hidden_dim, self.input_dim)
        self.b_dec2 = np.zeros(self.input_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to latent distribution parameters."""
        h = self._relu(x @ self.W_enc1 + self.b_enc1)
        mu = h @ self.W_mu + self.b_mu
        logvar = h @ self.W_logvar + self.b_logvar
        return mu, logvar

    def reparameterize(
        self, mu: np.ndarray, logvar: np.ndarray
    ) -> np.ndarray:
        """Sample z = μ + σ * ε."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent to reconstruction."""
        h = self._relu(z @ self.W_dec1 + self.b_dec1)
        x_recon = self._sigmoid(h @ self.W_dec2 + self.b_dec2)
        return x_recon

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def compute_loss(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        mu: np.ndarray,
        logvar: np.ndarray,
        beta: float = 1.0,
    ) -> Dict[str, float]:
        """Compute ELBO loss."""
        # Reconstruction (binary cross-entropy)
        eps = 1e-8
        recon = -np.mean(
            x * np.log(x_recon + eps) + (1 - x) * np.log(1 - x_recon + eps)
        )

        # KL divergence
        kl = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))

        return {
            "total": recon + beta * kl,
            "reconstruction": recon,
            "kl_divergence": kl,
        }
