# ara/dojo/encoder.py
"""
HDC-VAE Encoder for Thought Dojo
================================

Compresses HDC hypervectors (10,000 dim) to latent space (10D).

Architecture:
1. HDC input from cathedral sensors
2. VAE encoder -> mu, log_var
3. Reparameterization -> z (10D)
4. VAE decoder -> HDC reconstruction

Training uses ELBO loss:
    L = reconstruction_loss + beta * KL_divergence

Usage:
    # Training
    python -m ara.dojo.encoder --logs-path data/logs.pkl --output models/encoder

    # Loading
    encoder = load_encoder("models/encoder/hdc_vae_best.pt")
    z = encoder.encode(hdc_vector)
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HDCVAEConfig:
    """Configuration for HDC-VAE encoder."""
    hdc_dim: int = 10000           # HDC hypervector dimension
    latent_dim: int = 10           # Latent space dimension
    hidden_dims: List[int] = None  # Encoder/decoder hidden layers
    beta: float = 1.0              # KL divergence weight
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


# =============================================================================
# HDC-VAE Model
# =============================================================================

class HDCVAE(nn.Module):
    """
    Variational Autoencoder for HDC hypervectors.

    Compresses high-dimensional HDC representations to compact
    latent space suitable for world model dynamics.
    """

    def __init__(self, config: HDCVAEConfig):
        super().__init__()
        self.config = config

        # Encoder: HDC -> latent
        encoder_layers = []
        prev_dim = config.hdc_dim
        for h_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, config.latent_dim)

        # Decoder: latent -> HDC
        decoder_layers = []
        prev_dim = config.latent_dim
        for h_dim in reversed(config.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = h_dim

        self.decoder = nn.Sequential(*decoder_layers)
        self.fc_out = nn.Linear(prev_dim, config.hdc_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode HDC vector to latent space.

        Args:
            x: HDC vector (batch, hdc_dim)

        Returns:
            mu: Mean (batch, latent_dim)
            log_var: Log variance (batch, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to HDC space.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Reconstructed HDC (batch, hdc_dim)
        """
        h = self.decoder(z)
        return torch.tanh(self.fc_out(h))  # HDC is bipolar [-1, 1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            x_recon: Reconstructed HDC
            mu: Latent mean
            log_var: Latent log variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def get_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Get latent representation.

        Args:
            x: HDC vector
            deterministic: If True, return mu. If False, sample.

        Returns:
            z: Latent vector
        """
        mu, log_var = self.encode(x)
        if deterministic:
            return mu
        return self.reparameterize(mu, log_var)


# =============================================================================
# Loss Function
# =============================================================================

def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    VAE ELBO loss.

    L = reconstruction_loss + beta * KL_divergence
    """
    # Reconstruction loss (MSE for continuous HDC)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')

    # KL divergence
    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + beta * kl_div

    return total_loss, {
        "total": total_loss.item(),
        "recon": recon_loss.item(),
        "kl": kl_div.item(),
    }


# =============================================================================
# Dataset
# =============================================================================

class HDCDataset(Dataset):
    """Dataset of HDC hypervectors from logged cathedral states."""

    def __init__(self, logs: List[Dict], hdc_key: str = "H_state"):
        self.vectors = []

        for rec in logs:
            if hdc_key not in rec:
                continue

            h = np.asarray(rec[hdc_key], dtype="float32")

            # Normalize to [-1, 1] if binary {0, 1}
            if h.min() >= 0 and h.max() <= 1:
                h = 2 * h - 1

            self.vectors.append(h)

        if not self.vectors:
            raise ValueError(f"No HDC vectors found with key '{hdc_key}'")

        logger.info(f"HDCDataset: {len(self.vectors)} vectors, dim={len(self.vectors[0])}")

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.vectors[idx])


# =============================================================================
# Training
# =============================================================================

def train_encoder(
    logs_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[HDCVAEConfig] = None,
    device: str = "auto",
) -> HDCVAE:
    """
    Train HDC-VAE encoder on logged data.

    Args:
        logs_path: Path to pickled logs
        output_dir: Output directory for checkpoints
        config: Training configuration
        device: Device ("cpu", "cuda", or "auto")

    Returns:
        Trained encoder
    """
    config = config or HDCVAEConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load logs
    logger.info(f"Loading logs from {logs_path}")
    with open(logs_path, "rb") as f:
        logs = pickle.load(f)

    # Create dataset
    dataset = HDCDataset(logs)

    # Update config with actual HDC dimension
    if dataset.vectors:
        config.hdc_dim = len(dataset.vectors[0])

    # Split train/val
    n = len(dataset)
    n_val = max(int(0.1 * n), 1)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Model
    model = HDCVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')

    logger.info(f"Training HDC-VAE: {n_train} train, {n_val} val, {config.epochs} epochs")

    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        train_losses = []

        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            x_recon, mu, log_var = model(x)
            loss, metrics = vae_loss(x, x_recon, mu, log_var, config.beta)

            loss.backward()
            optimizer.step()

            train_losses.append(metrics)

        avg_train = {k: np.mean([m[k] for m in train_losses]) for k in train_losses[0]}

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_recon, mu, log_var = model(x)
                _, metrics = vae_loss(x, x_recon, mu, log_var, config.beta)
                val_losses.append(metrics)

        avg_val = {k: np.mean([m[k] for m in val_losses]) for k in val_losses[0]}

        scheduler.step(avg_val["total"])

        logger.info(
            f"[Epoch {epoch:03d}] train={avg_train['total']:.4f} "
            f"val={avg_val['total']:.4f} (recon={avg_val['recon']:.4f}, kl={avg_val['kl']:.4f})"
        )

        # Save best
        if avg_val["total"] < best_val_loss:
            best_val_loss = avg_val["total"]
            save_path = output_dir / "hdc_vae_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, save_path)
            logger.info(f"Saved best model to {save_path}")

    # Load best model
    best_ckpt = torch.load(output_dir / "hdc_vae_best.pt")
    model.load_state_dict(best_ckpt["model_state_dict"])

    return model


def load_encoder(ckpt_path: Union[str, Path], device: str = "cpu") -> HDCVAE:
    """Load trained encoder from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    model = HDCVAE(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"Loaded encoder from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train HDC-VAE encoder")
    parser.add_argument("--logs-path", type=str, required=True, help="Path to pickled logs")
    parser.add_argument("--output-dir", type=str, default="models/encoder", help="Output directory")
    parser.add_argument("--latent-dim", type=int, default=10, help="Latent dimension")
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = HDCVAEConfig(
        latent_dim=args.latent_dim,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    train_encoder(args.logs_path, args.output_dir, config, args.device)


if __name__ == "__main__":
    main()
