# ara/dojo/world_model.py
"""
Dojo World Model - Dynamics Learning for Imagination
=====================================================

Trains a world model f(z, u) -> z' on logged latent trajectories.

The world model is the core of Ara's predictive imagination:
- Given current latent state z and action u
- Predict next state z'
- Roll forward to simulate entire futures

Architecture options:
1. Linear: z' = Az + Bu (fast, interpretable)
2. MLP: z' = MLP([z; u]) (nonlinear)
3. Residual: z' = z + MLP([z; u]) (stable)
4. Ensemble: Multiple models for uncertainty

Training:
    python -m ara.dojo.world_model --logs-path data/logs.pkl --output models/world_model

Usage:
    model = load_world_model("models/world_model/best.pt")
    z_next = model.predict(z, u)
    trajectory = model.rollout(z_0, actions)
"""

from __future__ import annotations

import argparse
import logging
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
class DojoWorldModelConfig:
    """Configuration for Dojo world model."""
    latent_dim: int = 10           # Latent space dimension (from encoder)
    action_dim: int = 8            # Action space dimension
    hidden_dims: List[int] = None  # Hidden layer sizes
    model_type: str = "residual"   # "linear", "mlp", "residual"
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    weight_decay: float = 1e-4

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


# =============================================================================
# World Model
# =============================================================================

class DojoWorldModel(nn.Module):
    """
    World model for latent dynamics prediction.

    Learns z' = f(z, u) to enable mental simulation of futures.
    """

    def __init__(self, config: DojoWorldModelConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.action_dim = config.action_dim

        if config.model_type == "linear":
            self._build_linear()
        elif config.model_type == "mlp":
            self._build_mlp()
        elif config.model_type == "residual":
            self._build_residual()
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        # Track training stats
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _build_linear(self):
        """Linear dynamics: z' = Az + Bu"""
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(self.action_dim, self.latent_dim, bias=True)
        self.forward_fn = self._forward_linear

    def _forward_linear(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.A(z) + self.B(u)

    def _build_mlp(self):
        """MLP dynamics: z' = MLP([z; u])"""
        layers = []
        input_dim = self.latent_dim + self.action_dim

        for h_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
            ])
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, self.latent_dim))
        self.mlp = nn.Sequential(*layers)
        self.forward_fn = self._forward_mlp

    def _forward_mlp(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, u], dim=-1)
        return self.mlp(x)

    def _build_residual(self):
        """Residual dynamics: z' = z + MLP([z; u])"""
        layers = []
        input_dim = self.latent_dim + self.action_dim

        for h_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
            ])
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, self.latent_dim))
        self.residual = nn.Sequential(*layers)
        self.forward_fn = self._forward_residual

    def _forward_residual(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, u], dim=-1)
        return z + self.residual(x)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent state.

        Args:
            z: Current state (batch, latent_dim)
            u: Action (batch, action_dim)

        Returns:
            z_next: Predicted next state (batch, latent_dim)
        """
        return self.forward_fn(z, u)

    def predict(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """NumPy interface for prediction."""
        self.eval()
        with torch.no_grad():
            z_t = torch.from_numpy(z.astype(np.float32))
            u_t = torch.from_numpy(u.astype(np.float32))

            if z_t.dim() == 1:
                z_t = z_t.unsqueeze(0)
                u_t = u_t.unsqueeze(0)
                z_next = self(z_t, u_t).squeeze(0)
            else:
                z_next = self(z_t, u_t)

            return z_next.numpy()

    def rollout(
        self,
        z_init: np.ndarray,
        actions: np.ndarray,
        include_start: bool = True,
    ) -> np.ndarray:
        """
        Roll out trajectory given action sequence.

        Args:
            z_init: Initial state (latent_dim,)
            actions: Action sequence (H, action_dim)
            include_start: Include z_init in output

        Returns:
            trajectory: States (H+1, latent_dim) or (H, latent_dim)
        """
        self.eval()
        z = z_init.copy()
        trajectory = [z.copy()] if include_start else []

        for u in actions:
            z = self.predict(z, u)
            trajectory.append(z.copy())

        return np.array(trajectory)


# =============================================================================
# Ensemble World Model
# =============================================================================

class EnsembleWorldModel(nn.Module):
    """
    Ensemble of world models for uncertainty estimation.

    Disagreement between models indicates epistemic uncertainty -
    regions where imagination should be less confident.
    """

    def __init__(self, config: DojoWorldModelConfig, n_models: int = 5):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            DojoWorldModel(config) for _ in range(n_models)
        ])

    def forward(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty.

        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        predictions = torch.stack([m(z, u) for m in self.models])
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std

    def predict_with_uncertainty(
        self,
        z: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy interface with uncertainty."""
        self.eval()
        with torch.no_grad():
            z_t = torch.from_numpy(z.astype(np.float32))
            u_t = torch.from_numpy(u.astype(np.float32))

            if z_t.dim() == 1:
                z_t = z_t.unsqueeze(0)
                u_t = u_t.unsqueeze(0)

            mean, std = self(z_t, u_t)
            return mean.squeeze(0).numpy(), std.squeeze(0).numpy()


# =============================================================================
# Dataset
# =============================================================================

class TransitionDataset(Dataset):
    """Dataset of (z, u, z_next) transitions."""

    def __init__(
        self,
        logs: List[Dict],
        z_key: str = "z",
        u_key: str = "action",
        z_next_key: str = "z_next",
    ):
        self.transitions = []

        for rec in logs:
            if z_key not in rec or u_key not in rec:
                continue

            z = np.asarray(rec[z_key], dtype="float32")
            u = np.asarray(rec[u_key], dtype="float32")

            # z_next might be explicit or derived from next record
            if z_next_key in rec:
                z_next = np.asarray(rec[z_next_key], dtype="float32")
            else:
                continue  # Skip if no next state

            self.transitions.append((z, u, z_next))

        if not self.transitions:
            raise ValueError("No valid transitions found in logs")

        logger.info(f"TransitionDataset: {len(self.transitions)} transitions")

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, u, z_next = self.transitions[idx]
        return (
            torch.from_numpy(z),
            torch.from_numpy(u),
            torch.from_numpy(z_next),
        )


def prepare_transitions_from_trajectories(
    logs: List[Dict],
    z_key: str = "z",
    u_key: str = "action",
) -> List[Dict]:
    """
    Prepare transition dataset from sequential logs.

    Converts [z_0, z_1, z_2, ...] with [u_0, u_1, ...] into
    transitions [(z_0, u_0, z_1), (z_1, u_1, z_2), ...].
    """
    prepared = []

    for i in range(len(logs) - 1):
        if z_key not in logs[i] or u_key not in logs[i]:
            continue
        if z_key not in logs[i + 1]:
            continue

        prepared.append({
            "z": logs[i][z_key],
            "action": logs[i][u_key],
            "z_next": logs[i + 1][z_key],
        })

    return prepared


# =============================================================================
# Training
# =============================================================================

def train_world_model(
    logs_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[DojoWorldModelConfig] = None,
    device: str = "auto",
    from_trajectories: bool = True,
) -> DojoWorldModel:
    """
    Train world model on logged transitions.

    Args:
        logs_path: Path to pickled logs
        output_dir: Output directory for checkpoints
        config: Training configuration
        device: Device ("cpu", "cuda", "auto")
        from_trajectories: If True, prepare transitions from sequential logs

    Returns:
        Trained world model
    """
    config = config or DojoWorldModelConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load logs
    logger.info(f"Loading logs from {logs_path}")
    with open(logs_path, "rb") as f:
        logs = pickle.load(f)

    # Prepare transitions
    if from_trajectories:
        logs = prepare_transitions_from_trajectories(logs)

    dataset = TransitionDataset(logs)

    # Infer dimensions from data
    z, u, z_next = dataset[0]
    config.latent_dim = z.shape[0]
    config.action_dim = u.shape[0]

    # Split train/val
    n = len(dataset)
    n_val = max(int(0.1 * n), 1)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False, num_workers=2
    )

    # Model
    model = DojoWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_val_loss = float("inf")

    logger.info(
        f"Training DojoWorldModel ({config.model_type}): "
        f"{n_train} train, {n_val} val, {config.epochs} epochs"
    )
    logger.info(f"latent_dim={config.latent_dim}, action_dim={config.action_dim}")

    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0

        for z, u, z_next in train_loader:
            z, u, z_next = z.to(device), u.to(device), z_next.to(device)

            optimizer.zero_grad()
            z_pred = model(z, u)
            loss = F.mse_loss(z_pred, z_next)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * z.size(0)

        train_loss /= n_train
        model.train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for z, u, z_next in val_loader:
                z, u, z_next = z.to(device), u.to(device), z_next.to(device)
                z_pred = model(z, u)
                loss = F.mse_loss(z_pred, z_next)
                val_loss += loss.item() * z.size(0)

        val_loss /= n_val
        model.val_losses.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"[Epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f}"
            )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / "world_model_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                save_path,
            )
            logger.info(f"Saved best model to {save_path}")

    # Load best
    best_ckpt = torch.load(output_dir / "world_model_best.pt")
    model.load_state_dict(best_ckpt["model_state_dict"])

    return model


def load_world_model(
    ckpt_path: Union[str, Path],
    device: str = "cpu",
) -> DojoWorldModel:
    """Load trained world model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    model = DojoWorldModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"Loaded world model from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Dojo world model")
    parser.add_argument("--logs-path", type=str, required=True, help="Path to logs")
    parser.add_argument("--output-dir", type=str, default="models/world_model")
    parser.add_argument("--model-type", type=str, default="residual",
                        choices=["linear", "mlp", "residual"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = DojoWorldModelConfig(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    train_world_model(args.logs_path, args.output_dir, config, args.device)


if __name__ == "__main__":
    main()
