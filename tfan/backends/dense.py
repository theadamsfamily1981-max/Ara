# tfan/backends/dense.py
"""
Dense baseline backend (standard transformers).
"""

import torch
from torch import nn
from typing import Dict, Any

from .base import Backend, BackendHooks


class DenseHooks(BackendHooks):
    """Hooks for dense baseline."""

    def before_step(self, model: nn.Module):
        """Gradient clipping."""
        grad_clip = self.cfg.get('training', {}).get('grad_clip', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)


class DenseBackend(Backend):
    """
    Dense baseline backend.

    Standard transformer architecture without any sparsity or
    specialized components.
    """

    def build_model(self) -> nn.Module:
        """
        Build dense transformer model.

        Returns:
            model: Standard dense transformer
        """
        # For now, return a placeholder
        # In production, this would load from tfan/models/dense/
        model_cfg = self.cfg.get('model', {})

        # Placeholder: Simple linear layer for demonstration
        # Replace with actual transformer model
        hidden_size = model_cfg.get('hidden_size', 4096)

        class DenseModel(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.fc = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                return self.fc(x), {}

        return DenseModel(hidden_size)

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build AdamW optimizer."""
        train_cfg = self.cfg.get('training', {})

        lr = train_cfg.get('learning_rate', 1e-4)
        weight_decay = train_cfg.get('weight_decay', 0.01)
        betas = train_cfg.get('adam_betas', (0.9, 0.95))

        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )

    def build_hooks(self) -> BackendHooks:
        """Build hooks."""
        return DenseHooks(self.cfg)


__all__ = ['DenseBackend']
