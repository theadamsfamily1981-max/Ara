# tfan/backends/base.py
"""
Base backend class defining the interface for all model backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
from torch import nn


class BackendHooks:
    """
    Lifecycle hooks for training integration.

    Handles:
    - Pre-step operations (grad clipping, spectral norm)
    - Post-step operations (FDT updates, metric logging)
    - Logging and monitoring
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Args:
            cfg: Configuration dict
        """
        self.cfg = cfg
        self.step_count = 0

    def before_step(self, model: nn.Module):
        """
        Called before optimizer.step().

        Use for:
        - Gradient clipping
        - Spectral normalization
        - Gradient sanity checks
        """
        pass

    def after_step(self, model: nn.Module, aux: Dict[str, Any]):
        """
        Called after optimizer.step().

        Use for:
        - FDT PID updates
        - Metric collection
        - EPR-CV computation
        - Spike rate monitoring (SNN)
        """
        self.step_count += 1

    def log(self, step: int, aux: Dict[str, Any]):
        """
        Log metrics to console/tensorboard/wandb.

        Args:
            step: Training step
            aux: Auxiliary outputs from model
        """
        pass


class Backend(ABC):
    """
    Abstract base class for model backends.

    All backends must provide:
    - model: nn.Module
    - optim: torch.optim.Optimizer
    - hooks: BackendHooks for training lifecycle
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Args:
            cfg: Configuration dict
        """
        self.cfg = cfg
        self._model = None
        self._optim = None
        self._hooks = None

    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Build and return the model.

        Returns:
            model: nn.Module instance
        """
        pass

    @abstractmethod
    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Build and return the optimizer.

        Args:
            model: Model to optimize

        Returns:
            optimizer: torch.optim.Optimizer instance
        """
        pass

    @abstractmethod
    def build_hooks(self) -> BackendHooks:
        """
        Build and return lifecycle hooks.

        Returns:
            hooks: BackendHooks instance
        """
        pass

    @property
    def model(self) -> nn.Module:
        """Get model (lazy initialization)."""
        if self._model is None:
            self._model = self.build_model()
        return self._model

    @property
    def optim(self) -> torch.optim.Optimizer:
        """Get optimizer (lazy initialization)."""
        if self._optim is None:
            self._optim = self.build_optimizer(self.model)
        return self._optim

    @property
    def hooks(self) -> BackendHooks:
        """Get hooks (lazy initialization)."""
        if self._hooks is None:
            self._hooks = self.build_hooks()
        return self._hooks

    def to_device(self, device: str):
        """Move model to device."""
        self.model.to(device)

    def summary(self) -> Dict[str, Any]:
        """
        Return backend summary statistics.

        Returns:
            Dict with param counts, memory, etc.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'backend': self.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'device': next(self.model.parameters()).device.type,
        }


__all__ = ['Backend', 'BackendHooks']
