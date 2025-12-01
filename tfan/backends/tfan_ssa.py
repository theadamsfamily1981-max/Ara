# tfan/backends/tfan_ssa.py
"""
TF-A-N backend with Selective Sparse Attention (SSA).
"""

import torch
from torch import nn
from typing import Dict, Any

from .base import Backend, BackendHooks


class TFANHooks(BackendHooks):
    """
    Hooks for TF-A-N with FDT, topology, PGU integration.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.fdt_controller = None
        self.use_fdt = cfg.get('tfan', {}).get('use_fdt', True)

        if self.use_fdt:
            # Initialize FDT controller
            try:
                from training.fdt_controller import FDTControllerWithEmotion
                fdt_cfg = cfg.get('tfan', {}).get('fdt', {})
                self.fdt_controller = FDTControllerWithEmotion(
                    kp=fdt_cfg.get('kp', 0.30),
                    ki=fdt_cfg.get('ki', 0.02),
                    kd=fdt_cfg.get('kd', 0.10),
                    ema_alpha=fdt_cfg.get('ema_alpha', 0.85),
                    target_epr_cv=fdt_cfg.get('target_epr_cv', 0.15),
                )
            except ImportError:
                print("⚠ FDT controller not available")
                self.fdt_controller = None

    def before_step(self, model: nn.Module):
        """Gradient clipping."""
        grad_clip = self.cfg.get('training', {}).get('grad_clip', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    def after_step(self, model: nn.Module, aux: Dict[str, Any]):
        """FDT PID updates."""
        super().after_step(model, aux)

        if self.fdt_controller is not None and 'loss' in aux:
            # Compute gradient variance
            grad_variance = 0.0
            count = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_variance += p.grad.var().item()
                    count += 1
            if count > 0:
                grad_variance /= count

            # FDT step
            base_lr = self.cfg.get('training', {}).get('learning_rate', 1e-4)
            base_temp = 1.0

            fdt_outputs = self.fdt_controller.step(
                loss=aux['loss'],
                grad_variance=grad_variance,
                base_lr=base_lr,
                base_temp=base_temp
            )

            # Store FDT metrics in aux for logging
            aux['epr_cv'] = self.fdt_controller.epr_cv
            aux['lr_multiplier'] = fdt_outputs.get('lr_multiplier', 1.0)
            aux['temp_multiplier'] = fdt_outputs.get('temp_multiplier', 1.0)


class TFANBackend(Backend):
    """
    TF-A-N backend with SSA, FDT, topology regularization.

    Implements Selective Sparse Attention with O(N log N) complexity
    and TF-A-N control systems (FDT, PGU, topology).
    """

    def build_model(self) -> nn.Module:
        """
        Build TF-A-N 7B model with SSA.

        Returns:
            model: TFANForCausalLM with SSA
        """
        model_cfg = self.cfg.get('model', {})

        # Load TF-A-N 7B model
        try:
            from tfan.models.tfan7b import TFANForCausalLM, TFANConfig

            # Load config
            config_path = model_cfg.get('config_path', 'tfan/models/tfan7b/config.json')
            config = TFANConfig.from_json(config_path)

            # Create model
            model = TFANForCausalLM(config)

            return model

        except ImportError:
            print("⚠ TF-A-N 7B model not available, using placeholder")
            # Fallback to placeholder
            hidden_size = model_cfg.get('hidden_size', 4096)

            class PlaceholderTFAN(nn.Module):
                def __init__(self, hidden_size):
                    super().__init__()
                    self.fc = nn.Linear(hidden_size, hidden_size)

                def forward(self, x):
                    return self.fc(x), {}

            return PlaceholderTFAN(hidden_size)

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
        """Build TF-A-N hooks with FDT."""
        return TFANHooks(self.cfg)


__all__ = ['TFANBackend']
