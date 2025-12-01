# tfan/backends/__init__.py
"""
Backend factory for different model architectures.

Supports:
- dense: Standard dense transformers
- tfan: TF-A-N with SSA (Selective Sparse Attention)
- snn_emu: SNN emulation with low-rank masked synapses
"""

from .base import Backend
from .dense import DenseBackend
from .tfan_ssa import TFANBackend
from .snn_emu import SNNBackend


def build_backend(cfg):
    """
    Build backend from configuration.

    Args:
        cfg: Configuration object/dict with 'backend' field

    Returns:
        Backend instance with model, optimizer, and hooks

    Example:
        >>> cfg = {'backend': 'snn_emu', ...}
        >>> backend = build_backend(cfg)
        >>> model, optim, hooks = backend.model, backend.optim, backend.hooks
    """
    backend_type = cfg.get('backend', 'dense')

    if backend_type == 'snn_emu':
        return SNNBackend(cfg)
    elif backend_type == 'tfan':
        return TFANBackend(cfg)
    elif backend_type == 'dense':
        return DenseBackend(cfg)
    else:
        raise ValueError(f"Unknown backend: {backend_type}. Choose from: dense, tfan, snn_emu")


__all__ = [
    'Backend',
    'DenseBackend',
    'TFANBackend',
    'SNNBackend',
    'build_backend',
]
