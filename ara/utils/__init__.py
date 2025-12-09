"""
ARA Utilities Package

Common utilities for device management, seeding, configuration, etc.
"""

import os
import random
from typing import Optional

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def get_device(device: Optional[str] = None) -> str:
    """
    Get the best available device.

    Args:
        device: Specific device to use ('cuda', 'cpu', 'mps', etc.)
                If None, auto-detect best device.

    Returns:
        Device string suitable for PyTorch
    """
    if device is not None:
        return device

    if not TORCH_AVAILABLE:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic algorithms (slower)
    """
    random.seed(seed)
    if NUMPY_AVAILABLE:
        np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)


def count_parameters(model) -> dict:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    if not TORCH_AVAILABLE:
        return {"error": "torch not available"}

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_millions": total / 1e6,
        "total_billions": total / 1e9,
    }


def format_size(num_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


from .covenant import (
    Covenant,
    AutomationLevel,
    ContentScore,
    CovenantViolation,
    get_covenant,
    reset_covenant,
)


__all__ = [
    "get_device",
    "set_seed",
    "count_parameters",
    "format_size",
    "TORCH_AVAILABLE",
    "NUMPY_AVAILABLE",
    # Covenant
    "Covenant",
    "AutomationLevel",
    "ContentScore",
    "CovenantViolation",
    "get_covenant",
    "reset_covenant",
]
