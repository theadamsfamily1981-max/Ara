"""
Optimizer configuration for TF-A-N 7B training.
"""

import torch
from torch.optim import AdamW
from typing import Optional, List, Dict


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    no_decay_params: Optional[List[str]] = None,
) -> AdamW:
    """
    Create AdamW optimizer with weight decay applied selectively.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        eps: Adam epsilon
        no_decay_params: List of parameter name patterns to exclude from weight decay
                         (default: LayerNorm, RMSNorm, bias, embeddings)

    Returns:
        optimizer: Configured AdamW optimizer
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "norm", "embedding"]

    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params_list = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have no decay
        if any(nd in name.lower() for nd in no_decay_params):
            no_decay_params_list.append(param)
        else:
            decay_params.append(param)

    # Create parameter groups
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params_list,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=betas,
        eps=eps,
    )

    return optimizer


def get_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> Dict:
    """
    Get optimizer state dict for checkpointing.

    Args:
        optimizer: Optimizer instance

    Returns:
        state_dict: Optimizer state dictionary
    """
    return optimizer.state_dict()


def load_optimizer_state_dict(
    optimizer: torch.optim.Optimizer,
    state_dict: Dict,
):
    """
    Load optimizer state from checkpoint.

    Args:
        optimizer: Optimizer instance
        state_dict: State dictionary to load
    """
    optimizer.load_state_dict(state_dict)


__all__ = [
    "create_optimizer",
    "get_optimizer_state_dict",
    "load_optimizer_state_dict",
]
