"""
SwiGLU MLP (Swish-Gated Linear Unit) for TF-A-N 7B.

Gated feedforward network with SiLU activation, used in LLaMA/PaLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU MLP block.

    Consists of three projections:
        gate = SiLU(W_gate @ x)
        value = W_value @ x
        output = W_out @ (gate * value)

    This is more parameter-efficient than standard FFN while providing
    better expressiveness through gating.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension (typically 2.5-4× hidden_size)
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate and value projections (can be fused)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.value_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Output tensor [..., hidden_size]
        """
        gate = F.silu(self.gate_proj(x))  # SiLU activation
        value = self.value_proj(x)
        hidden = gate * value  # Gated multiplication
        output = self.out_proj(hidden)
        return output


class SwiGLUFused(nn.Module):
    """
    SwiGLU with fused gate/value projection for efficiency.

    Combines gate_proj and value_proj into a single 2*intermediate_size projection,
    then splits and applies gating. This reduces memory bandwidth.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Fused gate+value projection
        self.gate_value_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=bias
        )

        # Output projection
        self.out_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused projection.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Output tensor [..., hidden_size]
        """
        # Project to 2*intermediate_size and split
        gate_value = self.gate_value_proj(x)
        gate, value = gate_value.chunk(2, dim=-1)

        # Apply SiLU to gate and multiply
        hidden = F.silu(gate) * value

        # Project back to hidden_size
        output = self.out_proj(hidden)
        return output


class SwiGLUWithResidual(nn.Module):
    """
    SwiGLU MLP with optional residual connection and dropout.

    Used in transformer decoder layers with pre-norm architecture.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension
        bias: Whether to use bias
        dropout: Dropout probability (0 = no dropout)
        use_fused: Whether to use fused gate/value projection
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
        use_fused: bool = True,
    ):
        super().__init__()

        # Select MLP variant
        if use_fused:
            self.mlp = SwiGLUFused(hidden_size, intermediate_size, bias)
        else:
            self.mlp = SwiGLU(hidden_size, intermediate_size, bias)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional residual.

        Args:
            x: Input tensor (typically post-norm)
            residual: Optional residual to add after MLP

        Returns:
            Output tensor (with residual if provided)
        """
        output = self.mlp(x)

        if self.dropout is not None:
            output = self.dropout(output)

        if residual is not None:
            output = output + residual

        return output


def create_mlp(
    hidden_size: int,
    ffn_mult: float = 3.5,
    bias: bool = False,
    dropout: float = 0.0,
    use_fused: bool = True,
) -> SwiGLUWithResidual:
    """
    Factory function to create SwiGLU MLP with standard config.

    Args:
        hidden_size: Hidden dimension
        ffn_mult: Multiplier for intermediate size (typically 3.5-4.0)
        bias: Whether to use bias
        dropout: Dropout probability
        use_fused: Whether to use fused implementation

    Returns:
        SwiGLU MLP module
    """
    intermediate_size = int(ffn_mult * hidden_size)

    # Round to multiple of 128 for better hardware utilization
    intermediate_size = ((intermediate_size + 127) // 128) * 128

    return SwiGLUWithResidual(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bias=bias,
        dropout=dropout,
        use_fused=use_fused,
    )


__all__ = [
    "SwiGLU",
    "SwiGLUFused",
    "SwiGLUWithResidual",
    "create_mlp",
]
