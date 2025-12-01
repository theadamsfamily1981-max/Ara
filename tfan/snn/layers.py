"""
Spiking neural network layers for GPU-emulated SNN.

Implements:
- SpikingLinear: Dense layer + LIF neuron
- SpikingConv2d: Conv2d + LIF neuron
- SpikingResidualBlock: Residual connections in spike domain
- SpikingSelfAttention: Optional sparse attention for spikes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .neuron import LIF, PLIF, NeuronState


# ============================================================================
# Spiking Linear Layer
# ============================================================================

class SpikingLinear(nn.Module):
    """
    Spiking linear layer: Linear → LIF neuron.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        neuron_type: 'LIF' or 'PLIF'
        tau_ms: Membrane time constant
        v_threshold: Spike threshold
        surrogate_type: Surrogate gradient type
        surrogate_k: Surrogate width parameter
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        neuron_type: str = "LIF",
        tau_ms: float = 20.0,
        v_threshold: float = 1.0,
        surrogate_type: str = "plinear",
        surrogate_k: float = 1.0,
        dt_ms: float = 1.0,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Create neuron
        if neuron_type == "LIF":
            self.neuron = LIF(
                tau_ms=tau_ms,
                v_threshold=v_threshold,
                dt_ms=dt_ms,
                surrogate_type=surrogate_type,
                surrogate_k=surrogate_k,
            )
        elif neuron_type == "PLIF":
            self.neuron = PLIF(
                num_neurons=out_features,
                tau_init_ms=tau_ms,
                v_threshold=v_threshold,
                dt_ms=dt_ms,
                surrogate_type=surrogate_type,
                surrogate_k=surrogate_k,
            )
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        self.out_features = out_features

    def forward(
        self,
        s_t: torch.Tensor,
        state: Optional[NeuronState] = None,
    ) -> Tuple[torch.Tensor, NeuronState]:
        """
        Forward pass.

        Args:
            s_t: Input spikes [batch, in_features] or [batch, seq, in_features]
            state: Previous neuron state

        Returns:
            out_spikes: Output spikes [batch, out_features]
            new_state: Updated neuron state
        """
        # Linear transformation
        current = self.linear(s_t)

        # Neuron dynamics
        out_spikes, new_state = self.neuron(current, state)

        return out_spikes, new_state

    def reset_state(self, batch_size: int, device=None, dtype=None) -> NeuronState:
        """Create fresh neuron state."""
        return self.neuron.reset_state(batch_size, device=device, dtype=dtype)


# ============================================================================
# Spiking Conv2d Layer
# ============================================================================

class SpikingConv2d(nn.Module):
    """
    Spiking convolutional layer: Conv2d → LIF neuron.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding
        bias: Whether to use bias
        neuron_type: 'LIF' or 'PLIF'
        tau_ms: Membrane time constant
        v_threshold: Spike threshold
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        neuron_type: str = "LIF",
        tau_ms: float = 20.0,
        v_threshold: float = 1.0,
        surrogate_type: str = "plinear",
        surrogate_k: float = 1.0,
        dt_ms: float = 1.0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Create neuron
        if neuron_type == "LIF":
            self.neuron = LIF(
                tau_ms=tau_ms,
                v_threshold=v_threshold,
                dt_ms=dt_ms,
                surrogate_type=surrogate_type,
                surrogate_k=surrogate_k,
            )
        elif neuron_type == "PLIF":
            # For PLIF, we need spatial dimensions - will be set dynamically
            self.neuron_type = "PLIF"
            self.neuron_kwargs = {
                "tau_init_ms": tau_ms,
                "v_threshold": v_threshold,
                "dt_ms": dt_ms,
                "surrogate_type": surrogate_type,
                "surrogate_k": surrogate_k,
            }
            self.neuron = None  # Will be created on first forward
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        self.out_channels = out_channels
        self.neuron_type_str = neuron_type

    def _init_neuron_if_needed(self, out_shape):
        """Initialize PLIF neuron with correct number of neurons."""
        if self.neuron_type_str == "PLIF" and self.neuron is None:
            # out_shape: [batch, channels, H, W]
            # PLIF needs num_neurons = channels * H * W
            num_neurons = out_shape[1]  # Just channels for now
            self.neuron = PLIF(num_neurons=num_neurons, **self.neuron_kwargs)
            self.neuron = self.neuron.to(next(self.parameters()).device)

    def forward(
        self,
        s_t: torch.Tensor,
        state: Optional[NeuronState] = None,
    ) -> Tuple[torch.Tensor, NeuronState]:
        """
        Forward pass.

        Args:
            s_t: Input spikes [batch, in_channels, H, W]
            state: Previous neuron state

        Returns:
            out_spikes: Output spikes [batch, out_channels, H', W']
            new_state: Updated neuron state
        """
        # Convolution
        current = self.conv(s_t)

        # Initialize PLIF if needed
        self._init_neuron_if_needed(current.shape)

        # Neuron dynamics
        out_spikes, new_state = self.neuron(current, state)

        return out_spikes, new_state


# ============================================================================
# Spiking Residual Block
# ============================================================================

class SpikingResidualBlock(nn.Module):
    """
    Residual block in spike domain.

    Architecture:
        Conv1 → LIF → Conv2 → LIF → Add residual

    The residual connection must handle spike domain addition correctly.

    Args:
        channels: Number of channels
        kernel_size: Kernel size for convolutions
        stride: Stride (default: 1)
        neuron_kwargs: Arguments for neuron creation
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        neuron_type: str = "LIF",
        tau_ms: float = 20.0,
        v_threshold: float = 1.0,
        surrogate_type: str = "plinear",
        surrogate_k: float = 1.0,
        dt_ms: float = 1.0,
    ):
        super().__init__()

        # Main path: Conv → LIF → Conv → LIF
        self.conv1 = SpikingConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            neuron_type=neuron_type,
            tau_ms=tau_ms,
            v_threshold=v_threshold,
            surrogate_type=surrogate_type,
            surrogate_k=surrogate_k,
            dt_ms=dt_ms,
        )

        self.conv2 = SpikingConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            neuron_type=neuron_type,
            tau_ms=tau_ms,
            v_threshold=v_threshold,
            surrogate_type=surrogate_type,
            surrogate_k=surrogate_k,
            dt_ms=dt_ms,
        )

        # Shortcut connection (identity or conv if stride > 1)
        if stride > 1:
            self.shortcut = SpikingConv2d(
                channels,
                channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                neuron_type=neuron_type,
                tau_ms=tau_ms,
                v_threshold=v_threshold,
                surrogate_type=surrogate_type,
                surrogate_k=surrogate_k,
                dt_ms=dt_ms,
            )
        else:
            self.shortcut = None

    def forward(
        self,
        s_t: torch.Tensor,
        state: Optional[Tuple[NeuronState, NeuronState, Optional[NeuronState]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[NeuronState, NeuronState, Optional[NeuronState]]]:
        """
        Forward pass.

        Args:
            s_t: Input spikes [batch, channels, H, W]
            state: Tuple of (conv1_state, conv2_state, shortcut_state)

        Returns:
            out_spikes: Output spikes
            new_states: Updated states
        """
        if state is None:
            state1, state2, state_short = None, None, None
        else:
            state1, state2, state_short = state

        # Main path
        x, state1_new = self.conv1(s_t, state1)
        x, state2_new = self.conv2(x, state2)

        # Shortcut
        if self.shortcut is not None:
            shortcut, state_short_new = self.shortcut(s_t, state_short)
        else:
            shortcut = s_t
            state_short_new = None

        # Residual add in spike domain (OR operation or weighted sum)
        # Using weighted sum for differentiability
        out = 0.5 * x + 0.5 * shortcut

        return out, (state1_new, state2_new, state_short_new)


# ============================================================================
# Optional: Spiking Self-Attention (simplified, local window)
# ============================================================================

class SpikingSelfAttention(nn.Module):
    """
    Simplified spiking self-attention with local window.

    For spike inputs, we use binary Q/K/V and local attention only.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        window_size: Local attention window
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        window_size: int = 64,
        neuron_type: str = "LIF",
        tau_ms: float = 20.0,
        v_threshold: float = 1.0,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size

        # Q, K, V projections (spiking)
        self.q_proj = SpikingLinear(hidden_dim, hidden_dim, neuron_type=neuron_type, tau_ms=tau_ms, v_threshold=v_threshold)
        self.k_proj = SpikingLinear(hidden_dim, hidden_dim, neuron_type=neuron_type, tau_ms=tau_ms, v_threshold=v_threshold)
        self.v_proj = SpikingLinear(hidden_dim, hidden_dim, neuron_type=neuron_type, tau_ms=tau_ms, v_threshold=v_threshold)
        self.out_proj = SpikingLinear(hidden_dim, hidden_dim, neuron_type=neuron_type, tau_ms=tau_ms, v_threshold=v_threshold)

    def forward(
        self,
        s_t: torch.Tensor,
        state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.

        Args:
            s_t: Input spikes [batch, seq_len, hidden_dim]
            state: QKV projection states

        Returns:
            out_spikes: Output spikes
            new_states: Updated states
        """
        batch_size, seq_len, _ = s_t.shape

        # Initialize states
        if state is None:
            q_state, k_state, v_state, out_state = None, None, None, None
        else:
            q_state, k_state, v_state, out_state = state

        # Project Q, K, V (spike outputs)
        q, q_state_new = self.q_proj(s_t, q_state)
        k, k_state_new = self.k_proj(s_t, k_state)
        v, v_state_new = self.v_proj(s_t, v_state)

        # Reshape for multi-head: [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Local attention (simplified: just use local window)
        # For spike domain, attention weights are binary (spike coincidence)
        # attn_weights = q @ k^T (spike coincidence counting)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply local window mask
        mask = torch.ones(seq_len, seq_len, device=s_t.device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, :start] = 0
            mask[i, end:] = 0
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)

        # Softmax (still differentiable in spike domain via surrogate)
        attn_probs = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # [batch, heads, seq, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        out, out_state_new = self.out_proj(attn_output, out_state)

        new_states = (q_state_new, k_state_new, v_state_new, out_state_new)

        return out, new_states


__all__ = [
    "SpikingLinear",
    "SpikingConv2d",
    "SpikingResidualBlock",
    "SpikingSelfAttention",
]
