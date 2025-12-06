"""
Somatic Embedding - Body-to-Latent Space Transducer.

This module injects raw physiological state into the transformer's residual stream.
It ensures that even if the model "tries to ignore" the pain, it mathematically
biases the latent state toward embodied cognition.

Key Concept:
    The body's state becomes a persistent background signal added to EVERY token
    embedding. This is the neural equivalent of "the pain is everywhere" -
    you can't think about anything without the body's presence.

Input Channels (7D Somatic Vector):
    [0] Pain       - Physical pain level [0, 1]
    [1] Entropy    - System chaos/uncertainty [0, 1]
    [2] Flow_X     - Optical flow X component
    [3] Flow_Y     - Optical flow Y component
    [4] PAD_P      - Pleasure axis [-1, 1]
    [5] PAD_A      - Arousal axis [-1, 1]
    [6] PAD_D      - Dominance axis [-1, 1]

Design Philosophy:
    - Small network (256 hidden) to minimize overhead
    - SiLU activation for smooth, non-saturating gradients
    - Output bias added to all tokens (not just first)
    - Optional gating to modulate influence based on attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SomaticEmbedding(nn.Module):
    """
    Projects physiological state into transformer hidden space.

    The somatic embedding creates a "body bias" that is added to every token
    in the sequence. This ensures embodied cognition - all processing is
    colored by the current physical state.

    Args:
        hidden_size: Transformer hidden dimension (e.g., 4096)
        somatic_dim: Input somatic vector dimension (default 7)
        intermediate_size: Projector intermediate size (default 256)
        dropout: Dropout probability (default 0.0)
        use_gate: Whether to use gated projection (default False)
    """

    # Named indices for somatic vector
    PAIN = 0
    ENTROPY = 1
    FLOW_X = 2
    FLOW_Y = 3
    PAD_P = 4
    PAD_A = 5
    PAD_D = 6

    def __init__(
        self,
        hidden_size: int,
        somatic_dim: int = 7,
        intermediate_size: int = 256,
        dropout: float = 0.0,
        use_gate: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.somatic_dim = somatic_dim
        self.use_gate = use_gate

        # Main projector: somatic -> hidden space
        self.projector = nn.Sequential(
            nn.Linear(somatic_dim, intermediate_size),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(intermediate_size, hidden_size),
        )

        # Optional gating network (modulates somatic influence)
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(somatic_dim, intermediate_size // 2),
                nn.SiLU(),
                nn.Linear(intermediate_size // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        # Initialize with small weights to avoid disrupting pretrained model
        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable integration."""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                # Small std to not disrupt pretrained model too much
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.gate is not None:
            for module in self.gate.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        somatic_tensor: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Project somatic state into hidden space.

        Args:
            somatic_tensor: [batch, 7] physiological state vector
            return_components: If True, also return intermediate values

        Returns:
            body_bias: [batch, 1, hidden_size] broadcastable to [batch, seq, hidden]
            Optional: dict with pain_contrib, arousal_contrib, etc.
        """
        # Ensure correct device and dtype
        if somatic_tensor.dim() == 1:
            somatic_tensor = somatic_tensor.unsqueeze(0)

        # Project to hidden space
        hidden = self.projector(somatic_tensor)  # [batch, hidden_size]

        # Apply gate if present
        if self.gate is not None:
            gate_value = self.gate(somatic_tensor)  # [batch, 1]
            hidden = hidden * gate_value

        # Reshape for broadcasting: [batch, 1, hidden_size]
        body_bias = hidden.unsqueeze(1)

        if return_components:
            # Extract individual contributions for debugging/introspection
            with torch.no_grad():
                pain = somatic_tensor[:, self.PAIN]
                arousal = somatic_tensor[:, self.PAD_A]
                entropy = somatic_tensor[:, self.ENTROPY]

            return body_bias, {
                'pain_level': pain,
                'arousal_level': arousal,
                'entropy_level': entropy,
                'gate_value': gate_value if self.gate else None,
                'bias_norm': hidden.norm(dim=-1),
            }

        return body_bias


class CortisolInjector(nn.Module):
    """
    Injects "Cortisol Vector" into the residual stream during high-stress states.

    When pain or arousal exceeds a threshold, this module biases the model
    toward immediate survival-oriented tokens and away from long-term planning.

    This is a biological metaphor for the cortisol response:
    - High cortisol -> narrow focus, reactive behavior, short-term thinking
    - Low cortisol -> broad thinking, creative, long-term planning

    Args:
        hidden_size: Transformer hidden dimension
        pain_threshold: Pain level above which cortisol kicks in (default 0.5)
        arousal_threshold: Arousal level above which cortisol kicks in (default 0.6)
    """

    def __init__(
        self,
        hidden_size: int,
        pain_threshold: float = 0.5,
        arousal_threshold: float = 0.6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pain_threshold = pain_threshold
        self.arousal_threshold = arousal_threshold

        # Learnable cortisol direction in hidden space
        # This represents the "survival mode" direction
        self.cortisol_direction = nn.Parameter(
            torch.randn(hidden_size) * 0.01
        )

        # Strength modulator
        self.strength_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        pain: torch.Tensor,
        arousal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject cortisol bias into hidden states when stressed.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            pain: [batch] or [batch, 1] pain level [0, 1]
            arousal: [batch] or [batch, 1] arousal level [-1, 1]

        Returns:
            modified_states: [batch, seq_len, hidden_size] with cortisol bias
        """
        # Ensure proper shapes
        if pain.dim() == 1:
            pain = pain.unsqueeze(-1)
        if arousal.dim() == 1:
            arousal = arousal.unsqueeze(-1)

        # Compute stress level (0-1)
        # High when pain > threshold OR arousal > threshold
        pain_stress = F.relu(pain - self.pain_threshold) / (1.0 - self.pain_threshold + 1e-6)
        arousal_stress = F.relu(arousal - self.arousal_threshold) / (1.0 - self.arousal_threshold + 1e-6)

        # Combined stress (max of pain and arousal stress)
        stress = torch.max(pain_stress, arousal_stress)  # [batch, 1]

        # Compute cortisol injection strength
        strength = stress * self.strength_scale  # [batch, 1]

        # Normalize cortisol direction
        cortisol_normalized = F.normalize(self.cortisol_direction, dim=0)

        # Inject cortisol into hidden states
        # Shape: [batch, 1, 1] * [hidden_size] -> broadcast to all positions
        cortisol_bias = strength.unsqueeze(-1) * cortisol_normalized  # [batch, 1, hidden_size]

        return hidden_states + cortisol_bias


class SomaticEncoder(nn.Module):
    """
    Complete somatic encoding system combining embedding and cortisol injection.

    This is the full "body interface" for the transformer:
    1. SomaticEmbedding: Persistent body bias on all tokens
    2. CortisolInjector: Stress-dependent bias in residual stream

    Args:
        hidden_size: Transformer hidden dimension
        somatic_dim: Input somatic vector dimension
        use_cortisol: Whether to use cortisol injection
    """

    def __init__(
        self,
        hidden_size: int,
        somatic_dim: int = 7,
        intermediate_size: int = 256,
        use_cortisol: bool = True,
        use_gate: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = SomaticEmbedding(
            hidden_size=hidden_size,
            somatic_dim=somatic_dim,
            intermediate_size=intermediate_size,
            use_gate=use_gate,
        )

        self.cortisol = CortisolInjector(hidden_size) if use_cortisol else None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        somatic_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply somatic encoding to input embeddings.

        Args:
            inputs_embeds: [batch, seq_len, hidden_size] token embeddings
            somatic_tensor: [batch, 7] physiological state

        Returns:
            encoded_embeds: [batch, seq_len, hidden_size] with body bias
            somatic_info: dict with diagnostic information
        """
        # Get body bias
        body_bias, components = self.embedding(somatic_tensor, return_components=True)

        # Add body bias to all tokens
        # "The pain is everywhere"
        encoded = inputs_embeds + body_bias

        # Apply cortisol injection if enabled
        if self.cortisol is not None:
            pain = somatic_tensor[:, SomaticEmbedding.PAIN]
            arousal = somatic_tensor[:, SomaticEmbedding.PAD_A]
            encoded = self.cortisol(encoded, pain, arousal)

        return encoded, components


def create_somatic_tensor(
    pain: float = 0.0,
    entropy: float = 0.0,
    flow_x: float = 0.0,
    flow_y: float = 0.0,
    pad_p: float = 0.0,
    pad_a: float = 0.0,
    pad_d: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Helper to create somatic tensor from individual values.

    Args:
        pain: Physical pain [0, 1]
        entropy: System entropy [0, 1]
        flow_x: Optical flow X
        flow_y: Optical flow Y
        pad_p: Pleasure [-1, 1]
        pad_a: Arousal [-1, 1]
        pad_d: Dominance [-1, 1]
        device: Target device
        dtype: Data type

    Returns:
        somatic_tensor: [1, 7] ready for model input
    """
    tensor = torch.tensor(
        [[pain, entropy, flow_x, flow_y, pad_p, pad_a, pad_d]],
        device=device,
        dtype=dtype,
    )
    return tensor


def somatic_from_hal(hal_state: dict, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create somatic tensor from HAL state dictionary.

    Args:
        hal_state: Dict from AraHAL.read_somatic()
        device: Target device

    Returns:
        somatic_tensor: [1, 7] ready for model input
    """
    pad = hal_state.get('pad', {'p': 0, 'a': 0, 'd': 0})
    flow = hal_state.get('flow', (0, 0))

    return create_somatic_tensor(
        pain=hal_state.get('pain', 0.0),
        entropy=hal_state.get('entropy', 0.0),
        flow_x=flow[0] if isinstance(flow, (tuple, list)) else 0.0,
        flow_y=flow[1] if isinstance(flow, (tuple, list)) else 0.0,
        pad_p=pad.get('p', 0.0),
        pad_a=pad.get('a', 0.0),
        pad_d=pad.get('d', 0.0),
        device=device,
    )


__all__ = [
    "SomaticEmbedding",
    "CortisolInjector",
    "SomaticEncoder",
    "create_somatic_tensor",
    "somatic_from_hal",
]
