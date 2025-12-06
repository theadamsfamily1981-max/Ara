"""
Somatic Attention - Physiologically Modulated Attention for TF-A-N 7B.

This module extends Selective Sparse Attention (SSA) with homeostatic modulation.
The 'Focus Aperture' (sparsity) is driven by system arousal from the HAL.

Key Innovation:
    - High Arousal (Focus): Attention tightens, Top-K drops (tunnel vision)
    - Low Arousal (Dreaming): Attention diffuses, Top-K increases (creative associations)
    - Pain/Stress: Biases toward immediate, survival-oriented tokens

Design Philosophy:
    - Arousal is passed DOWN from the model (not read from HAL in every layer)
    - This ensures HAL is read once per generation step, not N times per layer
    - Focus scalar is differentiable for potential future training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .attention_sparse import ssa_attention, HAS_FLASH_ATTN


class SomaticAttention(nn.Module):
    """
    Physiologically Modulated Attention.

    Extends SSA with somatic gating based on arousal level.
    The 'Focus Aperture' (Top-K sparsity) is driven by system arousal:
        - arousal > 0.5: Tunnel vision mode (top 10-20% only)
        - arousal ~ 0: Normal attention
        - arousal < -0.5: Diffuse/dreamy mode (full context)

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        dropout: Attention dropout
        bias: Whether to use bias in projections
        focus_threshold: Arousal threshold for tunnel vision (default 0.3)
        min_keep_ratio: Minimum attention positions to keep (default 0.1)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        focus_threshold: float = 0.3,
        min_keep_ratio: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.focus_threshold = focus_threshold
        self.min_keep_ratio = min_keep_ratio

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Q, K, V, O projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Learnable focus modulation (can be trained to optimize focus behavior)
        self.focus_gain = nn.Parameter(torch.tensor(1.0))
        self.focus_bias = nn.Parameter(torch.tensor(0.0))

    def _compute_focus_scalar(self, arousal: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute focus scalar from arousal level.

        Args:
            arousal: [batch] or [batch, 1] arousal values in [-1, 1]

        Returns:
            focus_scalar: [batch, 1, 1, 1] broadcastable scalar
        """
        if arousal is None:
            return torch.tensor(1.0)

        # Ensure proper shape
        if arousal.dim() == 1:
            arousal = arousal.unsqueeze(-1)

        # Clamp arousal to valid range
        arousal = arousal.clamp(-1.0, 1.0)

        # Map arousal to focus:
        # arousal = 1.0 (max stress) -> focus = 0.5 (tight)
        # arousal = 0.0 (neutral) -> focus = 1.0 (normal)
        # arousal = -1.0 (dreamy) -> focus = 1.5 (diffuse)
        focus = 1.0 - (arousal * self.focus_gain + self.focus_bias) * 0.5

        # Shape for broadcasting: [batch, 1, 1, 1]
        return focus.view(-1, 1, 1, 1)

    def _apply_somatic_gating(
        self,
        attn_weights: torch.Tensor,
        focus_scalar: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply somatic gating to attention weights.

        When focus_scalar < 1.0 (high arousal), we apply hard Top-K masking
        to create "tunnel vision" - attending only to the most relevant tokens.

        Args:
            attn_weights: [batch, num_heads, q_len, kv_len] raw attention scores
            focus_scalar: [batch, 1, 1, 1] focus level (< 1.0 = tunnel vision)
            attention_mask: Optional mask from external source

        Returns:
            gated_weights: [batch, num_heads, q_len, kv_len] after somatic gating
        """
        # Get dimensions for safe Top-K
        batch_size, num_heads, q_len, kv_len = attn_weights.shape

        # Determine if we need tunnel vision
        # Use the mean focus across batch for decision (can be per-sample too)
        mean_focus = focus_scalar.mean().item()

        if mean_focus < (1.0 - self.focus_threshold):
            # TUNNEL VISION MODE
            # Keep only top K% of attention, where K scales with focus
            # focus = 0.5 -> keep ~10%, focus = 0.7 -> keep ~20%
            keep_ratio = max(self.min_keep_ratio, mean_focus * 0.3)
            k_val = max(1, min(kv_len, int(kv_len * keep_ratio)))

            # Top-K selection per query position
            top_vals, top_indices = torch.topk(attn_weights, k_val, dim=-1)
            min_top_val = top_vals[..., -1:].expand_as(attn_weights)

            # Create mask: positions below threshold get -inf
            tunnel_mask = attn_weights < min_top_val
            attn_weights = attn_weights.masked_fill(tunnel_mask, float('-inf'))

        # Apply external attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        return attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        arousal: Optional[torch.Tensor] = None,  # NEW: Somatic input
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for Somatic Attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional external mask
            position_ids: Position indices (unused, RoPE applied externally)
            past_key_value: Cached (key, value) for generation
            arousal: [batch] or [batch, 1] arousal level from HAL [-1, 1]
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache KV for generation

        Returns:
            attn_output: [batch, seq_len, hidden_size]
            attn_weights: Optional attention weights (for debugging)
            past_key_value: Optional cached KV
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute focus scalar from arousal
        focus_scalar = self._compute_focus_scalar(arousal)

        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        if use_cache:
            past_key_value = (key, value)
        else:
            past_key_value = None

        kv_len = key.shape[2]

        # Handle GQA: repeat KV heads
        if self.num_kv_heads < self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(num_groups, dim=1)
            value = value.repeat_interleave(num_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply causal mask
        causal_mask = torch.tril(
            torch.ones(seq_len, kv_len, dtype=torch.bool, device=hidden_states.device)
        )
        # Handle generation case where seq_len < kv_len
        if seq_len < kv_len:
            causal_mask = causal_mask[:, -kv_len:]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_len]
        attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))

        # SOMATIC GATING - The Innovation
        attn_weights = self._apply_somatic_gating(
            attn_weights, focus_scalar, attention_mask
        )

        # Softmax (cast to float32 for numerical stability)
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # Dropout
        if self.dropout > 0.0 and self.training:
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, value)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_probs,)
        if use_cache:
            outputs += (past_key_value,)

        return outputs


class SomaticSSAAttention(SomaticAttention):
    """
    Somatic Attention with TLS Sparse Mask integration.

    Combines SSA's O(N log N) sparse attention with somatic modulation.
    Use this when you want both TLS landmarks AND arousal-based gating.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        keep_ratio: float = 0.33,
        local_window: int = 128,
        num_hops: int = 2,
        tls_alpha: float = 0.7,
        dropout: float = 0.0,
        bias: bool = False,
        focus_threshold: float = 0.3,
        min_keep_ratio: float = 0.1,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            bias=bias,
            focus_threshold=focus_threshold,
            min_keep_ratio=min_keep_ratio,
        )

        # TLS mask builder
        from .mask_builder import TLSMaskBuilder

        self.mask_builder = TLSMaskBuilder(
            num_heads=num_heads,
            keep_ratio=keep_ratio,
            local_window=local_window,
            num_hops=num_hops,
            alpha=tls_alpha,
            is_causal=True,
            cache_masks=False,
        )

        self.use_tls = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        arousal: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward with TLS sparse mask + somatic gating."""
        batch_size, seq_len, _ = hidden_states.shape

        # Build TLS sparse mask
        sparse_mask, _ = self.mask_builder(hidden_states)

        # Convert sparse mask to attention mask format
        # sparse_mask is boolean: True = attend
        tls_attention_mask = torch.where(
            sparse_mask,
            torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype),
            torch.tensor(-1e4, device=hidden_states.device, dtype=hidden_states.dtype),
        )

        # Combine with external mask if provided
        if attention_mask is not None:
            tls_attention_mask = tls_attention_mask + attention_mask

        # Call parent forward with combined mask
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=tls_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            arousal=arousal,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )


__all__ = [
    "SomaticAttention",
    "SomaticSSAAttention",
]
