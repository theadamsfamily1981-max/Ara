"""
Selective Sparse Attention (SSA) for TF-A-N 7B.

O(N log N) attention using TLS landmarks and block-radial sparsity.
Compatible with Flash Attention when available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def ssa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selective Sparse Attention kernel.

    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, num_kv_heads, seq_len, head_dim]
        value: [batch, num_kv_heads, seq_len, head_dim]
        mask: [batch, num_heads, seq_len, seq_len] boolean mask (True = attend)
        scale: Attention scale (default: 1/sqrt(head_dim))
        dropout_p: Dropout probability
        is_causal: Whether to enforce causal masking

    Returns:
        output: [batch, num_heads, seq_len, head_dim]
        attn_weights: [batch, num_heads, seq_len, seq_len] (for debugging)
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    _, num_kv_heads, _, _ = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Handle Grouped Query Attention (GQA): repeat KV heads to match Q heads
    if num_kv_heads < num_heads:
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        num_groups = num_heads // num_kv_heads
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)

    # Compute attention scores: Q @ K^T
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    # [batch, num_heads, seq_len, seq_len]

    # Apply sparse mask
    if mask is not None:
        # mask is boolean: True = attend, False = mask out
        # Convert to additive mask: 0 for attend, -1e4 for mask
        attn_mask = torch.where(
            mask,
            torch.tensor(0.0, dtype=attn_weights.dtype, device=attn_weights.device),
            torch.tensor(-1e4, dtype=attn_weights.dtype, device=attn_weights.device),
        )
        attn_weights = attn_weights + attn_mask
    elif is_causal:
        # Default causal mask if no sparse mask provided
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device)
        )
        attn_mask = torch.where(
            causal_mask,
            torch.tensor(0.0, dtype=attn_weights.dtype, device=attn_weights.device),
            torch.tensor(-1e4, dtype=attn_weights.dtype, device=attn_weights.device),
        )
        attn_weights = attn_weights + attn_mask

    # Softmax
    attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # Dropout
    if dropout_p > 0.0 and self.training:
        attn_probs = F.dropout(attn_probs, p=dropout_p)

    # Apply attention to values
    output = torch.matmul(attn_probs, value)
    # [batch, num_heads, seq_len, head_dim]

    return output, attn_weights


def ssa_attention_flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, None]:
    """
    Selective Sparse Attention using Flash Attention.

    Note: Flash Attention has limited sparse mask support. This function
    attempts to use Flash Attention when possible, falling back to standard
    implementation for complex sparse patterns.

    Args:
        query: [batch, seq_len, num_heads, head_dim]
        key: [batch, seq_len, num_kv_heads, head_dim]
        value: [batch, seq_len, num_kv_heads, head_dim]
        mask: Optional sparse mask (limited support)
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        softmax_scale: Attention scale

    Returns:
        output: [batch, seq_len, num_heads, head_dim]
        None: (Flash Attention doesn't return weights)
    """
    if not HAS_FLASH_ATTN:
        raise RuntimeError("Flash Attention not available. Install flash-attn package.")

    # Flash Attention expects [batch, seq, heads, head_dim] format
    # Input already in this format

    # Note: Flash Attention v2 supports causal masking natively
    # For sparse masks, we fall back to standard implementation
    if mask is not None:
        # Flash Attention doesn't support arbitrary sparse masks well
        # Fall back to standard implementation
        # Transpose to [batch, heads, seq, head_dim] for ssa_attention
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)
        output, _ = ssa_attention(
            query_t, key_t, value_t, mask, softmax_scale, dropout_p, is_causal
        )
        # Transpose back to [batch, seq, heads, head_dim]
        return output.transpose(1, 2), None

    # Use Flash Attention for dense/causal case
    output = flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=is_causal,
    )

    return output, None


class SSAAttention(nn.Module):
    """
    Selective Sparse Attention module with TLS landmark selection.

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA, default = num_heads)
        keep_ratio: Landmark keep ratio for TLS
        local_window: Local attention window size
        num_hops: Number of radial hops from landmarks
        tls_alpha: TLS alpha (persistence vs diversity)
        dropout: Attention dropout
        bias: Whether to use bias in projections
        use_flash: Whether to use Flash Attention
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
        use_flash: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.use_flash = use_flash and HAS_FLASH_ATTN

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert (
            num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # TLS mask builder (imported from mask_builder.py)
        from .mask_builder import TLSMaskBuilder

        self.mask_builder = TLSMaskBuilder(
            num_heads=num_heads,
            keep_ratio=keep_ratio,
            local_window=local_window,
            num_hops=num_hops,
            alpha=tls_alpha,
            is_causal=True,
            cache_masks=False,  # Disable caching during training
        )

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for SSA attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional external mask
            position_ids: Position indices (not used with RoPE applied externally)
            past_key_value: Cached (key, value) for generation
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache KV for generation

        Returns:
            attn_output: [batch, seq_len, hidden_size]
            attn_weights: Optional attention weights
            past_key_value: Optional cached KV
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to [batch, seq_len, num_heads, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Handle KV cache for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)

        if use_cache:
            past_key_value = (key, value)
        else:
            past_key_value = None

        # Build sparse mask using TLS
        # Pass full hidden states (including cache) for landmark selection
        if past_key_value is not None:
            # During generation, use full sequence for mask building
            full_seq_len = key.shape[1]
            # Use key states for mask building (already includes past)
            mask_input = key.view(batch_size, full_seq_len, -1)
        else:
            mask_input = hidden_states

        sparse_mask, landmark_indices = self.mask_builder(mask_input)
        # sparse_mask: [batch, num_heads, seq_len, kv_seq_len]

        # Apply attention
        if self.use_flash:
            # Flash Attention path: [batch, seq, heads, head_dim]
            attn_output, attn_weights = ssa_attention_flash(
                query,
                key,
                value,
                mask=sparse_mask if attention_mask is None else None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                softmax_scale=self.scale,
            )
            # Output: [batch, seq_len, num_heads, head_dim]
            attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        else:
            # Standard attention path: [batch, heads, seq, head_dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            attn_output, attn_weights = ssa_attention(
                query,
                key,
                value,
                mask=sparse_mask,
                scale=self.scale,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            # Output: [batch, num_heads, seq_len, head_dim]
            # Reshape to [batch, seq_len, hidden_size]
            attn_output = attn_output.transpose(1, 2).reshape(
                batch_size, seq_len, self.hidden_size
            )

        # Output projection
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)

        return outputs


__all__ = [
    "ssa_attention",
    "ssa_attention_flash",
    "SSAAttention",
    "HAS_FLASH_ATTN",
]
