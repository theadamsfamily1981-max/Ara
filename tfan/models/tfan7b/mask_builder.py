"""
Mask construction and TLS (Topological Landmark Selection) for SSA.

Builds block-sparse attention masks with:
- Local windows (causal)
- Radial hops to selected landmarks
- Per-head landmark selection via TLS

TLS combines:
- Persistence: Distance to centroid (proxy for topological lifetime)
- Max-min diversity: Ensures well-distributed coverage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


def compute_tls_scores(
    hidden_states: torch.Tensor,
    alpha: float = 0.7,
    return_components: bool = False,
) -> torch.Tensor:
    """
    Compute TLS (Topological Landmark Selection) scores.

    Score = α * persistence + (1-α) * max_min_diversity

    Persistence approximation: ||x - centroid|| / max_dist
    Max-min diversity: distance to k nearest neighbors

    Args:
        hidden_states: Token representations [batch, seq_len, hidden_dim]
        alpha: Weight for persistence vs diversity (default 0.7)
        return_components: If True, return (scores, persistence, diversity)

    Returns:
        scores: TLS scores [batch, seq_len]
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Compute centroid
    centroid = hidden_states.mean(dim=1, keepdim=True)  # [batch, 1, hidden]

    # Persistence component: distance to centroid
    dist_to_centroid = torch.norm(
        hidden_states - centroid, dim=-1
    )  # [batch, seq_len]
    max_dist = dist_to_centroid.max(dim=1, keepdim=True)[0] + 1e-8
    persistence = dist_to_centroid / max_dist  # Normalize to [0, 1]

    # Max-min diversity: distance to nearest neighbors
    # Compute pairwise distances (for efficiency, sample if seq_len > 2048)
    if seq_len > 2048:
        # Sample subset for diversity computation
        sample_size = min(512, seq_len)
        indices = torch.randperm(seq_len, device=hidden_states.device)[:sample_size]
        sampled = hidden_states[:, indices, :]  # [batch, sample_size, hidden]

        # Pairwise distances within sample
        dists = torch.cdist(sampled, sampled, p=2)  # [batch, sample_size, sample_size]

        # Get k nearest distances (k=5)
        k = min(5, sample_size - 1)
        nearest_dists, _ = torch.topk(dists, k=k + 1, dim=-1, largest=False)
        # nearest_dists[:, :, 0] is self (distance 0), so take [:, :, 1:k+1]
        maxmin_diversity_sample = nearest_dists[:, :, -1]  # Max of k nearest

        # Map back to full sequence (use indices)
        maxmin_diversity = torch.zeros(
            batch_size, seq_len, device=hidden_states.device
        )
        maxmin_diversity[:, indices] = maxmin_diversity_sample

        # For non-sampled positions, use mean of sampled diversity
        mean_diversity = maxmin_diversity_sample.mean(dim=1, keepdim=True)
        mask = torch.zeros(batch_size, seq_len, device=hidden_states.device)
        mask[:, indices] = 1.0
        maxmin_diversity = maxmin_diversity + (1 - mask) * mean_diversity

    else:
        # Full pairwise distances for shorter sequences
        dists = torch.cdist(
            hidden_states, hidden_states, p=2
        )  # [batch, seq_len, seq_len]

        # Get k nearest distances
        k = min(5, seq_len - 1)
        nearest_dists, _ = torch.topk(dists, k=k + 1, dim=-1, largest=False)
        maxmin_diversity = nearest_dists[:, :, -1]  # Max of k nearest

    # Normalize diversity
    max_div = maxmin_diversity.max(dim=1, keepdim=True)[0] + 1e-8
    maxmin_diversity = maxmin_diversity / max_div

    # Combine with alpha weighting
    tls_scores = alpha * persistence + (1 - alpha) * maxmin_diversity

    if return_components:
        return tls_scores, persistence, maxmin_diversity
    return tls_scores


def select_landmarks_per_head(
    hidden_states: torch.Tensor,
    num_heads: int,
    keep_ratio: float = 0.33,
    alpha: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select landmarks for each attention head using TLS.

    Args:
        hidden_states: Token representations [batch, seq_len, hidden_dim]
        num_heads: Number of attention heads
        keep_ratio: Fraction of tokens to keep as landmarks
        alpha: TLS alpha parameter

    Returns:
        landmark_indices: [batch, num_heads, num_landmarks]
        landmark_scores: [batch, num_heads, num_landmarks]
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_landmarks = max(1, int(keep_ratio * seq_len))

    # Compute TLS scores
    tls_scores = compute_tls_scores(hidden_states, alpha=alpha)  # [batch, seq_len]

    # Add per-head diversity: split hidden states across heads and compute head-specific scores
    head_dim = hidden_dim // num_heads
    hidden_per_head = hidden_states.view(
        batch_size, seq_len, num_heads, head_dim
    )  # [batch, seq, heads, head_dim]
    hidden_per_head = hidden_per_head.transpose(1, 2)  # [batch, heads, seq, head_dim]

    # Compute per-head centroid distance for diversity
    head_centroids = hidden_per_head.mean(dim=2, keepdim=True)  # [batch, heads, 1, head_dim]
    head_dists = torch.norm(
        hidden_per_head - head_centroids, dim=-1
    )  # [batch, heads, seq]
    head_dists = head_dists / (head_dists.max(dim=2, keepdim=True)[0] + 1e-8)

    # Combine global TLS scores with per-head diversity
    # [batch, seq] -> [batch, 1, seq] + [batch, heads, seq]
    combined_scores = (
        0.7 * tls_scores.unsqueeze(1) + 0.3 * head_dists
    )  # [batch, heads, seq]

    # Select top-k per head
    landmark_scores, landmark_indices = torch.topk(
        combined_scores, k=num_landmarks, dim=-1
    )

    return landmark_indices, landmark_scores


def build_block_radial_mask(
    seq_len: int,
    landmark_indices: torch.Tensor,
    local_window: int = 128,
    num_hops: int = 2,
    is_causal: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build block-sparse attention mask with local + radial structure.

    Each token attends to:
    1. Local window (causal or bidirectional)
    2. Landmarks (selected per head)
    3. Neighbors of landmarks (radial hops)

    Args:
        seq_len: Sequence length
        landmark_indices: [batch, num_heads, num_landmarks]
        local_window: Size of local attention window
        num_hops: Number of radial hops from landmarks
        is_causal: If True, enforce causal masking
        device: Device for mask tensor

    Returns:
        mask: Attention mask [batch, num_heads, seq_len, seq_len]
              (True = attend, False = mask out)
    """
    if device is None:
        device = landmark_indices.device

    batch_size, num_heads, num_landmarks = landmark_indices.shape

    # Initialize mask: [batch, heads, seq_len, seq_len]
    mask = torch.zeros(
        batch_size, num_heads, seq_len, seq_len, dtype=torch.bool, device=device
    )

    # 1) Local window attention
    for i in range(seq_len):
        start = max(0, i - local_window + 1) if is_causal else max(0, i - local_window)
        end = i + 1 if is_causal else min(seq_len, i + local_window + 1)
        mask[:, :, i, start:end] = True

    # 2) Landmark attention: all tokens attend to landmarks
    for b in range(batch_size):
        for h in range(num_heads):
            lm_idx = landmark_indices[b, h]  # [num_landmarks]
            mask[b, h, :, lm_idx] = True  # All tokens attend to landmarks

            # Landmarks attend to all tokens (bidirectional)
            mask[b, h, lm_idx, :] = True

    # 3) Radial hops: attend to neighbors of landmarks
    if num_hops > 0:
        for b in range(batch_size):
            for h in range(num_heads):
                lm_idx = landmark_indices[b, h]  # [num_landmarks]
                for lm in lm_idx:
                    # Add neighbors within hop distance
                    for hop in range(1, num_hops + 1):
                        hop_dist = hop * local_window
                        start = max(0, lm - hop_dist)
                        end = min(seq_len, lm + hop_dist + 1)
                        mask[b, h, lm, start:end] = True  # Landmark attends to neighbors
                        mask[b, h, start:end, lm] = True  # Neighbors attend to landmark

    # 4) Enforce causal mask if needed
    if is_causal:
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        mask = mask & causal_mask.unsqueeze(0).unsqueeze(0)

    return mask


def compute_sparsity(mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute sparsity statistics for attention mask.

    Args:
        mask: [batch, num_heads, seq_len, seq_len] boolean mask

    Returns:
        dict with: sparsity (fraction of zeros), density, avg_nnz_per_row
    """
    total_elements = mask.numel()
    nonzero_elements = mask.sum().item()

    density = nonzero_elements / total_elements
    sparsity = 1.0 - density

    # Average non-zeros per row
    nnz_per_row = mask.float().sum(dim=-1).mean().item()

    return {
        "sparsity": sparsity,
        "density": density,
        "avg_nnz_per_row": nnz_per_row,
        "total_elements": total_elements,
        "nonzero_elements": nonzero_elements,
    }


class TLSMaskBuilder(nn.Module):
    """
    Stateful mask builder with TLS landmark selection.

    Caches landmarks and masks for efficiency during training.

    Args:
        num_heads: Number of attention heads
        keep_ratio: Fraction of landmarks to select
        local_window: Size of local attention window
        num_hops: Number of radial hops
        alpha: TLS alpha (persistence vs diversity)
        is_causal: Whether to enforce causal masking
        cache_masks: Whether to cache masks (saves computation)
    """

    def __init__(
        self,
        num_heads: int,
        keep_ratio: float = 0.33,
        local_window: int = 128,
        num_hops: int = 2,
        alpha: float = 0.7,
        is_causal: bool = True,
        cache_masks: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.keep_ratio = keep_ratio
        self.local_window = local_window
        self.num_hops = num_hops
        self.alpha = alpha
        self.is_causal = is_causal
        self.cache_masks = cache_masks

        # Cache for landmarks and masks
        self._landmark_cache = {}
        self._mask_cache = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build sparse attention mask with TLS landmarks.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            use_cache: Whether to use cached landmarks/masks

        Returns:
            mask: [batch, num_heads, seq_len, seq_len]
            landmark_indices: [batch, num_heads, num_landmarks]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        cache_key = (batch_size, seq_len, device)

        # Check cache
        if use_cache and self.cache_masks and cache_key in self._mask_cache:
            return self._mask_cache[cache_key], self._landmark_cache[cache_key]

        # Select landmarks
        landmark_indices, _ = select_landmarks_per_head(
            hidden_states,
            num_heads=self.num_heads,
            keep_ratio=self.keep_ratio,
            alpha=self.alpha,
        )

        # Build mask
        mask = build_block_radial_mask(
            seq_len=seq_len,
            landmark_indices=landmark_indices,
            local_window=self.local_window,
            num_hops=self.num_hops,
            is_causal=self.is_causal,
            device=device,
        )

        # Cache if enabled
        if self.cache_masks:
            self._landmark_cache[cache_key] = landmark_indices
            self._mask_cache[cache_key] = mask

        return mask, landmark_indices

    def clear_cache(self):
        """Clear landmark and mask caches."""
        self._landmark_cache.clear()
        self._mask_cache.clear()


__all__ = [
    "compute_tls_scores",
    "select_landmarks_per_head",
    "build_block_radial_mask",
    "compute_sparsity",
    "TLSMaskBuilder",
]
