"""
Encoders for converting TF-A-N metrics to visualization formats.
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

# Optional torch import for functions that need it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if TYPE_CHECKING:
    import torch


def encode_pd(
    diagram: np.ndarray,
    max_points: int = 200,
    normalize: bool = True
) -> List[Dict[str, float]]:
    """
    Encode persistence diagram for visualization.

    Args:
        diagram: [N, 2] array of (birth, death) pairs
        max_points: Maximum number of points to include
        normalize: Whether to normalize to [0, 1]

    Returns:
        List of {"b": birth, "d": death} dicts
    """
    if len(diagram) == 0:
        return []

    # Filter infinite points
    finite_mask = np.isfinite(diagram).all(axis=1)
    diagram = diagram[finite_mask]

    if len(diagram) == 0:
        return []

    # Sort by persistence (death - birth) descending
    persistence = diagram[:, 1] - diagram[:, 0]
    top_indices = np.argsort(persistence)[::-1][:max_points]
    diagram = diagram[top_indices]

    # Normalize if requested
    if normalize and len(diagram) > 0:
        min_val = diagram.min()
        max_val = diagram.max()
        if max_val > min_val:
            diagram = (diagram - min_val) / (max_val - min_val)

    # Convert to list of dicts
    return [
        {"b": float(b), "d": float(d)}
        for b, d in diagram
    ]


def encode_attention_matrix(
    attn: torch.Tensor,
    head_idx: int = 0,
    max_seq: int = 128,
    downsample: Optional[int] = None
) -> List[List[float]]:
    """
    Encode attention matrix for visualization.

    Args:
        attn: [batch, heads, seq, seq] or [heads, seq, seq] attention tensor
        head_idx: Which attention head to visualize
        max_seq: Maximum sequence length to encode
        downsample: If set, average-pool to this size

    Returns:
        2D list of attention weights
    """
    # Handle batch dimension
    if attn.dim() == 4:
        attn = attn[0]  # Take first in batch

    # Extract head
    if attn.dim() == 3:
        attn = attn[head_idx]

    # Limit sequence length
    seq_len = min(attn.size(0), max_seq)
    attn = attn[:seq_len, :seq_len]

    # Downsample if requested
    if downsample and seq_len > downsample:
        # Average pool
        factor = seq_len // downsample
        kernel = torch.ones(factor, factor) / (factor * factor)
        attn = torch.nn.functional.conv2d(
            attn.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            stride=factor
        )[0, 0]

    # Convert to list
    attn_np = attn.detach().cpu().numpy()
    return [[float(x) for x in row] for row in attn_np]


def encode_sparsity_metrics(
    kept_indices: torch.Tensor,
    total_length: int,
    compute_density_map: bool = False
) -> Dict[str, Any]:
    """
    Encode sparsity metrics.

    Args:
        kept_indices: Indices of kept tokens [N_kept]
        total_length: Total sequence length
        compute_density_map: Whether to compute spatial density

    Returns:
        Dict with sparsity info
    """
    kept_indices = kept_indices.detach().cpu().numpy()
    n_kept = len(kept_indices)
    sparsity = 1.0 - (n_kept / total_length)

    metrics = {
        "sparsity": float(sparsity),
        "kept_count": int(n_kept),
        "total_length": int(total_length),
        "kept_idx": [int(x) for x in kept_indices[:200]]  # Limit for transmission
    }

    if compute_density_map:
        # Compute density in bins
        n_bins = min(64, total_length)
        bin_size = total_length / n_bins
        density = np.zeros(n_bins)

        for idx in kept_indices:
            bin_idx = min(int(idx / bin_size), n_bins - 1)
            density[bin_idx] += 1

        metrics["density_map"] = [float(x) for x in density]

    return metrics


def encode_fdt_state(
    epr_cv: float,
    target_epr_cv: float,
    lr: float,
    lr_delta: float,
    integral: float,
    derivative: float
) -> Dict[str, float]:
    """
    Encode FDT controller state.

    Args:
        epr_cv: Current EPR-CV value
        target_epr_cv: Target EPR-CV
        lr: Current learning rate
        lr_delta: LR adjustment this step
        integral: Integral term
        derivative: Derivative term

    Returns:
        Dict with FDT state
    """
    return {
        "epr_cv": float(epr_cv),
        "target_epr_cv": float(target_epr_cv),
        "error": float(epr_cv - target_epr_cv),
        "lr": float(lr),
        "lr_delta": float(lr_delta),
        "integral": float(integral),
        "derivative": float(derivative),
    }


def encode_landmark_selection(
    scores: torch.Tensor,
    indices: torch.Tensor,
    method: str = "tls"
) -> Dict[str, Any]:
    """
    Encode landmark selection metrics.

    Args:
        scores: Selection scores [seq_len]
        indices: Selected indices [n_kept]
        method: Selection method ("tls", "uniform", "random")

    Returns:
        Dict with selection info
    """
    scores_np = scores.detach().cpu().numpy()
    indices_np = indices.detach().cpu().numpy()

    return {
        "method": method,
        "n_selected": len(indices_np),
        "mean_score": float(scores_np.mean()),
        "std_score": float(scores_np.std()),
        "min_score": float(scores_np.min()),
        "max_score": float(scores_np.max()),
        "selected_indices": [int(x) for x in indices_np[:200]],
        "score_histogram": _compute_histogram(scores_np, bins=20)
    }


def encode_layer_metrics(
    layer_idx: int,
    attn: torch.Tensor,
    pd: Optional[np.ndarray] = None,
    kept_indices: Optional[torch.Tensor] = None,
    seq_len: int = 0
) -> Dict[str, Any]:
    """
    Encode all metrics for a single layer.

    Args:
        layer_idx: Layer index
        attn: Attention tensor
        pd: Persistence diagram (optional)
        kept_indices: Kept token indices (optional)
        seq_len: Sequence length

    Returns:
        Comprehensive layer metrics
    """
    metrics = {
        "layer": layer_idx,
        "timestamp": float(np.datetime64('now').astype(float)),
    }

    # Attention
    if attn is not None:
        metrics["attention"] = encode_attention_matrix(attn, max_seq=64)
        metrics["attention_entropy"] = float(_compute_attention_entropy(attn))

    # Topology
    if pd is not None:
        metrics["pd"] = encode_pd(pd, max_points=100)

    # Sparsity
    if kept_indices is not None and seq_len > 0:
        metrics["sparsity"] = encode_sparsity_metrics(kept_indices, seq_len)

    return metrics


def _compute_histogram(data: np.ndarray, bins: int = 20) -> List[Dict[str, float]]:
    """Compute histogram for visualization."""
    counts, edges = np.histogram(data, bins=bins)
    return [
        {"bin_start": float(edges[i]), "bin_end": float(edges[i+1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]


def _compute_attention_entropy(attn: torch.Tensor) -> float:
    """Compute average entropy of attention distribution."""
    # attn: [heads, seq, seq]
    if attn.dim() == 4:
        attn = attn[0]  # First in batch

    # Entropy: -sum(p * log(p))
    eps = 1e-9
    log_attn = torch.log(attn + eps)
    entropy = -(attn * log_attn).sum(dim=-1)  # [heads, seq]

    return float(entropy.mean())
