"""
Visualization tools for TF-A-N.

Generates:
- Attention heatmaps & sparsity patterns
- Persistence landscapes and diagrams
- EPR/LR/Temperature curves
- Emotion trajectories
- TTW diagnostics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import torch


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Attention Heatmap",
):
    """
    Plot attention heatmap.

    Args:
        attention_weights: (n_heads, seq_len, seq_len) or (seq_len, seq_len)
        save_path: Path to save figure
        title: Plot title
    """
    if attention_weights.dim() == 3:
        # Average over heads
        attn = attention_weights.mean(dim=0).cpu().numpy()
    else:
        attn = attention_weights.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, cmap="viridis", square=True)
    plt.title(title)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_persistence_diagram(
    diagram: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Persistence Diagram",
):
    """
    Plot persistence diagram.

    Args:
        diagram: (n_features, 2) array of (birth, death) pairs
        save_path: Path to save figure
        title: Plot title
    """
    if len(diagram) == 0:
        return

    plt.figure(figsize=(8, 8))
    births = diagram[:, 0]
    deaths = diagram[:, 1]

    # Plot diagonal
    max_val = max(deaths.max(), births.max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

    # Plot points
    plt.scatter(births, deaths, alpha=0.6)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title(title)
    plt.axis("equal")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_training_curves(
    metrics_history: List[Dict[str, float]],
    keys: List[str] = ["loss", "epr", "epr_cv", "lr", "temperature"],
    save_path: Optional[str] = None,
):
    """
    Plot training curves.

    Args:
        metrics_history: List of metric dictionaries
        keys: Metric keys to plot
        save_path: Path to save figure
    """
    n_keys = len(keys)
    fig, axes = plt.subplots(n_keys, 1, figsize=(12, 3 * n_keys))

    if n_keys == 1:
        axes = [axes]

    for i, key in enumerate(keys):
        values = [m.get(key, float('nan')) for m in metrics_history]
        axes[i].plot(values)
        axes[i].set_ylabel(key)
        axes[i].set_xlabel("Step")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_emotion_trajectory(
    valence_history: List[float],
    arousal_history: List[float],
    save_path: Optional[str] = None,
):
    """
    Plot emotion trajectory in VA space.

    Args:
        valence_history: Valence values over time
        arousal_history: Arousal values over time
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 8))

    # Plot trajectory
    plt.plot(valence_history, arousal_history, 'b-', alpha=0.5)
    plt.scatter(valence_history, arousal_history, c=range(len(valence_history)),
                cmap="viridis", s=20)

    # Mark start and end
    plt.scatter(valence_history[0], arousal_history[0], c='green', s=100,
                marker='o', label='Start')
    plt.scatter(valence_history[-1], arousal_history[-1], c='red', s=100,
                marker='X', label='End')

    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Emotion Trajectory")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.colorbar(label="Time Step")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_sparsity_pattern(
    attention_mask: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Plot attention sparsity pattern.

    Args:
        attention_mask: Boolean mask (seq_len, seq_len)
        save_path: Path to save figure
    """
    mask = attention_mask.cpu().numpy().astype(float)

    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap="binary", interpolation="nearest")
    plt.title(f"Sparsity: {(1 - mask.mean()):.1%}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
