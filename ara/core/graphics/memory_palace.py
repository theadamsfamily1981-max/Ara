"""
Ara Memory Palace - Attractor Visualization
============================================

Projects HTC attractor matrix into 3D space for visualization.

The Memory Palace shows:
- Attractors as points in 3D space
- Colors indicate teleological alignment
- Clustering shows learned associations
- Active attractors glow/pulse

This is the "brain slice" view - seeing the soul's learned patterns.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


def project_attractors(
    attractor_matrix: np.ndarray,
    teleology_values: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Project attractor HVs into 3D space for visualization.

    Uses random projection (Johnson-Lindenstrauss style) to map
    high-dimensional attractors to 3D while preserving relative distances.

    Args:
        attractor_matrix: (R, D) matrix of attractor HVs (binary or bipolar)
        teleology_values: (R,) or (R, K) teleology scores per attractor
        seed: Random seed for reproducible projection

    Returns:
        {
            "points_3d": np.ndarray of shape (R, 3),
            "teleology_colors": np.ndarray of shape (R, 3) RGB in 0..1,
        }
    """
    if attractor_matrix is None or len(attractor_matrix) == 0:
        return {
            "points_3d": np.zeros((0, 3), dtype=np.float32),
            "teleology_colors": np.zeros((0, 3), dtype=np.float32),
        }

    R, D = attractor_matrix.shape

    # Random projection matrix (fixed seed for consistency)
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(D, 3)).astype(np.float32)
    proj /= np.sqrt(D)  # Scale for variance preservation

    # Convert to float and center (if binary)
    A = attractor_matrix.astype(np.float32)
    if A.max() <= 1 and A.min() >= 0:
        A = 2 * A - 1  # {0,1} -> {-1,+1}

    # Project to 3D
    pts = A @ proj

    # Normalize to unit sphere for display
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + 1e-9)

    # Compute colors from teleology values
    if teleology_values is not None:
        colors = _teleology_to_colors(teleology_values)
    else:
        # Default: uniform gray
        colors = np.full((R, 3), 0.5, dtype=np.float32)

    return {
        "points_3d": pts.astype(np.float32),
        "teleology_colors": colors.astype(np.float32),
    }


def _teleology_to_colors(teleology_values: np.ndarray) -> np.ndarray:
    """
    Map teleology scores to RGB colors.

    Color scheme:
    - High alignment: Gold/warm (protection, purpose)
    - Neutral: Blue/cool (background, exploration)
    - Low/negative: Red (danger, misalignment)
    """
    # Handle multi-dimensional teleology
    if teleology_values.ndim > 1:
        score = teleology_values.mean(axis=1)
    else:
        score = teleology_values.copy()

    # Normalize to [0, 1]
    score_min = score.min()
    score_range = score.max() - score_min
    if score_range > 1e-6:
        score_norm = (score - score_min) / score_range
    else:
        score_norm = np.full_like(score, 0.5)

    R = len(score)
    colors = np.zeros((R, 3), dtype=np.float32)

    # Color mapping: blue (0) -> white (0.5) -> gold (1)
    for i in range(R):
        s = score_norm[i]
        if s < 0.5:
            # Blue to white
            t = s * 2
            colors[i] = [t, t, 1.0]  # [0,0,1] -> [1,1,1]
        else:
            # White to gold
            t = (s - 0.5) * 2
            colors[i] = [1.0, 1.0 - 0.2 * t, 1.0 - t]  # [1,1,1] -> [1,0.8,0]

    return colors


def compute_attractor_activity(
    resonance_vector: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute which attractors are currently active.

    Args:
        resonance_vector: (R,) resonance values per attractor
        threshold: Minimum resonance to consider "active"

    Returns:
        (active_mask, activity_levels) where:
        - active_mask: bool array of which attractors are active
        - activity_levels: normalized activity for glow intensity
    """
    if resonance_vector is None or len(resonance_vector) == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float32)

    active_mask = resonance_vector > threshold

    # Normalize activity levels for display
    max_res = resonance_vector.max()
    if max_res > 1e-6:
        activity_levels = resonance_vector / max_res
    else:
        activity_levels = np.zeros_like(resonance_vector)

    return active_mask, activity_levels.astype(np.float32)


def cluster_attractors(
    attractor_matrix: np.ndarray,
    n_clusters: int = 8,
) -> np.ndarray:
    """
    Simple clustering of attractors for color grouping.

    Args:
        attractor_matrix: (R, D) matrix of attractor HVs
        n_clusters: Number of clusters

    Returns:
        (R,) array of cluster assignments
    """
    if attractor_matrix is None or len(attractor_matrix) < n_clusters:
        return np.zeros(len(attractor_matrix) if attractor_matrix is not None else 0,
                       dtype=np.int32)

    # Simple k-means style clustering
    # (In production, use proper k-means or HDBSCAN)
    R, D = attractor_matrix.shape

    # Initialize centroids randomly
    rng = np.random.default_rng(42)
    centroid_idx = rng.choice(R, size=n_clusters, replace=False)
    centroids = attractor_matrix[centroid_idx].astype(np.float32)

    # Convert to float
    A = attractor_matrix.astype(np.float32)
    if A.max() <= 1 and A.min() >= 0:
        A = 2 * A - 1

    # Single iteration assignment (fast approximation)
    distances = np.zeros((R, n_clusters))
    for k in range(n_clusters):
        diff = A - centroids[k]
        distances[:, k] = np.sum(diff ** 2, axis=1)

    assignments = np.argmin(distances, axis=1)

    return assignments.astype(np.int32)


__all__ = [
    'project_attractors',
    'compute_attractor_activity',
    'cluster_attractors',
]
