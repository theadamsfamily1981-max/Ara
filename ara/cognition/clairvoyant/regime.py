# ara/cognition/clairvoyant/regime.py
"""
Regime Classifier - Labeling States in 10D Hologram Space
==========================================================

Classifies the current 10D latent point into a "regime" - a named region
of state space with known characteristics.

Regimes emerge from clustering historical trajectories:
- FLOW: Deep work, everything smooth
- IDLE: System quiet, user away
- BUILDING: Active development, moderate load
- DEBUGGING: High attention, searching behavior
- PRE_CRASH: Warning signs, heading toward problems
- OVERLOAD: System stressed, intervention needed
- RECOVERY: Coming back from problems

The classifier uses:
1. K-means or GMM clustering (trained offline)
2. Per-cluster labels from historical outcomes
3. Confidence scores for uncertainty quantification

Usage:
    from ara.cognition.clairvoyant.regime import RegimeClassifier, RegimeType

    classifier = RegimeClassifier()
    classifier.load_from_training()

    regime = classifier.classify(z_t)
    print(f"Current regime: {regime.type.name}, confidence: {regime.confidence:.2f}")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Regime Types
# =============================================================================

class RegimeType(Enum):
    """
    Named operating regimes in Ara's state space.

    Good regimes (to seek):
    - FLOW: Deep work state, high productivity
    - BUILDING: Active development, good progress
    - IDLE: Healthy rest state

    Neutral regimes:
    - EXPLORING: Searching, researching
    - DEBUGGING: Problem-solving mode

    Warning regimes (to avoid):
    - PRE_CRASH: Early warning, heading toward problems
    - OVERLOAD: System stressed, needs intervention
    - BURNOUT: User showing fatigue signs
    - DOOMSCROLL: Unproductive browsing

    Recovery regimes:
    - RECOVERY: Coming back from problems
    - COOLDOWN: Post-intense-work stabilization

    Unknown:
    - UNKNOWN: Can't classify (out of distribution)
    """
    # Good
    FLOW = auto()
    BUILDING = auto()
    IDLE = auto()

    # Neutral
    EXPLORING = auto()
    DEBUGGING = auto()
    EXPERIMENTING = auto()

    # Warning
    PRE_CRASH = auto()
    OVERLOAD = auto()
    BURNOUT = auto()
    DOOMSCROLL = auto()
    CONTEXT_SWITCH_HELL = auto()

    # Recovery
    RECOVERY = auto()
    COOLDOWN = auto()

    # Unknown
    UNKNOWN = auto()


# Regime categories for policy decisions
GOOD_REGIMES = {RegimeType.FLOW, RegimeType.BUILDING, RegimeType.IDLE}
WARNING_REGIMES = {RegimeType.PRE_CRASH, RegimeType.OVERLOAD, RegimeType.BURNOUT, RegimeType.DOOMSCROLL}
NEUTRAL_REGIMES = {RegimeType.EXPLORING, RegimeType.DEBUGGING, RegimeType.EXPERIMENTING}
RECOVERY_REGIMES = {RegimeType.RECOVERY, RegimeType.COOLDOWN}


@dataclass
class Regime:
    """Classification result for a single latent point."""
    type: RegimeType
    confidence: float           # 0-1, how certain we are
    cluster_id: int             # Which cluster this maps to
    distance_to_center: float   # Distance to cluster centroid
    probabilities: Dict[RegimeType, float] = field(default_factory=dict)

    @property
    def is_good(self) -> bool:
        return self.type in GOOD_REGIMES

    @property
    def is_warning(self) -> bool:
        return self.type in WARNING_REGIMES

    @property
    def is_neutral(self) -> bool:
        return self.type in NEUTRAL_REGIMES

    @property
    def needs_intervention(self) -> bool:
        """Whether this regime suggests intervention."""
        return self.type in WARNING_REGIMES and self.confidence > 0.6


@dataclass
class ClusterInfo:
    """Information about a trained cluster."""
    cluster_id: int
    centroid: np.ndarray
    label: RegimeType
    sample_count: int = 0
    avg_distance: float = 0.0
    std_distance: float = 0.0


# =============================================================================
# Regime Classifier
# =============================================================================

class RegimeClassifier:
    """
    Classifies 10D latent points into named regimes.

    Training (offline):
    1. Collect many hypervectors / latent points
    2. Cluster them (k-means, GMM)
    3. Label each cluster based on outcomes (manual or semi-supervised)

    Inference (online):
    1. Find nearest cluster centroid
    2. Compute confidence (based on distance)
    3. Return regime label
    """

    def __init__(
        self,
        n_clusters: int = 12,
        latent_dim: int = 10,
    ):
        """
        Initialize classifier.

        Args:
            n_clusters: Number of clusters to use
            latent_dim: Dimensionality of latent space
        """
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        # Cluster parameters (set after training)
        self._centroids: Optional[np.ndarray] = None  # (n_clusters, latent_dim)
        self._labels: List[RegimeType] = []           # Label for each cluster
        self._cluster_stats: List[ClusterInfo] = []

        # For soft classification
        self._use_soft: bool = False
        self._covariances: Optional[np.ndarray] = None  # For GMM

    @property
    def is_trained(self) -> bool:
        return self._centroids is not None and len(self._labels) > 0

    def train(
        self,
        latent_points: np.ndarray,
        labels: Optional[List[RegimeType]] = None,
        method: str = "kmeans",
    ) -> None:
        """
        Train the classifier on latent point samples.

        Args:
            latent_points: Array of shape (n_samples, latent_dim)
            labels: Optional per-sample labels for semi-supervised
            method: "kmeans" or "gmm"
        """
        n_samples = len(latent_points)
        logger.info(f"Training regime classifier: {n_samples} samples, {method}")

        if method == "kmeans":
            self._train_kmeans(latent_points, labels)
        elif method == "gmm":
            self._train_gmm(latent_points, labels)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _train_kmeans(
        self,
        points: np.ndarray,
        labels: Optional[List[RegimeType]] = None,
    ) -> None:
        """Train using k-means clustering."""
        # Simple k-means implementation
        n_samples = len(points)
        k = min(self.n_clusters, n_samples)

        # Initialize centroids (k-means++ style)
        rng = np.random.default_rng(42)
        centroids = [points[rng.integers(n_samples)]]

        for _ in range(k - 1):
            # Distance to nearest centroid
            dists = np.min([
                np.sum((points - c) ** 2, axis=1)
                for c in centroids
            ], axis=0)

            # Sample proportional to squared distance
            probs = dists / dists.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(points[next_idx])

        centroids = np.array(centroids)

        # Iterate
        for iteration in range(100):
            # Assign points to nearest centroid
            distances = np.array([
                np.sum((points - c) ** 2, axis=1)
                for c in centroids
            ]).T  # (n_samples, k)

            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centroids[i] = points[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        self._centroids = centroids

        # Compute cluster stats
        self._cluster_stats = []
        for i in range(k):
            mask = assignments == i
            cluster_points = points[mask]

            stats = ClusterInfo(
                cluster_id=i,
                centroid=centroids[i],
                label=RegimeType.UNKNOWN,
                sample_count=len(cluster_points),
            )

            if len(cluster_points) > 0:
                dists = np.linalg.norm(cluster_points - centroids[i], axis=1)
                stats.avg_distance = float(dists.mean())
                stats.std_distance = float(dists.std())

            self._cluster_stats.append(stats)

        # Assign labels
        if labels is not None:
            self._assign_labels_from_samples(points, labels, assignments)
        else:
            self._assign_default_labels()

        logger.info(f"K-means trained: {k} clusters")

    def _train_gmm(
        self,
        points: np.ndarray,
        labels: Optional[List[RegimeType]] = None,
    ) -> None:
        """Train using Gaussian Mixture Model."""
        # Simplified GMM (diagonal covariance)
        # For production, use sklearn.mixture.GaussianMixture

        # First run k-means to initialize
        self._train_kmeans(points, labels)

        # Use k-means centroids as GMM means
        self._use_soft = True

        # Compute per-cluster covariances
        k = len(self._centroids)
        assignments = self._get_assignments(points)

        covariances = []
        for i in range(k):
            mask = assignments == i
            cluster_points = points[mask]

            if len(cluster_points) > 1:
                # Diagonal covariance
                cov = np.var(cluster_points, axis=0) + 1e-6
            else:
                cov = np.ones(self.latent_dim)

            covariances.append(cov)

        self._covariances = np.array(covariances)

    def _get_assignments(self, points: np.ndarray) -> np.ndarray:
        """Get cluster assignments for points."""
        distances = np.array([
            np.sum((points - c) ** 2, axis=1)
            for c in self._centroids
        ]).T
        return np.argmin(distances, axis=1)

    def _assign_labels_from_samples(
        self,
        points: np.ndarray,
        labels: List[RegimeType],
        assignments: np.ndarray,
    ) -> None:
        """Assign cluster labels based on majority vote from sample labels."""
        k = len(self._centroids)

        for i in range(k):
            mask = assignments == i
            cluster_labels = [labels[j] for j, m in enumerate(mask) if m]

            if cluster_labels:
                # Majority vote
                from collections import Counter
                counts = Counter(cluster_labels)
                majority_label = counts.most_common(1)[0][0]
                self._cluster_stats[i].label = majority_label
            else:
                self._cluster_stats[i].label = RegimeType.UNKNOWN

        self._labels = [s.label for s in self._cluster_stats]

    def _assign_default_labels(self) -> None:
        """Assign default labels based on cluster characteristics."""
        # Simple heuristic: label by distance from origin
        # (assumes origin = idle state)

        k = len(self._centroids)
        distances_from_origin = np.linalg.norm(self._centroids, axis=1)
        sorted_idx = np.argsort(distances_from_origin)

        # Assign labels in order of distance
        label_sequence = [
            RegimeType.IDLE,
            RegimeType.COOLDOWN,
            RegimeType.EXPLORING,
            RegimeType.BUILDING,
            RegimeType.FLOW,
            RegimeType.DEBUGGING,
            RegimeType.EXPERIMENTING,
            RegimeType.CONTEXT_SWITCH_HELL,
            RegimeType.PRE_CRASH,
            RegimeType.OVERLOAD,
            RegimeType.BURNOUT,
            RegimeType.RECOVERY,
        ]

        self._labels = [RegimeType.UNKNOWN] * k
        for rank, idx in enumerate(sorted_idx):
            if rank < len(label_sequence):
                self._labels[idx] = label_sequence[rank]
            self._cluster_stats[idx].label = self._labels[idx]

    def classify(self, z: np.ndarray) -> Regime:
        """
        Classify a latent point into a regime.

        Args:
            z: Latent vector of shape (latent_dim,)

        Returns:
            Regime object with classification result
        """
        if not self.is_trained:
            return Regime(
                type=RegimeType.UNKNOWN,
                confidence=0.0,
                cluster_id=-1,
                distance_to_center=float('inf'),
            )

        z = np.asarray(z)

        # Compute distances to all centroids
        distances = np.linalg.norm(self._centroids - z, axis=1)

        # Nearest cluster
        cluster_id = int(np.argmin(distances))
        min_dist = distances[cluster_id]

        # Confidence based on distance relative to cluster stats
        stats = self._cluster_stats[cluster_id]
        if stats.std_distance > 0:
            # Z-score based confidence
            z_score = (min_dist - stats.avg_distance) / stats.std_distance
            confidence = 1.0 / (1.0 + np.exp(z_score))  # Sigmoid
        else:
            confidence = 1.0 if min_dist < 1.0 else 0.5

        # Compute soft probabilities if GMM
        probabilities = {}
        if self._use_soft and self._covariances is not None:
            probs = self._compute_gmm_probs(z)
            for i, label in enumerate(self._labels):
                probabilities[label] = float(probs[i])

        return Regime(
            type=self._labels[cluster_id],
            confidence=float(np.clip(confidence, 0, 1)),
            cluster_id=cluster_id,
            distance_to_center=float(min_dist),
            probabilities=probabilities,
        )

    def _compute_gmm_probs(self, z: np.ndarray) -> np.ndarray:
        """Compute GMM posterior probabilities."""
        k = len(self._centroids)
        log_probs = []

        for i in range(k):
            diff = z - self._centroids[i]
            var = self._covariances[i]

            # Log probability under diagonal Gaussian
            log_p = -0.5 * np.sum(diff ** 2 / var) - 0.5 * np.sum(np.log(var))
            log_probs.append(log_p)

        # Softmax
        log_probs = np.array(log_probs)
        log_probs -= log_probs.max()  # Numerical stability
        probs = np.exp(log_probs)
        probs /= probs.sum()

        return probs

    def get_cluster_info(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Get information about a cluster."""
        if 0 <= cluster_id < len(self._cluster_stats):
            return self._cluster_stats[cluster_id]
        return None

    def distance_to_regime(self, z: np.ndarray, regime: RegimeType) -> float:
        """Compute distance from z to nearest cluster of given regime type."""
        if not self.is_trained:
            return float('inf')

        z = np.asarray(z)
        min_dist = float('inf')

        for i, label in enumerate(self._labels):
            if label == regime:
                dist = np.linalg.norm(z - self._centroids[i])
                min_dist = min(min_dist, dist)

        return min_dist

    def save(self, path: Union[str, Path]) -> None:
        """Save classifier to file."""
        path = Path(path)
        data = {
            "n_clusters": self.n_clusters,
            "latent_dim": self.latent_dim,
            "centroids": self._centroids,
            "labels": [l.value for l in self._labels],
            "cluster_stats": [
                {
                    "cluster_id": s.cluster_id,
                    "centroid": s.centroid,
                    "label": s.label.value,
                    "sample_count": s.sample_count,
                    "avg_distance": s.avg_distance,
                    "std_distance": s.std_distance,
                }
                for s in self._cluster_stats
            ],
            "use_soft": self._use_soft,
            "covariances": self._covariances,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Regime classifier saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load classifier from file."""
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.n_clusters = data["n_clusters"]
        self.latent_dim = data["latent_dim"]
        self._centroids = data["centroids"]
        self._labels = [RegimeType(v) for v in data["labels"]]
        self._cluster_stats = [
            ClusterInfo(
                cluster_id=s["cluster_id"],
                centroid=s["centroid"],
                label=RegimeType(s["label"]),
                sample_count=s["sample_count"],
                avg_distance=s["avg_distance"],
                std_distance=s["std_distance"],
            )
            for s in data["cluster_stats"]
        ]
        self._use_soft = data.get("use_soft", False)
        self._covariances = data.get("covariances")

        logger.info(f"Regime classifier loaded from {path}")


# =============================================================================
# Testing
# =============================================================================

def _test_classifier():
    """Test regime classifier."""
    print("=" * 60)
    print("Regime Classifier Test")
    print("=" * 60)

    # Generate synthetic data with 3 clear regimes
    np.random.seed(42)
    n_per_regime = 100
    latent_dim = 10

    # FLOW: clustered around [1, 1, 0, ...]
    flow_points = np.random.randn(n_per_regime, latent_dim) * 0.3
    flow_points[:, 0] += 2
    flow_points[:, 1] += 2

    # IDLE: near origin
    idle_points = np.random.randn(n_per_regime, latent_dim) * 0.3

    # OVERLOAD: far from origin
    overload_points = np.random.randn(n_per_regime, latent_dim) * 0.3
    overload_points[:, 0] += 5
    overload_points[:, 2] += 3

    # Combine
    all_points = np.vstack([flow_points, idle_points, overload_points])
    all_labels = (
        [RegimeType.FLOW] * n_per_regime +
        [RegimeType.IDLE] * n_per_regime +
        [RegimeType.OVERLOAD] * n_per_regime
    )

    # Train classifier
    classifier = RegimeClassifier(n_clusters=5)
    classifier.train(all_points, labels=all_labels)

    # Test classification
    test_points = [
        (np.array([2.0, 2.0] + [0.0] * 8), "Expected: FLOW"),
        (np.array([0.0] * 10), "Expected: IDLE"),
        (np.array([5.0, 0.0, 3.0] + [0.0] * 7), "Expected: OVERLOAD"),
    ]

    print("\nClassification results:")
    for point, expected in test_points:
        regime = classifier.classify(point)
        print(f"  {expected}")
        print(f"    -> {regime.type.name} (conf={regime.confidence:.2f}, dist={regime.distance_to_center:.2f})")

    # Save/load test
    classifier.save("/tmp/regime_test.pkl")
    classifier2 = RegimeClassifier()
    classifier2.load("/tmp/regime_test.pkl")
    print(f"\nSave/load: {classifier2.is_trained}")


if __name__ == "__main__":
    _test_classifier()
