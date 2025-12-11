# ara/cognition/clairvoyant/trajectory.py
"""
Trajectory Buffer - Sliding Window in 10D Hologram Space
=========================================================

Tracks Ara's path through the 10D latent space over time.

The trajectory is the key to clairvoyance:
- Where we are: current z_t point
- Where we've been: recent z_{t-k:t} history
- Where we're heading: velocity and predicted trajectory

This enables:
- Early warning: Approaching bad regions
- Regime detection: Which cluster we're in
- MPC planning: Choosing actions based on predicted futures

Usage:
    from ara.cognition.clairvoyant.trajectory import TrajectoryBuffer

    buffer = TrajectoryBuffer(max_len=600)  # 5 min at 0.5s ticks
    buffer.add(z_t)

    recent = buffer.get_recent(30)  # Last 15 seconds
    velocity = buffer.get_velocity()
    predicted = buffer.predict_future(steps=10)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """A single point in the trajectory with metadata."""
    z: np.ndarray                    # Latent vector
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_features: Optional[Dict[str, float]] = None
    regime: Optional[str] = None     # Classified regime at this point


@dataclass
class Trajectory:
    """A segment of trajectory with analysis."""
    points: List[TrajectoryPoint]
    start_time: datetime
    end_time: datetime

    @property
    def duration_seconds(self) -> float:
        """Duration of this trajectory segment."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def length(self) -> int:
        """Number of points."""
        return len(self.points)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array of shape (n_points, latent_dim)."""
        return np.array([p.z for p in self.points])

    def get_centroid(self) -> np.ndarray:
        """Get the centroid of this trajectory."""
        return np.mean(self.to_array(), axis=0)

    def get_total_distance(self) -> float:
        """Total Euclidean distance traveled."""
        arr = self.to_array()
        if len(arr) < 2:
            return 0.0
        diffs = np.diff(arr, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))


class TrajectoryBuffer:
    """
    Sliding window buffer for 10D trajectory points.

    Maintains a fixed-size history of recent latent space positions.
    Provides analysis methods for velocity, prediction, and curvature.
    """

    def __init__(
        self,
        max_len: int = 600,
        latent_dim: int = 10,
    ):
        """
        Initialize trajectory buffer.

        Args:
            max_len: Maximum points to keep (600 = 5 min at 0.5s)
            latent_dim: Dimensionality of latent space
        """
        self.max_len = max_len
        self.latent_dim = latent_dim
        self._points: deque = deque(maxlen=max_len)

        # Cached computations
        self._velocity_cache: Optional[np.ndarray] = None
        self._cache_valid: bool = False

    def add(
        self,
        z: np.ndarray,
        timestamp: Optional[datetime] = None,
        features: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None,
    ) -> None:
        """
        Add a new point to the trajectory.

        Args:
            z: Latent vector of shape (latent_dim,)
            timestamp: Time of this point
            features: Raw features that produced this point
            regime: Classified regime at this point
        """
        point = TrajectoryPoint(
            z=np.asarray(z),
            timestamp=timestamp or datetime.utcnow(),
            raw_features=features,
            regime=regime,
        )
        self._points.append(point)
        self._cache_valid = False

    def get_current(self) -> Optional[np.ndarray]:
        """Get the most recent latent point."""
        if not self._points:
            return None
        return self._points[-1].z

    def get_current_point(self) -> Optional[TrajectoryPoint]:
        """Get the most recent trajectory point with metadata."""
        if not self._points:
            return None
        return self._points[-1]

    def get_recent(self, n: Optional[int] = None) -> List[np.ndarray]:
        """
        Get recent latent vectors.

        Args:
            n: Number of points (None = all)

        Returns:
            List of latent vectors, oldest first
        """
        points = list(self._points)
        if n is not None and n < len(points):
            points = points[-n:]
        return [p.z for p in points]

    def get_recent_points(self, n: Optional[int] = None) -> List[TrajectoryPoint]:
        """Get recent trajectory points with metadata."""
        points = list(self._points)
        if n is not None and n < len(points):
            points = points[-n:]
        return points

    def to_array(self, n: Optional[int] = None) -> np.ndarray:
        """
        Get trajectory as numpy array.

        Args:
            n: Number of recent points (None = all)

        Returns:
            Array of shape (n_points, latent_dim)
        """
        vectors = self.get_recent(n)
        if not vectors:
            return np.zeros((0, self.latent_dim))
        return np.array(vectors)

    def get_velocity(self, window: int = 5) -> np.ndarray:
        """
        Compute velocity (direction of movement) in latent space.

        Uses linear regression over recent points for smoothing.

        Args:
            window: Number of recent points to use

        Returns:
            Velocity vector of shape (latent_dim,)
        """
        if self._cache_valid and self._velocity_cache is not None:
            return self._velocity_cache

        points = self.to_array(window)
        if len(points) < 2:
            self._velocity_cache = np.zeros(self.latent_dim)
            self._cache_valid = True
            return self._velocity_cache

        # Linear regression: z = a + b*t
        # b is the velocity
        n = len(points)
        t = np.arange(n)
        t_mean = t.mean()
        z_mean = points.mean(axis=0)

        # Slope in each dimension
        numerator = np.sum((t[:, None] - t_mean) * (points - z_mean), axis=0)
        denominator = np.sum((t - t_mean) ** 2)

        if denominator > 1e-10:
            velocity = numerator / denominator
        else:
            velocity = np.zeros(self.latent_dim)

        self._velocity_cache = velocity
        self._cache_valid = True
        return velocity

    def get_speed(self, window: int = 5) -> float:
        """Get magnitude of velocity (how fast we're moving)."""
        v = self.get_velocity(window)
        return float(np.linalg.norm(v))

    def get_acceleration(self, window: int = 10) -> np.ndarray:
        """
        Compute acceleration (change in velocity).

        Args:
            window: Number of points for velocity estimation

        Returns:
            Acceleration vector
        """
        if len(self._points) < window + 2:
            return np.zeros(self.latent_dim)

        # Velocity at two time points
        recent = self.to_array(window)
        older = self.to_array(window * 2)[:-window]

        if len(older) < 2:
            return np.zeros(self.latent_dim)

        # Velocity from recent half
        v1 = self._compute_velocity(older)
        v2 = self._compute_velocity(recent)

        return v2 - v1

    def _compute_velocity(self, points: np.ndarray) -> np.ndarray:
        """Helper to compute velocity from points array."""
        if len(points) < 2:
            return np.zeros(self.latent_dim)

        n = len(points)
        t = np.arange(n)
        t_mean = t.mean()
        z_mean = points.mean(axis=0)

        numerator = np.sum((t[:, None] - t_mean) * (points - z_mean), axis=0)
        denominator = np.sum((t - t_mean) ** 2)

        if denominator > 1e-10:
            return numerator / denominator
        return np.zeros(self.latent_dim)

    def predict_future(
        self,
        steps: int = 10,
        method: str = "linear",
    ) -> np.ndarray:
        """
        Predict future trajectory points.

        Args:
            steps: Number of future steps to predict
            method: Prediction method ("linear" or "quadratic")

        Returns:
            Array of shape (steps, latent_dim)
        """
        current = self.get_current()
        if current is None:
            return np.zeros((steps, self.latent_dim))

        velocity = self.get_velocity()

        if method == "quadratic":
            accel = self.get_acceleration()
            # z(t) = z0 + v*t + 0.5*a*t^2
            t = np.arange(1, steps + 1)[:, None]
            predictions = current + velocity * t + 0.5 * accel * (t ** 2)
        else:
            # Linear: z(t) = z0 + v*t
            t = np.arange(1, steps + 1)[:, None]
            predictions = current + velocity * t

        return predictions

    def distance_to_point(self, target: np.ndarray) -> float:
        """Compute current distance to a target point."""
        current = self.get_current()
        if current is None:
            return float('inf')
        return float(np.linalg.norm(current - target))

    def time_to_region(
        self,
        center: np.ndarray,
        radius: float,
        max_steps: int = 100,
    ) -> Optional[int]:
        """
        Estimate steps until entering a spherical region.

        Args:
            center: Center of target region
            radius: Radius of target region
            max_steps: Maximum steps to check

        Returns:
            Number of steps until entering region, or None if not approaching
        """
        predictions = self.predict_future(max_steps)

        for i, pred in enumerate(predictions):
            dist = np.linalg.norm(pred - center)
            if dist <= radius:
                return i + 1

        return None

    def get_trajectory_segment(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> Trajectory:
        """Extract a trajectory segment."""
        points = list(self._points)[start_idx:end_idx]
        if not points:
            return Trajectory(
                points=[],
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
            )

        return Trajectory(
            points=points,
            start_time=points[0].timestamp,
            end_time=points[-1].timestamp,
        )

    def clear(self) -> None:
        """Clear all points."""
        self._points.clear()
        self._cache_valid = False

    def __len__(self) -> int:
        return len(self._points)


# =============================================================================
# Trajectory Analysis
# =============================================================================

def compute_trajectory_curvature(points: np.ndarray) -> float:
    """
    Compute average curvature of a trajectory.

    High curvature = erratic movement
    Low curvature = smooth path

    Args:
        points: Array of shape (n_points, latent_dim)

    Returns:
        Average curvature value
    """
    if len(points) < 3:
        return 0.0

    # Compute tangent vectors
    tangents = np.diff(points, axis=0)

    # Normalize
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    tangents = tangents / norms

    # Curvature = rate of change of tangent direction
    if len(tangents) < 2:
        return 0.0

    curvatures = np.linalg.norm(np.diff(tangents, axis=0), axis=1)
    return float(np.mean(curvatures))


def compute_trajectory_entropy(points: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute entropy of trajectory (how spread out it is).

    High entropy = exploring widely
    Low entropy = concentrated in small region

    Args:
        points: Array of shape (n_points, latent_dim)
        n_bins: Number of histogram bins per dimension

    Returns:
        Entropy value
    """
    if len(points) < 2:
        return 0.0

    # Discretize each dimension
    total_entropy = 0.0
    for d in range(points.shape[1]):
        values = points[:, d]
        hist, _ = np.histogram(values, bins=n_bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        if len(hist) > 0:
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            total_entropy += entropy

    return total_entropy / points.shape[1]


# =============================================================================
# Testing
# =============================================================================

def _test_trajectory():
    """Test trajectory buffer."""
    print("=" * 60)
    print("Trajectory Buffer Test")
    print("=" * 60)

    buffer = TrajectoryBuffer(max_len=100, latent_dim=10)

    # Simulate a trajectory that moves then stabilizes
    np.random.seed(42)
    z = np.zeros(10)

    for i in range(50):
        # Move in direction [1, 0.5, 0, ...]
        z = z + np.array([0.1, 0.05, 0, 0, 0, 0, 0, 0, 0, 0]) + np.random.randn(10) * 0.01
        buffer.add(z)

    print(f"Buffer length: {len(buffer)}")
    print(f"Current position: {buffer.get_current()[:3]}...")
    print(f"Velocity: {buffer.get_velocity()[:3]}...")
    print(f"Speed: {buffer.get_speed():.4f}")

    # Predict future
    future = buffer.predict_future(steps=5)
    print(f"\nPredicted future (5 steps):")
    for i, f in enumerate(future):
        print(f"  t+{i+1}: {f[:3]}...")

    # Trajectory analysis
    arr = buffer.to_array()
    curvature = compute_trajectory_curvature(arr)
    entropy = compute_trajectory_entropy(arr)
    print(f"\nTrajectory curvature: {curvature:.4f}")
    print(f"Trajectory entropy: {entropy:.4f}")


if __name__ == "__main__":
    _test_trajectory()
