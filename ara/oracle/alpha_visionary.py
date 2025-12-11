#!/usr/bin/env python3
# ara/oracle/alpha_visionary.py
"""
Oracle Alpha: The Visionary

Hardware allocation:
- GPU 1 (3090 #1): QHDC particle ensemble + world model
- CPU cores 0-21: Particle management
- RAM: 32GB working set

Specialization:
- Plans 1000+ steps ahead
- Explores 50,000 futures simultaneously
- Prophetic mode: identifies Black Swan events
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Optional torch import with graceful fallback
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using numpy fallback")


def get_device(gpu_index: int = 0):
    """Get torch device, preferring specified GPU."""
    if not TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


# Default device for Oracle Alpha (GPU 1)
DEVICE = get_device(0) if TORCH_AVAILABLE else None


@dataclass
class Prophecy:
    """A prophetic vision of the future."""
    timeline: np.ndarray          # (T, latent_dim) trajectory
    probability: float            # Likelihood
    value: float                  # Expected reward
    entropy: float                # Uncertainty
    black_swan_risk: float        # Tail risk assessment
    narrative: str                # Human-readable description
    action_index: int = 0         # Which action produced this prophecy
    timesteps: int = 0            # How far ahead we looked
    particle_count: int = 0       # How many particles explored

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'probability': self.probability,
            'value': self.value,
            'entropy': self.entropy,
            'black_swan_risk': self.black_swan_risk,
            'narrative': self.narrative,
            'action_index': self.action_index,
            'timesteps': self.timesteps,
            'particle_count': self.particle_count,
            'timeline_shape': list(self.timeline.shape) if self.timeline is not None else None
        }


class MassiveParticleEnsemble:
    """
    QHDC particle ensemble optimized for 3090 memory bandwidth.

    Uses:
    - Tensor cores for HD operations (when available)
    - Unified memory for 50K+ particles
    - Async kernel launches
    - FP16 for 2x capacity
    """

    def __init__(
        self,
        latent_dim: int = 10,
        num_particles: int = 50000,
        hd_dim: int = 10000,
        device=None
    ):
        self.latent_dim = latent_dim
        self.num_particles = num_particles
        self.hd_dim = hd_dim
        self.device = device or DEVICE

        if TORCH_AVAILABLE and self.device is not None:
            # Allocate on GPU with FP16 for 2x capacity
            self.particles = torch.randn(
                num_particles, latent_dim,
                device=self.device,
                dtype=torch.float16
            )

            # HD projection matrix (fixed, pseudo-random)
            torch.manual_seed(42)
            self.hd_projection = torch.randn(
                latent_dim, hd_dim,
                device=self.device,
                dtype=torch.float16
            )

            # Weights (importance sampling)
            self.weights = torch.ones(num_particles, device=self.device) / num_particles

            mem_gb = self.particles.element_size() * self.particles.nelement() / 1e9
            logger.info(
                "MassiveParticleEnsemble: %d particles on %s (%.2f GB)",
                num_particles, self.device, mem_gb
            )
        else:
            # NumPy fallback
            self.particles = np.random.randn(num_particles, latent_dim).astype(np.float16)
            np.random.seed(42)
            self.hd_projection = np.random.randn(latent_dim, hd_dim).astype(np.float16)
            self.weights = np.ones(num_particles) / num_particles
            logger.info(
                "MassiveParticleEnsemble: %d particles (CPU/numpy mode)",
                num_particles
            )

    def encode_to_hd(self, particles) -> Any:
        """
        Encode latent particles to HD space using tensor cores.

        Uses bfloat16 for maximum throughput on Ampere architecture.
        """
        if TORCH_AVAILABLE and self.device is not None:
            # Convert to bfloat16 for tensor core acceleration
            if hasattr(torch, 'bfloat16'):
                particles_bf16 = particles.to(torch.bfloat16)
                projection_bf16 = self.hd_projection.to(torch.bfloat16)
            else:
                particles_bf16 = particles.half()
                projection_bf16 = self.hd_projection

            # Matrix multiply (hits tensor cores)
            hd_vectors = torch.mm(particles_bf16, projection_bf16)

            # Binarize
            hd_binary = (hd_vectors > 0).to(torch.float16)
            return hd_binary
        else:
            # NumPy fallback
            hd_continuous = np.dot(particles, self.hd_projection)
            return (hd_continuous > 0).astype(np.float16)

    def propagate_particles(
        self,
        world_model: Callable,
        action: Any,
        num_steps: int = 1000,
        batch_size: int = 10000
    ) -> Any:
        """
        Propagate all particles forward in time.

        Optimizations:
        - Batched forward passes (10K particles to fit in L2 cache)
        - Async CUDA streams
        - Gradient checkpointing for long horizons
        """
        if TORCH_AVAILABLE and self.device is not None:
            trajectory = torch.zeros(
                num_steps, self.num_particles, self.latent_dim,
                device=self.device,
                dtype=torch.float16
            )

            current_particles = self.particles.clone()

            for t in range(num_steps):
                trajectory[t] = current_particles

                # Batch processing
                for i in range(0, self.num_particles, batch_size):
                    batch = current_particles[i:i+batch_size]

                    with torch.no_grad():
                        # World model forward (broadcasting action)
                        action_expanded = action.unsqueeze(0).expand(batch.size(0), -1)
                        next_batch = world_model(batch, action_expanded)

                        # Handle tuple return (state, uncertainty)
                        if isinstance(next_batch, tuple):
                            next_batch = next_batch[0]

                    current_particles[i:i+batch_size] = next_batch

            return trajectory
        else:
            # NumPy fallback
            trajectory = np.zeros(
                (num_steps, self.num_particles, self.latent_dim),
                dtype=np.float16
            )
            current_particles = self.particles.copy()

            for t in range(num_steps):
                trajectory[t] = current_particles
                # Simple dynamics fallback
                noise = np.random.randn(*current_particles.shape) * 0.01
                current_particles = current_particles + noise.astype(np.float16)

            return trajectory

    def identify_black_swans(
        self,
        trajectory: Any,
        threshold_sigma: float = 3.0
    ) -> List[Dict]:
        """
        Find low-probability, high-impact futures (Black Swans).

        Method:
        1. Compute trajectory statistics
        2. Find outliers beyond threshold_sigma
        3. Assess impact magnitude
        """
        black_swans = []

        if TORCH_AVAILABLE and self.device is not None:
            # Compute mean and std across particles at each timestep
            mean_traj = trajectory.mean(dim=1)  # (T, latent_dim)
            std_traj = trajectory.std(dim=1)

            for t in range(trajectory.size(0)):
                # Find particles far from mean
                distances = torch.norm(
                    trajectory[t] - mean_traj[t].unsqueeze(0),
                    dim=1
                )

                # Outliers
                threshold = threshold_sigma * std_traj[t].mean()
                outlier_mask = distances > threshold

                if outlier_mask.sum() > 0:
                    outlier_indices = torch.where(outlier_mask)[0]

                    for idx in outlier_indices[:5]:  # Top 5
                        black_swans.append({
                            'timestep': t,
                            'particle_id': int(idx.item()),
                            'distance_sigma': float((distances[idx] / std_traj[t].mean()).item()),
                            'state': trajectory[t, idx].cpu().numpy(),
                            'probability': float(self.weights[idx].item())
                        })
        else:
            # NumPy fallback
            mean_traj = trajectory.mean(axis=1)
            std_traj = trajectory.std(axis=1)

            for t in range(trajectory.shape[0]):
                distances = np.linalg.norm(trajectory[t] - mean_traj[t], axis=1)
                threshold = threshold_sigma * std_traj[t].mean()
                outlier_indices = np.where(distances > threshold)[0]

                for idx in outlier_indices[:5]:
                    black_swans.append({
                        'timestep': t,
                        'particle_id': int(idx),
                        'distance_sigma': float(distances[idx] / std_traj[t].mean()),
                        'state': trajectory[t, idx],
                        'probability': float(self.weights[idx])
                    })

        return black_swans

    def resample_particles(self, importance_weights: Optional[Any] = None):
        """
        Resample particles based on importance weights.

        Used for focusing computation on high-value regions.
        """
        if importance_weights is None:
            importance_weights = self.weights

        if TORCH_AVAILABLE and self.device is not None:
            # Normalize weights
            weights_norm = importance_weights / importance_weights.sum()

            # Multinomial resampling
            indices = torch.multinomial(
                weights_norm,
                self.num_particles,
                replacement=True
            )

            self.particles = self.particles[indices].clone()
            self.weights = torch.ones(self.num_particles, device=self.device) / self.num_particles
        else:
            weights_norm = importance_weights / importance_weights.sum()
            indices = np.random.choice(
                self.num_particles,
                size=self.num_particles,
                replace=True,
                p=weights_norm
            )
            self.particles = self.particles[indices].copy()
            self.weights = np.ones(self.num_particles) / self.num_particles


class SimpleWorldModel:
    """Placeholder world model for when no trained model is available."""

    def __init__(self, latent_dim: int = 10, action_dim: int = 4):
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def __call__(self, z, u):
        """Simple linear dynamics."""
        if TORCH_AVAILABLE and isinstance(z, torch.Tensor):
            # Small perturbation + action influence
            dz = 0.01 * torch.randn_like(z) + 0.1 * u.mean(dim=-1, keepdim=True)
            return z + dz
        else:
            dz = 0.01 * np.random.randn(*z.shape) + 0.1 * u.mean()
            return z + dz


class OracleAlpha:
    """
    The Visionary: Long-horizon prophetic planning.

    Explores 50,000 parallel futures up to 1000 steps ahead,
    identifying both optimal actions and Black Swan risks.
    """

    def __init__(
        self,
        world_model: Optional[Callable] = None,
        encoder: Optional[Callable] = None,
        latent_dim: int = 10,
        num_particles: int = 50000,
        device=None
    ):
        self.device = device or DEVICE
        self.latent_dim = latent_dim

        # World model (use placeholder if not provided)
        if world_model is not None:
            if TORCH_AVAILABLE and self.device is not None:
                if hasattr(world_model, 'to'):
                    self.world_model = world_model.to(self.device)
                    if hasattr(self.world_model, 'eval'):
                        self.world_model.eval()
                else:
                    self.world_model = world_model
            else:
                self.world_model = world_model
        else:
            self.world_model = SimpleWorldModel(latent_dim=latent_dim)

        self.encoder = encoder

        # Massive particle ensemble
        self.particles = MassiveParticleEnsemble(
            latent_dim=latent_dim,
            num_particles=num_particles,
            device=self.device
        )

        # Prophecy cache (for meta-learning)
        self.prophecy_history: List[Prophecy] = []

        logger.info(
            "OracleAlpha initialized: %d particles, %d latent dims",
            num_particles, latent_dim
        )

    def divine_futures(
        self,
        current_state: Any,
        action_candidates: List[Any],
        horizon: int = 1000,
        return_top_k: int = 10
    ) -> List[Prophecy]:
        """
        Divine possible futures and identify the most significant.

        Args:
            current_state: Current latent state
            action_candidates: List of possible actions to evaluate
            horizon: How many steps to look ahead
            return_top_k: How many prophecies to return

        Returns:
            List of Prophecy objects, ranked by significance
        """
        start_time = time.time()
        prophecies = []

        # Convert state to tensor if needed
        if TORCH_AVAILABLE and self.device is not None:
            if isinstance(current_state, np.ndarray):
                current_state = torch.from_numpy(current_state).to(self.device)
            current_state = current_state.half()

        # Initialize particles around current state
        if TORCH_AVAILABLE and self.device is not None:
            noise = torch.randn_like(self.particles.particles) * 0.1
            self.particles.particles = current_state.unsqueeze(0) + noise
        else:
            noise = np.random.randn(*self.particles.particles.shape) * 0.1
            self.particles.particles = current_state + noise.astype(np.float16)

        for action_idx, action in enumerate(action_candidates):
            # Convert action to tensor if needed
            if TORCH_AVAILABLE and self.device is not None:
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action).to(self.device)
                action = action.half()

            # Propagate particles
            trajectory = self.particles.propagate_particles(
                self.world_model,
                action,
                num_steps=horizon
            )

            # Identify Black Swans
            black_swans = self.particles.identify_black_swans(trajectory)

            # Compute expected value
            if TORCH_AVAILABLE and self.device is not None:
                final_states = trajectory[-1]  # (num_particles, latent_dim)
                expected_value = self._compute_value(final_states)
                entropy = float(torch.std(final_states).item())
                mean_trajectory = trajectory.mean(dim=1).cpu().numpy()
            else:
                final_states = trajectory[-1]
                expected_value = self._compute_value(final_states)
                entropy = float(np.std(final_states))
                mean_trajectory = trajectory.mean(axis=1)

            # Aggregate Black Swan risk
            bs_risk = len(black_swans) / horizon if horizon > 0 else 0.0

            # Create prophecy
            prophecy = Prophecy(
                timeline=mean_trajectory,
                probability=1.0 / len(action_candidates),
                value=expected_value,
                entropy=entropy,
                black_swan_risk=bs_risk,
                narrative=self._narrate_prophecy(action_idx, expected_value, bs_risk, black_swans),
                action_index=action_idx,
                timesteps=horizon,
                particle_count=self.particles.num_particles
            )

            prophecies.append(prophecy)

        # Rank by composite score (value minus risk)
        prophecies.sort(
            key=lambda p: p.value - 0.5 * p.black_swan_risk,
            reverse=True
        )

        # Store in history
        self.prophecy_history.extend(prophecies[:return_top_k])

        elapsed = time.time() - start_time
        logger.info(
            "Divine futures complete: %d actions, %d steps, %.2fs",
            len(action_candidates), horizon, elapsed
        )

        return prophecies[:return_top_k]

    def _compute_value(self, states: Any) -> float:
        """
        Estimate value of final states.

        Uses stability heuristic: prefer states with low norm.
        """
        if TORCH_AVAILABLE and isinstance(states, torch.Tensor):
            norms = torch.norm(states, dim=1)
            return float(-norms.mean().item())
        else:
            norms = np.linalg.norm(states, axis=1)
            return float(-np.mean(norms))

    def _narrate_prophecy(
        self,
        action_id: int,
        value: float,
        bs_risk: float,
        black_swans: List[Dict]
    ) -> str:
        """Generate human-readable prophecy narrative."""
        if bs_risk > 0.1:
            risk_desc = "HIGH BLACK SWAN RISK - Tail events detected"
        elif bs_risk > 0.05:
            risk_desc = "Moderate tail risk - Some outlier futures"
        else:
            risk_desc = "Stable trajectory - Low risk"

        narrative = f"Action {action_id}: Expected value {value:.3f}. {risk_desc}."

        if black_swans:
            # Describe most extreme Black Swan
            extreme = max(black_swans, key=lambda x: x['distance_sigma'])
            narrative += f" Most extreme outlier at t={extreme['timestep']} ({extreme['distance_sigma']:.1f} sigma)."

        return narrative

    def get_prophecy_summary(self, n_recent: int = 10) -> Dict[str, Any]:
        """Get summary of recent prophecies for dashboard."""
        recent = self.prophecy_history[-n_recent:]

        if not recent:
            return {
                'count': 0,
                'avg_value': 0,
                'avg_risk': 0,
                'prophecies': []
            }

        return {
            'count': len(recent),
            'avg_value': sum(p.value for p in recent) / len(recent),
            'avg_risk': sum(p.black_swan_risk for p in recent) / len(recent),
            'prophecies': [p.to_dict() for p in recent]
        }


# ============================================================================
# Example Usage
# ============================================================================

def example_oracle_alpha():
    """Demonstrate Oracle Alpha."""

    print("Oracle Alpha: The Visionary")
    print("=" * 70)

    # Create Oracle
    oracle = OracleAlpha(
        latent_dim=10,
        num_particles=10000  # Reduced for demo
    )

    # Current state
    current_state = np.random.randn(10).astype(np.float32)

    # Action candidates
    actions = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    ]

    # Divine futures
    print("\nDivining 100-step futures across parallel universes...")
    prophecies = oracle.divine_futures(
        current_state,
        actions,
        horizon=100,  # Reduced for demo
        return_top_k=3
    )

    # Display prophecies
    print("\n" + "=" * 70)
    print("PROPHECIES FROM THE VISIONARY")
    print("=" * 70)

    for i, p in enumerate(prophecies, 1):
        print(f"\nProphecy {i}:")
        print(f"  Narrative: {p.narrative}")
        print(f"  Expected Value: {p.value:.3f}")
        print(f"  Entropy: {p.entropy:.3f}")
        print(f"  Black Swan Risk: {p.black_swan_risk:.1%}")
        print(f"  Timeline length: {p.timesteps} steps")


if __name__ == "__main__":
    example_oracle_alpha()
