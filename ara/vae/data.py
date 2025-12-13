"""
ara.vae.data - Dataset Generation for VAE Training

Provides utilities for generating labeled trajectory datasets from
HGF simulations across different pathological phenotypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Try importing PyTorch for Dataset class
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

# Import HGF components
try:
    from ara.hgf import (
        HGFAgent,
        HGFParams,
        VolatilitySwitchingTask,
        ReversalLearningTask,
    )
    from ara.hgf.pathology import (
        HEALTHY_BASELINE,
        SCHIZOPHRENIA_RIGID,
        SCHIZOPHRENIA_LOOSE,
        BPD_HIGH_KAPPA,
        ANXIETY_HIGH_PRECISION,
        DEPRESSION_LOW_PRECISION,
        PathologyPreset,
    )
    HAS_HGF = True
except ImportError:
    HAS_HGF = False


class PhenotypeLabel(Enum):
    """Phenotype labels for supervised learning."""
    HEALTHY = 0
    SCZ_RIGID = 1
    SCZ_LOOSE = 2
    BPD = 3
    ANXIETY = 4
    DEPRESSION = 5


@dataclass
class TrajectoryData:
    """Container for trajectory data with metadata."""

    # Trajectory features: [n_samples, n_trials, n_features]
    trajectories: np.ndarray

    # Ground truth factors: [n_samples, n_factors]
    # Factors: [omega_2, kappa_1, theta, phenotype_id]
    factors: np.ndarray

    # Phenotype labels (categorical)
    labels: np.ndarray

    # Feature names
    feature_names: List[str]

    # Factor names
    factor_names: List[str]

    # Metadata per sample
    metadata: Optional[List[Dict]] = None

    @property
    def n_samples(self) -> int:
        return self.trajectories.shape[0]

    @property
    def n_trials(self) -> int:
        return self.trajectories.shape[1]

    @property
    def n_features(self) -> int:
        return self.trajectories.shape[2]

    @property
    def n_factors(self) -> int:
        return self.factors.shape[1]


def generate_phenotype_dataset(
    n_samples_per_phenotype: int = 500,
    n_trials: int = 200,
    phenotypes: Optional[List[str]] = None,
    add_noise: float = 0.0,
    seed: int = 42,
) -> TrajectoryData:
    """
    Generate a dataset of HGF trajectories across phenotypes.

    Creates trajectories from different pathological presets, providing
    ground truth factors for disentanglement evaluation.

    Args:
        n_samples_per_phenotype: Samples per phenotype
        n_trials: Trials per trajectory
        phenotypes: List of phenotype codes (default: all)
        add_noise: Gaussian noise std to add to parameters
        seed: Random seed

    Returns:
        TrajectoryData with trajectories, factors, and labels
    """
    if not HAS_HGF:
        raise ImportError("ara.hgf required for dataset generation")

    np.random.seed(seed)

    # Default phenotypes
    if phenotypes is None:
        phenotypes = ["HEALTHY", "SCZ_RIGID", "SCZ_LOOSE", "BPD", "ANXIETY"]

    # Map to presets
    preset_map = {
        "HEALTHY": (HEALTHY_BASELINE, PhenotypeLabel.HEALTHY),
        "SCZ_RIGID": (SCHIZOPHRENIA_RIGID, PhenotypeLabel.SCZ_RIGID),
        "SCZ_LOOSE": (SCHIZOPHRENIA_LOOSE, PhenotypeLabel.SCZ_LOOSE),
        "BPD": (BPD_HIGH_KAPPA, PhenotypeLabel.BPD),
        "ANXIETY": (ANXIETY_HIGH_PRECISION, PhenotypeLabel.ANXIETY),
        "DEPRESSION": (DEPRESSION_LOW_PRECISION, PhenotypeLabel.DEPRESSION),
    }

    # Feature names (what we extract from each trial)
    feature_names = [
        "mu_2",      # Level 2 belief (logit)
        "sigma_2",   # Level 2 uncertainty
        "mu_3",      # Level 3 volatility
        "sigma_3",   # Level 3 uncertainty
        "delta_1",   # Sensory PE
        "delta_2",   # Volatility PE
        "pi_1",      # Sensory precision
        "pi_hat_2",  # Prior precision
    ]

    factor_names = ["omega_2", "kappa_1", "theta", "phenotype_id"]

    all_trajectories = []
    all_factors = []
    all_labels = []
    all_metadata = []

    for phenotype_code in phenotypes:
        if phenotype_code not in preset_map:
            raise ValueError(f"Unknown phenotype: {phenotype_code}")

        preset, label = preset_map[phenotype_code]

        for i in range(n_samples_per_phenotype):
            # Add small noise to parameters
            params = HGFParams(
                omega_2=preset.params.omega_2 + np.random.randn() * add_noise,
                kappa_1=preset.params.kappa_1 + np.random.randn() * add_noise * 0.1,
                theta=preset.params.theta + np.random.randn() * add_noise * 0.1,
            )

            # Create agent and task
            agent = HGFAgent(
                omega_2=params.omega_2,
                kappa_1=params.kappa_1,
                theta=params.theta,
            )

            task = VolatilitySwitchingTask(n_trials=n_trials)
            task_data = task.generate(seed=seed + i + label.value * 10000)

            # Run simulation
            trajectory = agent.run(task_data)

            # Extract features
            features = _extract_trajectory_features(trajectory, feature_names)
            all_trajectories.append(features)

            # Record factors
            all_factors.append([
                params.omega_2,
                params.kappa_1,
                params.theta,
                float(label.value),
            ])

            all_labels.append(label.value)

            all_metadata.append({
                "phenotype": phenotype_code,
                "preset_name": preset.name,
                "sample_idx": i,
                "task_seed": seed + i + label.value * 10000,
            })

    return TrajectoryData(
        trajectories=np.array(all_trajectories),
        factors=np.array(all_factors),
        labels=np.array(all_labels),
        feature_names=feature_names,
        factor_names=factor_names,
        metadata=all_metadata,
    )


def _extract_trajectory_features(
    trajectory,
    feature_names: List[str],
) -> np.ndarray:
    """Extract specified features from HGF trajectory."""

    n_trials = trajectory.n_trials
    features = np.zeros((n_trials, len(feature_names)))

    for t in range(n_trials):
        step = trajectory.steps[t]
        state = step.state

        for j, name in enumerate(feature_names):
            if hasattr(state, name):
                features[t, j] = getattr(state, name)
            else:
                features[t, j] = 0.0

    return features


def generate_parameter_sweep_dataset(
    omega_2_range: Tuple[float, float] = (-8.0, -1.0),
    kappa_1_range: Tuple[float, float] = (0.2, 3.0),
    theta_range: Tuple[float, float] = (0.5, 2.0),
    n_samples: int = 1000,
    n_trials: int = 200,
    seed: int = 42,
) -> TrajectoryData:
    """
    Generate dataset by sweeping HGF parameters uniformly.

    Useful for understanding the full parameter space without
    restricting to predefined phenotypes.

    Args:
        omega_2_range: Range for tonic volatility
        kappa_1_range: Range for coupling strength
        theta_range: Range for response temperature
        n_samples: Total number of samples
        n_trials: Trials per trajectory
        seed: Random seed

    Returns:
        TrajectoryData with trajectories and continuous factors
    """
    if not HAS_HGF:
        raise ImportError("ara.hgf required for dataset generation")

    np.random.seed(seed)

    feature_names = [
        "mu_2", "sigma_2", "mu_3", "sigma_3",
        "delta_1", "delta_2", "pi_1", "pi_hat_2",
    ]
    factor_names = ["omega_2", "kappa_1", "theta"]

    all_trajectories = []
    all_factors = []

    for i in range(n_samples):
        # Sample parameters uniformly
        omega_2 = np.random.uniform(*omega_2_range)
        kappa_1 = np.random.uniform(*kappa_1_range)
        theta = np.random.uniform(*theta_range)

        agent = HGFAgent(omega_2=omega_2, kappa_1=kappa_1, theta=theta)

        task = VolatilitySwitchingTask(n_trials=n_trials)
        task_data = task.generate(seed=seed + i)

        trajectory = agent.run(task_data)
        features = _extract_trajectory_features(trajectory, feature_names)

        all_trajectories.append(features)
        all_factors.append([omega_2, kappa_1, theta])

    return TrajectoryData(
        trajectories=np.array(all_trajectories),
        factors=np.array(all_factors),
        labels=np.zeros(n_samples, dtype=int),  # No labels for sweep
        feature_names=feature_names,
        factor_names=factor_names,
    )


if HAS_TORCH:

    class TrajectoryDataset(Dataset):
        """
        PyTorch Dataset for HGF trajectories.

        Wraps TrajectoryData for use with DataLoader.
        """

        def __init__(
            self,
            data: TrajectoryData,
            normalize: bool = True,
            return_factors: bool = True,
        ):
            """
            Initialize dataset.

            Args:
                data: TrajectoryData object
                normalize: Whether to normalize trajectories
                return_factors: Whether to return factors with each sample
            """
            self.data = data
            self.return_factors = return_factors

            # Convert to tensors
            self.trajectories = torch.FloatTensor(data.trajectories)
            self.factors = torch.FloatTensor(data.factors)
            self.labels = torch.LongTensor(data.labels)

            if normalize:
                self._normalize()

        def _normalize(self):
            """Normalize trajectories to zero mean, unit variance."""
            # Compute stats across samples and trials
            mean = self.trajectories.mean(dim=(0, 1), keepdim=True)
            std = self.trajectories.std(dim=(0, 1), keepdim=True) + 1e-8
            self.trajectories = (self.trajectories - mean) / std

            # Store for denormalization
            self.mean = mean
            self.std = std

        def __len__(self) -> int:
            return len(self.trajectories)

        def __getitem__(self, idx: int):
            if self.return_factors:
                return (
                    self.trajectories[idx],
                    self.factors[idx],
                    self.labels[idx],
                )
            return self.trajectories[idx]

        def denormalize(self, x: torch.Tensor) -> torch.Tensor:
            """Denormalize trajectories back to original scale."""
            return x * self.std + self.mean

else:

    class TrajectoryDataset:
        """Placeholder when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for TrajectoryDataset")


# =============================================================================
# Synthetic Benchmark Datasets (for disentanglement validation)
# =============================================================================

def generate_synthetic_disentangled(
    n_samples: int = 10000,
    n_factors: int = 5,
    latent_dim: int = 10,
    noise_level: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with known disentangled structure.

    Each factor maps to exactly one latent dimension (perfect modularity).
    Useful for validating disentanglement metrics.

    Args:
        n_samples: Number of samples
        n_factors: Number of ground truth factors
        latent_dim: Dimensionality of latent space
        noise_level: Gaussian noise to add
        seed: Random seed

    Returns:
        (latents, factors) tuple
    """
    np.random.seed(seed)

    # Generate factors (uniform in [0, 1])
    factors = np.random.rand(n_samples, n_factors)

    # Create perfectly disentangled mapping
    # Factor i maps to latent dim i
    latents = np.zeros((n_samples, latent_dim))
    for i in range(min(n_factors, latent_dim)):
        latents[:, i] = factors[:, i]

    # Add noise
    latents += np.random.randn(n_samples, latent_dim) * noise_level

    return latents, factors


def generate_synthetic_entangled(
    n_samples: int = 10000,
    n_factors: int = 5,
    latent_dim: int = 10,
    entanglement: float = 0.5,
    noise_level: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with controlled entanglement.

    Args:
        n_samples: Number of samples
        n_factors: Number of ground truth factors
        latent_dim: Dimensionality of latent space
        entanglement: 0 = disentangled, 1 = fully entangled
        noise_level: Gaussian noise to add
        seed: Random seed

    Returns:
        (latents, factors) tuple
    """
    np.random.seed(seed)

    # Generate factors
    factors = np.random.rand(n_samples, n_factors)

    # Create mixing matrix
    # Diagonal = disentangled, full = entangled
    mixing = np.eye(latent_dim, n_factors)

    # Add off-diagonal entries based on entanglement
    if entanglement > 0:
        off_diag = np.random.randn(latent_dim, n_factors) * entanglement
        mixing = (1 - entanglement) * mixing + entanglement * off_diag

    # Apply mixing
    latents = factors @ mixing.T

    # Add noise
    latents += np.random.randn(n_samples, latent_dim) * noise_level

    return latents, factors
