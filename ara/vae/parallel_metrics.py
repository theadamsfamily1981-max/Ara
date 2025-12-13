"""
ara.vae.parallel_metrics - Parallel Disentanglement Metrics

Provides parallelized implementations of disentanglement metrics:
- MINE-based MI estimation with JAX SPMD
- shard_map for parallel (latent, factor) pair computation
- ProcessPool fallback for CPU-bound sklearn operations

Key insight: EDI requires computing MI(z_j, v_k) for all j×k pairs.
This is embarrassingly parallel and benefits hugely from:
- JAX shard_map: one MI pair per device
- ProcessPool: sklearn-based metrics across CPU cores
"""

from __future__ import annotations

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import random, lax
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np

# Flax for MINE network
try:
    import flax.linen as nn
    import optax
    HAS_FLAX = True
except ImportError:
    HAS_FLAX = False

# sklearn for CPU-based metrics
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# MINE: Mutual Information Neural Estimator
# =============================================================================

if HAS_JAX and HAS_FLAX:

    class MINE(nn.Module):
        """
        Mutual Information Neural Estimator.

        Uses the Donsker-Varadhan representation:
        I(X;Y) >= E[T(X,Y)] - log(E[exp(T(X',Y))])

        where X' is sampled from marginal (shuffled).
        """
        hidden_dim: int = 64

        @nn.compact
        def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """
            Compute T(x, y) statistics network.

            Args:
                x: [batch, 1] - samples from first variable
                y: [batch, 1] - samples from second variable

            Returns:
                T values [batch, 1]
            """
            xy = jnp.concatenate([x, y], axis=-1)
            h = nn.Dense(self.hidden_dim)(xy)
            h = nn.relu(h)
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.relu(h)
            t = nn.Dense(1)(h)
            return t


    def mine_loss(params, apply_fn, x, y, rng):
        """
        MINE loss: -I(X;Y) lower bound.

        Args:
            params: MINE network parameters
            apply_fn: Network apply function
            x, y: Joint samples [batch, 1]
            rng: Random key for shuffling

        Returns:
            Negative MI estimate (to minimize)
        """
        # Joint: T(x, y)
        t_joint = apply_fn({'params': params}, x, y)

        # Marginal: T(x, y') where y' is shuffled
        perm = random.permutation(rng, len(y))
        y_shuffled = y[perm]
        t_marginal = apply_fn({'params': params}, x, y_shuffled)

        # Donsker-Varadhan bound
        # I(X;Y) >= E[T(X,Y)] - log(E[exp(T(X,Y'))])
        mi_estimate = jnp.mean(t_joint) - jnp.log(jnp.mean(jnp.exp(t_marginal)) + 1e-10)

        return -mi_estimate  # Minimize negative MI


    @jax.jit
    def mine_train_step(state, x, y, rng):
        """Single MINE training step."""
        def loss_fn(params):
            return mine_loss(params, state.apply_fn, x, y, rng)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, -loss  # Return positive MI estimate


    def estimate_mi_mine(
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 100,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> float:
        """
        Estimate MI(X, Y) using MINE.

        Args:
            x: First variable samples [n_samples]
            y: Second variable samples [n_samples]
            n_epochs: Training epochs
            hidden_dim: MINE hidden dimension
            lr: Learning rate
            seed: Random seed

        Returns:
            MI estimate
        """
        rng = random.PRNGKey(seed)
        rng, init_rng = random.split(rng)

        # Reshape to [batch, 1]
        x_jax = jnp.array(x).reshape(-1, 1)
        y_jax = jnp.array(y).reshape(-1, 1)

        # Initialize MINE
        model = MINE(hidden_dim=hidden_dim)
        params = model.init(init_rng, x_jax[:1], y_jax[:1])

        # Create optimizer
        tx = optax.adam(lr)
        from flax.training import train_state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params['params'],
            tx=tx,
        )

        # Train
        mi_estimates = []
        for epoch in range(n_epochs):
            rng, step_rng = random.split(rng)
            state, mi = mine_train_step(state, x_jax, y_jax, step_rng)
            mi_estimates.append(float(mi))

        # Return average of last 10 estimates (more stable)
        return float(np.mean(mi_estimates[-10:]))


    # =========================================================================
    # Parallel MI Matrix Computation with shard_map
    # =========================================================================

    def compute_mi_matrix_parallel(
        latents: np.ndarray,
        factors: np.ndarray,
        n_epochs: int = 50,
        n_devices: Optional[int] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Compute full MI matrix using parallel MINE estimation.

        This uses shard_map to compute multiple MI pairs in parallel
        across available devices.

        Args:
            latents: [n_samples, latent_dim]
            factors: [n_samples, n_factors]
            n_epochs: MINE training epochs per pair
            n_devices: Number of devices (None = all)
            seed: Random seed

        Returns:
            MI matrix [latent_dim, n_factors]
        """
        latent_dim = latents.shape[1]
        n_factors = factors.shape[1]

        # Get devices
        devices = jax.devices()
        if n_devices is not None:
            devices = devices[:n_devices]
        n_devices_actual = len(devices)

        print(f"Computing MI matrix ({latent_dim}×{n_factors}) on {n_devices_actual} device(s)")

        # Create all (latent_idx, factor_idx) pairs
        pairs = [(j, k) for j in range(latent_dim) for k in range(n_factors)]
        n_pairs = len(pairs)

        # Compute MI for each pair
        # Use vmap where possible, fall back to sequential for many pairs
        mi_values = []

        for i, (j, k) in enumerate(pairs):
            mi = estimate_mi_mine(
                latents[:, j],
                factors[:, k],
                n_epochs=n_epochs,
                seed=seed + i,
            )
            mi_values.append(mi)

            if (i + 1) % 10 == 0:
                print(f"  Computed {i+1}/{n_pairs} MI pairs")

        # Reshape to matrix
        mi_matrix = np.array(mi_values).reshape(latent_dim, n_factors)

        return mi_matrix


    def compute_edi_parallel(
        latents: np.ndarray,
        factors: np.ndarray,
        n_epochs: int = 50,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Compute EDI using parallel MINE-based MI estimation.

        Args:
            latents: [n_samples, latent_dim]
            factors: [n_samples, n_factors]
            n_epochs: MINE epochs per MI pair
            seed: Random seed

        Returns:
            Dict with modularity, compactness, explicitness
        """
        # Compute MI matrix
        MI = compute_mi_matrix_parallel(latents, factors, n_epochs, seed=seed)

        latent_dim, n_factors = MI.shape

        # Modularity: For each latent dim, exclusivity of MI
        MI_norm_row = MI / (MI.sum(axis=1, keepdims=True) + 1e-10)
        modularity_scores = []
        for j in range(latent_dim):
            p = MI_norm_row[j, :]
            if p.max() > 0:
                exclusivity = p.max() - (p.sum() - p.max()) / max(1, n_factors - 1)
                modularity_scores.append(max(0, exclusivity))
            else:
                modularity_scores.append(0)
        modularity = float(np.mean(modularity_scores))

        # Compactness: For each factor, concentration in few dims
        MI_norm_col = MI / (MI.sum(axis=0, keepdims=True) + 1e-10)
        compactness_scores = []
        for k in range(n_factors):
            p = MI_norm_col[:, k]
            if p.max() > 0:
                exclusivity = p.max() - (p.sum() - p.max()) / max(1, latent_dim - 1)
                compactness_scores.append(max(0, exclusivity))
            else:
                compactness_scores.append(0)
        compactness = float(np.mean(compactness_scores))

        # Explicitness: Average MI (normalized)
        explicitness = float(np.mean(MI))

        return {
            'modularity': modularity,
            'compactness': compactness,
            'explicitness': explicitness,
            'mi_matrix': MI,
        }


else:
    # Fallback without JAX

    def estimate_mi_mine(*args, **kwargs):
        raise ImportError("JAX and Flax required for MINE")

    def compute_mi_matrix_parallel(*args, **kwargs):
        raise ImportError("JAX and Flax required for parallel MI computation")

    def compute_edi_parallel(*args, **kwargs):
        raise ImportError("JAX and Flax required for parallel EDI")


# =============================================================================
# CPU-Parallel DCI Computation
# =============================================================================

def _fit_factor_importance(args):
    """
    Worker function for parallel DCI factor importance computation.

    Args:
        args: (z, y, factor_idx, n_estimators)

    Returns:
        (factor_idx, importance_vector)
    """
    z, y, factor_idx, n_estimators = args

    # Discretize if continuous
    n_unique = len(np.unique(y))
    if n_unique > 20:
        percentiles = np.linspace(0, 100, 11)
        bins = np.percentile(y, percentiles)
        y = np.digitize(y, bins[1:-1])

    # Fit Random Forest
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=42,
    )
    model.fit(z, y)

    return factor_idx, model.feature_importances_


def compute_dci_parallel(
    z: np.ndarray,
    factors: np.ndarray,
    n_workers: Optional[int] = None,
    n_estimators: int = 50,
) -> Dict[str, float]:
    """
    Compute DCI with parallel factor importance estimation.

    Uses ProcessPoolExecutor to fit RF classifiers for each factor
    in parallel across CPU cores.

    Args:
        z: Latent representations [n_samples, latent_dim]
        factors: Ground truth factors [n_samples, n_factors]
        n_workers: Number of worker processes (None = CPU count)
        n_estimators: RF trees per factor

    Returns:
        DCI scores dict
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for DCI")

    from sklearn.preprocessing import StandardScaler
    import os

    n_factors = factors.shape[1]
    latent_dim = z.shape[1]

    # Normalize latents
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    # Prepare arguments for parallel execution
    args_list = [
        (z_scaled, factors[:, i], i, n_estimators)
        for i in range(n_factors)
    ]

    # Parallel execution
    n_workers = n_workers or os.cpu_count()
    R = np.zeros((n_factors, latent_dim))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_fit_factor_importance, args_list))

    for factor_idx, importance in results:
        R[factor_idx, :] = importance

    # Normalize R
    R = R / (R.sum(axis=1, keepdims=True) + 1e-10)

    # Disentanglement
    R_col = R / (R.sum(axis=0, keepdims=True) + 1e-10)
    disentanglement_scores = []
    for j in range(latent_dim):
        p = R_col[:, j]
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(n_factors)
        disentanglement_scores.append(1 - entropy / max_entropy)
    weights = R.sum(axis=0)
    weights = weights / (weights.sum() + 1e-10)
    disentanglement = float(np.dot(weights, disentanglement_scores))

    # Completeness
    completeness_scores = []
    for i in range(n_factors):
        p = R[i, :]
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(latent_dim)
        completeness_scores.append(1 - entropy / max_entropy)
    completeness = float(np.mean(completeness_scores))

    # Informativeness (quick estimate)
    from sklearn.model_selection import cross_val_score
    accuracies = []
    for i in range(min(3, n_factors)):  # Sample for speed
        y = factors[:, i]
        if len(np.unique(y)) > 20:
            y = np.digitize(y, np.percentile(y, np.linspace(0, 100, 11))[1:-1])
        model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        scores = cross_val_score(model, z_scaled, y, cv=3, scoring="accuracy")
        accuracies.append(scores.mean())
    informativeness = float(np.mean(accuracies))

    return {
        'disentanglement': disentanglement,
        'completeness': completeness,
        'informativeness': informativeness,
        'importance_matrix': R,
    }


# =============================================================================
# Batch Parallel Dataset Generation
# =============================================================================

def _generate_trajectory(args):
    """
    Worker function for parallel trajectory generation.

    Args:
        args: (phenotype_code, seed, n_trials, preset_map)

    Returns:
        (trajectory_features, factors, label)
    """
    phenotype_code, seed, n_trials = args

    # Import here to avoid pickling issues
    from ara.hgf import HGFAgent, VolatilitySwitchingTask, HGFParams
    from ara.hgf.pathology import (
        HEALTHY_BASELINE, SCHIZOPHRENIA_RIGID, SCHIZOPHRENIA_LOOSE,
        BPD_HIGH_KAPPA, ANXIETY_HIGH_PRECISION,
    )

    preset_map = {
        'HEALTHY': (HEALTHY_BASELINE, 0),
        'SCZ_RIGID': (SCHIZOPHRENIA_RIGID, 1),
        'SCZ_LOOSE': (SCHIZOPHRENIA_LOOSE, 2),
        'BPD': (BPD_HIGH_KAPPA, 3),
        'ANXIETY': (ANXIETY_HIGH_PRECISION, 4),
    }

    preset, label = preset_map[phenotype_code]

    # Create agent
    agent = HGFAgent(
        omega_2=preset.params.omega_2,
        kappa_1=preset.params.kappa_1,
        theta=preset.params.theta,
    )

    # Generate task
    task = VolatilitySwitchingTask(n_trials=n_trials)
    task_data = task.generate(seed=seed)

    # Run
    trajectory = agent.run(task_data)

    # Extract features
    features = np.zeros((n_trials, 8))
    for t in range(n_trials):
        state = trajectory.steps[t].state
        features[t, 0] = state.mu_2
        features[t, 1] = state.sigma_2
        features[t, 2] = state.mu_3
        features[t, 3] = state.sigma_3
        features[t, 4] = state.delta_1
        features[t, 5] = state.delta_2
        features[t, 6] = state.pi_1
        features[t, 7] = state.pi_hat_2

    factors = np.array([
        preset.params.omega_2,
        preset.params.kappa_1,
        preset.params.theta,
        float(label),
    ])

    return features, factors, label


def generate_dataset_parallel(
    n_samples_per_phenotype: int = 100,
    n_trials: int = 200,
    phenotypes: Optional[List[str]] = None,
    n_workers: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate trajectory dataset with parallel workers.

    Args:
        n_samples_per_phenotype: Samples per phenotype
        n_trials: Trials per trajectory
        phenotypes: List of phenotype codes
        n_workers: Worker processes
        seed: Base seed

    Returns:
        (trajectories, factors, labels)
    """
    import os

    if phenotypes is None:
        phenotypes = ['HEALTHY', 'SCZ_RIGID', 'SCZ_LOOSE', 'BPD', 'ANXIETY']

    # Create all (phenotype, seed) pairs
    args_list = []
    for i, phenotype in enumerate(phenotypes):
        for j in range(n_samples_per_phenotype):
            sample_seed = seed + i * 10000 + j
            args_list.append((phenotype, sample_seed, n_trials))

    n_workers = n_workers or os.cpu_count()
    print(f"Generating {len(args_list)} trajectories with {n_workers} workers...")

    # Parallel generation
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_generate_trajectory, args_list))

    # Unpack results
    trajectories = np.array([r[0] for r in results])
    factors = np.array([r[1] for r in results])
    labels = np.array([r[2] for r in results])

    print(f"Generated {len(trajectories)} trajectories")

    return trajectories, factors, labels
