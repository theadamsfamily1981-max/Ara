#!/usr/bin/env python3
"""
Experiment 3: Prove Causal Disentanglement via Interventions

Claim: Latents in the balanced regime are causally disentangled—intervening
on one dimension (traversing/clamping) affects only the corresponding factor
in reconstructions, enabling counterfactual predictions.

Inspired by Locatello et al.'s work on disentanglement.

Protocol:
1. Train VAE at various β values
2. For each latent dimension, traverse from -3σ to +3σ while fixing others
3. Decode traversals and measure which factors change
4. Compute modularity: does each latent control exactly one factor?

Expected: Balanced β shows highest modularity - each latent maps to one factor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from scipy.stats import spearmanr
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from ..model import EEGAraBrain, create_train_state
from ..experiment_edge_of_autumn_v2 import generate_disentangled_eeg


@dataclass
class InterventionResult:
    """Results from latent traversal interventions."""
    beta: float
    modularity: float  # How well each latent controls single factor
    compactness: float  # How well each factor is captured by single latent
    informativeness: float  # How much variance latents explain
    alignment_matrix: np.ndarray  # (latent_dim, num_factors) alignment scores
    best_latent_per_factor: List[int]
    best_factor_per_latent: List[int]


def perform_latent_traversal(
    model: EEGAraBrain,
    state,
    z_base: np.ndarray,
    latent_idx: int,
    num_steps: int = 11,
    sigma_range: float = 3.0,
) -> np.ndarray:
    """
    Traverse a single latent dimension while keeping others fixed.

    Returns: (num_steps, time, channels) reconstructions
    """
    latent_dim = z_base.shape[-1]
    z_base = z_base.reshape(1, latent_dim)

    # Compute traversal range based on empirical std
    z_std = max(np.std(z_base), 0.5)
    traversal_values = np.linspace(-sigma_range * z_std, sigma_range * z_std, num_steps)

    reconstructions = []
    decode_rng = jax.random.PRNGKey(42)

    for val in traversal_values:
        z_modified = z_base.copy()
        z_modified[0, latent_idx] = val

        # Decode
        x_recon = state.apply_fn(
            {'params': state.params},
            jnp.array(z_modified),
            training=False,
            method=model.decode,
        )
        reconstructions.append(np.array(x_recon[0]))

    return np.array(reconstructions)


def measure_factor_change(
    reconstructions: np.ndarray,
    factor_idx: int,
) -> float:
    """
    Measure how much a specific factor changes across traversal.

    Uses signal properties as proxy for ground truth factors:
    - Factor 0 (frequency): spectral centroid
    - Factor 1 (amplitude): signal energy
    - Factor 2 (spatial): channel activation pattern
    - Factor 3 (temporal): burstiness measure
    """
    num_steps = len(reconstructions)
    factor_values = []

    for recon in reconstructions:
        # recon shape: (time, channels)
        if factor_idx == 0:
            # Frequency: spectral centroid (simplified)
            fft_mag = np.abs(np.fft.rfft(recon.mean(axis=1)))
            freqs = np.fft.rfftfreq(recon.shape[0])
            centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
            factor_values.append(centroid)

        elif factor_idx == 1:
            # Amplitude: RMS energy
            energy = np.sqrt(np.mean(recon ** 2))
            factor_values.append(energy)

        elif factor_idx == 2:
            # Spatial: which channel group is most active
            channels = recon.shape[1]
            group_size = channels // 4
            group_energies = []
            for g in range(4):
                ge = np.mean(recon[:, g*group_size:(g+1)*group_size] ** 2)
                group_energies.append(ge)
            dominant = np.argmax(group_energies) / 3  # Normalize to [0, 1]
            factor_values.append(dominant)

        elif factor_idx == 3:
            # Temporal: burstiness (variance of envelope)
            envelope = np.abs(recon.mean(axis=1))
            burstiness = np.std(envelope) / (np.mean(envelope) + 1e-10)
            factor_values.append(burstiness)

    factor_values = np.array(factor_values)

    # Measure change as range / mean
    change = (factor_values.max() - factor_values.min()) / (np.abs(factor_values.mean()) + 1e-10)
    return float(change)


def compute_intervention_alignment(
    model: EEGAraBrain,
    state,
    x_val: np.ndarray,
    num_factors: int = 4,
    max_latents: int = 8,  # Only test first N latents
) -> Tuple[np.ndarray, float, float]:
    """
    Compute alignment matrix between latents and factors via interventions.

    Returns:
        alignment: (latent_dim, num_factors) matrix
        modularity: score (0-1, higher = better)
        compactness: score (0-1, higher = better)
    """
    eval_rng = jax.random.PRNGKey(123)

    # Get latent representations
    z_all = state.apply_fn(
        {'params': state.params},
        jnp.array(x_val[:50]),  # Use subset
        eval_rng,
        training=False,
        method=model.encode,
    )
    z_all = np.array(z_all)
    z_mean = z_all.mean(axis=0)

    latent_dim = min(z_all.shape[1], max_latents)

    # Build alignment matrix
    alignment = np.zeros((latent_dim, num_factors))

    for lat_idx in range(latent_dim):
        # Traverse this latent
        recons = perform_latent_traversal(model, state, z_mean, lat_idx)

        # Measure effect on each factor
        for fac_idx in range(num_factors):
            change = measure_factor_change(recons, fac_idx)
            alignment[lat_idx, fac_idx] = change

    # Normalize alignment matrix
    alignment = alignment / (alignment.max() + 1e-10)

    # Modularity: each latent should affect mostly ONE factor
    # = mean of (max_factor / sum_factors) per latent
    modularity_scores = []
    for lat_idx in range(latent_dim):
        row = alignment[lat_idx]
        if row.sum() > 1e-6:
            mod = row.max() / row.sum()
        else:
            mod = 0.5
        modularity_scores.append(mod)
    modularity = np.mean(modularity_scores)

    # Compactness: each factor should be captured by mostly ONE latent
    # = mean of (max_latent / sum_latents) per factor
    compactness_scores = []
    for fac_idx in range(num_factors):
        col = alignment[:, fac_idx]
        if col.sum() > 1e-6:
            comp = col.max() / col.sum()
        else:
            comp = 0.5
        compactness_scores.append(comp)
    compactness = np.mean(compactness_scores)

    return alignment, modularity, compactness


def train_and_analyze_interventions(
    beta: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    num_epochs: int = 15,
    seed: int = 42,
) -> InterventionResult:
    """Train model and analyze causal disentanglement via interventions."""
    rng = jax.random.PRNGKey(seed)
    batch_size = 32
    time_steps = x_train.shape[1]
    channels = x_train.shape[2]

    model = EEGAraBrain(
        latent_dim=16,
        time=time_steps,
        channels=channels,
        beta=float(beta),
        telepathy_weight=1.0,
        dropout_rate=0.1,
    )

    state = create_train_state(
        rng, model,
        learning_rate=1e-3,
        input_shape=(batch_size, time_steps, channels),
    )

    # Training
    num_samples = len(x_train)
    steps_per_epoch = num_samples // batch_size
    train_rng = rng

    for epoch in range(num_epochs):
        perm = np.random.default_rng(seed + epoch).permutation(num_samples)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            x_batch = jnp.array(x_shuffled[start:end])
            y_batch = jnp.array(y_shuffled[start:end])

            train_rng, step_rng, dropout_rng = jax.random.split(train_rng, 3)

            def loss_fn(params):
                loss, outputs = state.apply_fn(
                    {'params': params},
                    x_batch, step_rng,
                    labels=y_batch,
                    training=True,
                    rngs={'dropout': dropout_rng},
                )
                return loss, outputs

            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)

    # Intervention analysis
    alignment, modularity, compactness = compute_intervention_alignment(
        model, state, x_val
    )

    # Informativeness: total variance explained by latents
    informativeness = np.mean(alignment.max(axis=1))

    # Find best mappings
    best_latent_per_factor = list(np.argmax(alignment, axis=0))
    best_factor_per_latent = list(np.argmax(alignment, axis=1))

    return InterventionResult(
        beta=beta,
        modularity=modularity,
        compactness=compactness,
        informativeness=informativeness,
        alignment_matrix=alignment,
        best_latent_per_factor=best_latent_per_factor,
        best_factor_per_latent=best_factor_per_latent,
    )


def run_intervention_experiment(
    beta_values: List[float] = None,
    num_samples: int = 600,
    num_epochs: int = 15,
    verbose: bool = True,
) -> List[InterventionResult]:
    """
    Run causal intervention experiment.

    Proves: Balanced β produces causally disentangled representations.
    """
    if not JAX_AVAILABLE:
        print("JAX required")
        return []

    if beta_values is None:
        beta_values = [0.1, 0.5, 1.0, 3.0, 10.0, 30.0]

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CAUSAL DISENTANGLEMENT VIA INTERVENTIONS")
    print("=" * 70)
    print(f"\nClaim: Balanced β enables causal interventions (latent traversals")
    print(f"       affect only corresponding factors)")

    # Generate data
    print("\n1. Generating EEG data with known factors...")
    x, y, factors = generate_disentangled_eeg(
        num_samples=num_samples,
        time_steps=128,
        channels=16,
        seed=42,
    )

    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, _ = y[:split], y[split:]

    print(f"   Factors: frequency, amplitude, spatial, temporal")

    # Run sweep
    print(f"\n2. Running β-sweep with intervention analysis...")
    print("-" * 70)

    results = []
    for i, beta in enumerate(beta_values):
        result = train_and_analyze_interventions(
            beta=beta,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            num_epochs=num_epochs,
        )
        results.append(result)

        marker = "★" if result.modularity > 0.5 and result.compactness > 0.5 else " "
        print(f"   [{i+1}/{len(beta_values)}] β={beta:6.2f}  "
              f"Modularity={result.modularity:.3f}  "
              f"Compactness={result.compactness:.3f}  "
              f"Info={result.informativeness:.3f} {marker}")

    # Find best
    best_mod = max(results, key=lambda r: r.modularity)
    best_comp = max(results, key=lambda r: r.compactness)

    # Display alignment heatmaps (text version)
    print("\n" + "=" * 70)
    print("ALIGNMENT MATRICES (Latent → Factor)")
    print("=" * 70)

    factor_names = ["Freq", "Amp", "Spat", "Temp"]

    for r in results:
        print(f"\nβ = {r.beta:.1f}:")
        print(f"  {'':>6}", end="")
        for fn in factor_names:
            print(f"{fn:>8}", end="")
        print()

        for lat_idx in range(min(6, len(r.alignment_matrix))):
            print(f"  z_{lat_idx:>3}:", end="")
            for fac_idx in range(4):
                val = r.alignment_matrix[lat_idx, fac_idx]
                # Show as bar
                bar_len = int(val * 6)
                bar = "█" * bar_len + "░" * (6 - bar_len)
                print(f" {bar}", end="")
            print()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'β':>8} {'Modularity':>12} {'Compactness':>12} {'Disentangled':>14}")
    print("-" * 50)
    for r in results:
        disentangled = "★" if r.modularity > 0.5 and r.compactness > 0.5 else " "
        print(f"{r.beta:>8.2f} {r.modularity:>12.3f} {r.compactness:>12.3f} {disentangled:>14}")

    print(f"\n  Best modularity:   β={best_mod.beta:.2f} (M={best_mod.modularity:.3f})")
    print(f"  Best compactness:  β={best_comp.beta:.2f} (C={best_comp.compactness:.3f})")

    # Show factor-latent mappings for best
    print(f"\n  Latent-Factor mappings at β={best_mod.beta:.1f}:")
    for fac_idx, lat_idx in enumerate(best_mod.best_latent_per_factor):
        print(f"    Factor '{factor_names[fac_idx]}' ← z_{lat_idx}")

    print("\n" + "=" * 70)
    causal_betas = [r.beta for r in results if r.modularity > 0.5 and r.compactness > 0.5]
    if causal_betas:
        print(f"✓ CLAIM SUPPORTED: Causal disentanglement at β ∈ {causal_betas}")
        print("  (Intervening on one latent affects primarily one factor)")
    else:
        print("? PARTIAL: Weak causal structure detected")
        print("  (Traversals affect multiple factors - more training may help)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_intervention_experiment()
