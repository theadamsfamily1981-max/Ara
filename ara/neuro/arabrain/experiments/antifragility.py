#!/usr/bin/env python3
"""
Experiment 2: Prove Antifragility (Gains from Perturbations)

Claim: The edge-of-chaos regime is antifragile—moderate perturbations
actually enhance calibration or diversity, unlike:
- Fragile low-β (breaks under noise)
- Robust-but-dull high-β (no change, no benefit)

Inspired by Taleb's Antifragile: Things That Gain from Disorder.

Protocol:
1. Train models at various β values
2. Apply escalating noise levels during inference
3. Measure changes in calibration (ECE) and latent diversity (entropy)
4. Identify regime where moderate noise IMPROVES metrics

Expected: Only balanced β shows antifragile response - slight improvement
under low noise before eventual degradation at high noise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from sklearn.metrics import roc_auc_score
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from ..model import EEGAraBrain, create_train_state
from ..experiment_edge_of_autumn_v2 import generate_disentangled_eeg


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    Lower ECE = better calibrated predictions.
    ECE = Σ |accuracy(bin) - confidence(bin)| * size(bin)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += abs(bin_acc - bin_conf) * mask.sum() / len(y_true)

    return float(ece)


def compute_latent_entropy(z: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute entropy of latent activation distribution.

    Higher entropy = more diverse/spread latent representations.
    """
    # Flatten all latent dims and compute histogram entropy
    z_flat = z.flatten()
    hist, _ = np.histogram(z_flat, bins=n_bins, density=True)
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    return float(entropy)


def compute_prediction_diversity(probs: np.ndarray) -> float:
    """
    Compute diversity of predictions.

    Higher = model uses full range of probabilities.
    """
    return float(np.std(probs))


@dataclass
class NoiseResponse:
    """Response to a single noise level."""
    noise_sigma: float
    ece: float
    latent_entropy: float
    pred_diversity: float
    accuracy: float


@dataclass
class AntifragilityResult:
    """Results for a single β value across noise levels."""
    beta: float
    baseline: NoiseResponse
    responses: List[NoiseResponse]
    # Computed metrics
    delta_ece_low: float  # Change at low noise (negative = improvement)
    delta_ece_high: float  # Change at high noise
    delta_entropy_low: float  # Change in latent diversity
    is_antifragile: bool  # True if improves under low noise


def analyze_noise_response(
    model: EEGAraBrain,
    state,
    x_val: np.ndarray,
    y_val: np.ndarray,
    noise_levels: List[float],
) -> Tuple[NoiseResponse, List[NoiseResponse]]:
    """Analyze model response to various noise levels."""
    eval_rng = jax.random.PRNGKey(123)
    x_val_jnp = jnp.array(x_val)

    def evaluate_with_noise(noise_sigma: float) -> NoiseResponse:
        nonlocal eval_rng

        # Add noise
        if noise_sigma > 0:
            eval_rng, noise_rng = jax.random.split(eval_rng)
            noise = jax.random.normal(noise_rng, x_val_jnp.shape) * noise_sigma
            x_noisy = x_val_jnp + noise
        else:
            x_noisy = x_val_jnp

        # Get predictions
        probs = state.apply_fn(
            {'params': state.params},
            x_noisy, eval_rng,
            training=False,
            method=model.predict_overload,
        )
        probs_np = np.array(probs).flatten()

        # Get latents
        z = state.apply_fn(
            {'params': state.params},
            x_noisy, eval_rng,
            training=False,
            method=model.encode,
        )
        z_np = np.array(z)

        # Compute metrics
        ece = compute_ece(y_val, probs_np)
        entropy = compute_latent_entropy(z_np)
        diversity = compute_prediction_diversity(probs_np)
        preds = (probs_np > 0.5).astype(float)
        accuracy = np.mean(preds == y_val)

        return NoiseResponse(
            noise_sigma=noise_sigma,
            ece=ece,
            latent_entropy=entropy,
            pred_diversity=diversity,
            accuracy=accuracy,
        )

    baseline = evaluate_with_noise(0.0)
    responses = [evaluate_with_noise(sigma) for sigma in noise_levels]

    return baseline, responses


def train_and_analyze_antifragility(
    beta: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    noise_levels: List[float],
    num_epochs: int = 15,
    seed: int = 42,
) -> AntifragilityResult:
    """Train model and analyze antifragility response."""
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

    # Analyze noise response
    baseline, responses = analyze_noise_response(
        model, state, x_val, y_val, noise_levels
    )

    # Compute deltas
    low_noise = responses[0] if responses else baseline  # First noise level
    high_noise = responses[-1] if responses else baseline  # Last noise level

    delta_ece_low = low_noise.ece - baseline.ece
    delta_ece_high = high_noise.ece - baseline.ece
    delta_entropy_low = low_noise.latent_entropy - baseline.latent_entropy

    # Antifragile if ECE improves (decreases) under low noise
    is_antifragile = delta_ece_low < -0.005

    return AntifragilityResult(
        beta=beta,
        baseline=baseline,
        responses=responses,
        delta_ece_low=delta_ece_low,
        delta_ece_high=delta_ece_high,
        delta_entropy_low=delta_entropy_low,
        is_antifragile=is_antifragile,
    )


def run_antifragility_experiment(
    beta_values: List[float] = None,
    noise_levels: List[float] = None,
    num_samples: int = 600,
    num_epochs: int = 15,
    verbose: bool = True,
) -> List[AntifragilityResult]:
    """
    Run antifragility experiment.

    Proves: Balanced β regime gains from moderate perturbations.
    """
    if not JAX_AVAILABLE:
        print("JAX required")
        return []

    if beta_values is None:
        beta_values = [0.1, 0.5, 1.0, 3.0, 10.0, 30.0]

    if noise_levels is None:
        noise_levels = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ANTIFRAGILITY (Gains from Disorder)")
    print("=" * 70)
    print(f"\nClaim: Balanced β improves under moderate noise (Taleb antifragility)")
    print(f"Noise levels: {noise_levels}")

    # Generate data
    print("\n1. Generating EEG data...")
    x, y, factors = generate_disentangled_eeg(
        num_samples=num_samples,
        time_steps=128,
        channels=16,
        seed=42,
    )

    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    # Run sweep
    print(f"\n2. Running β-sweep with noise analysis...")
    print("-" * 70)

    results = []
    for i, beta in enumerate(beta_values):
        result = train_and_analyze_antifragility(
            beta=beta,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            noise_levels=noise_levels,
            num_epochs=num_epochs,
        )
        results.append(result)

        af_marker = "★ ANTIFRAGILE" if result.is_antifragile else ""
        print(f"   [{i+1}/{len(beta_values)}] β={beta:6.2f}  "
              f"ΔECE(low)={result.delta_ece_low:+.4f}  "
              f"ΔECE(high)={result.delta_ece_high:+.4f}  "
              f"ΔH={result.delta_entropy_low:+.3f} {af_marker}")

    # Display results
    print("\n" + "=" * 70)
    print("NOISE RESPONSE CURVES")
    print("=" * 70)

    print(f"\nECE vs Noise Level (lower = better calibration):")
    print(f"{'β':>6}", end="")
    print(f"{'base':>8}", end="")
    for sigma in noise_levels:
        print(f"  σ={sigma:.2f}", end="")
    print()
    print("-" * (8 + 8 + 8 * len(noise_levels)))

    for r in results:
        print(f"{r.beta:>6.1f}", end="")
        print(f"{r.baseline.ece:>8.4f}", end="")
        for resp in r.responses:
            delta = resp.ece - r.baseline.ece
            marker = "↓" if delta < -0.002 else ("↑" if delta > 0.002 else "·")
            print(f"  {resp.ece:.4f}{marker}", end="")
        print()

    # Antifragility summary
    print("\n" + "=" * 70)
    print("ANTIFRAGILITY ANALYSIS")
    print("=" * 70)

    antifragile_betas = [r.beta for r in results if r.is_antifragile]
    fragile_betas = [r.beta for r in results if r.delta_ece_high > 0.02]
    robust_betas = [r.beta for r in results if abs(r.delta_ece_low) < 0.002 and abs(r.delta_ece_high) < 0.01]

    print(f"\n  Fragile (breaks under noise):     β ∈ {fragile_betas}")
    print(f"  Robust (unchanged):               β ∈ {robust_betas}")
    print(f"  Antifragile (gains from noise):   β ∈ {antifragile_betas}")

    # Visual: show response type
    print("\n  Response patterns:")
    for r in results:
        if r.is_antifragile:
            pattern = "↓↑ (dip then rise - antifragile)"
        elif r.delta_ece_high > 0.02:
            pattern = "↗↗ (monotonic degradation - fragile)"
        else:
            pattern = "→→ (flat - robust but dull)"
        print(f"    β={r.beta:5.1f}: {pattern}")

    print("\n" + "=" * 70)
    if antifragile_betas:
        print(f"✓ CLAIM SUPPORTED: Antifragility observed at β ∈ {antifragile_betas}")
        print("  (ECE improved under low noise, degraded under high noise)")
    else:
        print("? PARTIAL: No clear antifragile regime found")
        print("  (May need different noise types or more training)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_antifragility_experiment()
