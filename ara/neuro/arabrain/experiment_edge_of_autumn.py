#!/usr/bin/env python3
"""
Experiment: Can a Computer Find the Edge of Autumn?

Empirical test of the existence theorem by training EEGAraBrain
at multiple β values and measuring S(β), P(β), R(β).

This script:
1. Generates synthetic EEG data with known structure
2. Trains EEGAraBrain at various β values
3. Measures Structure, Performance, Robustness
4. Finds the balanced regime (if it exists)
5. Verifies the theorem's predictions

Usage:
    python -m ara.neuro.arabrain.experiment_edge_of_autumn
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available")
    exit(1)

from .model import EEGAraBrain, create_train_state
from .edge_of_autumn import (
    MetricPoint,
    find_edge_of_autumn,
    compute_structure_metric,
    compute_performance_metric,
)


# =============================================================================
# Data Generation with Known Ground Truth
# =============================================================================

def generate_structured_eeg(
    num_samples: int = 1000,
    time_steps: int = 256,
    channels: int = 32,
    num_factors: int = 4,
    noise_level: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG with known latent factors.

    Creates EEG-like signals where:
    - Factor 1: Frequency (low vs high)
    - Factor 2: Amplitude
    - Factor 3: Phase offset
    - Factor 4: Overload indicator (what telepathy head predicts)

    Returns:
        x: EEG data (N, T, C)
        y: Overload labels (N,)
        factors: Ground truth factors (N, num_factors)
    """
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 4 * np.pi, time_steps)

    # Generate latent factors
    factors = rng.uniform(0, 1, (num_samples, num_factors))

    # Factor meanings:
    # 0: Base frequency (5-40 Hz mapped from [0,1])
    # 1: Amplitude (0.5-2.0)
    # 2: Phase offset (0-2π)
    # 3: Overload probability

    x = np.zeros((num_samples, time_steps, channels))
    y = np.zeros(num_samples)

    for i in range(num_samples):
        freq = 5 + 35 * factors[i, 0]  # 5-40 Hz
        amp = 0.5 + 1.5 * factors[i, 1]  # Amplitude
        phase = 2 * np.pi * factors[i, 2]  # Phase

        # Overload: high freq + high amp → overload
        overload_prob = factors[i, 0] * factors[i, 1]
        y[i] = float(rng.random() < overload_prob)

        # Generate signal for each channel
        for ch in range(channels):
            ch_phase = phase + ch * 0.1  # Slight phase shift per channel
            ch_freq = freq * (1 + 0.05 * rng.standard_normal())  # Slight freq variation

            signal = amp * np.sin(ch_freq * t / (2 * np.pi) + ch_phase)
            # Add harmonics
            signal += 0.3 * amp * np.sin(2 * ch_freq * t / (2 * np.pi) + ch_phase)
            # Add noise
            signal += noise_level * rng.standard_normal(time_steps)

            x[i, :, ch] = signal

    # Normalize to [0, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    return x.astype(np.float32), y.astype(np.float32), factors.astype(np.float32)


# =============================================================================
# Training Function
# =============================================================================

def train_model(
    model: EEGAraBrain,
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    verbose: bool = False,
) -> Tuple[any, Dict]:
    """Train model and return final state + metrics."""
    rng = jax.random.PRNGKey(42)

    # Create train state
    state = create_train_state(
        rng,
        model,
        learning_rate,
        input_shape=(batch_size, model.time, model.channels),
    )

    # Training loop
    num_samples = len(x_train)
    steps_per_epoch = num_samples // batch_size

    final_metrics = {}

    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_acc = []

        # Shuffle
        perm = np.random.permutation(num_samples)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size

            x_batch = jnp.array(x_shuffled[start:end])
            y_batch = jnp.array(y_shuffled[start:end])

            rng, step_rng, dropout_rng = jax.random.split(rng, 3)

            # Train step
            def loss_fn(params):
                loss, outputs = state.apply_fn(
                    {'params': params},
                    x_batch, step_rng,
                    labels=y_batch,
                    training=True,
                    rngs={'dropout': dropout_rng},
                )
                return loss, outputs

            (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)

            epoch_loss.append(float(loss))
            if 'telepathy_accuracy' in outputs:
                epoch_acc.append(float(outputs['telepathy_accuracy']))

        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={np.mean(epoch_loss):.4f}, "
                  f"acc={np.mean(epoch_acc):.2%}")

        final_metrics = {
            'loss': np.mean(epoch_loss),
            'accuracy': np.mean(epoch_acc) if epoch_acc else 0.0,
        }

    return state, final_metrics


# =============================================================================
# Metric Evaluation
# =============================================================================

def evaluate_model(
    model: EEGAraBrain,
    state: any,
    x: np.ndarray,
    y: np.ndarray,
    factors: np.ndarray,
) -> Tuple[float, float, float, Dict]:
    """
    Evaluate model for S, P, R metrics.

    Returns:
        S: Structure score
        P: Performance score
        R: Robustness score
        details: Detailed metrics
    """
    rng = jax.random.PRNGKey(123)

    # Get latent representations
    x_jnp = jnp.array(x)

    # Encode to latent
    z = state.apply_fn(
        {'params': state.params},
        x_jnp, rng,
        training=False,
        method=model.encode,
    )
    z_np = np.array(z)

    # Get predictions
    probs = state.apply_fn(
        {'params': state.params},
        x_jnp, rng,
        training=False,
        method=model.predict_overload,
    )
    probs_np = np.array(probs).flatten()

    # === Structure S ===
    S, s_details = compute_structure_metric(z_np, factors)

    # === Performance P ===
    P, p_details = compute_performance_metric(probs_np, y)

    # === Robustness R ===
    # Test prediction stability under noise
    noise_levels = [0.01, 0.05, 0.1]
    stabilities = []

    for noise_std in noise_levels:
        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, x_jnp.shape) * noise_std
        x_noisy = x_jnp + noise

        probs_noisy = state.apply_fn(
            {'params': state.params},
            x_noisy, rng,
            training=False,
            method=model.predict_overload,
        )
        probs_noisy_np = np.array(probs_noisy).flatten()

        # Stability = correlation between clean and noisy predictions
        corr = np.corrcoef(probs_np, probs_noisy_np)[0, 1]
        stabilities.append(max(0, corr))  # Clip negative correlations

    R = np.mean(stabilities)

    details = {
        **s_details,
        **p_details,
        'stability_0.01': stabilities[0],
        'stability_0.05': stabilities[1],
        'stability_0.10': stabilities[2],
    }

    return S, P, R, details


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    beta_min: float = 0.1,
    beta_max: float = 16.0,
    num_betas: int = 12,
    num_samples: int = 500,
    num_epochs: int = 15,
    verbose: bool = True,
):
    """
    Run the Edge of Autumn experiment.

    Tests whether a neural network can find the balanced regime.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: CAN A COMPUTER FIND THE EDGE OF AUTUMN?")
    print("=" * 70)

    # Generate data
    print("\n1. Generating structured EEG data...")
    x, y, factors = generate_structured_eeg(
        num_samples=num_samples,
        time_steps=128,  # Shorter for speed
        channels=16,
        num_factors=4,
    )
    print(f"   Data shape: {x.shape}")
    print(f"   Labels: {y.sum():.0f}/{len(y)} overload")
    print(f"   Factors shape: {factors.shape}")

    # Split
    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    factors_val = factors[split:]

    # β sweep
    betas = np.logspace(np.log10(beta_min), np.log10(beta_max), num_betas)

    print(f"\n2. Running β-sweep ({num_betas} values from {beta_min} to {beta_max})...")
    print("-" * 70)

    results = []
    start_time = time.time()

    for i, beta in enumerate(betas):
        print(f"\n   [{i+1}/{num_betas}] β = {beta:.3f}")

        # Create model
        model = EEGAraBrain(
            latent_dim=16,
            time=128,
            channels=16,
            beta=float(beta),
            telepathy_weight=1.0,
            dropout_rate=0.1,
        )

        # Train
        state, train_metrics = train_model(
            model, x_train, y_train,
            num_epochs=num_epochs,
            batch_size=32,
            verbose=False,
        )

        # Evaluate
        S, P, R, details = evaluate_model(model, state, x_val, y_val, factors_val)

        point = MetricPoint(
            beta=float(beta),
            S=S,
            P=P,
            R=R,
            accuracy=details.get('accuracy'),
        )
        results.append(point)

        print(f"       S={S:.3f}  P={P:.3f}  R={R:.3f}  "
              f"(acc={details.get('accuracy', 0):.1%})")

    elapsed = time.time() - start_time
    print(f"\n   Sweep completed in {elapsed:.1f}s")

    # Find Edge of Autumn
    print("\n3. Searching for Edge of Autumn...")
    print("-" * 70)

    result = find_edge_of_autumn(results, threshold_percentile=50)

    print(f"\n   {result.message}")
    print(f"   Min max-deficit F = {result.F_min:.4f} at β = {result.F_argmin_beta:.3f}")

    print("\n   Assumptions check:")
    for assumption, satisfied in result.assumptions_satisfied.items():
        status = "✓" if satisfied else "✗"
        print(f"     {status} {assumption}")

    # Results visualization
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'β':>8} {'S':>8} {'P':>8} {'R':>8} {'In 𝓑':>8}")
    print("-" * 44)

    for p in results:
        in_B = "  ★" if result.found and result.regime and \
               result.regime.beta_range[0] <= p.beta <= result.regime.beta_range[1] else ""
        print(f"{p.beta:>8.2f} {p.S:>8.3f} {p.P:>8.3f} {p.R:>8.3f} {in_B}")

    if result.found:
        regime = result.regime
        print(f"\n" + "=" * 70)
        print("✓ EDGE OF AUTUMN FOUND!")
        print("=" * 70)
        print(f"\n  Optimal β*:      {regime.beta_star:.3f}")
        print(f"  Balanced range:  [{regime.beta_range[0]:.3f}, {regime.beta_range[1]:.3f}]")
        print(f"  Region width:    {regime.width:.3f}")
        print(f"  Safety margin:   {regime.margin:.3f}")
        print(f"\n  Thresholds:")
        print(f"    S* = {regime.S_star:.3f}")
        print(f"    P* = {regime.P_star:.3f}")
        print(f"    R* = {regime.R_star:.3f}")
        print(f"\n  Points in balanced region: {len(regime.points)}")

        print("\n" + "=" * 70)
        print("CONCLUSION: The computer found the balanced regime!")
        print("The Edge of Autumn exists and is empirically locatable.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ BALANCED REGIME NOT FOUND")
        print("=" * 70)
        print("\nPossible reasons:")
        print("  - β range too narrow")
        print("  - Training insufficient")
        print("  - Thresholds too strict")
        print("  - Data too simple/hard")

    # Metric curves (text visualization)
    print("\n" + "=" * 70)
    print("METRIC CURVES")
    print("=" * 70)

    for metric_name, metric_key in [('Structure (S)', 'S'),
                                     ('Performance (P)', 'P'),
                                     ('Robustness (R)', 'R')]:
        print(f"\n{metric_name}:")
        threshold = getattr(result.regime, f'{metric_key}_star', 0.5) if result.found else 0.5

        for p in results:
            val = getattr(p, metric_key)
            bar_len = int(val * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            marker = "★" if result.found and result.regime and \
                     result.regime.beta_range[0] <= p.beta <= result.regime.beta_range[1] else " "
            thresh_mark = "│" if abs(val - threshold) < 0.05 else " "
            print(f"  β={p.beta:6.2f} |{bar}| {val:.2f} {marker}{thresh_mark}")

    return result


def main():
    """Run the experiment."""
    if not JAX_AVAILABLE:
        print("JAX required for this experiment")
        return

    run_experiment(
        beta_min=0.1,
        beta_max=16.0,
        num_betas=12,
        num_samples=500,
        num_epochs=15,
    )


if __name__ == "__main__":
    main()
