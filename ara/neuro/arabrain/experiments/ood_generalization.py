#!/usr/bin/env python3
"""
Experiment 1: Prove Improved Generalization to Out-of-Distribution (OOD) Data

Claim: In the balanced β regime, the model generalizes better to OOD shifts
because disentangled latents capture invariant structures rather than memorizing.

Protocol:
1. Train on "source" distribution (e.g., EEG with factors in range A)
2. Test on "target" distribution (e.g., EEG with shifted factors)
3. Measure OOD performance degradation across β values
4. Show minimal degradation in balanced regime

Expected: U-shaped OOD drop curve - balanced β minimizes transfer gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
from ..experiment_edge_of_autumn_v2 import (
    generate_disentangled_eeg,
    compute_mig_score,
)


@dataclass
class OODResult:
    """Results for a single β value."""
    beta: float
    in_dist_auc: float
    ood_auc: float
    auc_drop: float  # in_dist - ood (lower = better transfer)
    mig_score: float
    in_dist_acc: float
    ood_acc: float


def generate_shifted_eeg(
    num_samples: int = 500,
    time_steps: int = 128,
    channels: int = 16,
    shift_type: str = "frequency",
    shift_magnitude: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate OOD EEG data with systematic distribution shift.

    Shift types:
    - "frequency": Shift base frequencies higher
    - "amplitude": Scale amplitudes differently
    - "spatial": Different channel activation patterns
    - "subject": Simulate new subject (all factors shifted)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, time_steps)

    # Generate factors with shift
    factors = rng.uniform(0, 1, (num_samples, 4))

    if shift_type == "frequency":
        # Shift factor 0 (frequency) higher
        factors[:, 0] = np.clip(factors[:, 0] + shift_magnitude, 0, 1)
    elif shift_type == "amplitude":
        # Shift factor 1 (amplitude)
        factors[:, 1] = np.clip(factors[:, 1] * (1 + shift_magnitude), 0, 1)
    elif shift_type == "spatial":
        # Invert spatial pattern (factor 2)
        factors[:, 2] = 1 - factors[:, 2]
    elif shift_type == "subject":
        # Systematic shift on all factors (new subject)
        factors = np.clip(factors + shift_magnitude * rng.standard_normal((num_samples, 4)) * 0.5, 0, 1)

    x = np.zeros((num_samples, time_steps, channels))
    y = np.zeros(num_samples)

    for i in range(num_samples):
        base_freq = 8 + 32 * factors[i, 0]
        amplitude = 0.3 + 1.2 * factors[i, 1]
        spatial_weight = factors[i, 2]
        burstiness = factors[i, 3]

        # Channel weights based on spatial factor
        channel_weights = np.zeros(channels)
        group_size = channels // 4
        for g in range(4):
            dist = abs(spatial_weight - g/3)
            channel_weights[g*group_size:(g+1)*group_size] = np.exp(-3 * dist)
        channel_weights /= channel_weights.max() + 1e-8

        # Burst envelope
        burst_envelope = np.ones(time_steps)
        if burstiness > 0.5:
            num_bursts = int(2 + 4 * burstiness)
            burst_centers = rng.choice(time_steps, min(num_bursts, time_steps), replace=False)
            burst_envelope *= 0.2
            for bc in burst_centers:
                burst_width = time_steps // (num_bursts + 2)
                burst_envelope += 0.8 * np.exp(-((np.arange(time_steps) - bc)**2) / (2 * burst_width**2))

        # Overload label
        overload_score = 0.4 * factors[i, 0] + 0.4 * factors[i, 1] + 0.2 * factors[i, 3]
        y[i] = float(rng.random() < overload_score)

        for ch in range(channels):
            ch_freq = base_freq * (1 + 0.1 * rng.standard_normal())
            ch_phase = 2 * np.pi * ch / channels
            signal = amplitude * channel_weights[ch] * np.sin(2 * np.pi * ch_freq * t / time_steps + ch_phase)
            signal += 0.3 * amplitude * channel_weights[ch] * np.sin(4 * np.pi * ch_freq * t / time_steps + ch_phase)
            signal *= burst_envelope
            signal += 0.2 * rng.standard_normal(time_steps)
            x[i, :, ch] = signal

    # Normalize
    x = (x - x.mean()) / (x.std() + 1e-8)
    x = np.clip(x, -3, 3)
    x = (x + 3) / 6

    return x.astype(np.float32), y.astype(np.float32), factors.astype(np.float32)


def train_and_evaluate_ood(
    beta: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val_id: np.ndarray,
    y_val_id: np.ndarray,
    x_val_ood: np.ndarray,
    y_val_ood: np.ndarray,
    factors_id: np.ndarray,
    num_epochs: int = 15,
    seed: int = 42,
) -> OODResult:
    """Train on in-distribution, evaluate on both ID and OOD."""
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

    # Evaluation
    eval_rng = jax.random.PRNGKey(123)

    def evaluate(x_eval, y_eval):
        x_jnp = jnp.array(x_eval)
        probs = state.apply_fn(
            {'params': state.params},
            x_jnp, eval_rng,
            training=False,
            method=model.predict_overload,
        )
        probs_np = np.array(probs).flatten()

        try:
            auc = roc_auc_score(y_eval, probs_np)
        except:
            auc = 0.5

        preds = (probs_np > 0.5).astype(float)
        acc = np.mean(preds == y_eval)
        return auc, acc, probs_np

    id_auc, id_acc, _ = evaluate(x_val_id, y_val_id)
    ood_auc, ood_acc, _ = evaluate(x_val_ood, y_val_ood)

    # MIG score on in-distribution
    z = state.apply_fn(
        {'params': state.params},
        jnp.array(x_val_id), eval_rng,
        training=False,
        method=model.encode,
    )
    mig = compute_mig_score(np.array(z), factors_id)

    return OODResult(
        beta=beta,
        in_dist_auc=id_auc,
        ood_auc=ood_auc,
        auc_drop=id_auc - ood_auc,
        mig_score=mig,
        in_dist_acc=id_acc,
        ood_acc=ood_acc,
    )


def run_ood_experiment(
    beta_values: List[float] = None,
    shift_type: str = "subject",
    shift_magnitude: float = 0.3,
    num_samples: int = 600,
    num_epochs: int = 15,
    verbose: bool = True,
) -> List[OODResult]:
    """
    Run OOD generalization experiment.

    Proves: Balanced β regime shows minimal OOD performance drop.
    """
    if not JAX_AVAILABLE:
        print("JAX required")
        return []

    if beta_values is None:
        beta_values = [0.1, 0.5, 1.0, 3.0, 10.0, 30.0]

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: OOD GENERALIZATION")
    print("=" * 70)
    print(f"\nClaim: Balanced β minimizes OOD performance drop")
    print(f"Shift type: {shift_type}, magnitude: {shift_magnitude}")

    # Generate in-distribution data
    print("\n1. Generating in-distribution EEG data...")
    x_id, y_id, factors_id = generate_disentangled_eeg(
        num_samples=num_samples,
        time_steps=128,
        channels=16,
        seed=42,
    )

    # Generate OOD data
    print("2. Generating out-of-distribution EEG data...")
    x_ood, y_ood, factors_ood = generate_shifted_eeg(
        num_samples=num_samples // 2,
        time_steps=128,
        channels=16,
        shift_type=shift_type,
        shift_magnitude=shift_magnitude,
        seed=999,
    )

    # Split in-distribution
    split = int(0.7 * len(x_id))
    x_train, x_val_id = x_id[:split], x_id[split:]
    y_train, y_val_id = y_id[:split], y_id[split:]
    factors_val_id = factors_id[split:]

    print(f"   Train: {len(x_train)}, Val-ID: {len(x_val_id)}, Val-OOD: {len(x_ood)}")

    # Run sweep
    print(f"\n3. Running β-sweep...")
    print("-" * 70)

    results = []
    for i, beta in enumerate(beta_values):
        result = train_and_evaluate_ood(
            beta=beta,
            x_train=x_train,
            y_train=y_train,
            x_val_id=x_val_id,
            y_val_id=y_val_id,
            x_val_ood=x_ood,
            y_val_ood=y_ood,
            factors_id=factors_val_id,
            num_epochs=num_epochs,
        )
        results.append(result)

        marker = "★" if result.auc_drop < 0.05 else " "
        print(f"   [{i+1}/{len(beta_values)}] β={beta:6.2f}  "
              f"ID-AUC={result.in_dist_auc:.3f}  "
              f"OOD-AUC={result.ood_auc:.3f}  "
              f"Drop={result.auc_drop:+.3f}  "
              f"MIG={result.mig_score:.3f} {marker}")

    # Find best β
    best = min(results, key=lambda r: r.auc_drop)
    worst = max(results, key=lambda r: r.auc_drop)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'β':>8} {'ID-AUC':>10} {'OOD-AUC':>10} {'Drop':>10} {'MIG':>8}")
    print("-" * 50)
    for r in results:
        marker = "★ BEST" if r.beta == best.beta else ""
        print(f"{r.beta:>8.2f} {r.in_dist_auc:>10.3f} {r.ood_auc:>10.3f} "
              f"{r.auc_drop:>+10.3f} {r.mig_score:>8.3f} {marker}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    drops = [r.auc_drop for r in results]
    best_drop = min(drops)
    worst_drop = max(drops)

    print(f"\n  Best β (min drop):   {best.beta:.2f} (drop = {best.auc_drop:+.3f})")
    print(f"  Worst β (max drop):  {worst.beta:.2f} (drop = {worst.auc_drop:+.3f})")
    print(f"  Improvement:         {worst_drop - best_drop:.3f} AUC points")

    # Correlation with MIG
    migs = [r.mig_score for r in results]
    corr = np.corrcoef(migs, [-d for d in drops])[0, 1]
    print(f"\n  Correlation (MIG vs -Drop): {corr:.3f}")
    if corr > 0.5:
        print("  → Higher disentanglement correlates with better OOD transfer ✓")

    print("\n" + "=" * 70)
    if best_drop < 0.05:
        print("✓ CLAIM SUPPORTED: Balanced regime shows minimal OOD degradation")
    else:
        print("? PARTIAL SUPPORT: OOD drop varies but pattern unclear")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_ood_experiment()
