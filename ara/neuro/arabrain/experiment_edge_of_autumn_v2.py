#!/usr/bin/env python3
"""
Experiment V2: Enhanced Edge of Autumn with Proper Empirical Validation

Following the empirical roadmap:
1. Show there IS a sweet spot (balanced regime) with proper metrics
2. Show edge-of-chaos-like behavior (sensitivity analysis)
3. Show latents are actually disentangled & useful
4. Show robustness and antifragility
5. Statistical significance across multiple seeds

Key improvements over V1:
- Proper MIG-style disentanglement metric
- Sensitivity analysis (Jacobian proxy)
- Information-theoretic structure metric
- Multiple seeds for error bars
- Harder synthetic data with true latent factors

Usage:
    python -m ara.neuro.arabrain.experiment_edge_of_autumn_v2
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available")
    exit(1)

from .model import EEGAraBrain, create_train_state
from .edge_of_autumn import (
    MetricPoint,
    find_edge_of_autumn,
)


# =============================================================================
# Enhanced Data Generation with True Latent Factors
# =============================================================================

def generate_disentangled_eeg(
    num_samples: int = 1000,
    time_steps: int = 256,
    channels: int = 32,
    num_factors: int = 4,
    noise_level: float = 0.2,
    correlation_strength: float = 0.0,  # 0 = fully disentangled, 1 = fully correlated
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG with KNOWN disentangled latent factors.

    The key improvement: factors independently control different aspects
    of the signal, allowing us to measure true disentanglement.

    Factor assignments (designed for measurable disentanglement):
    - Factor 0: Controls FREQUENCY (alpha/beta/gamma band)
    - Factor 1: Controls AMPLITUDE envelope
    - Factor 2: Controls SPATIAL pattern (which channels active)
    - Factor 3: Controls TEMPORAL dynamics (burst vs sustained)

    Returns:
        x: EEG data (N, T, C)
        y: Overload labels (N,) - high freq + high amp → overload
        factors: Ground truth factors (N, num_factors) normalized to [0,1]
    """
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 4 * np.pi, time_steps)

    # Generate independent latent factors
    if correlation_strength > 0:
        # Add some correlation between factors (makes disentanglement harder)
        base = rng.uniform(0, 1, (num_samples, 1))
        factors = rng.uniform(0, 1, (num_samples, num_factors))
        factors = (1 - correlation_strength) * factors + correlation_strength * base
    else:
        factors = rng.uniform(0, 1, (num_samples, num_factors))

    x = np.zeros((num_samples, time_steps, channels))
    y = np.zeros(num_samples)

    for i in range(num_samples):
        # Factor 0: Frequency (8-40 Hz range, covering alpha/beta/gamma)
        base_freq = 8 + 32 * factors[i, 0]

        # Factor 1: Amplitude (0.3-1.5 range)
        amplitude = 0.3 + 1.2 * factors[i, 1]

        # Factor 2: Spatial pattern (controls which channel groups are active)
        # Split channels into 4 groups, factor controls which are dominant
        spatial_weight = factors[i, 2]
        channel_weights = np.zeros(channels)
        group_size = channels // 4
        for g in range(4):
            # Smooth activation based on factor distance
            dist = abs(spatial_weight - g/3)
            channel_weights[g*group_size:(g+1)*group_size] = np.exp(-3 * dist)
        channel_weights /= channel_weights.max() + 1e-8

        # Factor 3: Temporal dynamics (0=sustained, 1=bursty)
        burstiness = factors[i, 3]
        burst_envelope = np.ones(time_steps)
        if burstiness > 0.5:
            # Create burst pattern
            num_bursts = int(2 + 4 * burstiness)
            burst_centers = rng.choice(time_steps, num_bursts, replace=False)
            burst_envelope *= 0.2
            for bc in burst_centers:
                burst_width = time_steps // (num_bursts + 2)
                burst_envelope += 0.8 * np.exp(-((np.arange(time_steps) - bc)**2) / (2 * burst_width**2))

        # Overload: combination of high frequency and high amplitude
        # This is the "telepathy" target - can the model predict cognitive load?
        overload_score = 0.4 * factors[i, 0] + 0.4 * factors[i, 1] + 0.2 * factors[i, 3]
        y[i] = float(rng.random() < overload_score)

        # Generate signal for each channel
        for ch in range(channels):
            # Channel-specific frequency jitter
            ch_freq = base_freq * (1 + 0.1 * rng.standard_normal())
            # Phase offset per channel (spatial coherence)
            ch_phase = 2 * np.pi * ch / channels

            # Base oscillation
            signal = amplitude * channel_weights[ch] * np.sin(2 * np.pi * ch_freq * t / time_steps + ch_phase)
            # Add harmonic
            signal += 0.3 * amplitude * channel_weights[ch] * np.sin(4 * np.pi * ch_freq * t / time_steps + ch_phase)
            # Apply temporal envelope
            signal *= burst_envelope
            # Add noise (scaled by signal power)
            signal += noise_level * rng.standard_normal(time_steps)

            x[i, :, ch] = signal

    # Normalize to reasonable range but preserve relative differences
    x = (x - x.mean()) / (x.std() + 1e-8)
    # Clip and rescale to [0, 1]
    x = np.clip(x, -3, 3)
    x = (x + 3) / 6

    return x.astype(np.float32), y.astype(np.float32), factors.astype(np.float32)


# =============================================================================
# Enhanced Metrics
# =============================================================================

def compute_mig_score(z: np.ndarray, factors: np.ndarray) -> float:
    """
    Compute Mutual Information Gap (MIG) - proper disentanglement metric.

    MIG measures whether each factor is captured by a SINGLE latent dimension.
    High MIG = disentangled (each latent encodes one factor).
    Low MIG = entangled (factors spread across latents).

    Simplified implementation using empirical mutual information estimation.
    """
    num_samples, latent_dim = z.shape
    _, num_factors = factors.shape

    if num_samples < 100:
        return 0.5  # Not enough samples

    # Discretize latents and factors for MI estimation
    num_bins = min(20, num_samples // 10)

    mig_scores = []

    for k in range(num_factors):
        # Get factor k
        fk = factors[:, k]
        fk_binned = np.digitize(fk, np.linspace(0, 1, num_bins)) - 1

        # Compute MI with each latent dimension
        mis = []
        for j in range(latent_dim):
            zj = z[:, j]
            zj_binned = np.digitize(zj, np.linspace(zj.min(), zj.max(), num_bins)) - 1

            # Empirical MI via joint and marginal histograms
            joint_hist = np.histogram2d(fk_binned, zj_binned, bins=num_bins)[0]
            joint_hist = joint_hist / joint_hist.sum() + 1e-10

            marginal_f = joint_hist.sum(axis=1, keepdims=True)
            marginal_z = joint_hist.sum(axis=0, keepdims=True)

            mi = np.sum(joint_hist * np.log(joint_hist / (marginal_f * marginal_z + 1e-10)))
            mis.append(max(0, mi))

        mis = np.array(mis)
        # MIG: difference between top two MI values
        sorted_mis = np.sort(mis)[::-1]
        if sorted_mis[0] > 1e-6:
            gap = (sorted_mis[0] - sorted_mis[1]) / sorted_mis[0]
        else:
            gap = 0
        mig_scores.append(gap)

    return float(np.mean(mig_scores))


def compute_dci_disentanglement(z: np.ndarray, factors: np.ndarray) -> float:
    """
    Compute DCI Disentanglement score.

    Uses linear regression from latents to factors and measures
    how "spread out" the importance weights are.
    """
    from scipy.stats import spearmanr

    num_samples, latent_dim = z.shape
    _, num_factors = factors.shape

    if num_samples < 50:
        return 0.5

    # Compute importance matrix: how much does each z_j predict factor_k?
    importance = np.zeros((latent_dim, num_factors))

    for k in range(num_factors):
        for j in range(latent_dim):
            corr, _ = spearmanr(z[:, j], factors[:, k])
            importance[j, k] = abs(corr) if not np.isnan(corr) else 0

    # DCI: for each factor, how concentrated is the importance across latents?
    dci_scores = []
    for k in range(num_factors):
        imp_k = importance[:, k]
        if imp_k.sum() > 1e-6:
            # Normalize to probability distribution
            p = imp_k / imp_k.sum()
            # Entropy (lower = more disentangled)
            entropy = -np.sum(p * np.log(p + 1e-10))
            max_entropy = np.log(latent_dim)
            # DCI = 1 - normalized entropy
            dci = 1 - entropy / max_entropy if max_entropy > 0 else 0
            dci_scores.append(dci)

    return float(np.mean(dci_scores)) if dci_scores else 0.5


def compute_sensitivity(
    model: EEGAraBrain,
    state,
    x: np.ndarray,
    noise_scale: float = 0.01,
) -> float:
    """
    Compute input-to-latent sensitivity.

    Measures how much small input perturbations affect the latent representation.
    - Low sensitivity → frozen/collapsed (over-regularized)
    - High sensitivity → unstable (under-regularized)
    - Moderate sensitivity → edge-of-chaos sweet spot

    Returns normalized sensitivity score.
    """
    rng = jax.random.PRNGKey(999)
    x_jnp = jnp.array(x[:100])  # Use subset for speed

    # Get base latents
    z_base = state.apply_fn(
        {'params': state.params},
        x_jnp, rng,
        training=False,
        method=model.encode,
    )

    # Add small noise and get perturbed latents
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x_jnp.shape) * noise_scale
    x_noisy = x_jnp + noise

    z_noisy = state.apply_fn(
        {'params': state.params},
        x_noisy, rng,
        training=False,
        method=model.encode,
    )

    # Measure latent displacement relative to input displacement
    input_delta = float(jnp.sqrt(jnp.mean(noise**2)))
    latent_delta = float(jnp.sqrt(jnp.mean((z_noisy - z_base)**2)))

    # Sensitivity ratio (Jacobian norm proxy)
    sensitivity = latent_delta / (input_delta + 1e-10)

    return float(sensitivity)


def compute_information_content(z: np.ndarray, y: np.ndarray) -> float:
    """
    Compute information content: I(Z; Y) / H(Y).

    Measures how much predictive information about the label
    is captured in the latent representation.
    """
    num_samples = len(y)
    if num_samples < 50:
        return 0.5

    # Discretize latents
    num_bins = 10
    z_binned = np.zeros((num_samples, z.shape[1]), dtype=int)
    for j in range(z.shape[1]):
        z_binned[:, j] = np.digitize(z[:, j], np.linspace(z[:, j].min(), z[:, j].max(), num_bins))

    # Create composite latent index
    z_composite = np.sum(z_binned * (num_bins ** np.arange(min(4, z.shape[1]))), axis=1)

    # Compute MI(Z, Y)
    y_binary = (y > 0.5).astype(int)

    # Joint and marginal distributions
    joint = np.zeros((2, len(np.unique(z_composite))))
    unique_z = np.unique(z_composite)
    z_map = {v: i for i, v in enumerate(unique_z)}

    for i in range(num_samples):
        joint[y_binary[i], z_map[z_composite[i]]] += 1

    joint = joint / joint.sum() + 1e-10
    p_y = joint.sum(axis=1, keepdims=True)
    p_z = joint.sum(axis=0, keepdims=True)

    mi = np.sum(joint * np.log(joint / (p_y * p_z + 1e-10)))

    # H(Y)
    p_y_flat = p_y.flatten()
    h_y = -np.sum(p_y_flat * np.log(p_y_flat + 1e-10))

    # Normalized MI
    return float(mi / (h_y + 1e-10)) if h_y > 0 else 0.5


# =============================================================================
# Training with Enhanced Metrics
# =============================================================================

def train_and_evaluate(
    beta: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    factors_val: np.ndarray,
    num_epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    """
    Train model at given β and compute comprehensive metrics.
    """
    rng = jax.random.PRNGKey(seed)

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

    # Create train state
    state = create_train_state(
        rng,
        model,
        learning_rate=1e-3,
        input_shape=(batch_size, time_steps, channels),
    )

    # Training loop
    num_samples = len(x_train)
    steps_per_epoch = num_samples // batch_size

    train_rng = rng

    for epoch in range(num_epochs):
        # Shuffle
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

            (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)

    # === Evaluation ===
    eval_rng = jax.random.PRNGKey(123)
    x_val_jnp = jnp.array(x_val)

    # Get latent representations
    z = state.apply_fn(
        {'params': state.params},
        x_val_jnp, eval_rng,
        training=False,
        method=model.encode,
    )
    z_np = np.array(z)

    # Get predictions
    probs = state.apply_fn(
        {'params': state.params},
        x_val_jnp, eval_rng,
        training=False,
        method=model.predict_overload,
    )
    probs_np = np.array(probs).flatten()

    # === Structure Metrics ===
    mig = compute_mig_score(z_np, factors_val)
    dci = compute_dci_disentanglement(z_np, factors_val)
    info = compute_information_content(z_np, y_val)

    # Composite structure score
    S = (mig + dci + info) / 3

    # === Performance Metrics ===
    # AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_val, probs_np)
    except:
        auc = 0.5

    # Accuracy
    preds = (probs_np > 0.5).astype(float)
    accuracy = np.mean(preds == y_val)

    P = (auc + accuracy) / 2

    # === Sensitivity (Edge-of-Chaos) ===
    sensitivity = compute_sensitivity(model, state, x_val)

    # Normalize sensitivity to [0, 1] with optimal around 1.0
    # sensitivity < 0.1 → frozen, > 10 → chaotic
    sens_score = np.exp(-((np.log10(sensitivity + 1e-6) - 0)**2) / 2)

    # === Robustness ===
    # Test under increasing noise
    noise_levels = [0.01, 0.05, 0.1]
    stabilities = []

    for noise_std in noise_levels:
        noise_rng, eval_rng = jax.random.split(eval_rng)
        noise = jax.random.normal(noise_rng, x_val_jnp.shape) * noise_std
        x_noisy = x_val_jnp + noise

        probs_noisy = state.apply_fn(
            {'params': state.params},
            x_noisy, eval_rng,
            training=False,
            method=model.predict_overload,
        )
        probs_noisy_np = np.array(probs_noisy).flatten()

        corr = np.corrcoef(probs_np, probs_noisy_np)[0, 1]
        stabilities.append(max(0, corr) if not np.isnan(corr) else 0)

    R = np.mean(stabilities)

    return {
        'beta': beta,
        'S': S,
        'P': P,
        'R': R,
        'sensitivity': sensitivity,
        'sensitivity_score': sens_score,
        'mig': mig,
        'dci': dci,
        'info': info,
        'auc': auc,
        'accuracy': accuracy,
        'stability_0.01': stabilities[0],
        'stability_0.05': stabilities[1],
        'stability_0.10': stabilities[2],
    }


# =============================================================================
# Main Experiment with Multiple Seeds
# =============================================================================

@dataclass
class ExperimentResult:
    """Results from a single β value across seeds."""
    beta: float
    S_mean: float
    S_std: float
    P_mean: float
    P_std: float
    R_mean: float
    R_std: float
    sensitivity_mean: float
    details: List[Dict] = field(default_factory=list)


def run_experiment_v2(
    beta_values: List[float] = None,
    num_seeds: int = 3,
    num_samples: int = 800,
    num_epochs: int = 20,
    correlation_strength: float = 0.1,  # Slight correlation makes it harder
    verbose: bool = True,
):
    """
    Run enhanced Edge of Autumn experiment with error bars.

    Key improvements:
    1. Multiple seeds for statistical significance
    2. Proper disentanglement metrics (MIG, DCI)
    3. Sensitivity analysis for edge-of-chaos detection
    4. Harder synthetic data with correlated factors
    """
    if beta_values is None:
        beta_values = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

    print("\n" + "=" * 70)
    print("EXPERIMENT V2: ENHANCED EDGE OF AUTUMN VALIDATION")
    print("=" * 70)

    # Generate data
    print("\n1. Generating structured EEG data with true latent factors...")
    x, y, factors = generate_disentangled_eeg(
        num_samples=num_samples,
        time_steps=128,
        channels=16,
        num_factors=4,
        noise_level=0.2,
        correlation_strength=correlation_strength,
    )
    print(f"   Data shape: {x.shape}")
    print(f"   Labels: {y.sum():.0f}/{len(y)} overload ({y.mean()*100:.1f}%)")
    print(f"   Factor correlation: {correlation_strength}")

    # Split
    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    factors_val = factors[split:]

    print(f"\n2. Running β-sweep with {num_seeds} seeds per β...")
    print(f"   β values: {beta_values}")
    print("-" * 70)

    results = []
    start_time = time.time()

    for i, beta in enumerate(beta_values):
        print(f"\n   [{i+1}/{len(beta_values)}] β = {beta:.3f}")

        seed_results = []
        for seed in range(num_seeds):
            metrics = train_and_evaluate(
                beta=beta,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                factors_val=factors_val,
                num_epochs=num_epochs,
                seed=42 + seed * 100,
            )
            seed_results.append(metrics)
            print(f"       seed {seed}: S={metrics['S']:.3f} P={metrics['P']:.3f} "
                  f"R={metrics['R']:.3f} sens={metrics['sensitivity']:.2f}")

        # Aggregate across seeds
        S_vals = [r['S'] for r in seed_results]
        P_vals = [r['P'] for r in seed_results]
        R_vals = [r['R'] for r in seed_results]
        sens_vals = [r['sensitivity'] for r in seed_results]

        result = ExperimentResult(
            beta=beta,
            S_mean=np.mean(S_vals),
            S_std=np.std(S_vals),
            P_mean=np.mean(P_vals),
            P_std=np.std(P_vals),
            R_mean=np.mean(R_vals),
            R_std=np.std(R_vals),
            sensitivity_mean=np.mean(sens_vals),
            details=seed_results,
        )
        results.append(result)

        print(f"       → S={result.S_mean:.3f}±{result.S_std:.3f}  "
              f"P={result.P_mean:.3f}±{result.P_std:.3f}  "
              f"R={result.R_mean:.3f}±{result.R_std:.3f}")

    elapsed = time.time() - start_time
    print(f"\n   Completed in {elapsed/60:.1f} min")

    # Convert to MetricPoints for Edge of Autumn analysis
    metric_points = [
        MetricPoint(beta=r.beta, S=r.S_mean, P=r.P_mean, R=r.R_mean)
        for r in results
    ]

    # Find balanced regime
    print("\n3. Searching for Edge of Autumn...")
    print("-" * 70)

    edge_result = find_edge_of_autumn(metric_points, threshold_percentile=50)

    print(f"\n   {edge_result.message}")

    # Display results table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'β':>8} {'S':>12} {'P':>12} {'R':>12} {'Sens':>8} {'Balanced':>8}")
    print("-" * 64)

    for r in results:
        in_balanced = "★" if edge_result.found and edge_result.regime and \
            edge_result.regime.beta_range[0] <= r.beta <= edge_result.regime.beta_range[1] else ""
        print(f"{r.beta:>8.2f} {r.S_mean:>6.3f}±{r.S_std:>4.3f} "
              f"{r.P_mean:>6.3f}±{r.P_std:>4.3f} "
              f"{r.R_mean:>6.3f}±{r.R_std:>4.3f} "
              f"{r.sensitivity_mean:>8.2f} {in_balanced:>8}")

    # Sensitivity curve (edge-of-chaos indicator)
    print("\n" + "=" * 70)
    print("SENSITIVITY CURVE (Edge-of-Chaos Indicator)")
    print("=" * 70)
    print("\nOptimal sensitivity ≈ 1.0 (not frozen, not exploding)")

    for r in results:
        bar_len = min(40, int(np.log10(r.sensitivity_mean + 0.01) * 10 + 20))
        bar_len = max(1, bar_len)
        bar = "█" * bar_len
        marker = "★" if 0.5 < r.sensitivity_mean < 5.0 else " "
        print(f"  β={r.beta:6.2f} |{bar:<40}| {r.sensitivity_mean:6.2f} {marker}")

    # Component metrics
    print("\n" + "=" * 70)
    print("COMPONENT METRICS (averaged over seeds)")
    print("=" * 70)

    print(f"\n{'β':>8} {'MIG':>8} {'DCI':>8} {'Info':>8} {'AUC':>8} {'Acc':>8}")
    print("-" * 52)

    for r in results:
        avg = lambda k: np.mean([d[k] for d in r.details])
        print(f"{r.beta:>8.2f} {avg('mig'):>8.3f} {avg('dci'):>8.3f} "
              f"{avg('info'):>8.3f} {avg('auc'):>8.3f} {avg('accuracy'):>8.1%}")

    # Conclusion
    print("\n" + "=" * 70)
    if edge_result.found:
        regime = edge_result.regime
        print("✓ BALANCED REGIME FOUND")
        print(f"  Optimal β*: {regime.beta_star:.3f}")
        print(f"  Range: [{regime.beta_range[0]:.3f}, {regime.beta_range[1]:.3f}]")

        # Check for edge-of-chaos signature
        optimal_results = [r for r in results
                          if regime.beta_range[0] <= r.beta <= regime.beta_range[1]]
        if optimal_results:
            sens_in_regime = np.mean([r.sensitivity_mean for r in optimal_results])
            if 0.3 < sens_in_regime < 10:
                print(f"  Edge-of-chaos signature: ✓ (sensitivity={sens_in_regime:.2f})")
            else:
                print(f"  Edge-of-chaos signature: ? (sensitivity={sens_in_regime:.2f})")
    else:
        print("✗ BALANCED REGIME NOT FOUND")
        print("  Possible issues:")
        print("    - β range may need adjustment")
        print("    - More training epochs needed")
        print("    - Data may need more structure")

    print("=" * 70)

    return results, edge_result


def main():
    """Run the enhanced experiment."""
    if not JAX_AVAILABLE:
        print("JAX required")
        return

    results, edge_result = run_experiment_v2(
        beta_values=[0.1, 0.5, 1.0, 3.0, 10.0, 30.0],
        num_seeds=3,
        num_samples=800,
        num_epochs=15,
        correlation_strength=0.1,
    )


if __name__ == "__main__":
    main()
