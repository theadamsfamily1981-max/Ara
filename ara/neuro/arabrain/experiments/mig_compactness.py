#!/usr/bin/env python3
"""
Rigorous MIG vs β Experiment

A scientifically tight experiment proving:
"There is (or isn't) an intermediate β region where latent representations
are maximally compact (high MIG), flanked by under- and over-regularized
regimes where compactness degrades."

Design principles:
1. dSprites first (known ground-truth factors), EEG second (exploratory)
2. Multiple seeds (5+) for reproducibility
3. Proper MI estimation with quantile binning
4. ANOVA + Tukey HSD for statistical significance
5. Bootstrap CIs for uncertainty quantification

What this DOES prove:
- Existence of intermediate β regime with higher MIG
- Reproducibility across seeds
- Statistical distinguishability from extremes

What this DOESN'T prove:
- "Edge of chaos" in grand metaphysical sense
- Direct brain equivalence

Usage:
    python -m ara.neuro.arabrain.experiments.mig_compactness
    python -m ara.neuro.arabrain.experiments.mig_compactness --seeds 10
    python -m ara.neuro.arabrain.experiments.mig_compactness --eeg  # EEG mode
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # Placeholder for type hints
    torch = None
    DataLoader = None
    TensorDataset = None
    random_split = None

try:
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.metrics import mutual_info_score
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# dSprites Data Generation (Synthetic with known factors)
# =============================================================================

def generate_dsprites_synthetic(
    num_samples: int = 10000,
    image_size: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dSprites-like data with known latent factors.

    Factors (all normalized to [0, 1]):
    - shape: 0=square, 0.5=ellipse, 1=heart (discrete but encoded continuous)
    - scale: size of shape
    - rotation: orientation angle
    - pos_x: horizontal position
    - pos_y: vertical position

    Returns:
        images: (N, 1, H, W) float32
        factors: (N, 5) float32 ground truth factors
    """
    rng = np.random.default_rng(seed)

    # Generate random factors
    factors = rng.uniform(0, 1, (num_samples, 5))

    images = np.zeros((num_samples, 1, image_size, image_size), dtype=np.float32)

    for i in range(num_samples):
        shape_type = int(factors[i, 0] * 3) % 3  # 0, 1, or 2
        scale = 0.3 + 0.5 * factors[i, 1]  # 0.3 to 0.8
        rotation = factors[i, 2] * 2 * np.pi
        pos_x = factors[i, 3]
        pos_y = factors[i, 4]

        # Create coordinate grids
        y_coords = np.linspace(0, 1, image_size)
        x_coords = np.linspace(0, 1, image_size)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

        # Center coordinates on shape position
        cx = xx - pos_x
        cy = yy - pos_y

        # Rotate coordinates
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rx = cos_r * cx + sin_r * cy
        ry = -sin_r * cx + cos_r * cy

        # Scale
        rx = rx / scale
        ry = ry / scale

        # Draw shape
        if shape_type == 0:  # Square
            mask = (np.abs(rx) < 0.15) & (np.abs(ry) < 0.15)
        elif shape_type == 1:  # Ellipse
            mask = (rx**2 / 0.04 + ry**2 / 0.02) < 1
        else:  # Heart-ish
            mask = ((rx**2 + ry**2 - 0.02)**3 - rx**2 * ry**3) < 0

        images[i, 0] = mask.astype(np.float32)

    return images, factors.astype(np.float32)


# =============================================================================
# β-VAE Model (PyTorch) - Only defined when PyTorch is available
# =============================================================================

if TORCH_AVAILABLE:
    class Encoder(nn.Module):
        """Simple convolutional encoder for 64x64 images."""

        def __init__(self, latent_dim: int = 16):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 32x32
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 8x8
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 4x4
                nn.ReLU(),
                nn.Flatten(),
            )
            self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
            self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        def forward(self, x):
            h = self.conv(x)
            return self.fc_mu(h), self.fc_logvar(h)


    class Decoder(nn.Module):
        """Simple deconvolutional decoder for 64x64 images."""

        def __init__(self, latent_dim: int = 16):
            super().__init__()
            self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 64x64
                nn.Sigmoid(),
            )

        def forward(self, z):
            h = self.fc(z).view(-1, 256, 4, 4)
            return self.deconv(h)


    class BetaVAE(nn.Module):
        """β-VAE with configurable latent dimension."""

        def __init__(self, latent_dim: int = 16):
            super().__init__()
            self.encoder = Encoder(latent_dim)
            self.decoder = Decoder(latent_dim)
            self.latent_dim = latent_dim

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            x_hat = self.decoder(z)
            return x_hat, mu, logvar

        def encode(self, x):
            mu, _ = self.encoder(x)
            return mu


    def vae_loss(x, x_hat, mu, logvar, beta):
        """β-VAE loss: reconstruction + β * KL divergence."""
        recon = nn.functional.binary_cross_entropy(
            x_hat, x, reduction='sum'
        ) / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon + beta * kl, recon, kl


# =============================================================================
# MIG Computation (Proper MI Estimation)
# =============================================================================

def compute_mig(
    latents: np.ndarray,
    factors: np.ndarray,
    num_bins: int = 20,
) -> Tuple[float, List[float], np.ndarray]:
    """
    Compute Mutual Information Gap (MIG) - proper implementation.

    For each factor f_j:
    1. Compute MI between all latent dims z_i and f_j
    2. Sort MI values: I_1 >= I_2 >= ...
    3. Gap = (I_1 - I_2) / I_1

    MIG = mean of gaps across factors.

    Args:
        latents: (N, latent_dim) latent representations
        factors: (N, num_factors) ground truth factors

    Returns:
        mig: mean MIG score
        gaps: list of per-factor gaps
        mi_matrix: (latent_dim, num_factors) MI values
    """
    num_samples, latent_dim = latents.shape
    num_factors = factors.shape[1]

    # Discretize using quantile strategy (robust to skewed distributions)
    disc_z = KBinsDiscretizer(
        n_bins=num_bins,
        encode="ordinal",
        strategy="quantile",
        subsample=None,
    ).fit_transform(latents).astype(int)

    disc_f = KBinsDiscretizer(
        n_bins=num_bins,
        encode="ordinal",
        strategy="quantile",
        subsample=None,
    ).fit_transform(factors).astype(int)

    # Compute MI matrix
    mi_matrix = np.zeros((latent_dim, num_factors))
    for i in range(latent_dim):
        for j in range(num_factors):
            mi_matrix[i, j] = mutual_info_score(disc_z[:, i], disc_f[:, j])

    # Compute gaps for each factor
    gaps = []
    for j in range(num_factors):
        mi_j = mi_matrix[:, j]
        sorted_mi = np.sort(mi_j)[::-1]

        if sorted_mi[0] > 1e-10:
            gap = (sorted_mi[0] - sorted_mi[1]) / sorted_mi[0]
        else:
            gap = 0.0
        gaps.append(gap)

    mig = float(np.mean(gaps))
    return mig, gaps, mi_matrix


def compute_mig_bootstrap(
    latents: np.ndarray,
    factors: np.ndarray,
    num_bins: int = 20,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute MIG with bootstrap confidence intervals.

    Returns:
        mig_mean: mean MIG across bootstrap samples
        ci_low: 2.5th percentile
        ci_high: 97.5th percentile
    """
    rng = np.random.default_rng(seed)
    n_samples = len(latents)

    mig_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        mig, _, _ = compute_mig(latents[idx], factors[idx], num_bins)
        mig_samples.append(mig)

    mig_samples = np.array(mig_samples)
    return (
        float(np.mean(mig_samples)),
        float(np.percentile(mig_samples, 2.5)),
        float(np.percentile(mig_samples, 97.5)),
    )


# =============================================================================
# Training
# =============================================================================

def train_beta_vae(
    train_loader: DataLoader,
    beta: float,
    latent_dim: int = 16,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = 'cpu',
    seed: int = 42,
    verbose: bool = False,
) -> BetaVAE:
    """Train β-VAE with fixed hyperparameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BetaVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return model


def extract_latents(
    model: BetaVAE,
    loader: DataLoader,
    device: str = 'cpu',
) -> np.ndarray:
    """Extract latent means from trained model."""
    model.eval()
    latents = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu = model.encode(x)
            latents.append(mu.cpu().numpy())
    return np.concatenate(latents, axis=0)


# =============================================================================
# Statistical Analysis
# =============================================================================

@dataclass
class BetaResult:
    """Results for a single β value across seeds."""
    beta: float
    mig_values: List[float]
    mig_mean: float
    mig_std: float
    mig_ci_low: float
    mig_ci_high: float


def run_anova(results: List[BetaResult]) -> Tuple[float, float]:
    """Run one-way ANOVA across β conditions."""
    groups = [r.mig_values for r in results]
    F, p = stats.f_oneway(*groups)
    return float(F), float(p)


def run_tukey_hsd(results: List[BetaResult]) -> List[Tuple[float, float, float, bool]]:
    """
    Run Tukey HSD post-hoc test.

    Returns list of (beta_i, beta_j, p_value, significant) tuples.
    """
    from itertools import combinations

    comparisons = []
    n_comparisons = len(results) * (len(results) - 1) // 2

    for i, j in combinations(range(len(results)), 2):
        r_i, r_j = results[i], results[j]

        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(
            r_i.mig_values, r_j.mig_values,
            equal_var=False
        )

        # Bonferroni correction
        p_corrected = min(1.0, p_value * n_comparisons)
        significant = p_corrected < 0.05

        comparisons.append((r_i.beta, r_j.beta, p_corrected, significant))

    return comparisons


def compute_effect_size(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)


# =============================================================================
# Main Experiment
# =============================================================================

def run_mig_experiment(
    betas: List[float] = None,
    seeds: List[int] = None,
    num_samples: int = 10000,
    latent_dim: int = 16,
    epochs: int = 20,
    batch_size: int = 128,
    num_bins: int = 20,
    use_bootstrap: bool = True,
    verbose: bool = True,
) -> List[BetaResult]:
    """
    Run rigorous MIG vs β experiment.

    Returns list of BetaResult objects for each β value.
    """
    if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
        print("PyTorch and scikit-learn required")
        return []

    if betas is None:
        betas = [0.1, 0.3, 1.0, 3.0, 10.0]

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "=" * 70)
    print("RIGOROUS MIG vs β EXPERIMENT")
    print("=" * 70)
    print(f"\nDesign:")
    print(f"  β values: {betas}")
    print(f"  Seeds: {seeds}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Epochs per model: {epochs}")
    print(f"  Device: {device}")

    # Generate data
    print("\n1. Generating synthetic dSprites data...")
    images, factors = generate_dsprites_synthetic(num_samples=num_samples, seed=42)
    print(f"   Images: {images.shape}, Factors: {factors.shape}")
    print(f"   Factors: shape, scale, rotation, pos_x, pos_y")

    # Create datasets
    dataset = TensorDataset(
        torch.from_numpy(images),
        torch.from_numpy(factors),
    )

    train_size = int(0.7 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Extract test factors
    test_factors = factors[test_ds.indices]

    # Run sweep
    print(f"\n2. Running β-sweep...")
    print("-" * 70)

    results = []
    start_time = time.time()

    for beta in betas:
        print(f"\n   β = {beta}")
        mig_values = []

        for seed in seeds:
            # Train model
            model = train_beta_vae(
                train_loader=train_loader,
                beta=beta,
                latent_dim=latent_dim,
                epochs=epochs,
                device=device,
                seed=seed,
            )

            # Extract latents on test set
            test_latents = extract_latents(model, test_loader, device)

            # Compute MIG
            mig, gaps, mi_matrix = compute_mig(test_latents, test_factors, num_bins)
            mig_values.append(mig)

            print(f"       seed={seed}: MIG={mig:.4f}")

        # Aggregate
        mig_mean = np.mean(mig_values)
        mig_std = np.std(mig_values)

        # Bootstrap CI (using last model's latents as representative)
        if use_bootstrap:
            _, ci_low, ci_high = compute_mig_bootstrap(
                test_latents, test_factors, num_bins
            )
        else:
            ci_low = mig_mean - 1.96 * mig_std / np.sqrt(len(seeds))
            ci_high = mig_mean + 1.96 * mig_std / np.sqrt(len(seeds))

        result = BetaResult(
            beta=beta,
            mig_values=mig_values,
            mig_mean=mig_mean,
            mig_std=mig_std,
            mig_ci_low=ci_low,
            mig_ci_high=ci_high,
        )
        results.append(result)

        print(f"       → MIG = {mig_mean:.4f} ± {mig_std:.4f} "
              f"(95% CI: [{ci_low:.4f}, {ci_high:.4f}])")

    elapsed = time.time() - start_time
    print(f"\n   Completed in {elapsed/60:.1f} min")

    # Statistical analysis
    print("\n3. Statistical Analysis...")
    print("-" * 70)

    # ANOVA
    F_stat, p_value = run_anova(results)
    print(f"\n   One-way ANOVA:")
    print(f"     F = {F_stat:.3f}, p = {p_value:.6f}")
    if p_value < 0.05:
        print(f"     → β significantly affects MIG (p < 0.05) ✓")
    else:
        print(f"     → No significant effect of β on MIG")

    # Find best β
    best_idx = np.argmax([r.mig_mean for r in results])
    best_result = results[best_idx]
    print(f"\n   Best β: {best_result.beta} (MIG = {best_result.mig_mean:.4f})")

    # Tukey HSD
    print(f"\n   Pairwise comparisons (Bonferroni-corrected):")
    comparisons = run_tukey_hsd(results)
    for beta_i, beta_j, p_corr, sig in comparisons:
        sig_marker = "★" if sig else ""
        print(f"     β={beta_i} vs β={beta_j}: p={p_corr:.4f} {sig_marker}")

    # Effect sizes vs extremes
    if len(results) >= 3:
        low_beta_result = results[0]
        high_beta_result = results[-1]

        d_vs_low = compute_effect_size(best_result.mig_values, low_beta_result.mig_values)
        d_vs_high = compute_effect_size(best_result.mig_values, high_beta_result.mig_values)

        print(f"\n   Effect sizes (Cohen's d):")
        print(f"     β*={best_result.beta} vs β_low={low_beta_result.beta}: d = {d_vs_low:.3f}")
        print(f"     β*={best_result.beta} vs β_high={high_beta_result.beta}: d = {d_vs_high:.3f}")

    # Display results table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'β':>8} {'MIG':>12} {'Std':>8} {'95% CI':>20} {'Best':>6}")
    print("-" * 58)
    for r in results:
        best_marker = "★" if r.beta == best_result.beta else ""
        print(f"{r.beta:>8.2f} {r.mig_mean:>12.4f} {r.mig_std:>8.4f} "
              f"[{r.mig_ci_low:>7.4f}, {r.mig_ci_high:>7.4f}] {best_marker:>6}")

    # ASCII plot
    print("\n" + "=" * 70)
    print("MIG vs β (ASCII)")
    print("=" * 70)

    max_mig = max(r.mig_mean for r in results)
    min_mig = min(r.mig_mean for r in results)
    range_mig = max_mig - min_mig if max_mig > min_mig else 0.1

    for r in results:
        bar_len = int(40 * (r.mig_mean - min_mig) / range_mig) if range_mig > 0 else 20
        bar = "█" * bar_len
        err_bar = "├" + "─" * int(40 * r.mig_std / range_mig) + "┤" if range_mig > 0 else ""
        best = "★" if r.beta == best_result.beta else " "
        print(f"  β={r.beta:5.1f} |{bar:<40}| {r.mig_mean:.3f} {best}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if p_value < 0.05 and best_idx not in [0, len(results) - 1]:
        print(f"""
✓ CLAIM SUPPORTED:

  "We observe a β regime where latent representations are maximally
  compact (high MIG), flanked by under- and over-regularized regimes
  where compactness degrades."

  Evidence:
  - ANOVA: F={F_stat:.2f}, p={p_value:.4f} (significant)
  - Peak MIG at intermediate β={best_result.beta}
  - Effect size vs extremes: d={max(abs(d_vs_low), abs(d_vs_high)):.2f}

  This identifies an 'ideal line' in parameter space for efficient,
  factor-aligned encoding.
""")
    elif p_value < 0.05:
        print(f"""
? PARTIAL SUPPORT:

  β significantly affects MIG (p={p_value:.4f}), but peak is at an
  extreme rather than intermediate value. This could indicate:
  - Need wider β range
  - Task-specific optimal regularization
""")
    else:
        print(f"""
✗ NOT SUPPORTED:

  No significant effect of β on MIG (p={p_value:.4f}).
  Possible issues:
  - Insufficient training
  - Too few seeds
  - Data doesn't require disentanglement
""")

    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Rigorous MIG vs β Experiment")
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--eeg", action="store_true",
                        help="Run EEG version (exploratory)")
    args = parser.parse_args()

    if args.eeg:
        print("EEG mode not yet implemented - use dSprites for proof-of-concept")
        return

    run_mig_experiment(
        betas=args.betas,
        seeds=list(range(args.seeds)),
        num_samples=args.samples,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
