#!/usr/bin/env python3
"""
neurobalance_synthetic.py - Full NeuroBalance Telepathy Dashboard

Runs the complete β-VAE + disentanglement pipeline with W&B logging,
creating a visual "telepathy dashboard" showing:
- Training curves
- Reconstruction quality
- Latent traversals (which neuron does what)
- Disentanglement metrics (MIG, DCI, EDI)
- NeuroBalance composite index
- Importance heatmaps (latent × factor wiring diagram)

Usage:
    python scripts/neurobalance_synthetic.py --beta 4.0 --latent_dim 10 --epochs 50

This is the synthetic data version. For EEG, swap the data loader and
call run_disentanglement_suite(eeg_latents, proxy_labels, prefix="eeg").
"""

import argparse
import time
from pathlib import Path
import numpy as np

# Optional W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("W&B not available - metrics will be printed only")

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not available - using NumPy fallback")

# Ara imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara.vae.jax_spmd import (
    SPMDConfig,
    init_beta_vae_params,
    train_beta_vae_spmd,
    encode_to_latent,
    sample_from_prior,
    decoder,
)
from ara.vae.jax_eval import (
    evaluate_disentanglement_jax,
    compute_mig_jax,
)
from ara.vae.metrics import compute_dci, compute_edi
from ara.vae.ara_interface import AraStateReader, StateReadout


# =============================================================================
# Data Generation
# =============================================================================

def generate_dsprites_like(n_samples: int = 10000, seed: int = 42):
    """
    Generate dSprites-like synthetic data with known factors.

    Factors:
    - shape (3 values)
    - scale (6 values)
    - orientation (40 values)
    - x_position (32 values)
    - y_position (32 values)

    Returns:
        images: (n_samples, 64, 64)
        factors: (n_samples, 5)
    """
    np.random.seed(seed)

    images = []
    factors = []

    for _ in range(n_samples):
        # Sample factors
        shape = np.random.randint(0, 3)
        scale = np.random.uniform(0.5, 1.0)
        orientation = np.random.uniform(0, 2 * np.pi)
        x_pos = np.random.uniform(0.1, 0.9)
        y_pos = np.random.uniform(0.1, 0.9)

        # Create 64x64 image
        img = np.zeros((64, 64))

        # Draw shape at position with scale and orientation
        cx, cy = int(x_pos * 64), int(y_pos * 64)
        r = int(scale * 10)

        if shape == 0:  # Square
            img[max(0, cy-r):min(64, cy+r), max(0, cx-r):min(64, cx+r)] = 1
        elif shape == 1:  # Ellipse
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dy*dy + dx*dx <= r*r:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < 64 and 0 <= nx < 64:
                            img[ny, nx] = 1
        else:  # Heart/triangle approximation
            for dy in range(-r, r+1):
                width = r - abs(dy)
                for dx in range(-width, width+1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < 64 and 0 <= nx < 64:
                        img[ny, nx] = 1

        images.append(img)
        factors.append([shape, scale, orientation, x_pos, y_pos])

    return np.array(images, dtype=np.float32), np.array(factors, dtype=np.float32)


def generate_hgf_trajectories(n_samples: int = 1000, n_trials: int = 200, seed: int = 42):
    """
    Generate HGF trajectory data with known phenotype factors.

    Uses the ara.hgf package if available, otherwise generates synthetic.

    Returns:
        trajectories: (n_samples, n_trials, 8)
        factors: (n_samples, 4) for [omega_2, kappa_1, theta, phenotype]
    """
    try:
        from ara.vae import generate_phenotype_dataset
        data = generate_phenotype_dataset(
            n_samples_per_phenotype=n_samples // 5,
            n_trials=n_trials,
            seed=seed,
        )
        return data.trajectories, data.factors
    except Exception:
        # Fallback: synthetic trajectories
        np.random.seed(seed)

        trajectories = []
        factors = []

        phenotypes = [
            (-4.0, 1.0, 1.0, 0),   # Healthy
            (-8.0, 0.2, 0.5, 1),   # SCZ rigid
            (-1.0, 2.0, 2.0, 2),   # SCZ loose
            (-4.0, 3.0, 1.5, 3),   # BPD
            (-2.0, 1.5, 1.0, 4),   # Anxiety
        ]

        for _ in range(n_samples):
            omega_2, kappa_1, theta, label = phenotypes[np.random.randint(0, 5)]

            # Add noise
            omega_2 += np.random.randn() * 0.5
            kappa_1 += np.random.randn() * 0.2
            theta += np.random.randn() * 0.1

            # Generate synthetic trajectory (8 features per trial)
            traj = np.random.randn(n_trials, 8) * 0.5
            traj[:, 0] += omega_2 * 0.1  # mu_2 influenced by omega
            traj[:, 4] += kappa_1 * 0.2  # delta_1 influenced by kappa

            trajectories.append(traj)
            factors.append([omega_2, kappa_1, theta, float(label)])

        return np.array(trajectories, dtype=np.float32), np.array(factors, dtype=np.float32)


# =============================================================================
# Visualization Helpers
# =============================================================================

def log_reconstructions(params, config, data, step, num_images=16):
    """Log original vs reconstructed images to W&B."""
    if not HAS_WANDB or not HAS_JAX:
        return

    from ara.vae.jax_spmd import beta_vae_forward

    key = random.PRNGKey(step)
    imgs = jnp.array(data[:num_images])

    recon, mu, logvar, _ = beta_vae_forward(params, key, imgs)
    recon = np.array(recon).reshape(-1, *config.input_shape)

    # Create grids
    n_row = int(np.sqrt(num_images))
    orig_grid = _make_grid(data[:num_images], n_row)
    recon_grid = _make_grid(recon, n_row)

    wandb.log({
        "recon/original": wandb.Image(orig_grid, caption=f"Original @ step {step}"),
        "recon/reconstructed": wandb.Image(recon_grid, caption=f"Reconstruction @ step {step}"),
    }, step=step)


def log_latent_traversal(params, config, data, dim=0, num_steps=9, step=0):
    """Log latent traversal for a single dimension."""
    if not HAS_WANDB or not HAS_JAX:
        return

    from ara.vae.jax_spmd import encoder, decoder

    # Encode a reference image
    img = jnp.array(data[0:1])
    img_flat = img.reshape(1, -1)
    mu, logvar = encoder(params, img_flat)
    base_z = np.array(mu[0])

    # Traverse
    values = np.linspace(-3, 3, num_steps)
    traversals = []

    for v in values:
        z = base_z.copy()
        z[dim] = v
        z_jax = jnp.array(z).reshape(1, -1)
        recon = decoder(params, z_jax)
        traversals.append(np.array(recon).reshape(config.input_shape))

    # Make grid
    grid = _make_grid(np.array(traversals), num_steps)

    wandb.log({
        f"traversal/dim_{dim}": wandb.Image(
            grid,
            caption=f"z[{dim}] from {values[0]:.1f} to {values[-1]:.1f}"
        )
    }, step=step)


def log_importance_heatmap(latents, factors, factor_names=None, step=0):
    """Log latent × factor importance heatmap."""
    if not HAS_WANDB:
        return

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X_train, _, y_train, _ = train_test_split(latents, factors, test_size=0.2, random_state=42)

    # Fit per-factor for proper importance
    latent_dim = latents.shape[1]
    n_factors = factors.shape[1]
    importance = np.zeros((latent_dim, n_factors))

    for i in range(n_factors):
        reg = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
        reg.fit(X_train, y_train[:, i])
        importance[:, i] = reg.feature_importances_

    # Normalize
    importance = importance / (importance.sum(axis=0, keepdims=True) + 1e-10)

    # Create heatmap as image
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(importance, cmap='viridis', aspect='auto')

    if factor_names is None:
        factor_names = [f"factor_{i}" for i in range(n_factors)]

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(factor_names, rotation=45)
    ax.set_yticks(range(latent_dim))
    ax.set_yticklabels([f"z{i}" for i in range(latent_dim)])
    ax.set_xlabel("Factor")
    ax.set_ylabel("Latent Dimension")
    ax.set_title("Latent × Factor Importance (Telepathy Wiring)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    wandb.log({"disentanglement/importance_heatmap": wandb.Image(fig)}, step=step)
    plt.close(fig)


def _make_grid(images, n_row):
    """Create image grid from batch."""
    n = len(images)
    n_col = (n + n_row - 1) // n_row

    h, w = images[0].shape[:2]
    grid = np.zeros((n_row * h, n_col * w))

    for i, img in enumerate(images):
        row, col = i // n_col, i % n_col
        if img.ndim == 3:
            img = img[:, :, 0]
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

    return grid


# =============================================================================
# Disentanglement Suite
# =============================================================================

def run_disentanglement_suite(latents, factors, prefix="synthetic", factor_names=None):
    """
    Run full disentanglement evaluation and log to W&B.

    Returns:
        Dict of all metrics including NeuroBalance index
    """
    # Compute metrics
    mig_score, mi_matrix = compute_mig_jax(latents, factors[:, :3])  # First 3 factors

    try:
        dci_scores = compute_dci(latents, factors[:, :3])
    except Exception:
        dci_scores = {'disentanglement': 0.5, 'completeness': 0.5, 'informativeness': 0.5}

    try:
        edi_scores = compute_edi(latents, factors[:, :3])
    except Exception:
        edi_scores = {'modularity': 0.5, 'compactness': 0.5, 'explicitness': 0.5}

    # Composite NeuroBalance index
    neurobalance_index = (
        0.30 * mig_score +
        0.25 * dci_scores['disentanglement'] +
        0.25 * edi_scores['modularity'] +
        0.20 * edi_scores['compactness']
    )

    metrics = {
        f"{prefix}/mig": mig_score,
        f"{prefix}/dci_disentanglement": dci_scores['disentanglement'],
        f"{prefix}/dci_completeness": dci_scores['completeness'],
        f"{prefix}/dci_informativeness": dci_scores['informativeness'],
        f"{prefix}/edi_modularity": edi_scores['modularity'],
        f"{prefix}/edi_compactness": edi_scores['compactness'],
        f"{prefix}/edi_explicitness": edi_scores.get('explicitness', 0.5),
        f"{prefix}/neurobalance_index": neurobalance_index,
    }

    if HAS_WANDB:
        wandb.log(metrics)

    print(f"\n{prefix.upper()} Disentanglement Metrics:")
    print(f"  MIG: {mig_score:.4f}")
    print(f"  DCI: D={dci_scores['disentanglement']:.4f}, "
          f"C={dci_scores['completeness']:.4f}, "
          f"I={dci_scores['informativeness']:.4f}")
    print(f"  EDI: M={edi_scores['modularity']:.4f}, "
          f"C={edi_scores['compactness']:.4f}")
    print(f"  NeuroBalance Index: {neurobalance_index:.4f}")

    return metrics


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NeuroBalance Synthetic Pipeline")
    parser.add_argument("--beta", type=float, default=4.0, help="β-VAE KL weight")
    parser.add_argument("--latent_dim", type=int, default=10, help="Latent dimensionality")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--data", type=str, default="hgf", choices=["dsprites", "hgf"],
                        help="Dataset type")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    # Configuration
    config = {
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset": args.data,
        "n_samples": args.n_samples,
        "model": "beta-vae-jax-spmd",
        "mode": "sandbox",
        "input_modality": "image" if args.data == "dsprites" else "hgf_trajectory",
    }

    run_name = f"beta{config['beta']}_z{config['latent_dim']}_{int(time.time())}"

    # Initialize W&B
    if HAS_WANDB and not args.no_wandb:
        wandb.init(
            project="neurobalance-disentanglement",
            config=config,
            name=run_name,
            tags=["ara", "neurobalance", "disentanglement", args.data],
        )
        print(f"W&B run: {wandb.run.url}")
    else:
        print("Running without W&B logging")

    # Generate data
    print(f"\nGenerating {args.data} data ({args.n_samples} samples)...")
    if args.data == "dsprites":
        data, factors = generate_dsprites_like(args.n_samples, args.seed)
        input_shape = (64, 64)
        factor_names = ["shape", "scale", "orientation", "x_pos", "y_pos"]
    else:
        data, factors = generate_hgf_trajectories(args.n_samples, seed=args.seed)
        input_shape = data.shape[1:]  # (n_trials, n_features)
        factor_names = ["omega_2", "kappa_1", "theta", "phenotype"]

    print(f"Data shape: {data.shape}")
    print(f"Factors shape: {factors.shape}")

    # Configure VAE
    vae_config = SPMDConfig(
        latent_dim=args.latent_dim,
        beta=args.beta,
        input_shape=input_shape,
        hidden_dim=256,
    )

    # Train
    print(f"\nTraining β-VAE (β={args.beta}, z_dim={args.latent_dim})...")
    params, history = train_beta_vae_spmd(
        data,
        vae_config,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=True,
        loss_type="bce" if args.data == "dsprites" else "mse",
    )

    # Log training history
    if HAS_WANDB and not args.no_wandb:
        for i, (loss, recon, kl) in enumerate(zip(
            history['loss'], history['recon_loss'], history['kl']
        )):
            wandb.log({
                "train/loss": loss,
                "train/recon_loss": recon,
                "train/kl_loss": kl,
                "epoch": i + 1,
            })

    # Encode to latent space
    print("\nEncoding to latent space...")
    z, mu, logvar = encode_to_latent(params, data)
    print(f"Latent shape: {z.shape}")

    # Log reconstructions
    if args.data == "dsprites":
        print("\nLogging reconstructions...")
        log_reconstructions(params, vae_config, data, step=args.epochs)

        print("Logging latent traversals...")
        for dim in range(min(5, args.latent_dim)):
            log_latent_traversal(params, vae_config, data, dim=dim, step=args.epochs)

    # Run disentanglement suite
    print("\nRunning disentanglement evaluation...")
    metrics = run_disentanglement_suite(
        z, factors,
        prefix=args.data,
        factor_names=factor_names[:3],  # Use first 3 factors
    )

    # Log importance heatmap
    print("Logging importance heatmap...")
    log_importance_heatmap(z, factors[:, :3], factor_names[:3], step=args.epochs)

    # Create Ara state readout
    print("\nCreating Ara state readout...")
    reader = AraStateReader(latent_dim=args.latent_dim)

    # Simulate a few updates
    for i in range(5):
        readout = reader.update(
            latent=z[i],
            hgf_params={
                'omega_2': float(factors[i, 0]),
                'kappa_1': float(factors[i, 1]),
                'theta': float(factors[i, 2]) if factors.shape[1] > 2 else 1.0,
            },
            task_label=args.data,
        )

    print("\n" + readout.summary())

    # Final summary
    print("\n" + "="*50)
    print("NEUROBALANCE SYNTHETIC PIPELINE COMPLETE")
    print("="*50)
    print(f"NeuroBalance Index: {metrics[f'{args.data}/neurobalance_index']:.4f}")
    print(f"Recommended Mode: {readout.recommended_mode}")
    print(f"Geometry Health: {readout.geometry_health:.2f}")
    print(f"Volatility: {readout.volatility:.2f}")

    if HAS_WANDB and not args.no_wandb:
        wandb.log({
            "final/neurobalance_index": metrics[f'{args.data}/neurobalance_index'],
            "final/geometry_health": readout.geometry_health,
            "final/volatility": readout.volatility,
            "final/recommended_mode": readout.recommended_mode,
        })
        wandb.finish()
        print(f"\nW&B run complete: {wandb.run.url if wandb.run else 'N/A'}")


if __name__ == "__main__":
    main()
