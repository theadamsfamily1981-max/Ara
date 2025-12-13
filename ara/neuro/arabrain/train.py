#!/usr/bin/env python3
"""
Sharded Training Script for EEGAraBrain

Implements 2D mesh parallelism (batch + model) for training
EEGAraBrain on EEG data with cognitive overload labels.

Features:
- 2D device mesh: batch parallel × model parallel
- JIT-compiled train step with sharding constraints
- WandB integration for experiment tracking
- Checkpoint saving/loading
- Disentanglement metric evaluation (MIG, DCI)

Usage:
    # Single GPU (no parallelism)
    python -m ara.neuro.arabrain.train --batch_size 32

    # Multi-GPU with data parallelism
    python -m ara.neuro.arabrain.train --data_parallel 4 --batch_size 128

    # Full 2D mesh (batch + model parallel)
    python -m ara.neuro.arabrain.train --data_parallel 4 --model_parallel 2

    # Demo mode (synthetic data)
    python -m ara.neuro.arabrain.train --demo
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    from flax.training import checkpoints
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .mesh import MeshContext, get_data_sharding, shard_params, ShardingConfig
from .model import EEGAraBrain, create_train_state, EEGAraBrainTrainState


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""

    # Model
    latent_dim: int = 32
    beta: float = 4.0
    telepathy_weight: float = 1.0
    encoder_type: str = "standard"
    decoder_type: str = "mlp"

    # Data
    time: int = 256         # EEG window samples
    channels: int = 32      # EEG channels
    batch_size: int = 32

    # Parallelism
    data_parallel: int = 1
    model_parallel: int = 1

    # Training
    learning_rate: float = 1e-3
    num_epochs: int = 100
    steps_per_epoch: int = 100
    eval_every: int = 10
    checkpoint_every: int = 25

    # Paths
    checkpoint_dir: str = "./checkpoints/arabrain"
    data_dir: str = "./data/eeg"

    # Logging
    use_wandb: bool = False
    wandb_project: str = "arabrain"
    wandb_run_name: Optional[str] = None

    # Debug
    demo: bool = False  # Use synthetic data


# =============================================================================
# Data Loading
# =============================================================================

def create_synthetic_data(
    config: TrainConfig,
    num_samples: int = 1000,
    rng: Optional[jnp.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic EEG data for demo/testing.

    Generates EEG-like signals with different patterns for
    'overload' vs 'normal' conditions.
    """
    if rng is None:
        rng = jax.random.PRNGKey(42)

    rng, data_rng, label_rng = jax.random.split(rng, 3)

    # Generate labels (30% overload)
    labels = jax.random.bernoulli(label_rng, 0.3, (num_samples,))

    # Generate EEG data
    # Normal: lower amplitude, slower oscillations
    # Overload: higher amplitude, faster oscillations

    t = np.linspace(0, 2 * np.pi, config.time)

    eeg_data = []
    for i in range(num_samples):
        if labels[i]:
            # Overload: high-frequency, high amplitude
            freq_base = 20 + np.random.rand(config.channels) * 30
            amplitude = 1.5 + np.random.rand(config.channels) * 0.5
        else:
            # Normal: lower frequency, lower amplitude
            freq_base = 5 + np.random.rand(config.channels) * 10
            amplitude = 0.5 + np.random.rand(config.channels) * 0.3

        # Generate multi-channel signal
        signal = np.zeros((config.time, config.channels))
        for ch in range(config.channels):
            signal[:, ch] = amplitude[ch] * np.sin(freq_base[ch] * t)
            # Add harmonics
            signal[:, ch] += 0.3 * amplitude[ch] * np.sin(2 * freq_base[ch] * t)
            # Add noise
            signal[:, ch] += 0.1 * np.random.randn(config.time)

        eeg_data.append(signal)

    eeg_data = np.array(eeg_data, dtype=np.float32)

    # Normalize to [0, 1] range
    eeg_data = (eeg_data - eeg_data.min()) / (eeg_data.max() - eeg_data.min() + 1e-8)

    return eeg_data, np.array(labels, dtype=np.float32)


def create_data_iterator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Create batched data iterator."""
    num_samples = len(x)
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        # Pad last batch if needed
        if len(batch_indices) < batch_size:
            pad_size = batch_size - len(batch_indices)
            batch_indices = np.concatenate([
                batch_indices,
                batch_indices[:pad_size]
            ])

        yield x[batch_indices], y[batch_indices]


# =============================================================================
# Training Step
# =============================================================================

def make_train_step(mesh: Optional[Mesh] = None):
    """
    Create JIT-compiled train step with optional sharding.

    Returns a function that performs one training step.
    """

    def train_step(
        state: EEGAraBrainTrainState,
        x: jnp.ndarray,
        y: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> Tuple[EEGAraBrainTrainState, Dict[str, Any]]:
        """
        Single training step.

        Args:
            state: Current train state
            x: EEG batch (B, T, C)
            y: Labels (B,)
            rng: Random key

        Returns:
            new_state: Updated train state
            metrics: Training metrics dict
        """

        def loss_fn(params):
            loss, outputs = state.apply_fn(
                {'params': params},
                x, rng,
                labels=y,
                training=True,
            )
            return loss, outputs

        # Compute gradients
        (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # Update parameters
        new_state = state.apply_gradients(grads=grads)

        metrics = {
            'loss': loss,
            'recon_loss': outputs['recon_loss'],
            'kl_loss': outputs['kl_loss'],
            'telepathy_loss': outputs.get('telepathy_loss', 0.0),
            'telepathy_accuracy': outputs.get('telepathy_accuracy', 0.0),
        }

        return new_state, metrics

    # JIT compile
    if mesh is not None:
        # With sharding
        data_sharding = get_data_sharding(mesh)
        train_step_jit = jax.jit(
            train_step,
            in_shardings=(
                None,  # state (replicated)
                data_sharding,  # x
                NamedSharding(mesh, P('data')),  # y
                NamedSharding(mesh, P()),  # rng
            ),
            out_shardings=(
                None,  # new_state
                None,  # metrics (replicated)
            ),
        )
    else:
        train_step_jit = jax.jit(train_step)

    return train_step_jit


def make_eval_step(mesh: Optional[Mesh] = None):
    """Create JIT-compiled evaluation step."""

    def eval_step(
        state: EEGAraBrainTrainState,
        x: jnp.ndarray,
        y: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate without gradient computation."""
        loss, outputs = state.apply_fn(
            {'params': state.params},
            x, rng,
            labels=y,
            training=False,
        )
        return {
            'loss': loss,
            'recon_loss': outputs['recon_loss'],
            'kl_loss': outputs['kl_loss'],
            'telepathy_loss': outputs.get('telepathy_loss', 0.0),
            'telepathy_accuracy': outputs.get('telepathy_accuracy', 0.0),
            'z': outputs['z'],
        }

    if mesh is not None:
        data_sharding = get_data_sharding(mesh)
        eval_step_jit = jax.jit(
            eval_step,
            in_shardings=(
                None,
                data_sharding,
                NamedSharding(mesh, P('data')),
                NamedSharding(mesh, P()),
            ),
        )
    else:
        eval_step_jit = jax.jit(eval_step)

    return eval_step_jit


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: TrainConfig):
    """Main training function."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX required for training")

    print("\n" + "=" * 70)
    print("EEGAraBrain TRAINING")
    print("=" * 70)

    # Initialize RNG
    rng = jax.random.PRNGKey(42)

    # -------------------------------------------------------------------------
    # Setup mesh parallelism
    # -------------------------------------------------------------------------
    print(f"\nDevice setup:")
    print(f"  Available devices: {len(jax.devices())}")
    print(f"  Data parallel: {config.data_parallel}")
    print(f"  Model parallel: {config.model_parallel}")

    mesh_ctx = MeshContext(
        data_parallel=config.data_parallel,
        model_parallel=config.model_parallel,
    )

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    print(f"\nModel config:")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Beta: {config.beta}")
    print(f"  Encoder: {config.encoder_type}")
    print(f"  EEG shape: ({config.time}, {config.channels})")

    model = EEGAraBrain(
        latent_dim=config.latent_dim,
        time=config.time,
        channels=config.channels,
        beta=config.beta,
        telepathy_weight=config.telepathy_weight,
        encoder_type=config.encoder_type,
        decoder_type=config.decoder_type,
    )

    # Initialize train state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        init_rng,
        model,
        config.learning_rate,
        input_shape=(config.batch_size, config.time, config.channels),
    )

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"  Parameters: {param_count:,}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print(f"\nData setup:")
    if config.demo:
        print("  Using synthetic data (demo mode)")
        rng, data_rng = jax.random.split(rng)
        train_x, train_y = create_synthetic_data(
            config,
            num_samples=config.batch_size * config.steps_per_epoch,
            rng=data_rng,
        )
        val_x, val_y = create_synthetic_data(
            config,
            num_samples=config.batch_size * 10,
            rng=jax.random.PRNGKey(123),
        )
    else:
        # Load real data
        data_path = Path(config.data_dir)
        if not data_path.exists():
            print(f"  Data directory not found: {data_path}")
            print("  Falling back to synthetic data")
            train_x, train_y = create_synthetic_data(config)
            val_x, val_y = create_synthetic_data(config, num_samples=500)
        else:
            # Load from npz files
            train_data = np.load(data_path / "train.npz")
            train_x, train_y = train_data['x'], train_data['y']
            val_data = np.load(data_path / "val.npz")
            val_x, val_y = val_data['x'], val_data['y']

    print(f"  Train samples: {len(train_x)}")
    print(f"  Val samples: {len(val_x)}")

    # -------------------------------------------------------------------------
    # Setup logging
    # -------------------------------------------------------------------------
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # -------------------------------------------------------------------------
    # Create train/eval steps
    # -------------------------------------------------------------------------
    with mesh_ctx:
        train_step = make_train_step(mesh_ctx.mesh)
        eval_step = make_eval_step(mesh_ctx.mesh)

        # -------------------------------------------------------------------------
        # Training loop
        # -------------------------------------------------------------------------
        print(f"\nTraining:")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Steps per epoch: {config.steps_per_epoch}")

        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(1, config.num_epochs + 1):
            epoch_start = time.time()
            epoch_metrics = []

            # Training epoch
            train_iter = create_data_iterator(
                train_x, train_y, config.batch_size, shuffle=True
            )

            for step, (x_batch, y_batch) in enumerate(train_iter):
                if step >= config.steps_per_epoch:
                    break

                rng, step_rng = jax.random.split(rng)

                # Shard data to devices
                x_batch = jnp.array(x_batch)
                y_batch = jnp.array(y_batch)
                x_batch = mesh_ctx.shard_data(x_batch)
                y_batch = jax.device_put(
                    y_batch,
                    NamedSharding(mesh_ctx.mesh, P('data'))
                )

                # Train step
                state, metrics = train_step(state, x_batch, y_batch, step_rng)
                epoch_metrics.append(metrics)

            # Aggregate epoch metrics
            avg_metrics = {
                k: float(np.mean([m[k] for m in epoch_metrics]))
                for k in epoch_metrics[0].keys()
            }

            epoch_time = time.time() - epoch_start

            # Logging
            log_str = (
                f"Epoch {epoch:3d} | "
                f"Loss: {avg_metrics['loss']:.4f} | "
                f"Recon: {avg_metrics['recon_loss']:.4f} | "
                f"KL: {avg_metrics['kl_loss']:.4f} | "
                f"Telepathy: {avg_metrics['telepathy_accuracy']:.2%} | "
                f"Time: {epoch_time:.1f}s"
            )
            print(log_str)

            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': avg_metrics['loss'],
                    'train/recon_loss': avg_metrics['recon_loss'],
                    'train/kl_loss': avg_metrics['kl_loss'],
                    'train/telepathy_loss': avg_metrics['telepathy_loss'],
                    'train/telepathy_accuracy': avg_metrics['telepathy_accuracy'],
                })

            # Evaluation
            if epoch % config.eval_every == 0:
                val_metrics = []
                val_iter = create_data_iterator(
                    val_x, val_y, config.batch_size, shuffle=False
                )

                for x_batch, y_batch in val_iter:
                    rng, eval_rng = jax.random.split(rng)
                    x_batch = jnp.array(x_batch)
                    y_batch = jnp.array(y_batch)
                    x_batch = mesh_ctx.shard_data(x_batch)
                    y_batch = jax.device_put(
                        y_batch,
                        NamedSharding(mesh_ctx.mesh, P('data'))
                    )

                    metrics = eval_step(state, x_batch, y_batch, eval_rng)
                    val_metrics.append({k: v for k, v in metrics.items() if k != 'z'})

                avg_val = {
                    k: float(np.mean([m[k] for m in val_metrics]))
                    for k in val_metrics[0].keys()
                }

                print(f"  Val | Loss: {avg_val['loss']:.4f} | "
                      f"Telepathy: {avg_val['telepathy_accuracy']:.2%}")

                if config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'val/loss': avg_val['loss'],
                        'val/recon_loss': avg_val['recon_loss'],
                        'val/telepathy_accuracy': avg_val['telepathy_accuracy'],
                    })

                # Best model tracking
                if avg_val['loss'] < best_val_loss:
                    best_val_loss = avg_val['loss']
                    print(f"  New best validation loss: {best_val_loss:.4f}")

            # Checkpointing
            if epoch % config.checkpoint_every == 0:
                ckpt_path = Path(config.checkpoint_dir)
                ckpt_path.mkdir(parents=True, exist_ok=True)
                checkpoints.save_checkpoint(
                    ckpt_path,
                    state,
                    step=epoch,
                    keep=3,
                )
                print(f"  Checkpoint saved: epoch {epoch}")

        # -------------------------------------------------------------------------
        # Training complete
        # -------------------------------------------------------------------------
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("Training complete!")
        print(f"  Total time: {total_time / 60:.1f} minutes")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print("=" * 70)

        if config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        return state


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train EEGAraBrain with 2D mesh parallelism"
    )

    # Model args
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--beta', type=float, default=4.0)
    parser.add_argument('--telepathy_weight', type=float, default=1.0)
    parser.add_argument('--encoder_type', type=str, default='standard',
                       choices=['standard', 'spatial_temporal', 'frequency_aware'])
    parser.add_argument('--decoder_type', type=str, default='mlp',
                       choices=['mlp', 'conv_transpose'])

    # Data args
    parser.add_argument('--time', type=int, default=256)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='./data/eeg')

    # Parallelism
    parser.add_argument('--data_parallel', type=int, default=1)
    parser.add_argument('--model_parallel', type=int, default=1)

    # Training args
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--checkpoint_every', type=int, default=25)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/arabrain')

    # Logging
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='arabrain')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Debug
    parser.add_argument('--demo', action='store_true',
                       help='Use synthetic data for testing')

    args = parser.parse_args()

    config = TrainConfig(
        latent_dim=args.latent_dim,
        beta=args.beta,
        telepathy_weight=args.telepathy_weight,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        time=args.time,
        channels=args.channels,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        data_parallel=args.data_parallel,
        model_parallel=args.model_parallel,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        demo=args.demo,
    )

    train(config)


if __name__ == "__main__":
    main()
