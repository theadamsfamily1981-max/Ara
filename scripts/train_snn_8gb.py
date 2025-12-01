#!/usr/bin/env python
"""
SNN Training Script for 8GB Dataset

Trains SNN emulation model on large-scale temporal dataset with:
- Efficient streaming from HDF5
- FDT-controlled learning rate adaptation
- Comprehensive metrics and logging
- Checkpointing and early stopping

Usage:
    python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml
    python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml --resume checkpoints/snn_8gb/latest.pt
    python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml --epochs 100 --batch-size 32
"""

import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from typing import Dict, Any

from tfan.backends import build_backend
from tfan.data import create_data_loaders


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor, task_type: str = 'classification') -> float:
    """
    Compute accuracy based on task type.

    Args:
        outputs: Model outputs [batch, ...]
        labels: Ground truth [batch, ...]
        task_type: 'classification', 'multilabel', or 'regression'

    Returns:
        accuracy: Accuracy metric
    """
    if task_type == 'classification':
        # Multi-class classification
        preds = outputs.argmax(dim=1)
        targets = labels.argmax(dim=1) if labels.dim() > 1 else labels
        accuracy = (preds == targets).float().mean().item()
    elif task_type == 'multilabel':
        # Multi-label classification
        preds = (outputs > 0.5).float()
        accuracy = (preds == labels).float().mean().item()
    else:
        # Regression: use R^2 score
        ss_res = ((outputs - labels) ** 2).sum().item()
        ss_tot = ((labels - labels.mean()) ** 2).sum().item()
        accuracy = 1 - (ss_res / (ss_tot + 1e-8))

    return accuracy


def train_epoch(
    backend,
    train_loader,
    epoch: int,
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        backend: SNN backend
        train_loader: Training data loader
        epoch: Current epoch
        config: Configuration dict
        device: Device

    Returns:
        metrics: Dict of epoch metrics
    """
    model = backend.model
    optimizer = backend.optim
    hooks = backend.hooks

    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    total_spike_rate = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    # Determine task type
    dataset_config = config.get('dataset', {})
    if dataset_config.get('type') == 'rate':
        task_type = 'classification'
        loss_fn = nn.CrossEntropyLoss()
    else:
        task_type = 'multilabel'
        loss_fn = nn.BCEWithLogitsLoss()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Move to device
        inputs = inputs.to(device)  # [batch, T, N]
        labels = labels.to(device)  # [batch, num_classes] or [batch, N]

        # Forward pass
        # SNN expects inputs sequentially, but we can process batch at once
        # For simplicity, we average over time for the output
        batch_size, T, N = inputs.shape

        # Initialize state
        if hasattr(model, 'lif'):
            # SNN model
            v = torch.zeros(batch_size, N, device=device)
            spike_counts = torch.zeros(batch_size, N, device=device)

            # Run through time
            for t in range(T):
                external_input = inputs[:, t, :]  # [batch, N]
                v, spikes = model.lif(v, external_input=external_input)
                spike_counts += spikes

            # Output: average spike rate
            outputs = spike_counts / T
        else:
            # Dense model fallback
            outputs = model(inputs.mean(dim=1))  # Average over time

        # Compute loss
        if task_type == 'classification':
            loss = loss_fn(outputs, labels)
        else:
            loss = loss_fn(outputs, labels)

        # Backward
        loss.backward()

        # Pre-step hooks (gradient clipping, spectral norm)
        hooks.before_step(model)

        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float('inf')
        )

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Post-step hooks (FDT, PGU)
        aux = {
            'loss': loss.item(),
            'spike_rate': (spike_counts / T).mean().item() if hasattr(model, 'lif') else 0.0,
        }
        hooks.after_step(model, aux)

        # Compute accuracy
        with torch.no_grad():
            accuracy = compute_accuracy(outputs, labels, task_type)

        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += accuracy
        if hasattr(model, 'lif'):
            total_spike_rate += (spike_counts / T).mean().item()
        total_grad_norm += grad_norm.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy:.4f}",
            'spike': f"{(spike_counts / T).mean().item():.3f}" if hasattr(model, 'lif') else "N/A",
        })

        # Logging
        if batch_idx % config['logging']['log_every_n_steps'] == 0:
            hooks.log(epoch * len(train_loader) + batch_idx, aux)

    # Epoch metrics
    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'spike_rate': total_spike_rate / num_batches,
        'grad_norm': total_grad_norm / num_batches,
    }

    return metrics


def validate(
    backend,
    val_loader,
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Validate model.

    Args:
        backend: SNN backend
        val_loader: Validation data loader
        config: Configuration dict
        device: Device

    Returns:
        metrics: Dict of validation metrics
    """
    model = backend.model
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_spike_rate = 0.0
    num_batches = 0

    # Determine task type
    dataset_config = config.get('dataset', {})
    if dataset_config.get('type') == 'rate':
        task_type = 'classification'
        loss_fn = nn.CrossEntropyLoss()
    else:
        task_type = 'multilabel'
        loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            batch_size, T, N = inputs.shape

            # Forward pass
            if hasattr(model, 'lif'):
                v = torch.zeros(batch_size, N, device=device)
                spike_counts = torch.zeros(batch_size, N, device=device)

                for t in range(T):
                    external_input = inputs[:, t, :]
                    v, spikes = model.lif(v, external_input=external_input)
                    spike_counts += spikes

                outputs = spike_counts / T
                spike_rate = (spike_counts / T).mean().item()
            else:
                outputs = model(inputs.mean(dim=1))
                spike_rate = 0.0

            # Compute loss
            if task_type == 'classification':
                loss = loss_fn(outputs, labels)
            else:
                loss = loss_fn(outputs, labels)

            # Compute accuracy
            accuracy = compute_accuracy(outputs, labels, task_type)

            total_loss += loss.item()
            total_accuracy += accuracy
            total_spike_rate += spike_rate
            num_batches += 1

    metrics = {
        'val_loss': total_loss / num_batches,
        'val_accuracy': total_accuracy / num_batches,
        'val_spike_rate': total_spike_rate / num_batches,
    }

    return metrics


def save_checkpoint(
    backend,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint_dir = Path(config['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': backend.model.state_dict(),
        'optimizer_state_dict': backend.optim.state_dict(),
        'metrics': metrics,
        'config': config,
    }

    # Save latest
    latest_path = checkpoint_dir / 'latest.pt'
    torch.save(checkpoint, latest_path)

    # Save epoch checkpoint
    if (epoch + 1) % config['checkpoint']['save_every_n_epochs'] == 0:
        epoch_path = checkpoint_dir / f'epoch_{epoch+1:03d}.pt'
        torch.save(checkpoint, epoch_path)

    # Save best
    if is_best and config['checkpoint']['save_best']:
        best_path = checkpoint_dir / 'best.pt'
        torch.save(checkpoint, best_path)

    # Cleanup old checkpoints
    if config['checkpoint']['keep_last_n'] > 0:
        checkpoints = sorted(checkpoint_dir.glob('epoch_*.pt'))
        if len(checkpoints) > config['checkpoint']['keep_last_n']:
            for old_ckpt in checkpoints[:-config['checkpoint']['keep_last_n']]:
                old_ckpt.unlink()


def main():
    parser = argparse.ArgumentParser(description="Train SNN on 8GB dataset")
    parser.add_argument('--config', type=str, required=True, help="Config YAML path")
    parser.add_argument('--resume', type=str, default=None, help="Resume from checkpoint")
    parser.add_argument('--epochs', type=int, default=None, help="Override max epochs")
    parser.add_argument('--batch-size', type=int, default=None, help="Override batch size")
    parser.add_argument('--device', type=str, default=None, help="Override device")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.epochs is not None:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size is not None:
        config['dataloader']['batch_size'] = args.batch_size
        config['training']['batch_size'] = args.batch_size
    if args.device is not None:
        config['device'] = args.device

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print("SNN Training on 8GB Dataset")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Epochs: {config['training']['max_epochs']}")
    print(f"Batch size: {config['dataloader']['batch_size']}")
    print(f"{'='*60}\n")

    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader = create_data_loaders(
        hdf5_path=config['dataset']['path'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        cache_size=config['dataloader']['cache_size'],
        augment_train=config['dataloader']['augment_train'],
        temporal_jitter=config['dataloader']['temporal_jitter'],
        spike_dropout=config['dataloader']['spike_dropout'],
    )

    # Build backend
    print("\nBuilding SNN backend...")
    backend = build_backend(config)
    backend.to_device(device)

    # Print model summary
    if hasattr(backend.model, 'lif'):
        summary = backend.model.lif.summary()
        print(f"\nSNN Model Summary:")
        print(f"  N: {summary['N']:,}")
        print(f"  Rank: {summary['rank']}")
        print(f"  Avg degree: {summary['avg_degree']:.1f}")
        print(f"  Parameters: {summary['lowrank_params']:,} (vs {summary['dense_params']:,} dense)")
        print(f"  Reduction: {summary['reduction_pct']:.2f}%")
        print(f"  Sparsity: {summary['sparsity']:.2%}")

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        backend.model.load_state_dict(checkpoint['model_state_dict'])
        backend.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }

    epochs_without_improvement = 0

    for epoch in range(start_epoch, config['training']['max_epochs']):
        # Train
        train_metrics = train_epoch(backend, train_loader, epoch, config, device)

        print(f"\nEpoch {epoch+1}/{config['training']['max_epochs']} - Train:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Validate
        if (epoch + 1) % config['validation']['val_every_n_epochs'] == 0:
            val_metrics = validate(backend, val_loader, config, device)

            print(f"Epoch {epoch+1}/{config['training']['max_epochs']} - Validation:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Update history
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_accuracy'].append(val_metrics['val_accuracy'])

            # Check for improvement
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
                epochs_without_improvement = 0
                print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            all_metrics = {**train_metrics, **val_metrics}
            save_checkpoint(backend, epoch, all_metrics, config, is_best)

            # Early stopping
            patience = config['training'].get('early_stopping_patience', 10)
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break
        else:
            # Save checkpoint without validation
            save_checkpoint(backend, epoch, train_metrics, config, False)

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])

    # Save final history
    history_path = Path(config['checkpoint']['save_dir']) / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config['checkpoint']['save_dir']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
