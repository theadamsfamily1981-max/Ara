#!/usr/bin/env python
"""
Accuracy and Energy Benchmark Script

Compares Dense, TF-A-N, and SNN emulation backends on:
- Accuracy (loss convergence)
- EPR-CV (epistemic uncertainty)
- Parameter count
- VRAM usage
- Energy efficiency (throughput/watt proxy)

Usage:
    python scripts/bench_accuracy_energy.py --steps 1000 --output artifacts/comparison.json
    python scripts/bench_accuracy_energy.py --quick  # Fast smoke test
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from tfan.backends import build_backend


def create_dummy_dataset(num_samples=1000, N=4096, seed=42):
    """
    Create dummy dataset for benchmarking.

    In production, replace with real dataset loader.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    inputs = torch.randn(num_samples, N)
    labels = torch.randint(0, 2, (num_samples, N)).float()

    return inputs, labels


def measure_vram_usage(model, device='cuda'):
    """
    Measure VRAM usage in MB.

    Returns:
        vram_mb: VRAM allocated in megabytes
    """
    if device == 'cpu':
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)

    # Dummy forward pass
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'lif'):
            # SNN model
            N = model.lif.N
            dummy_input = torch.randn(2, N, device=device)
        else:
            # Dense or TF-A-N
            dummy_input = torch.randn(2, 4096, device=device)

        try:
            _ = model(dummy_input)
        except:
            pass  # May fail for placeholder models

    vram_bytes = torch.cuda.max_memory_allocated(device)
    vram_mb = vram_bytes / (1024 ** 2)

    return vram_mb


def train_backend(
    backend_name: str,
    config: Dict[str, Any],
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Train a backend and collect metrics.

    Args:
        backend_name: 'dense', 'tfan', or 'snn_emu'
        config: Backend configuration
        inputs: Input data [num_samples, N]
        labels: Target labels [num_samples, N]
        num_steps: Training steps
        device: 'cpu' or 'cuda'

    Returns:
        results: Dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {backend_name.upper()} backend")
    print(f"{'='*60}")

    # Build backend
    backend = build_backend(config)
    model = backend.model
    optimizer = backend.optim
    hooks = backend.hooks

    # Move to device
    backend.to_device(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())

    # Measure VRAM
    vram_mb = measure_vram_usage(model, device)

    # Training loop
    losses = []
    epr_cvs = []
    spike_rates = []
    throughputs = []

    batch_size = config.get('training', {}).get('batch_size', 2)
    num_samples = inputs.shape[0]

    t_start = time.perf_counter()

    for step in range(num_steps):
        # Sample batch
        idx = torch.randint(0, num_samples, (batch_size,))
        batch_input = inputs[idx]
        batch_labels = labels[idx]

        # Forward
        step_start = time.perf_counter()
        output, aux = model(batch_input)

        # Loss (simple MSE for demonstration)
        loss = ((output - batch_labels) ** 2).mean()
        aux['loss'] = loss.item()

        # Backward
        loss.backward()

        # Pre-step hooks
        hooks.before_step(model)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Post-step hooks
        hooks.after_step(model, aux)

        step_end = time.perf_counter()
        step_time = step_end - step_start

        # Collect metrics
        losses.append(loss.item())

        if 'epr_cv' in aux:
            epr_cvs.append(aux['epr_cv'])

        if 'spike_rate' in aux:
            spike_rates.append(aux['spike_rate'])

        # Throughput (samples/sec)
        throughput = batch_size / step_time
        throughputs.append(throughput)

        # Logging
        if step % 100 == 0:
            hooks.log(step, aux)

    t_end = time.perf_counter()
    total_time = t_end - t_start

    # Compute final metrics
    final_loss = np.mean(losses[-100:])  # Last 100 steps
    initial_loss = np.mean(losses[:10])
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    avg_epr_cv = np.mean(epr_cvs[-100:]) if epr_cvs else None
    avg_spike_rate = np.mean(spike_rates[-100:]) if spike_rates else None
    avg_throughput = np.mean(throughputs)

    results = {
        'backend': backend_name,
        'param_count': param_count,
        'vram_mb': vram_mb,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction_pct': loss_reduction,
        'avg_epr_cv': avg_epr_cv,
        'avg_spike_rate': avg_spike_rate,
        'avg_throughput': avg_throughput,
        'total_time_sec': total_time,
        'steps': num_steps,
    }

    # Backend-specific metrics
    if backend_name == 'snn_emu':
        summary = model.lif.summary()
        results['param_reduction_pct'] = summary['reduction_pct']
        results['avg_degree'] = summary['avg_degree']
        results['sparsity'] = summary['sparsity']

    print(f"\n{backend_name.upper()} Results:")
    print(f"  Parameters: {param_count:,}")
    print(f"  VRAM: {vram_mb:.1f} MB")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {loss_reduction:.1f}%")
    if avg_epr_cv is not None:
        print(f"  EPR-CV: {avg_epr_cv:.4f}")
    if avg_spike_rate is not None:
        print(f"  Spike rate: {avg_spike_rate:.3f}")
    print(f"  Throughput: {avg_throughput:.1f} samples/sec")
    print(f"  Total time: {total_time:.2f} sec")

    return results


def compare_backends(
    num_steps: int = 1000,
    device: str = 'cpu',
    output_path: str = None
):
    """
    Compare Dense, TF-A-N, and SNN backends.

    Args:
        num_steps: Training steps per backend
        device: 'cpu' or 'cuda'
        output_path: Path to save JSON results
    """
    # Create dataset
    print("Creating dataset...")
    N = 4096
    inputs, labels = create_dummy_dataset(num_samples=1000, N=N)

    # Backend configurations
    configs = {
        'dense': {
            'backend': 'dense',
            'model': {'hidden_size': N},
            'training': {
                'learning_rate': 1e-4,
                'grad_clip': 1.0,
                'batch_size': 2,
            },
            'device': device,
        },
        'tfan': {
            'backend': 'tfan',
            'model': {'hidden_size': N},
            'tfan': {
                'use_fdt': True,
                'fdt': {
                    'kp': 0.30,
                    'ki': 0.02,
                    'target_epr_cv': 0.15,
                },
            },
            'training': {
                'learning_rate': 1e-4,
                'grad_clip': 1.0,
                'batch_size': 2,
            },
            'device': device,
        },
        'snn_emu': {
            'backend': 'snn_emu',
            'model': {
                'N': N,
                'lowrank_rank': 32,
                'k_per_row': 64,
            },
            'snn': {
                'v_th': 1.0,
                'alpha': 0.95,
                'surrogate_scale': 0.3,
                'time_steps': 256,
                'use_spectral_norm': False,
            },
            'tfan': {
                'use_fdt': True,
                'fdt': {
                    'kp': 0.30,
                    'ki': 0.02,
                    'target_epr_cv': 0.15,
                },
            },
            'training': {
                'learning_rate': 1.5e-3,
                'grad_clip': 1.0,
                'batch_size': 2,
            },
            'device': device,
        },
    }

    # Train each backend
    all_results = {}

    for backend_name, config in configs.items():
        try:
            results = train_backend(
                backend_name=backend_name,
                config=config,
                inputs=inputs,
                labels=labels,
                num_steps=num_steps,
                device=device
            )
            all_results[backend_name] = results
        except Exception as e:
            print(f"\n⚠ {backend_name} failed: {e}")
            all_results[backend_name] = {'error': str(e)}

    # Print comparison table
    print(f"\n{'='*80}")
    print("BACKEND COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Dense':>15} {'TF-A-N':>15} {'SNN':>15}")
    print(f"{'-'*80}")

    metrics = [
        ('param_count', 'Parameters', lambda x: f"{x:,}"),
        ('vram_mb', 'VRAM (MB)', lambda x: f"{x:.1f}"),
        ('final_loss', 'Final Loss', lambda x: f"{x:.4f}"),
        ('loss_reduction_pct', 'Loss Reduction (%)', lambda x: f"{x:.1f}"),
        ('avg_epr_cv', 'EPR-CV', lambda x: f"{x:.4f}" if x else "N/A"),
        ('avg_throughput', 'Throughput (samp/s)', lambda x: f"{x:.1f}"),
        ('total_time_sec', 'Time (sec)', lambda x: f"{x:.1f}"),
    ]

    for metric_key, metric_name, fmt in metrics:
        row = f"{metric_name:<30}"
        for backend_name in ['dense', 'tfan', 'snn_emu']:
            if backend_name in all_results and metric_key in all_results[backend_name]:
                value = all_results[backend_name][metric_key]
                row += f"{fmt(value):>15}"
            else:
                row += f"{'N/A':>15}"
        print(row)

    print(f"{'-'*80}")

    # Efficiency metrics
    if 'dense' in all_results and 'snn_emu' in all_results:
        dense_params = all_results['dense'].get('param_count', 1)
        snn_params = all_results['snn_emu'].get('param_count', 1)
        param_reduction = (1 - snn_params / dense_params) * 100

        dense_vram = all_results['dense'].get('vram_mb', 1)
        snn_vram = all_results['snn_emu'].get('vram_mb', 1)
        vram_reduction = (1 - snn_vram / dense_vram) * 100 if dense_vram > 0 else 0

        print(f"\nSNN vs Dense:")
        print(f"  Parameter reduction: {param_reduction:.1f}%")
        print(f"  VRAM reduction: {vram_reduction:.1f}%")

    print(f"{'='*80}\n")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Results saved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare backend accuracy and energy")
    parser.add_argument('--steps', type=int, default=1000, help="Training steps per backend")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device")
    parser.add_argument('--output', type=str, default='artifacts/comparison.json', help="Output JSON path")
    parser.add_argument('--quick', action='store_true', help="Quick mode (100 steps)")

    args = parser.parse_args()

    num_steps = 100 if args.quick else args.steps

    print(f"Running accuracy/energy benchmark:")
    print(f"  Steps: {num_steps}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output}")

    results = compare_backends(
        num_steps=num_steps,
        device=args.device,
        output_path=args.output
    )

    # Check gates for SNN
    if 'snn_emu' in results and 'param_reduction_pct' in results['snn_emu']:
        snn = results['snn_emu']
        gates = {
            'param_reduction >= 97%': snn.get('param_reduction_pct', 0) >= 97.0,
            'avg_degree <= 0.02*N': snn.get('avg_degree', float('inf')) <= 0.02 * 4096,
        }

        print("\nSNN Gate Validation:")
        for gate_name, passed in gates.items():
            status = '✓' if passed else '✗'
            print(f"  {status} {gate_name}")

        if all(gates.values()):
            print("\n✅ All SNN gates passed!")
        else:
            print("\n❌ Some SNN gates failed!")
            return 1

    return 0


if __name__ == '__main__':
    exit(main())
