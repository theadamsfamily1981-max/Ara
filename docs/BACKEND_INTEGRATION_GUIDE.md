# Backend Integration Guide

Complete guide to using the TF-A-N backend factory system for switching between Dense, TF-A-N (SSA), and SNN emulation backends.

## Overview

The backend factory provides a unified interface for training with different model architectures:

- **dense**: Standard dense transformers (baseline)
- **tfan**: TF-A-N with Selective Sparse Attention (SSA)
- **snn_emu**: SNN emulation with low-rank masked synapses (97-99% param reduction)

All backends integrate with TF-A-N control systems (FDT, PGU, topology) and provide consistent training hooks.

## Quick Start

### 1. Select Backend via Configuration

**SNN Emulation** (97-99% parameter reduction):
```yaml
# configs/my_snn_config.yaml
backend: snn_emu

model:
  N: 4096
  lowrank_rank: 32
  k_per_row: 64

snn:
  v_th: 1.0
  alpha: 0.95
  time_steps: 256

training:
  learning_rate: 1.5e-3
  grad_clip: 1.0
```

**TF-A-N with SSA**:
```yaml
# configs/my_tfan_config.yaml
backend: tfan

model:
  config_path: "tfan/models/tfan7b/config.json"

tfan:
  use_fdt: true
  fdt:
    kp: 0.30
    ki: 0.02
    target_epr_cv: 0.15

training:
  learning_rate: 1e-4
  grad_clip: 1.0
```

**Dense Baseline**:
```yaml
# configs/my_dense_config.yaml
backend: dense

model:
  hidden_size: 4096

training:
  learning_rate: 1e-4
  grad_clip: 1.0
```

### 2. Training Script Integration

#### Simple Training Loop

```python
from tfan.backends import build_backend
import torch

# Load config
config = {
    'backend': 'snn_emu',
    'model': {'N': 4096, 'lowrank_rank': 32, 'k_per_row': 64},
    'snn': {'v_th': 1.0, 'alpha': 0.95, 'time_steps': 256},
    'training': {'learning_rate': 1.5e-3, 'grad_clip': 1.0},
    'device': 'cuda',
    'dtype': 'float16',
}

# Build backend
backend = build_backend(config)

# Access model, optimizer, hooks
model = backend.model
optimizer = backend.optim
hooks = backend.hooks

# Move to device
backend.to_device('cuda')

# Print summary
print(backend.summary())

# Training loop
for step, batch in enumerate(dataloader):
    # Forward
    output, aux = model(batch['input'])
    loss = compute_loss(output, batch['labels'])
    aux['loss'] = loss.item()

    # Backward
    loss.backward()

    # Pre-step hooks (grad clipping, spectral norm)
    hooks.before_step(model)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Post-step hooks (FDT, spike EMA, metrics)
    hooks.after_step(model, aux)

    # Logging
    if step % 100 == 0:
        hooks.log(step, aux)
```

#### Full Training Script with YAML Config

```python
#!/usr/bin/env python
"""
Training script with backend factory.

Usage:
    python train_backend.py --config configs/snn_emu_4096.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
from tfan.backends import build_backend


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_dummy_dataloader(config, num_batches=1000):
    """Create dummy dataloader for demonstration."""
    batch_size = config.get('training', {}).get('batch_size', 2)
    N = config.get('model', {}).get('N', 4096)

    class DummyDataset:
        def __len__(self):
            return num_batches

        def __getitem__(self, idx):
            return {
                'input': torch.randn(N),
                'labels': torch.randint(0, 2, (N,)),
            }

    return torch.utils.data.DataLoader(
        DummyDataset(),
        batch_size=batch_size,
        shuffle=True
    )


def main(config_path):
    """Main training loop."""
    # Load config
    config = load_config(config_path)

    # Build backend
    print("Building backend...")
    backend = build_backend(config)

    # Print summary
    summary = backend.summary()
    print("\n" + "="*60)
    print("BACKEND SUMMARY")
    print("="*60)
    for key, value in summary.items():
        print(f"{key:20s}: {value}")
    print("="*60 + "\n")

    # Get components
    model = backend.model
    optimizer = backend.optim
    hooks = backend.hooks

    # Move to device
    device = config.get('device', 'cpu')
    backend.to_device(device)

    # Create dataloader
    dataloader = create_dummy_dataloader(config)

    # Training config
    train_cfg = config.get('training', {})
    max_steps = train_cfg.get('max_steps', 1000)
    log_interval = train_cfg.get('log_interval', 100)

    # Training loop
    print(f"Starting training for {max_steps} steps...")
    step = 0

    for epoch in range(100):  # Outer loop
        for batch in dataloader:
            if step >= max_steps:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward
            output, aux = model(batch['input'])

            # Compute loss (dummy: just sum)
            loss = output.sum()
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

            # Logging
            if step % log_interval == 0:
                hooks.log(step, aux)

            step += 1

        if step >= max_steps:
            break

    print(f"\nTraining complete! Trained for {step} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    main(args.config)
```

## Backend-Specific Features

### SNN Emulation Backend

**Unique features**:
- 97-99% parameter reduction via low-rank masked synapses
- Event-driven processing (10-100× throughput gain)
- Spike rate monitoring and EMA
- FDT integration with spike-rate aware temperature
- Numerical stability (grad clip, spectral norm, NaN detection)

**Configuration**:
```yaml
backend: snn_emu

model:
  N: 4096                  # Number of neurons
  lowrank_rank: 32         # Rank (2Nr parameters)
  k_per_row: 64            # Avg outgoing degree

snn:
  v_th: 1.0                # Spike threshold
  alpha: 0.95              # Membrane leak decay
  surrogate_scale: 0.3     # Surrogate gradient scale
  time_steps: 256          # Simulation duration
  use_spectral_norm: false # Spectral norm on U, V

training:
  learning_rate: 1.5e-3    # Higher for SNN
  grad_clip: 1.0
  weight_decay: 0.01

tfan:
  use_fdt: true            # Enable FDT homeostasis
  fdt:
    kp: 0.30
    ki: 0.02
    target_epr_cv: 0.15
```

**Accessing SNN-specific metrics**:
```python
backend = build_backend(config)
model, hooks = backend.model, backend.hooks

# Forward pass
output, aux = model(input_batch)

# SNN metrics in aux
print(f"Spike rate: {aux['spike_rate']:.3f}")
print(f"Spike sparsity: {aux['spike_sparsity']:.2%}")
print(f"Active events: {aux['active_events']}")

# FDT metrics (if enabled)
if 'epr_cv' in aux:
    print(f"EPR-CV: {aux['epr_cv']:.4f}")
    print(f"v_th multiplier: {aux.get('v_th_multiplier', 1.0):.3f}")
```

### TF-A-N Backend

**Unique features**:
- Selective Sparse Attention (SSA) with O(N log N)
- FDT PID controller for EPR-CV ≤ 0.15
- Topology regularization
- PGU proof-gated updates

**Configuration**:
```yaml
backend: tfan

model:
  config_path: "tfan/models/tfan7b/config.json"

tfan:
  use_fdt: true
  fdt:
    kp: 0.30
    ki: 0.02
    kd: 0.10
    target_epr_cv: 0.15

  use_topology: true
  topology:
    lambda_topo: 0.1

  use_pgu: false  # Optional formal verification

training:
  learning_rate: 1e-4
  grad_clip: 1.0
```

**Accessing TF-A-N metrics**:
```python
# FDT metrics
print(f"EPR-CV: {aux['epr_cv']:.4f}")
print(f"LR multiplier: {aux['lr_multiplier']:.3f}")
print(f"Temp multiplier: {aux['temp_multiplier']:.3f}")
```

### Dense Baseline Backend

Simplest backend for comparison.

```yaml
backend: dense

model:
  hidden_size: 4096

training:
  learning_rate: 1e-4
  grad_clip: 1.0
```

## Lifecycle Hooks

All backends provide lifecycle hooks for training integration:

### before_step(model)

Called before `optimizer.step()`.

**Operations**:
- Gradient clipping (all backends)
- Spectral normalization on U, V (SNN only)
- NaN/Inf gradient detection (SNN only)

**Example**:
```python
# In training loop, before optimizer.step()
hooks.before_step(model)
optimizer.step()
```

### after_step(model, aux)

Called after `optimizer.step()`.

**Operations**:
- FDT PID updates (TF-A-N, SNN)
- Spike rate EMA (SNN)
- Metric collection
- EPR-CV computation

**Example**:
```python
# In training loop, after optimizer.step()
optimizer.step()
hooks.after_step(model, aux)

# aux is updated with metrics
print(f"EPR-CV: {aux.get('epr_cv', 0):.4f}")
```

### log(step, aux)

Called for periodic logging.

**Example**:
```python
if step % 100 == 0:
    hooks.log(step, aux)
```

## Switching Backends

### During Development

Simply change the `backend` field in your config:

```bash
# Try SNN emulation
python train.py --config configs/snn_emu_4096.yaml

# Compare with TF-A-N
python train.py --config configs/7b/quanta_focus.yaml  # backend: tfan

# Compare with dense baseline
python train.py --config configs/dense_baseline.yaml
```

### Programmatic Switching

```python
configs = {
    'snn': {'backend': 'snn_emu', ...},
    'tfan': {'backend': 'tfan', ...},
    'dense': {'backend': 'dense', ...},
}

results = {}
for name, cfg in configs.items():
    backend = build_backend(cfg)
    # Train and evaluate...
    results[name] = evaluate(backend)

# Compare
print("Parameter counts:")
for name, res in results.items():
    print(f"  {name}: {res['params']:,}")
```

## Validation and Gates

### SNN Gates (CI-enforced)

All SNN configurations must pass these gates:

```python
from tfan.snn import verify_all_gates

# After building SNN backend
if config['backend'] == 'snn_emu':
    # Get model parameters
    N = config['model']['N']
    r = config['model']['lowrank_rank']
    mask = ...  # Get mask from model

    # Verify gates
    gates = verify_all_gates(
        N=N,
        r=r,
        indptr=mask['indptr'],
        param_reduction_min=97.0,
        degree_frac_max=0.02,
        rank_frac_max=0.02
    )

    assert all(gates.values()), f"SNN gates failed: {gates}"
```

**CI automatically validates these gates** on every push.

### TF-A-N Gates

```python
# EPR-CV gate (FDT)
assert aux['epr_cv'] <= 0.15, f"EPR-CV {aux['epr_cv']:.4f} > 0.15"

# SSA speedup gate (benchmark)
assert speedup_vs_dense >= 3.0, f"SSA speedup {speedup_vs_dense:.1f}× < 3×"
```

## Troubleshooting

### SNN: Silent Network (No Spikes)

**Symptom**: `spike_rate` near zero, `spike_sparsity` near 100%.

**Solutions**:
1. Lower `v_th` (try 0.5-0.8)
2. Increase `surrogate_scale` (try 0.5-1.0)
3. Check input magnitude (normalize to [-1, 1])
4. Increase `time_steps` (try 512)

### SNN: Event Burst Fallback

**Symptom**: High `active_events`, low throughput gain.

**Solutions**:
1. Increase `v_th` to reduce spike rate
2. Adjust `alpha` (higher = longer integration, fewer spikes)
3. Check `sparsity_threshold` in EventDrivenStepper

### TF-A-N: EPR-CV Not Converging

**Symptom**: `epr_cv` oscillates or exceeds 0.15.

**Solutions**:
1. Tune FDT PID gains (reduce `kp`, increase `ki`)
2. Increase `ema_alpha` for smoother response
3. Check gradient variance (should be stable)

### Dense: Out of Memory

**Solutions**:
1. Reduce batch size
2. Use gradient checkpointing
3. Switch to SNN backend (97% less memory)

## Performance Comparison

Expected metrics (N=4096):

| Backend | Parameters | Memory | Forward Latency | Throughput |
|---------|------------|--------|-----------------|------------|
| Dense   | 16.7M      | 67MB   | ~3ms            | 1× (baseline) |
| TF-A-N  | 16.7M      | 67MB   | ~1ms            | 3× (SSA) |
| SNN     | 262k       | 1MB    | ~0.8ms          | 10-100× (event-driven) |

## Next Steps

1. **Run benchmarks**: `python scripts/bench_snn.py --roofline`
2. **Validate gates**: CI automatically checks on every push
3. **Compare backends**: Train all three, compare accuracy/efficiency
4. **Optimize**: Use roofline data to guide CUDA kernel development

## See Also

- [SNN Implementation Guide](SNN_IMPLEMENTATION_GUIDE.md) - SNN details
- [Production Guide](PRODUCTION_GUIDE.md) - Deployment best practices
- [QUANTA Config Guide](../configs/README.md) - Configuration reference
