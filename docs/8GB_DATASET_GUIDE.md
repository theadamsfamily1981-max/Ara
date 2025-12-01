# 8GB Dataset Training Guide for SNN

Complete guide to generating, loading, and training on large-scale temporal datasets with SNN emulation.

## Overview

This system provides infrastructure for training SNNs on **8GB temporal datasets** with:
- ✅ **98.44% parameter reduction** (N=4096, r=32, k=64)
- ✅ **Efficient HDF5 streaming** (doesn't load all data into RAM)
- ✅ **Data augmentation** for temporal sequences
- ✅ **FDT-controlled learning** for stability
- ✅ **Multi-task support** (classification, multi-label, regression)

## Quick Start

```bash
# 1. Generate 8GB dataset (takes ~15-30 minutes)
python scripts/generate_8gb_dataset.py \
  --output data/snn_8gb.h5 \
  --type rate \
  --size-gb 8.0 \
  --num-classes 10

# 2. Inspect dataset
python scripts/generate_8gb_dataset.py --output data/snn_8gb.h5 --inspect

# 3. Train SNN
python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml

# 4. Resume training
python scripts/train_snn_8gb.py \
  --config configs/snn_8gb_training.yaml \
  --resume checkpoints/snn_8gb/best.pt
```

---

## Dataset Generation

### Data Types

The generator supports 4 data types optimized for SNNs:

#### 1. **Poisson Spike Trains** (`--type poisson`)

Random Poisson spike trains with varying firing rates.

**Use case**: Baseline for testing SNN dynamics
**Label**: Firing rates per neuron [N]

```bash
python scripts/generate_8gb_dataset.py \
  --output data/snn_poisson_8gb.h5 \
  --type poisson \
  --size-gb 8.0
```

**Characteristics**:
- Independent Poisson processes per neuron
- Firing rates: 0.05 to 0.3 Hz
- Sparse (~10-20% spikes)

#### 2. **Temporal Patterns** (`--type temporal`)

Embedded temporal patterns in background noise.

**Use case**: Pattern detection, sequence learning
**Label**: Binary pattern presence per neuron [N]

```bash
python scripts/generate_8gb_dataset.py \
  --output data/snn_temporal_8gb.h5 \
  --type temporal \
  --size-gb 8.0
```

**Characteristics**:
- 5 embedded patterns per sequence
- Pattern length: 20 time steps
- Background noise: 5% firing rate
- Patterns localized in neuron space

#### 3. **Rate-Encoded Classification** (`--type rate`) ⭐ **RECOMMENDED**

Rate-coded temporal sequences for multi-class classification.

**Use case**: Classification tasks (like MNIST, CIFAR-10)
**Label**: One-hot class labels [num_classes]

```bash
python scripts/generate_8gb_dataset.py \
  --output data/snn_8gb.h5 \
  --type rate \
  --size-gb 8.0 \
  --num-classes 10
```

**Characteristics**:
- Class-specific spatial rate patterns
- Temporal modulation (sinusoidal)
- Background rate: 5%
- Active neurons per class: N/num_classes
- Firing rates: 30-60% for active neurons

#### 4. **Event-Driven Sequences** (`--type event`)

Sparse event-driven data (like DVS cameras).

**Use case**: Event-based vision, neuromorphic sensing
**Label**: Event density per neuron [N]

```bash
python scripts/generate_8gb_dataset.py \
  --output data/snn_event_8gb.h5 \
  --type event \
  --size-gb 8.0
```

**Characteristics**:
- 50 events per sequence
- Localized spatiotemporal events
- Temporal spread: 5 time steps
- Very sparse (~1-2% spikes)

### Dataset Parameters

#### Size Calculation

For **N=4096** neurons, **T=256** time steps, **float32** (4 bytes):

```python
# Input size per sequence
input_size = T × N × 4 bytes = 256 × 4096 × 4 = 4,194,304 bytes = 4 MB

# Label size (multi-label)
label_size = N × 4 bytes = 4096 × 4 = 16,384 bytes = 16 KB

# Total per sequence
sequence_size = 4 MB + 16 KB ≈ 4.016 MB

# Number of sequences for 8 GB
num_sequences = 8 GB / 4.016 MB ≈ 2,038 sequences
  - Training: 1,834 sequences (90%)
  - Validation: 204 sequences (10%)
```

#### Custom Parameters

```bash
# Smaller dataset (1 GB for testing)
python scripts/generate_8gb_dataset.py \
  --output data/snn_1gb.h5 \
  --type rate \
  --size-gb 1.0 \
  --quick

# Different network size
python scripts/generate_8gb_dataset.py \
  --output data/snn_large.h5 \
  --type rate \
  --N 8192 \
  --T 512 \
  --size-gb 16.0

# Different time resolution
python scripts/generate_8gb_dataset.py \
  --output data/snn_highres.h5 \
  --type rate \
  --T 512 \
  --size-gb 8.0
```

### HDF5 Structure

```
snn_8gb.h5
├── train/
│   ├── inputs   [num_train, T, N]  float32  # Spike sequences
│   └── labels   [num_train, ...]   float32  # Labels
├── val/
│   ├── inputs   [num_val, T, N]    float32
│   └── labels   [num_val, ...]     float32
└── attrs/
    ├── data_type: "rate"
    ├── N: 4096
    ├── T: 256
    ├── num_train: 1834
    ├── num_val: 204
    ├── num_classes: 10
    ├── size_gb: 8.0
    └── created_at: "2025-11-16 12:34:56"
```

---

## Data Loading

### PyTorch Dataset

```python
from tfan.data import SNNTemporalDataset, get_snn_dataloader

# Create dataset
dataset = SNNTemporalDataset(
    hdf5_path='data/snn_8gb.h5',
    split='train',  # 'train' or 'val'
    cache_size=256,  # Cache first 256 samples in RAM
    temporal_jitter=8,  # ±8 time steps jitter
    spike_dropout=0.05,  # 5% spike dropout
)

# Create data loader
loader = get_snn_dataloader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

# Iterate
for batch_inputs, batch_labels in loader:
    # batch_inputs: [batch, T, N] spike sequences
    # batch_labels: [batch, num_classes] or [batch, N]
    ...
```

### Data Augmentation

The dataset supports temporal augmentation:

```python
from tfan.data import TemporalAugmentation

augment = TemporalAugmentation(
    temporal_jitter=10,  # Random shift ±10 time steps
    spike_dropout=0.1,  # Drop 10% of spikes
    temporal_scaling=(0.9, 1.1),  # Compress/expand time 90-110%
    spatial_permute=False,  # Permute neuron ordering
)

dataset = SNNTemporalDataset(
    hdf5_path='data/snn_8gb.h5',
    split='train',
    transform=augment,
)
```

### Efficient Streaming

**No RAM overload**: Data streams from disk via HDF5 chunking.

```python
# Only loads batches into RAM, not entire 8 GB
for batch in loader:
    # Process batch
    ...
# Total RAM usage: batch_size × sequence_size = 16 × 4 MB = 64 MB per batch
```

**Caching**: Cache frequently accessed samples for speedup.

```python
dataset = SNNTemporalDataset(
    hdf5_path='data/snn_8gb.h5',
    cache_size=512,  # Cache first 512 samples (~2 GB RAM)
)

# Check cache stats
stats = dataset.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

---

## Model Configuration

### SNN Parameters for 8GB Dataset

**File**: `configs/snn_8gb_training.yaml`

#### Network Architecture

```yaml
model:
  N: 4096  # Must match dataset N
  lowrank_rank: 32  # r = 32 → 0.78% of N
  k_per_row: 64  # k = 64 → 1.56% of N
```

**Parameter reduction**:
```python
dense_params = N² = 4096² = 16,777,216
snn_params = 2×N×r + k×N = 2×4096×32 + 64×4096 = 262,144

reduction = (1 - 262,144 / 16,777,216) × 100% = 98.44%
```

**Gates enforced**:
- ✓ param_reduction ≥ 97%
- ✓ avg_degree = 64 ≤ 0.02×4096 = 81.92
- ✓ rank = 32 ≤ 0.02×4096 = 81.92
- ✓ sparsity = 98.44% ≥ 98%

#### SNN Dynamics

```yaml
snn:
  v_th: 1.0  # Spike threshold
  alpha: 0.95  # Membrane decay (longer memory)
  surrogate_scale: 0.3  # Gradient scale
  time_steps: 256  # Must match dataset T
  use_spectral_norm: true  # Prevent exploding gradients
  tau_mem: 20.0  # Membrane time constant (ms)
  tau_syn: 5.0  # Synaptic time constant (ms)
```

#### Training Configuration

```yaml
training:
  learning_rate: 2.0e-3  # Higher LR for sparse gradients
  weight_decay: 1e-4
  grad_clip: 1.0
  batch_size: 16  # Adjust for GPU memory
  max_epochs: 50
  warmup_epochs: 5
  lr_schedule: cosine
```

#### FDT Control

```yaml
tfan:
  use_fdt: true
  fdt:
    kp: 0.30  # Proportional gain
    ki: 0.02  # Integral gain
    kd: 0.10  # Derivative gain
    target_epr_cv: 0.15  # Target uncertainty
```

---

## Training

### Basic Training

```bash
python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml
```

**Output**:
```
============================================================
SNN Training on 8GB Dataset
============================================================
Config: configs/snn_8gb_training.yaml
Device: cuda
Epochs: 50
Batch size: 16
============================================================

Loading dataset...
Data loaders created:
  Train: 1,834 samples
  Val: 204 samples
  Batch size: 16
  Num workers: 8

Building SNN backend...

SNN Model Summary:
  N: 4,096
  Rank: 32
  Avg degree: 64.0
  Parameters: 262,144 (vs 16,777,216 dense)
  Reduction: 98.44%
  Sparsity: 98.44%

============================================================
Starting Training
============================================================

Epoch 1/50 [Train]: 100%|█████████| 115/115 [01:23<00:00]
  loss: 0.4123
  accuracy: 0.7234
  spike_rate: 0.152
  grad_norm: 0.832

Epoch 1/50 - Validation:
  val_loss: 0.3987
  val_accuracy: 0.7456
  val_spike_rate: 0.148
  ✓ New best validation loss: 0.3987

...
```

### Advanced Options

```bash
# Resume training
python scripts/train_snn_8gb.py \
  --config configs/snn_8gb_training.yaml \
  --resume checkpoints/snn_8gb/best.pt

# Override epochs
python scripts/train_snn_8gb.py \
  --config configs/snn_8gb_training.yaml \
  --epochs 100

# Override batch size
python scripts/train_snn_8gb.py \
  --config configs/snn_8gb_training.yaml \
  --batch-size 32

# CPU training
python scripts/train_snn_8gb.py \
  --config configs/snn_8gb_training.yaml \
  --device cpu
```

### Checkpointing

**Automatic checkpointing**:
- `latest.pt` - Latest epoch
- `best.pt` - Best validation loss
- `epoch_XXX.pt` - Every N epochs

**Config**:
```yaml
checkpoint:
  save_dir: checkpoints/snn_8gb
  save_every_n_epochs: 5
  keep_last_n: 3  # Keep last 3 epoch checkpoints
  save_best: true
```

**Load checkpoint**:
```python
checkpoint = torch.load('checkpoints/snn_8gb/best.pt')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
metrics = checkpoint['metrics']
```

---

## Performance

### Expected Metrics

For **N=4096, r=32, k=64, batch=16** on **NVIDIA A100 (40GB)**:

| Metric | Value |
|--------|-------|
| Parameters | 262,144 (98.44% reduction) |
| VRAM (training) | ~4-6 GB |
| Training time per epoch | ~2-3 minutes |
| Throughput | ~600-800 samples/sec |
| Spike rate | 10-20% |
| Final accuracy | 75-85% (dataset dependent) |

### Memory Usage

**GPU memory breakdown** (batch=16):

```
Model parameters:        262,144 × 4 bytes = 1.0 MB
Optimizer state (Adam):  262,144 × 8 bytes = 2.0 MB  (m, v)
Batch input:             16 × 256 × 4096 × 4 = 64 MB
Batch gradients:         ~64 MB
Activations:             ~100-200 MB
──────────────────────────────────────────────────
Total:                   ~250-350 MB

With overhead:           ~1-2 GB typical
Large batch (32):        ~2-4 GB
```

**Scaling to larger networks**:

| N | r | k | Params | VRAM (batch=16) | Dataset Size (8GB) |
|---|---|---|--------|-----------------|-------------------|
| 2048 | 16 | 32 | 65,536 | ~1 GB | ~32K sequences |
| 4096 | 32 | 64 | 262,144 | ~4 GB | ~2K sequences |
| 8192 | 64 | 128 | 1,048,576 | ~16 GB | ~512 sequences |

### Optimization Tips

#### 1. Increase Batch Size (if GPU memory allows)

```yaml
dataloader:
  batch_size: 32  # Double throughput
```

#### 2. More DataLoader Workers

```yaml
dataloader:
  num_workers: 16  # Use more CPU cores
  pin_memory: true  # Faster GPU transfer
  prefetch_factor: 4  # Prefetch more batches
```

#### 3. Mixed Precision Training

```python
# In train_snn_8gb.py, add:
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 4. Gradient Accumulation

```python
# Effective batch size = batch_size × accumulation_steps
accumulation_steps = 4

for i, (inputs, labels) in enumerate(loader):
    loss = compute_loss(inputs, labels)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Examples

### Example 1: Train on Rate-Coded Dataset

```bash
# Generate dataset
python scripts/generate_8gb_dataset.py \
  --output data/snn_rate_8gb.h5 \
  --type rate \
  --num-classes 10 \
  --size-gb 8.0

# Train
python scripts/train_snn_8gb.py \
  --config configs/snn_8gb_training.yaml
```

### Example 2: Quick Test with 1GB Dataset

```bash
# Generate small dataset
python scripts/generate_8gb_dataset.py \
  --output data/snn_1gb.h5 \
  --type rate \
  --quick  # Generates 1 GB

# Create test config
cp configs/snn_8gb_training.yaml configs/snn_1gb_test.yaml
# Edit: dataset.path = data/snn_1gb.h5

# Train
python scripts/train_snn_8gb.py \
  --config configs/snn_1gb_test.yaml \
  --epochs 10
```

### Example 3: Custom Data Type

```bash
# Event-driven dataset
python scripts/generate_8gb_dataset.py \
  --output data/snn_event_8gb.h5 \
  --type event \
  --size-gb 8.0

# Update config for multi-label task
# In snn_8gb_training.yaml:
#   dataset.type: event
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'h5py'"

**Solution**: Install HDF5 dependencies
```bash
pip install h5py
```

### Issue: "OSError: Unable to create file (unable to open file)"

**Cause**: Insufficient disk space

**Solution**: Check available space
```bash
df -h data/
# Need at least ~10 GB free for 8 GB dataset (with compression)
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size
```yaml
dataloader:
  batch_size: 8  # or 4
```

### Issue: Training is very slow

**Possible causes**:
1. **Disk I/O bottleneck**: Use SSD for dataset storage
2. **Too few workers**: Increase `num_workers`
3. **No caching**: Enable `cache_size`

**Solution**:
```yaml
dataloader:
  num_workers: 16
  cache_size: 512  # Cache ~2 GB in RAM
```

### Issue: "RuntimeError: DataLoader worker exited unexpectedly"

**Cause**: HDF5 file corruption or worker crash

**Solution**:
```bash
# Regenerate dataset
python scripts/generate_8gb_dataset.py --output data/snn_8gb.h5 --type rate --size-gb 8.0

# Reduce workers
num_workers: 4
```

---

## See Also

- [Backend Integration Guide](BACKEND_INTEGRATION_GUIDE.md) - Backend factory usage
- [Testing Guide](TESTING_GUIDE.md) - Comprehensive testing
- [SNN Implementation Details](../tfan/snn/README.md) - Low-level SNN implementation
