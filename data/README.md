# SNN Dataset Directory

This directory contains generated datasets for SNN training.

## Quick Start

```bash
# Generate 8GB dataset (recommended for production)
python ../scripts/generate_8gb_dataset.py \
  --output snn_8gb.h5 \
  --type rate \
  --num-classes 10

# Generate 1GB dataset (for testing)
python ../scripts/generate_8gb_dataset.py \
  --output snn_1gb.h5 \
  --type rate \
  --num-classes 10 \
  --quick

# Inspect dataset
python ../scripts/generate_8gb_dataset.py \
  --output snn_8gb.h5 \
  --inspect
```

## Dataset Types

| Type | Size | Use Case | Label Shape |
|------|------|----------|-------------|
| `rate` | 8 GB | Classification (MNIST-like) | [num_classes] |
| `poisson` | 8 GB | Baseline testing | [N] |
| `temporal` | 8 GB | Pattern detection | [N] |
| `event` | 8 GB | Event-based vision | [N] |

## Dataset Structure

```
snn_8gb.h5
├── train/
│   ├── inputs   [1834, 256, 4096]  # Training sequences
│   └── labels   [1834, 10]         # Training labels
├── val/
│   ├── inputs   [204, 256, 4096]   # Validation sequences
│   └── labels   [204, 10]          # Validation labels
└── attrs/
    ├── N: 4096
    ├── T: 256
    ├── num_classes: 10
    └── ...
```

## Parameters for 8GB Dataset

**Network size**: N=4096, T=256
**Sequences**: ~2,038 total (1,834 train, 204 val)
**Sequence size**: 4 MB each
**Total size**: ~8 GB

## Training

```bash
# Train on 8GB dataset
python ../scripts/train_snn_8gb.py \
  --config ../configs/snn_8gb_training.yaml

# Resume training
python ../scripts/train_snn_8gb.py \
  --config ../configs/snn_8gb_training.yaml \
  --resume ../checkpoints/snn_8gb/best.pt
```

## SNN Parameters (Optimized for 8GB)

```yaml
model:
  N: 4096
  lowrank_rank: 32     # 98.44% param reduction
  k_per_row: 64

snn:
  time_steps: 256      # Must match dataset T
  v_th: 1.0
  alpha: 0.95

training:
  batch_size: 16
  learning_rate: 2.0e-3
```

**Expected metrics**:
- Parameters: 262,144 (vs 16,777,216 dense)
- Reduction: 98.44%
- VRAM: ~4-6 GB (batch=16)
- Training time: ~2-3 min/epoch (A100)

## See Also

- [8GB Dataset Guide](../docs/8GB_DATASET_GUIDE.md) - Complete documentation
- [Backend Integration Guide](../docs/BACKEND_INTEGRATION_GUIDE.md) - Model usage
- [Testing Guide](../docs/TESTING_GUIDE.md) - Validation and testing
