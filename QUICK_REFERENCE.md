# Quick Reference: SNN Training with 8GB Dataset

One-page reference for SNN emulation training.

## 🚀 Complete Workflow

```bash
# 1. Generate dataset (15-30 min)
python scripts/generate_8gb_dataset.py --output data/snn_8gb.h5 --type rate --num-classes 10

# 2. Train SNN
python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml

# 3. Validate gates
python scripts/validate_all_gates.py --config configs/snn_8gb_training.yaml
```

## 📊 Dataset Generation

```bash
# Production (8 GB)
python scripts/generate_8gb_dataset.py --output data/snn_8gb.h5 --type rate --size-gb 8.0

# Quick test (1 GB)
python scripts/generate_8gb_dataset.py --output data/snn_1gb.h5 --type rate --quick

# Custom size
python scripts/generate_8gb_dataset.py --output data/snn_custom.h5 --type rate --size-gb 16.0

# Inspect dataset
python scripts/generate_8gb_dataset.py --output data/snn_8gb.h5 --inspect
```

**Data types**: `poisson`, `temporal`, `rate` (recommended), `event`

## 🎯 Training Commands

```bash
# Basic training
python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml

# Resume from checkpoint
python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml --resume checkpoints/snn_8gb/best.pt

# Override parameters
python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml --epochs 100 --batch-size 32

# CPU training
python scripts/train_snn_8gb.py --config configs/snn_8gb_training.yaml --device cpu
```

## 🧪 Testing Commands

```bash
# Run all tests
./scripts/run_all_tests.sh

# Quick smoke test
./scripts/run_all_tests.sh --quick

# Gates only
./scripts/run_all_tests.sh --gates-only

# Specific tests
pytest tests/snn/test_param_audit.py -v
pytest tests/snn/test_grad_stability.py -v
pytest tests/snn/test_ablate_tls.py -v

# Benchmarks
python scripts/bench_snn.py --audit
python scripts/bench_snn.py --roofline
python scripts/bench_accuracy_energy.py --quick
```

## 📈 Dataset Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **N** | 4096 | Number of neurons |
| **T** | 256 | Time steps per sequence |
| **r** | 32 | Low-rank factor (0.78% of N) |
| **k** | 64 | Connections per neuron (1.56% of N) |
| **Sequences** | ~2,038 | Total sequences (1,834 train, 204 val) |
| **Size** | 8 GB | Total dataset size |
| **Sequence size** | ~4 MB | Size per sequence |

## 🔧 Model Configuration

```yaml
# configs/snn_8gb_training.yaml

backend: snn_emu

dataset:
  path: data/snn_8gb.h5
  type: rate
  num_classes: 10

model:
  N: 4096
  lowrank_rank: 32
  k_per_row: 64

snn:
  v_th: 1.0
  alpha: 0.95
  time_steps: 256

training:
  learning_rate: 2.0e-3
  batch_size: 16
  max_epochs: 50

tfan:
  use_fdt: true
  fdt:
    target_epr_cv: 0.15
```

## 📊 Expected Metrics

| Metric | Value |
|--------|-------|
| **Parameters** | 262,144 (98.44% reduction) |
| **Dense params** | 16,777,216 |
| **VRAM** | 4-6 GB (batch=16) |
| **Sparsity** | 98.44% |
| **Avg degree** | 64 ≤ 81.92 (0.02×N) |
| **Training time** | 2-3 min/epoch (A100) |
| **Throughput** | 600-800 samples/sec |

## ✅ Gates Validated

```python
✓ param_reduction_pct >= 97.0%
✓ avg_degree <= 0.02 × N
✓ rank <= 0.02 × N
✓ sparsity >= 0.98
✓ epr_cv <= 0.15 (with FDT)
```

## 🐍 Python API

```python
# Load dataset
from tfan.data import SNNTemporalDataset, create_data_loaders

dataset = SNNTemporalDataset('data/snn_8gb.h5', split='train')
train_loader, val_loader = create_data_loaders('data/snn_8gb.h5', batch_size=16)

# Build backend
from tfan.backends import build_backend

config = {'backend': 'snn_emu', 'model': {'N': 4096, 'lowrank_rank': 32, 'k_per_row': 64}}
backend = build_backend(config)

# Train
for inputs, labels in train_loader:
    outputs, aux = backend.model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    backend.hooks.before_step(backend.model)
    backend.optim.step()
    backend.hooks.after_step(backend.model, aux)
```

## 📁 File Structure

```
Quanta-meis-nib-cis/
├── scripts/
│   ├── generate_8gb_dataset.py    # Dataset generator
│   ├── train_snn_8gb.py           # Training script
│   ├── validate_all_gates.py      # Gate validator
│   ├── bench_snn.py               # SNN benchmarks
│   └── run_all_tests.sh           # Test runner
├── configs/
│   ├── snn_8gb_training.yaml      # 8GB training config
│   └── ci/ci_quick.yaml           # CI quick config
├── tfan/
│   ├── backends/                  # Backend factory
│   ├── snn/                       # SNN implementation
│   └── data/                      # Dataset loaders
├── tests/snn/                     # Test suite
├── data/                          # Generated datasets
└── docs/
    ├── 8GB_DATASET_GUIDE.md       # Complete guide
    ├── BACKEND_INTEGRATION_GUIDE.md
    └── TESTING_GUIDE.md
```

## 🔍 Troubleshooting

### CUDA OOM
```yaml
dataloader:
  batch_size: 8  # Reduce batch size
```

### Slow training
```yaml
dataloader:
  num_workers: 16  # More workers
  cache_size: 512  # Cache samples
```

### Import errors
```bash
pip install -r requirements.txt
pip install -e .
```

## 📚 Documentation

- **[8GB Dataset Guide](docs/8GB_DATASET_GUIDE.md)** - Complete dataset documentation
- **[Backend Integration Guide](docs/BACKEND_INTEGRATION_GUIDE.md)** - Model usage
- **[Testing Guide](docs/TESTING_GUIDE.md)** - Comprehensive testing
- **[Integration Summary](docs/INTEGRATION_SUMMARY.md)** - System overview

## 🎯 Gates Enforcement

All gates are automatically enforced in CI:

```bash
.github/workflows/integration_tests.yml
├── snn-gates job
│   ├── pytest tests/snn/ -v
│   ├── python scripts/bench_snn.py --audit
│   └── Validate gates (fail CI if violated)
```

## 💾 Checkpoint Management

```bash
checkpoints/snn_8gb/
├── latest.pt         # Latest epoch
├── best.pt           # Best validation loss
├── epoch_005.pt      # Epoch checkpoints
├── epoch_010.pt
└── history.json      # Training history
```

Load checkpoint:
```python
checkpoint = torch.load('checkpoints/snn_8gb/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

**For detailed documentation, see**: `docs/8GB_DATASET_GUIDE.md`
