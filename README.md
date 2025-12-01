# TF-A-N: Transformer with Formal Alignment and Neuromodulation

**Long-sequence, multi-modal inference at production latencies with provable structural fidelity and homeostatic stability.**

## Overview

TF-A-N is a research framework for transformer models with:

- **Sub-quadratic attention** via Topological Landmark Selection (TLS)
- **Multi-modal fusion** with trainable time warping (TTW-Sentry)
- **Formal safety** via Proof-Gated Updates (PGU-MAK)
- **Homeostatic stability** via FDT (Fluctuation-Dissipation Theorem) control
- **Neuromodulation** through emotion-aware learning rate/temperature adaptation
- **Topological regularization** with differentiable persistent homology
- **Adaptive tool-use control** via AEPO (Adaptive Entropy Policy Optimizer)

## Hard Gates (Non-Negotiable)

All components must meet these quantitative requirements:

| Component | Metric | Threshold |
|-----------|--------|-----------|
| **TTW-Sentry** | p95 latency | < 5 ms |
| **TTW-Sentry** | Coverage | ≥ 90% |
| **PGU-MAK** | p95 latency | ≤ 200 ms |
| **PGU-MAK** | Cache hit rate | ≥ 50% |
| **SSA (Sparse Attention)** | Speedup @ 16k/32k | ≥ 3× |
| **SSA** | Accuracy delta | ≤ 2% |
| **FDT Homeostat** | EPR-CV | ≤ 0.15 |
| **Topology** | Wasserstein gap | ≤ 2% |
| **Topology** | Cosine similarity | ≥ 0.90 |
| **Memory** | Scaling exponent α | < 1.0 |
| **AEPO** | Tool-call reduction | ≥ 50% |
| **AEPO** | Reward delta | ≤ 1% |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

```python
import torch
from tfan import TFANConfig, TFANTrainer
from tfan.datasets import create_dataloader

# Load config
config = TFANConfig.from_yaml("config_examples/default.yaml")

# Create trainer
trainer = TFANTrainer(model, config)

# Training loop
for batch in train_loader:
    metrics = trainer.training_step(batch)
```

See full documentation in the code and docstrings.

## Architecture

```
Input → Modality Adapters → TTW Alignment → Fusion → TLS Landmarks → Sparse Attention
                                                                           ↓
                                                                  Transformer Layers
                                                                           ↓
                                                                  Task + Emotion Heads
                                                                           ↓
                                                                     FDT Homeostat
                                                                           ↓
                                                                    PGU Verification
                                                                           ↓
                                                                    Optimizer Step
```

## Key Components

- `tfan/topo.py` - Topological regularization
- `tfan/ttw.py` - Trainable time warping alignment
- `tfan/attention.py` - TLS + sparse attention
- `tfan/pgu.py` - Proof-gated updates
- `tfan/mm/` - Multi-modal processing
- `tfan/emotion/` - Emotion prediction & control
- `tfan/trainer.py` - FDT homeostat + training loop
- `tfan/ctd.py` - Hyperbolic geometry
- `tfan/pareto.py` - Multi-objective optimization
- `tfan/agent/` - AEPO tool-use policy optimization

## AEPO Tool-Use Control

Train adaptive policies to minimize tool calls while maintaining performance:

```bash
# Train AEPO policy
python scripts/train_aepo.py --iterations 200 --seed 42

# Evaluate trained policy
python scripts/eval_aepo.py --checkpoint artifacts/aepo/final.pt --seeds 10
```

See [AEPO Guide](docs/AEPO_GUIDE.md) for detailed documentation.

## License

MIT License
