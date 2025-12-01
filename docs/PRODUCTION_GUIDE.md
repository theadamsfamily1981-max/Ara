# TF-A-N Production Guide

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Installation & Setup](#installation--setup)
3. [Configuration](#configuration)
4. [Multi-Modal Processing](#multi-modal-processing)
5. [Training Workflow](#training-workflow)
6. [Monitoring & Observability](#monitoring--observability)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)
9. [Security & Governance](#security--governance)

## System Architecture

### Dataflow Overview
```
Ingest → TTW-Sentry → Fusion → TLS Landmarks → Sparse Attention
    ↓         ↓          ↓           ↓                ↓
  Text     Align     Pack       Masks           O(N log N)
  Audio                                              ↓
  Video                                    Transformer Layers
  IMU                                                ↓
                                          Task + Emotion Heads
                                                     ↓
                                              Topology KL
                                                     ↓
                                              FDT PI-D Control
                                                     ↓
                                              PGU Verification
                                                     ↓
                                              Optimizer Step
```

### Hard Gates (Non-Negotiable)
- **TTW-Sentry**: p95 < 5 ms, coverage ≥ 90%
- **PGU-MAK**: p95 ≤ 200 ms, cache hit ≥ 50%
- **SSA**: ≥ 3× speedup @ 16k/32k, ≤ 2% accuracy delta
- **FDT**: EPR-CV ≤ 0.15 sustained
- **Topology**: Wasserstein gap ≤ 2%, cosine ≥ 0.90
- **Memory**: Scaling exponent α < 1.0

## Installation & Setup

### System Requirements
- **GPU**: NVIDIA RTX 3090 or better (24GB+ VRAM for 32k sequences)
- **CUDA**: 12.0+
- **Python**: 3.10 or 3.11
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for datasets and artifacts

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/your-org/Quanta-meis-nib-cis.git
cd Quanta-meis-nib-cis

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install core dependencies
pip install -r requirements.txt

# 5. Install package in development mode
pip install -e .

# 6. Verify installation
python -c "import tfan; print(tfan.__version__)"
```

### Optional Optimizations

```bash
# Flash Attention (for supported GPUs)
pip install flash-attn --no-build-isolation

# Apex (mixed precision training)
git clone https://github.com/NVIDIA/apex
cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

## Configuration

### Default Configuration (QUANTA-Focused)

**As of latest version, TF-A-N defaults to QUANTA-focused training configuration.**

The training script automatically uses `configs/7b/quanta_focus.yaml` by default, which is optimized for:
- Quantum computing & algorithms
- Neuromorphic & SNN architectures
- HPC, GPU kernels, control systems
- Formal verification & safety

**Data sources** (QUANTA focus):
- 30% QUANTA internal docs
- 20% arXiv (systems, neuro, HPC, quantum)
- 15% Technical manuals & specs
- 20% HPC/CUDA code
- 10% Long-form books (PG-19)
- 5% Instruction SFT

**Environment variables for QUANTA data**:
```bash
export QUANTA_DATA_ROOT=/data/shards/          # Local data directory
export QUANTA_S3_BUCKET=s3://quanta-datasets/  # S3 bucket (optional)
export QUANTA_MANIFEST=/path/to/manifest.json  # Data manifest (optional)
```

If QUANTA data is not available, the system safely falls back to WikiText-103 or dummy data for smoke tests.

**Quick start**:
```bash
# QUANTA-focused (default)
python training/train.py

# Generic baseline (for comparison)
python training/train.py --data configs/7b/datasets/generic_base.yaml

# CI quick smoke test
python training/train.py --config configs/ci/ci_quick.yaml
```

See `configs/README.md` for complete configuration documentation.

### Configuration File Structure

Create a YAML configuration file (e.g., `configs/my_config.yaml`):

```yaml
# Model architecture
d_model: 1024
n_heads: 16
n_layers: 12
d_ff: 4096
max_seq_len: 32768

# Multi-modal settings
modality:
  use_text: true
  use_audio: true
  use_video: false
  use_imu: false

# TTW-Sentry
ttw:
  budget_ms: 5.0
  triggers:
    vfe_spike: true
    entropy_jump: true
  thresholds:
    vfe_spike: 0.15
    entropy_jump: 0.25

# Sparse Attention (SSA + TLS)
ssa:
  keep_ratio: 0.33
  alpha_tls: 0.70
  window_size: 128
  per_head_masks: true
  degree_floor: 2

# Topological Regularization
topology:
  lambda_topo: 0.10
  filtration_type: rips
  homology_degrees: [0, 1]
  landscape_levels: 5
  gates:
    wasserstein_gap_max: 0.02
    cosine_min: 0.90

# CTD Hyperbolic
ctd:
  enable: true
  manifold: poincare
  thresholds:
    tree_likeness: 0.6
    ndcg_improvement: 0.05
    overhead_max: 0.12

# FDT Homeostat
fdt:
  pid:
    kp: 0.30
    ki: 0.02
    kd: 0.10
    ema_alpha: 0.85
  epr_cv_max: 0.15
  temperature_range: [0.7, 1.8]
  lr_range: [1.0e-6, 1.0e-2]

# Emotion
emotion:
  enable: true
  mode: VA  # or PAD
  loss_type: CCC  # or MSE
  temporal_smoothness_weight: 0.1
  controller:
    arousal_temp_coupling: [0.8, 1.3]
    valence_lr_coupling: [0.7, 1.2]
    controller_weight: 0.3

# PGU-MAK
pgu:
  mode: soft  # or hard
  timeout_ms: 120
  fallback_timeout_ms: 180
  cache_size: 10000
  rule_cap: 20
  safety_domain_hard_mode: true

# Training
batch_size: 8
gradient_accumulation_steps: 4
base_lr: 3.0e-4
base_temperature: 1.0
warmup_steps: 1000
max_steps: 100000
eval_interval: 500
save_interval: 2000

# Hardware
device: cuda
mixed_precision: true
compile_model: false

# Monitoring
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
  enable_grafana: true
  grafana_port: 3000
```

### Loading Configuration

```python
from tfan import TFANConfig

# Load from YAML
config = TFANConfig.from_yaml("configs/my_config.yaml")

# Validate gates
violations = config.validate_gates()
if violations:
    print("Configuration violations:", violations)
else:
    print("Configuration valid ✓")
```

## Multi-Modal Processing

### Ingesting Modalities

```python
from tfan.mm import MultiModalIngestor

# Create ingestor
ingestor = MultiModalIngestor(
    modalities=["text", "audio", "video"],
    output_dim=1024,
)

# Prepare inputs
inputs = {
    "text": ["This is a test sentence"],
    "audio": torch.randn(1, 48000),  # 3 seconds @ 16kHz
    "video": torch.randn(1, 90, 3, 224, 224),  # 90 frames @ 30fps
}

# Ingest
streams = ingestor.ingest(inputs)

# Each stream has: features, timestamps, modality, confidence
for mod, stream in streams.items():
    print(f"{mod}: {stream.features.shape}, conf={stream.confidence:.2f}")
```

### Alignment with TTW-Sentry

```python
from tfan.mm import align_streams

# Align with automatic trigger detection
aligned_streams, metrics = align_streams(
    streams,
    max_iter=50,
    p95_latency_ms=5.0,
    coverage_target=0.90,
)

# Check alignment quality
print(f"p95 latency: {metrics.p95_latency_ms:.2f} ms")
print(f"Coverage: {metrics.coverage:.1%}")

# Validate gates
assert metrics.p95_latency_ms < 5.0, "TTW p95 latency gate failed"
assert metrics.coverage >= 0.90, "TTW coverage gate failed"
```

### Fusion and Packing

```python
from tfan.mm import pack_and_mask

# Fuse modalities with TLS landmark selection
fused = pack_and_mask(
    tokens_by_mod={k: v.features for k, v in aligned_streams.items()},
    timestamps_by_mod={k: v.timestamps for k, v in aligned_streams.items()},
    keep_ratio=0.33,
    alpha=0.7,
    per_head=True,
    d_model=1024,
    n_heads=16,
)

# Use in model
# fused.tokens: (batch, total_seq_len, d_model)
# fused.landmark_candidates: (batch, n_heads, total_seq_len)
# fused.modality_map: (batch, total_seq_len) - modality IDs
```

## Training Workflow

### Minimal Training Loop

```python
import torch
from tfan import TFANConfig, TFANTrainer
from tfan.datasets import create_dataloader

# 1. Load configuration
config = TFANConfig.from_yaml("configs/prod_longseq.yaml")

# 2. Create model (your transformer)
model = YourTransformerModel(config).to(config.device)

# 3. Create trainer
trainer = TFANTrainer(model, config)

# 4. Load data
train_loader = create_dataloader(
    "multimodal",
    batch_size=config.batch_size,
    split="train",
)

# 5. Training loop
for epoch in range(100):
    for batch in train_loader:
        # Training step (handles FDT, PGU, emotion modulation)
        metrics = trainer.training_step(batch)

        # Log metrics
        if trainer.current_step % 100 == 0:
            print(f"Step {trainer.current_step}: "
                  f"loss={metrics['loss']:.4f}, "
                  f"epr_cv={metrics['epr_cv']:.4f}, "
                  f"lr={metrics['lr']:.2e}")

        # Validate gates
        if trainer.current_step % 500 == 0:
            passes, results = trainer.validate_gates()
            if not passes:
                print(f"⚠️  Gate violations: {results}")
            else:
                print("✓ All gates passing")

        # Save checkpoint
        if trainer.current_step % 2000 == 0:
            trainer.save_checkpoint(
                f"checkpoints/step_{trainer.current_step}.pt"
            )
```

### Advanced Training Features

#### Custom PGU Rules

```python
from tfan.pgu import example_gradient_bound_rule, example_weight_range_rule

# Add custom safety rules
trainer.pgu.add_safety_rule(
    name="gradient_bound",
    description="Gradient norm must be < 10.0",
    constraint_fn=example_gradient_bound_rule(max_grad_norm=10.0),
)

trainer.pgu.add_safety_rule(
    name="weight_range",
    description="Weights must be in [-5, 5]",
    constraint_fn=example_weight_range_rule(min_val=-5.0, max_val=5.0),
)
```

#### Topology Target Setting

```python
# Set target topology for regularization
target_diagrams = {
    0: np.array([[0.0, 0.5], [0.1, 0.6], [0.2, 0.8]]),  # β₀
    1: np.array([[0.3, 0.7], [0.4, 0.9]]),  # β₁
}

trainer.topo_regularizer.set_target_topology(target_diagrams)
```

## Monitoring & Observability

### Prometheus Metrics

Start Prometheus exporter:

```python
from tfan.monitoring import PrometheusExporter

exporter = PrometheusExporter(port=9090)
exporter.register_trainer(trainer)
exporter.start()

# Metrics available:
# - tfan_ttw_p95_latency_ms
# - tfan_ttw_coverage
# - tfan_ssa_speedup
# - tfan_ssa_sparsity
# - tfan_epr_cv
# - tfan_lr
# - tfan_temperature
# - tfan_pgu_p95_latency_ms
# - tfan_pgu_cache_hit_rate
# - tfan_topology_wasserstein_gap
# - tfan_topology_cosine_similarity
```

### Grafana Dashboards

Import the pre-built dashboard:

```bash
# Copy Grafana dashboard JSON
cp monitoring/grafana_dashboards.json /var/lib/grafana/dashboards/

# Or import via UI:
# Grafana → Dashboards → Import → Upload JSON file
```

### Visualization

```python
from tfan.viz import (
    plot_attention_heatmap,
    plot_persistence_diagram,
    plot_training_curves,
    plot_emotion_trajectory,
)

# Plot attention patterns
plot_attention_heatmap(
    attn_weights,
    save_path="artifacts/viz/attention.png",
)

# Plot topology
plot_persistence_diagram(
    diagrams[0],  # β₀ diagram
    save_path="artifacts/viz/ph_diagram.png",
)

# Plot training curves
plot_training_curves(
    trainer.metrics_history,
    keys=["loss", "epr_cv", "lr", "temperature"],
    save_path="artifacts/viz/training.png",
)

# Plot emotion trajectory
valence_hist = [m.get("emotion_valence", 0) for m in trainer.metrics_history]
arousal_hist = [m.get("emotion_arousal", 0.5) for m in trainer.metrics_history]
plot_emotion_trajectory(
    valence_hist,
    arousal_hist,
    save_path="artifacts/viz/emotion.png",
)
```

## Performance Tuning

### Memory Optimization

For 32k sequences on 24GB GPUs:

```yaml
# Reduce keep_ratio for sparser attention
ssa:
  keep_ratio: 0.25  # from 0.33

# Smaller batch size
batch_size: 4

# Gradient accumulation
gradient_accumulation_steps: 8

# Enable mixed precision
mixed_precision: true

# Reduce window size
ssa:
  window_size: 96  # from 128
```

### Latency Optimization

Reduce TTW overhead:

```yaml
ttw:
  budget_ms: 3.0  # tighter budget
  triggers:
    # Increase thresholds to reduce false positives
    vfe_spike_threshold: 0.20
    entropy_jump_threshold: 0.30
```

Optimize PGU:

```yaml
pgu:
  timeout_ms: 80  # faster timeout
  cache_size: 20000  # larger cache
  mode: soft  # for non-safety-critical
```

### Speedup Validation

```bash
# Run benchmark
python scripts/bench_attention.py \
    --seq 8192 16384 32768 \
    --batch 4 \
    --report artifacts/bench/attention.json

# Check results
python -c "
import json
data = json.load(open('artifacts/bench/attention.json'))
for seq, speedup in zip(data['seq_lengths'], data['speedups']):
    print(f'{seq}: {speedup:.2f}×')
    assert speedup >= 3.0, f'Speedup gate failed at {seq}'
"
```

## Troubleshooting

### Common Issues

#### 1. TTW p95 > 5 ms

**Symptoms**: Alignment latency exceeds budget

**Causes**:
- Triggers firing too frequently
- Sinc kernel iterations too high
- Large stream length mismatches

**Solutions**:
```yaml
# Increase trigger thresholds
ttw:
  triggers:
    vfe_spike_threshold: 0.20  # up from 0.15
    entropy_jump_threshold: 0.30  # up from 0.25

# Reduce max iterations
ttw:
  max_iter: 30  # down from 50

# Skip alignment on low-priority streams
ttw:
  skip_modalities: ["imu"]  # if present
```

#### 2. EPR-CV > 0.15

**Symptoms**: Training instability, high EPR coefficient of variation

**Causes**:
- PID gains too low
- No gradient clipping
- High batch variance

**Solutions**:
```yaml
# Increase PID proportional gain
fdt:
  pid:
    kp: 0.40  # up from 0.30

# Enable gradient clipping
fdt:
  grad_clip_norm: 1.0

# Add batch jitter
training:
  batch_jitter: 0.1  # ±10% variation
```

#### 3. PGU p95 > 200 ms

**Symptoms**: Proof verification slow

**Causes**:
- Cold cache
- Complex formulas
- No substitution canonicalization

**Solutions**:
```yaml
# Increase cache size
pgu:
  cache_size: 20000

# Reduce timeout for faster fallback
pgu:
  timeout_ms: 100
  fallback_timeout_ms: 150

# Use soft mode for non-critical
pgu:
  mode: soft
```

#### 4. Topology Gate Failures

**Symptoms**: Wasserstein gap > 2% or cosine < 0.90

**Causes**:
- keep_ratio too small
- Noisy latent representations
- Incorrect filtration

**Solutions**:
```yaml
# Increase landmark retention
ssa:
  keep_ratio: 0.40  # up from 0.33

# Smooth latents (add dropout/layer norm)
model:
  dropout: 0.15

# Queue for nightly audit
topology:
  enable_nightly_exact: true
```

#### 5. OOM on 32k Sequences

**Symptoms**: CUDA out of memory

**Solutions**:
```yaml
# Reduce batch size
batch_size: 2

# Smaller attention window
ssa:
  window_size: 64

# Enable gradient checkpointing
model:
  gradient_checkpointing: true

# Use FP16 for KV cache
mixed_precision: true
```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Set specific loggers
logging.getLogger("tfan.ttw").setLevel(logging.DEBUG)
logging.getLogger("tfan.pgu").setLevel(logging.DEBUG)
```

## Security & Governance

### PGU Policies

**Hard Mode** (safety-critical domains):
```yaml
pgu:
  mode: hard
  safety_domain_hard_mode: true
  # Rejects any update that fails verification
```

**Soft Mode** (research/development):
```yaml
pgu:
  mode: soft
  # Warns but allows updates that timeout
```

### Secrets Management

Never commit secrets to repository:

```bash
# Use environment variables
export WANDB_API_KEY=xxx
export HF_TOKEN=xxx

# Or use secrets manager
vault kv put secret/tfan \
    wandb_key=xxx \
    hf_token=xxx
```

### Data Privacy

For PII-sensitive data:

```python
from tfan.datasets import MultiModalDataset

# Enable anonymization
dataset = MultiModalDataset(
    data_dir="data/",
    anonymize_audio=True,  # Redact speech transcripts
    anonymize_video=True,  # Blur faces
)
```

### Model Cards

Document your trained models:

```yaml
# model_card.yaml
model_name: TF-A-N-1B-MultiModal
version: 1.0
gates_validated:
  ttw_p95: 4.2 ms
  pgu_p95: 185 ms
  ssa_speedup: 5.4x
  epr_cv: 0.12
  topology_wass: 0.015
  topology_cos: 0.93
datasets:
  - WikiText-103
  - Common Voice (audio)
limitations:
  - Extreme non-manifold topologies may trigger CAT fallback
  - Nightly PH bounded to 20 min / 5k samples
safe_use:
  - PGU hard mode for safety-critical deployments
  - Emotion coupling bounded and low-gain
```

## Production Checklist

Before deploying:

- [ ] All hard gates validated on representative data
- [ ] Nightly PH audit passing (Wass ≤ 2%, Cos ≥ 0.90)
- [ ] Benchmarks confirm ≥ 3× speedup @ 16k/32k
- [ ] Memory scaling α < 1.0 verified
- [ ] TTW p95 < 5 ms under load
- [ ] PGU p95 ≤ 200 ms with ≥ 50% cache hit
- [ ] EPR-CV ≤ 0.15 sustained over 10k steps
- [ ] Prometheus metrics exporting correctly
- [ ] Grafana dashboards rendering
- [ ] Runbooks documented for failures
- [ ] Model card completed
- [ ] Security review passed
- [ ] Secrets not in repository
- [ ] CI/CD green across all workflows
