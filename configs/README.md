# TF-A-N Configuration System

Comprehensive configuration for training TF-A-N 7B models with domain-specific datasets.

## Quick Start

### QUANTA-Focused Training (Default)

```bash
# Train with QUANTA domain data (quantum, neuromorphic, HPC)
accelerate launch training/train.py --config configs/7b/default.yaml

# Or explicitly:
accelerate launch training/train.py --config configs/7b/quanta_focus.yaml
```

### Generic Baseline Training

```bash
# Train with general internet-scale data
accelerate launch training/train.py \
  --config configs/7b/quanta_focus.yaml \
  --data configs/7b/datasets/generic_base.yaml
```

## Configuration Structure

```
configs/
├── README.md                        # This file
├── 7b/
│   ├── default.yaml → quanta_focus.yaml  # Default points to QUANTA
│   ├── quanta_focus.yaml            # QUANTA-focused training
│   └── datasets/
│       ├── quanta_focus.yaml        # QUANTA domain data mix
│       └── generic_base.yaml        # Generic baseline mix
```

## Dataset Configurations

### QUANTA Focus (Default)

**Use case**: Domain-specific training for QUANTA applications

**Data mix**:
- 30% QUANTA internal docs (if available)
- 20% arXiv (systems, neuro, HPC, quantum)
- 15% Technical manuals & specs
- 20% HPC/CUDA code
- 10% Long-form books (PG-19)
- 5% Instruction SFT

**Sequence length**: 16k (supports up to 32k)

**Config**: `configs/7b/datasets/quanta_focus.yaml`

### Generic Base

**Use case**: General-purpose pretraining baseline

**Data mix**:
- 40% SlimPajama/Pile-like web text
- 15% Books (long-form)
- 15% arXiv (math, CS, physics)
- 20% GitHub code (permissive)
- 10% Dialogue/conversation

**Sequence length**: 8k (supports up to 16k)

**Config**: `configs/7b/datasets/generic_base.yaml`

## Data Directory Structure

Place your dataset shards in `/data/shards/`:

```
/data/shards/
├── quanta_internal_docs/
│   ├── part-00000.jsonl.zst
│   ├── part-00001.jsonl.zst
│   └── ...
├── arxiv_sys_neuro_hpc/
│   ├── part-00000.jsonl.zst
│   └── ...
├── manuals_specs_kernel/
│   ├── part-00000.jsonl.zst
│   └── ...
├── code_permissive_hpc_cuda/
│   ├── part-00000.jsonl.zst
│   └── ...
├── long_books_pg19/
│   ├── part-00000.jsonl.zst
│   └── ...
├── instruction_sft_clean/
│   ├── part-00000.jsonl.zst
│   └── ...
├── slim_pile_clean/          # For generic baseline
│   ├── part-00000.jsonl.zst
│   └── ...
└── wikitext_103/              # Fallback for smoke tests
    ├── part-00000.jsonl.zst
    └── ...
```

### Data Format

Each `.jsonl.zst` file contains compressed JSONL with one document per line:

```json
{
  "text": "Full document text...",
  "source": "arxiv",
  "id": "2301.12345",
  "metadata": {
    "title": "Quantum Error Correction...",
    "authors": ["Author1", "Author2"],
    "category": "quant-ph",
    "date": "2023-01-15"
  }
}
```

**Required field**: `text`
**Optional fields**: `source`, `id`, `metadata`

## Configuration Hierarchy

### 1. Main Config (`configs/7b/quanta_focus.yaml`)

Top-level training configuration:
- Model architecture (points to `tfan/models/tfan7b/config.json`)
- Training hyperparameters (LR, batch size, steps)
- TF-A-N components (FDT, PGU, topology, TTW)
- SNN integration (optional)
- Logging & checkpointing
- Gate validation

### 2. Dataset Config (`configs/7b/datasets/*.yaml`)

Dataset-specific configuration:
- Data source mixture with weights
- Packing & sequence length
- Filtering & preprocessing
- Tokenization settings
- Data loading parameters

### 3. Model Config (`tfan/models/tfan7b/config.json`)

Model architecture specification:
- Layers, hidden size, heads
- Attention configuration (SSA)
- MLP configuration (SwiGLU)
- RoPE settings
- Vocabulary size

## Key Configuration Options

### Training Hyperparameters

```yaml
training:
  batch_size: 1                    # Per-GPU batch size
  gradient_accumulation_steps: 16  # Effective batch = 1 × 16 = 16
  seq_length: 16384                # QUANTA uses 16k (32k max)
  max_steps: 100000

  learning_rate: 1.0e-4
  weight_decay: 0.1
  warmup_steps: 2000
  lr_schedule: "cosine"

  grad_clip: 1.0
  precision: "bf16"
```

### TF-A-N Components

```yaml
tfan:
  use_fdt: true         # FDT homeostasis (EPR-CV ≤ 0.15)
  use_pgu: false        # Formal verification (optional)
  use_topology: true    # Topology regularization
  use_ttw: false        # Multi-modal alignment
  use_emotion: false    # Emotion modulation
```

### SNN Integration

```yaml
snn:
  enable: false         # Set to true for SNN mode
  time_steps: 256
  encoder: "rate"       # rate | latency | delta
  neuron: "LIF"         # LIF | PLIF
  target_sparsity: 0.75
```

### Data Mixture Weights

```yaml
mixture:
  - name: "quanta_internal_docs/*"
    weight: 0.30        # 30% of training data
    path: "/data/shards/quanta_internal_docs/"
    enabled: true

  - name: "arxiv_sys_neuro_hpc/*"
    weight: 0.20        # 20% of training data
    path: "/data/shards/arxiv_sys_neuro_hpc/"
    enabled: true
```

**Note**: Weights are normalized automatically. Total doesn't need to sum to 1.0.

## Smoke Testing (No Real Data)

If you don't have dataset shards yet, use the fallback:

```yaml
# In configs/7b/datasets/quanta_focus.yaml
fallback:
  use_fallback: true
  fallback_dataset: "wikitext-103"
  fallback_path: "/data/shards/wikitext_103/"
```

Or run with dummy data:

```bash
python training/train.py \
  --config configs/7b/quanta_focus.yaml \
  --dummy_data  # Uses random tokens for testing
```

## Switching Configurations

### Option 1: Change Default

```bash
cd configs/7b/
rm default.yaml
ln -s generic_base.yaml default.yaml  # Point to generic

# Then train with default:
accelerate launch training/train.py --config configs/7b/default.yaml
```

### Option 2: Explicit Config

```bash
# Always specify which config:
accelerate launch training/train.py --config configs/7b/quanta_focus.yaml

# Or with custom data:
accelerate launch training/train.py \
  --config configs/7b/quanta_focus.yaml \
  --data configs/7b/datasets/my_custom_mix.yaml
```

## Creating Custom Data Mixes

Copy and modify an existing dataset config:

```bash
cp configs/7b/datasets/quanta_focus.yaml configs/7b/datasets/my_mix.yaml
```

Then edit `my_mix.yaml`:

```yaml
mixture:
  - name: "my_domain_data/*"
    weight: 0.50
    path: "/data/shards/my_domain/"
    enabled: true

  - name: "arxiv_my_field/*"
    weight: 0.30
    path: "/data/shards/arxiv_my_field/"
    enabled: true

  - name: "code_my_lang/*"
    weight: 0.20
    path: "/data/shards/code_my_lang/"
    enabled: true
```

## Gate Validation

TF-A-N validates quality gates every 500 steps:

```yaml
gates:
  check_interval: 500

  hard_gates:
    - name: "epr_cv"
      threshold: 0.15
      operator: "<="

    - name: "topology_wasserstein"
      threshold: 0.02
      operator: "<="

    - name: "spike_sparsity"
      threshold: 0.75
      operator: ">="
      enabled_if: "snn.enable"
```

If a hard gate fails, training will log a warning. Configure actions:

```yaml
gates:
  on_hard_gate_fail: "warn"  # warn | pause | stop
  on_soft_gate_fail: "log"   # log | warn
```

## Distributed Training

### DeepSpeed ZeRO-3

```bash
accelerate launch \
  --config_file training/accelerate_config.yaml \
  training/train.py \
  --config configs/7b/quanta_focus.yaml
```

### Multi-Node

```yaml
# In configs/7b/quanta_focus.yaml
distributed:
  num_gpus: 32          # 4 nodes × 8 GPUs
  zero_stage: 3
  offload_optimizer: false
  offload_param: false
```

## Monitoring

### Tensorboard

```bash
tensorboard --logdir runs/quanta_7b/
```

### Weights & Biases

```yaml
logging:
  use_wandb: true
  wandb_project: "tfan-quanta-7b"
  wandb_entity: "your-entity"
```

## FAQ

### Q: How do I switch from QUANTA to generic?

**A**: Either:
1. Change symlink: `ln -sf generic_base.yaml configs/7b/default.yaml`
2. Or specify: `--config configs/7b/quanta_focus.yaml --data configs/7b/datasets/generic_base.yaml`

### Q: What if I don't have QUANTA data yet?

**A**: Set `fallback.use_fallback: true` in the dataset config, or use `--dummy_data` flag for smoke tests.

### Q: Can I mix QUANTA and generic data?

**A**: Yes! Create a custom mix:

```yaml
mixture:
  - name: "quanta_internal_docs/*"
    weight: 0.20
  - name: "slim_pile_clean/*"
    weight: 0.50
  - name: "arxiv_sys_neuro_hpc/*"
    weight: 0.30
```

### Q: How do I enable SNN mode?

**A**: In `configs/7b/quanta_focus.yaml`:

```yaml
snn:
  enable: true
  time_steps: 256
  encoder: "rate"
  neuron: "LIF"
```

### Q: What's the difference between QUANTA and generic?

**A**:
- **QUANTA**: Domain-focused (quantum, neuro, HPC), 16k-32k context, specialized tokenization
- **Generic**: Broad internet data, 8k context, general-purpose baseline

---

**Default Configuration**: `configs/7b/default.yaml` → `quanta_focus.yaml`
**Status**: ✅ QUANTA-focused by default
**Maintained by**: TF-A-N Team
