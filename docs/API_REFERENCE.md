# TF-A-N API Reference

## Table of Contents
- [Configuration](#configuration)
- [Multi-Modal Processing](#multi-modal-processing)
- [Attention](#attention)
- [Topology](#topology)
- [Emotion](#emotion)
- [Training](#training)
- [PGU](#pgu)
- [CTD](#ctd)
- [Visualization](#visualization)

## Configuration

### `TFANConfig`

Main configuration class for TF-A-N system.

```python
from tfan import TFANConfig

config = TFANConfig(
    d_model=1024,
    n_heads=16,
    n_layers=12,
    batch_size=8,
    base_lr=3e-4,
    # ... (see config.py for all parameters)
)

# Load from YAML
config = TFANConfig.from_yaml("path/to/config.yaml")

# Save to YAML
config.to_yaml("path/to/output.yaml")

# Validate gates
violations = config.validate_gates()
```

**Attributes**:
- `d_model` (int): Model dimension
- `n_heads` (int): Number of attention heads
- `n_layers` (int): Number of transformer layers
- `ttw` (TTWConfig): TTW-Sentry configuration
- `attention` (AttentionConfig): Sparse attention configuration
- `topology` (TopologyConfig): Topology regularization configuration
- `fdt` (FDTConfig): FDT homeostat configuration
- `emotion` (EmotionConfig): Emotion system configuration
- `pgu` (PGUConfig): PGU configuration

**Methods**:
- `from_yaml(path: str) -> TFANConfig`: Load from YAML file
- `to_yaml(path: str)`: Save to YAML file
- `validate_gates() -> List[str]`: Validate all hard gates

## Multi-Modal Processing

### `ModalityAdapter`

Base class for modality-specific adapters.

```python
from tfan.mm import ModalityAdapter

class CustomAdapter(ModalityAdapter):
    def forward(self, input_data):
        # Process input
        features = ...
        timestamps = ...
        return ModalityStream(
            features=features,
            timestamps=timestamps,
            modality="custom",
            confidence=1.0,
        )
```

### `TextAdapter`

```python
from tfan.mm import TextAdapter

adapter = TextAdapter(
    output_dim=768,
    tokenizer_name="bert-base-uncased",
    max_length=512,
    deterministic=True,
)

stream = adapter(
    text=["This is a test sentence"],
    timestamps=None,  # Optional
)
```

### `AudioAdapter`

```python
from tfan.mm import AudioAdapter

adapter = AudioAdapter(
    output_dim=768,
    sample_rate=16000,
    n_mels=80,
    hop_length=160,
)

stream = adapter(
    audio=torch.randn(1, 48000),  # (batch, samples)
    timestamps=None,
)
```

### `VideoAdapter`

```python
from tfan.mm import VideoAdapter

adapter = VideoAdapter(
    output_dim=768,
    fps=30,
    patch_size=16,
    backbone="vit_base_patch16_224",
)

stream = adapter(
    video=torch.randn(1, 90, 3, 224, 224),  # (batch, frames, C, H, W)
    timestamps=None,
)
```

### `align_streams`

Align multi-modal streams using TTW-Sentry.

```python
from tfan.mm import align_streams

aligned_streams, metrics = align_streams(
    streams: Dict[str, ModalityStream],
    trigger: Optional[Dict[str, bool]] = None,
    max_iter: int = 50,
    p95_latency_ms: float = 5.0,
    coverage_target: float = 0.90,
) -> Tuple[Dict[str, ModalityStream], AlignmentMetrics]
```

**Parameters**:
- `streams`: Dictionary mapping modality name to ModalityStream
- `trigger`: Optional manual trigger override
- `max_iter`: Maximum alignment iterations
- `p95_latency_ms`: Target p95 latency (gate)
- `coverage_target`: Minimum coverage (gate)

**Returns**:
- Tuple of (aligned_streams, alignment_metrics)

### `pack_and_mask`

Fuse and pack multi-modal tokens with TLS landmark selection.

```python
from tfan.mm import pack_and_mask

fused = pack_and_mask(
    tokens_by_mod: Dict[str, torch.Tensor],
    timestamps_by_mod: Dict[str, torch.Tensor],
    keep_ratio: float = 0.33,
    alpha: float = 0.7,
    per_head: bool = True,
    d_model: int = 768,
    n_heads: int = 12,
) -> FusedRepresentation
```

**Returns**: `FusedRepresentation` with:
- `tokens`: (batch, total_seq_len, d_model)
- `modality_map`: (batch, total_seq_len)
- `timestamps`: (batch, total_seq_len)
- `landmark_candidates`: (batch, n_heads, total_seq_len)
- `metadata`: Dict

## Attention

### `TLSLandmarkSelector`

Topological Landmark Selection for sparse attention.

```python
from tfan.attention import TLSLandmarkSelector

selector = TLSLandmarkSelector(
    keep_ratio=0.33,
    alpha=0.7,
    degree_floor=2,
    per_head=True,
)

landmark_mask = selector.select_landmarks(
    hidden_states: torch.Tensor,  # (batch, seq_len, hidden_dim)
    n_heads: int = 12,
) -> torch.Tensor  # (batch, n_heads, seq_len) or (batch, 1, seq_len)
```

**Parameters**:
- `keep_ratio`: Fraction of tokens to keep as landmarks (default 0.33)
- `alpha`: TLS blend factor (α·lifetime + (1-α)·max-min, default 0.7)
- `degree_floor`: Minimum connectivity degree (default 2)
- `per_head`: Use different landmarks per head

### `SparseAttention`

Complete sparse attention module with TLS.

```python
from tfan.attention import SparseAttention

attn = SparseAttention(
    d_model=768,
    n_heads=12,
    keep_ratio=0.33,
    alpha=0.7,
    window_size=128,
    per_head_masks=True,
    degree_floor=2,
    dropout=0.1,
    mask_value=-1e4,
    enable_cat_fallback=True,
    cat_fallback_ratio=0.50,
)

output, metrics = attn(
    x: torch.Tensor,  # (batch, seq_len, d_model)
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]
```

**Returns**:
- `output`: (batch, seq_len, d_model)
- `metrics`: Dict with timing and sparsity info

**Methods**:
- `activate_cat_fallback()`: Activate denser CAT fallback mode
- `deactivate_cat_fallback()`: Deactivate CAT fallback

### `benchmark_attention`

Benchmark sparse vs full attention.

```python
from tfan.attention import benchmark_attention

results = benchmark_attention(
    seq_lengths: List[int] = [1024, 2048, 4096, 8192, 16384, 32768],
    d_model: int = 768,
    n_heads: int = 12,
    batch_size: int = 4,
    device: str = "cuda",
) -> Dict
```

**Returns**: Dict with:
- `seq_lengths`: List of tested sequence lengths
- `sparse_times`: Times for sparse attention
- `full_times`: Times for full attention
- `speedups`: Speedup factors
- `memory_sparse`: Memory usage (sparse)
- `memory_full`: Memory usage (full)

## Topology

### `TopologyRegularizer`

Topological regularization via differentiable persistent homology.

```python
from tfan.topo import TopologyRegularizer

topo_reg = TopologyRegularizer(
    lambda_topo=0.01,
    filtration_type="rips",
    homology_degrees=[0, 1],
    landscape_levels=5,
    wasserstein_gap_max=0.02,
    cosine_min=0.90,
    device="cuda",
)

# Set target topology
topo_reg.set_target_topology(target_diagrams)

# Compute regularization
penalty, metrics = topo_reg(
    latents: torch.Tensor,  # (batch, seq_len, d_model)
) -> Tuple[torch.Tensor, TopologyMetrics]
```

**Methods**:
- `set_target_topology(target_diagrams: Dict[int, np.ndarray])`: Set target PH
- `compute_landscape(latents, return_diagrams=False)`: Compute persistence landscapes
- `kl_penalty(current_landscapes, target_landscapes=None)`: Compute topological KL
- `validate_against_exact(approx_diagrams, exact_diagrams)`: Validate against exact PH

### `TopologyGate`

Validate fusion quality using topology.

```python
from tfan.mm import TopologyGate

gate = TopologyGate(
    d_model=768,
    wasserstein_gap_max=0.02,
    cosine_min=0.90,
    cat_fallback_ratio=0.50,
    max_retries=2,
)

# Set target
gate.set_target_topology(target_diagrams)

# Validate
validated_fused, metrics = gate(
    fused: FusedRepresentation,
    recompute_landmarks_fn: Optional[Callable] = None,
) -> Tuple[FusedRepresentation, Dict]
```

## Emotion

### `EmotionHead`

Emotion prediction head (VA or PAD).

```python
from tfan.emotion import EmotionHead

head = EmotionHead(
    d_model=768,
    mode="VA",  # or "PAD"
    loss_type="CCC",  # or "MSE"
    temporal_smoothness_weight=0.1,
    enable_topo_trajectory=False,
    dropout=0.1,
)

# Forward pass
prediction = head(
    latents: torch.Tensor,  # (batch, seq_len, d_model) or (batch, d_model)
) -> EmotionPrediction
```

**EmotionPrediction** attributes:
- `valence`: (batch, seq_len) or (batch,) in [-1, 1]
- `arousal`: (batch, seq_len) or (batch,) in [0, 1]
- `dominance`: (batch, seq_len) or (batch,) (PAD mode only)
- `confidence`: (batch, seq_len) or (batch,)

**Methods**:
- `compute_loss(predictions, targets)`: Compute emotion loss

### `EmotionController`

Safe policy modulation based on emotion.

```python
from tfan.emotion import EmotionController

controller = EmotionController(
    arousal_temp_coupling=(0.8, 1.3),
    valence_lr_coupling=(0.7, 1.2),
    controller_weight=0.3,
    jerk_threshold=0.1,
    confidence_threshold=0.5,
)

modulation = controller.modulate_policy(
    fdt_metrics: Dict[str, float],
    emotion: EmotionPrediction,
    base_lr: float = 1.0,
    base_temperature: float = 1.0,
) -> ControlModulation
```

**ControlModulation** attributes:
- `lr_multiplier`: Learning rate multiplier
- `temperature_multiplier`: Temperature multiplier
- `weight`: Overall modulation weight
- `reason`: Explanation string

## Training

### `TFANTrainer`

Main trainer with FDT homeostat.

```python
from tfan import TFANTrainer

trainer = TFANTrainer(
    model: nn.Module,
    config: TFANConfig,
    optimizer: Optional[optim.Optimizer] = None,
    pgu: Optional[ProofGatedUpdater] = None,
    emotion_head: Optional[EmotionHead] = None,
    emotion_controller: Optional[EmotionController] = None,
    topo_regularizer: Optional[TopologyRegularizer] = None,
)

# Training step
metrics = trainer.training_step(
    batch: Dict[str, torch.Tensor],
    emotion_targets: Optional[EmotionPrediction] = None,
) -> Dict[str, float]

# Validate gates
passes, results = trainer.validate_gates() -> Tuple[bool, Dict]

# Save/load checkpoints
trainer.save_checkpoint(path: str)
trainer.load_checkpoint(path: str)
```

**Attributes**:
- `current_step`: Current training step
- `current_epoch`: Current epoch
- `current_lr`: Current learning rate
- `current_temperature`: Current temperature
- `metrics_history`: List of all step metrics

### `FDTHomeostat`

Fluctuation-Dissipation Theorem controller.

```python
from tfan.trainer import FDTHomeostat

fdt = FDTHomeostat(
    kp=0.30,
    ki=0.02,
    kd=0.10,
    ema_alpha=0.95,
    epr_cv_max=0.15,
    temperature_min=0.5,
    temperature_max=2.0,
    lr_min=1e-6,
    lr_max=1e-2,
)

metrics = fdt.step(
    loss: float,
    grad_variance: float,
) -> FDTMetrics

# Check if should pause due to instability
should_pause = fdt.should_pause(epr_cv: float, threshold_mult=2.0)
```

## PGU

### `ProofGatedUpdater`

Formal verification for model updates.

```python
from tfan.pgu import ProofGatedUpdater

pgu = ProofGatedUpdater(
    timeout_ms=120,
    fallback_timeout_ms=180,
    p95_latency_max_ms=200.0,
    cache_size=10000,
    cache_cycle_batches=1000,
    rule_cap=20,
    mode="soft",  # or "hard"
    safety_domain_hard_mode=True,
)

# Add safety rule
pgu.add_safety_rule(
    name="gradient_bound",
    description="Gradient norm must be bounded",
    constraint_fn=lambda ctx: ...,  # Z3 constraint function
)

# Verify update
result = pgu.verify_update(
    update_payload: Dict[str, Any],
    is_safety_critical: bool = False,
) -> ProofResult

# Context manager for guarded updates
with pgu.guard(update_payload, is_safety_critical=False):
    optimizer.step()

# Get metrics
metrics = pgu.get_metrics() -> Dict[str, float]

# Validate gates
passes, metrics = pgu.validate_gates() -> Tuple[bool, Dict]
```

**ProofResult** attributes:
- `proven`: Whether proof succeeded
- `timeout`: Whether proof timed out
- `cached`: Whether result was cached
- `latency_ms`: Latency in milliseconds
- `rule_violations`: List of violated rule names

## CTD

### `HyperbolicEmbedding`

Hyperbolic embedding layer (Poincaré or Lorentz).

```python
from tfan.ctd import HyperbolicEmbedding

embedding = HyperbolicEmbedding(
    num_embeddings=10000,
    embedding_dim=768,
    manifold="poincare",  # or "lorentz"
    tree_likeness_threshold=0.6,
    enable=True,
)

# Forward pass
embeddings = embedding(
    indices: torch.Tensor,  # (batch, ...)
) -> torch.Tensor  # (batch, ..., embedding_dim)

# Distance in hyperbolic space
dist = embedding.distance(x, y)
```

### `TreeLikenessDetector`

Detect hierarchical structure in data.

```python
from tfan.ctd import TreeLikenessDetector

detector = TreeLikenessDetector(threshold=0.6)

tree_likeness = detector.compute_tree_likeness(
    embeddings: torch.Tensor,
    betti_1: Optional[int] = None,
) -> float

should_use, score = detector.should_use_hyperbolic(
    embeddings: torch.Tensor,
    betti_1: Optional[int] = None,
) -> Tuple[bool, float]
```

## Visualization

### Plotting Functions

```python
from tfan.viz import (
    plot_attention_heatmap,
    plot_persistence_diagram,
    plot_training_curves,
    plot_emotion_trajectory,
    plot_sparsity_pattern,
)

# Attention heatmap
plot_attention_heatmap(
    attention_weights: torch.Tensor,  # (n_heads, seq, seq) or (seq, seq)
    save_path: Optional[str] = None,
    title: str = "Attention Heatmap",
)

# Persistence diagram
plot_persistence_diagram(
    diagram: np.ndarray,  # (n_features, 2) of (birth, death)
    save_path: Optional[str] = None,
    title: str = "Persistence Diagram",
)

# Training curves
plot_training_curves(
    metrics_history: List[Dict[str, float]],
    keys: List[str] = ["loss", "epr", "lr"],
    save_path: Optional[str] = None,
)

# Emotion trajectory (VA space)
plot_emotion_trajectory(
    valence_history: List[float],
    arousal_history: List[float],
    save_path: Optional[str] = None,
)

# Sparsity pattern
plot_sparsity_pattern(
    attention_mask: torch.Tensor,  # (seq_len, seq_len)
    save_path: Optional[str] = None,
)
```

## Datasets

### `create_dataloader`

Create dataloader for various datasets.

```python
from tfan.datasets import create_dataloader

loader = create_dataloader(
    dataset_name: str,  # "wikitext", "fb15k", "wordnet", "multimodal"
    batch_size: int = 8,
    split: str = "train",  # "train", "valid", "test"
    num_workers: int = 4,
    **kwargs,  # Dataset-specific arguments
) -> DataLoader
```

## Pareto Optimization

### `ParetoOptimizer`

Multi-objective Pareto optimization.

```python
from tfan.pareto import ParetoOptimizer

optimizer = ParetoOptimizer(
    algorithm="NSGA2",  # or "EHVI"
    objectives=["accuracy", "latency", "epr_cv", "topo_gap", "energy"],
    n_iterations=50,
    population_size=24,
    min_non_dominated=6,
)

# Define evaluation function
def evaluate_config(config: Dict) -> Dict[str, float]:
    # Train and evaluate
    return {
        "accuracy": 0.85,
        "latency": 120,
        "epr_cv": 0.12,
        "topo_gap": 0.015,
        "energy": 100,
    }

# Run optimization
pareto_points = optimizer.optimize(
    evaluate_fn=evaluate_config,
    n_var=5,  # Number of decision variables
) -> List[ParetoPoint]

# Validate gate
passes, metrics = optimizer.validate_gate(pareto_points)

# Export results
optimizer.export_pareto_front(
    pareto_points,
    output_path="artifacts/pareto/front.json",
)
```

**ParetoPoint** attributes:
- `config`: Configuration dict
- `objectives`: Objective values dict
- `crowding_distance`: Crowding distance metric
