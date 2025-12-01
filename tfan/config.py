"""
Central configuration system for TF-A-N.

All hard gates, thresholds, and system parameters are defined here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml


@dataclass
class TTWConfig:
    """Trainable Time Warping configuration."""
    max_iter: int = 50
    p95_latency_ms: float = 5.0  # Hard gate
    coverage_target: float = 0.90
    trigger_vfe_spike_threshold: float = 0.15
    trigger_entropy_jump_threshold: float = 0.25
    auroc_threshold: float = 0.8


@dataclass
class AttentionConfig:
    """Sparse attention and TLS configuration."""
    keep_ratio: float = 0.33
    alpha: float = 0.7  # TLS: α·lifetime + (1-α)·max-min
    window_size: int = 128
    per_head_masks: bool = True
    degree_floor: int = 2
    speedup_target: float = 3.0  # Hard gate: ≥3× at 16k/32k
    accuracy_delta_max: float = 0.02  # Hard gate: ≤2%
    mask_value: float = -1e4  # Numerical stability
    enable_cat_fallback: bool = True
    cat_fallback_ratio: float = 0.50


@dataclass
class TopologyConfig:
    """Topological regularization configuration."""
    lambda_topo: float = 0.01
    filtration_type: str = "rips"
    homology_degrees: List[int] = field(default_factory=lambda: [0, 1])
    landscape_levels: int = 5
    wasserstein_gap_max: float = 0.02  # Hard gate
    cosine_min: float = 0.90  # Hard gate
    enable_nightly_exact: bool = True
    nightly_max_samples: int = 5000
    nightly_timeout_min: int = 20


@dataclass
class MemoryConfig:
    """Memory scaling configuration."""
    alpha_max: float = 1.0  # Hard gate: α < 1.0
    test_sequence_lengths: List[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192, 16384, 32768])


@dataclass
class HyperbolicConfig:
    """CTD hyperbolic geometry configuration."""
    enable: bool = True
    manifold: str = "poincare"  # or "lorentz"
    tree_likeness_threshold: float = 0.6
    ndcg_improvement_target: float = 0.05  # +5% vs Euclidean
    overhead_max: float = 0.12  # ≤12%


@dataclass
class FDTConfig:
    """Fluctuation-Dissipation Theorem homeostat configuration."""
    kp: float = 0.30  # PID proportional gain
    ki: float = 0.02  # PID integral gain
    kd: float = 0.10  # PID derivative gain
    ema_alpha: float = 0.95
    grad_clip_norm: float = 1.0
    epr_cv_max: float = 0.15  # Hard gate: EPR-CV ≤ 0.15
    temperature_min: float = 0.5
    temperature_max: float = 2.0
    lr_min: float = 1e-6
    lr_max: float = 1e-2


@dataclass
class EmotionConfig:
    """Emotion prediction and control configuration."""
    mode: str = "VA"  # "VA" (valence-arousal) or "PAD" (pleasure-arousal-dominance)
    loss_type: str = "CCC"  # "MSE" or "CCC" (concordance correlation coefficient)
    temporal_smoothness_weight: float = 0.1
    enable_topo_trajectory: bool = True
    topo_trajectory_beta0_target: int = 1
    topo_trajectory_beta1_target: int = 0

    # Controller bounds
    arousal_temp_coupling: Tuple[float, float] = (0.8, 1.3)  # (min_mult, max_mult)
    valence_lr_coupling: Tuple[float, float] = (0.7, 1.2)
    controller_weight: float = 0.3  # Blend factor
    jerk_threshold: float = 0.1  # ||Δ²state||
    confidence_threshold: float = 0.5  # Low confidence → reduce controller weight


@dataclass
class PGUConfig:
    """Proof-Gated Updater configuration."""
    timeout_ms: int = 120
    fallback_timeout_ms: int = 180
    p95_latency_max_ms: float = 200.0  # Hard gate
    cache_size: int = 10000
    cache_cycle_batches: int = 1000
    cache_hit_target: float = 0.50  # Hard gate: ≥50%
    rule_cap: int = 20
    mode: str = "soft"  # "hard" (block) or "soft" (warn) for timeouts
    safety_domain_hard_mode: bool = True


@dataclass
class MultiModalConfig:
    """Multi-modal ingest and fusion configuration."""
    modalities: List[str] = field(default_factory=lambda: ["text", "audio", "video"])

    # Audio settings
    audio_sample_rate: int = 16000
    audio_n_mels: int = 80
    audio_hop_length: int = 160

    # Video settings
    video_fps: int = 30
    video_patch_size: int = 16
    video_backbone: str = "vit_base_patch16_224"

    # Fusion settings
    packing_order: List[str] = field(default_factory=lambda: ["text", "audio", "video"])
    late_stream_degradation: float = 0.5


@dataclass
class ParetoConfig:
    """Multi-objective Pareto optimization configuration."""
    algorithm: str = "EHVI"  # "EHVI" or "MOEAD"
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "latency", "epr_cv", "topo_gap", "energy"])
    n_iterations: int = 50
    population_size: int = 24
    min_non_dominated: int = 6  # Hard gate


@dataclass
class MonitoringConfig:
    """Observability and monitoring configuration."""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    scrape_interval_sec: int = 30
    enable_grafana: bool = True
    grafana_port: int = 3000
    enable_wandb: bool = False
    wandb_project: Optional[str] = None


@dataclass
class TFANConfig:
    """Master TF-A-N configuration."""
    # Component configs
    ttw: TTWConfig = field(default_factory=TTWConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    hyperbolic: HyperbolicConfig = field(default_factory=HyperbolicConfig)
    fdt: FDTConfig = field(default_factory=FDTConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    pgu: PGUConfig = field(default_factory=PGUConfig)
    multimodal: MultiModalConfig = field(default_factory=MultiModalConfig)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Model architecture
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 32768

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    base_lr: float = 3e-4
    base_temperature: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_interval: int = 500
    save_interval: int = 2000

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    compile_model: bool = False

    # Artifacts
    artifact_dir: str = "./artifacts"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    @classmethod
    def from_yaml(cls, path: str) -> "TFANConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Recursively build nested dataclass configs
        return cls(**data)

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj

        data = dataclass_to_dict(self)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate_gates(self) -> List[str]:
        """
        Validate all hard gates and return list of violations.

        Returns:
            List of violation messages (empty if all gates pass).
        """
        violations = []

        # TTW p95 < 5 ms
        if self.ttw.p95_latency_ms >= 5.0:
            violations.append(f"TTW p95 latency {self.ttw.p95_latency_ms} >= 5.0 ms")

        # PGU p95 ≤ 200 ms with ≥50% cache hit
        if self.pgu.p95_latency_max_ms > 200.0:
            violations.append(f"PGU p95 latency {self.pgu.p95_latency_max_ms} > 200 ms")
        if self.pgu.cache_hit_target < 0.50:
            violations.append(f"PGU cache hit target {self.pgu.cache_hit_target} < 50%")

        # SSA speedup ≥3×, accuracy delta ≤2%
        if self.attention.speedup_target < 3.0:
            violations.append(f"SSA speedup target {self.attention.speedup_target} < 3.0×")
        if self.attention.accuracy_delta_max > 0.02:
            violations.append(f"SSA accuracy delta max {self.attention.accuracy_delta_max} > 2%")

        # EPR-CV ≤ 0.15
        if self.fdt.epr_cv_max > 0.15:
            violations.append(f"FDT EPR-CV max {self.fdt.epr_cv_max} > 0.15")

        # Topology: Wasserstein ≤ 2%, cosine ≥ 0.90
        if self.topology.wasserstein_gap_max > 0.02:
            violations.append(f"Topology Wasserstein gap {self.topology.wasserstein_gap_max} > 2%")
        if self.topology.cosine_min < 0.90:
            violations.append(f"Topology cosine min {self.topology.cosine_min} < 0.90")

        # Memory scaling α < 1.0
        if self.memory.alpha_max >= 1.0:
            violations.append(f"Memory alpha max {self.memory.alpha_max} >= 1.0")

        return violations
