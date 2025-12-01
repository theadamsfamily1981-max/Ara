"""
ARA Unified Configuration System

Consolidates all configuration sources:
- config/: Avatar and voice settings
- configs/: Training and experiment configs
- tfan/config.py: System hard gates
- src/models/tfan/config.py: Model architecture

Usage:
    from ara.configs import load_config, AraConfig

    # Load full config
    config = load_config()

    # Access components
    model_config = config.model
    training_config = config.training
    system_config = config.system
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Add parent paths for imports
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Try yaml import
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class ModelConfig:
    """TFAN model architecture configuration."""
    model_type: str = "tfan7b"
    vocab_size: int = 32768
    hidden_size: int = 4096
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 13312
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    use_bias: bool = False
    attention_impl: str = "ssa_radial_v1"
    ssa_keep_ratio: float = 0.33
    ssa_local: int = 128
    ssa_hops: int = 2
    torch_dtype: str = "bfloat16"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    base_lr: float = 3e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_interval: int = 500
    save_interval: int = 2000
    mixed_precision: bool = True
    compile_model: bool = False


@dataclass
class SystemConfig:
    """System and hard gate configuration."""
    # TTW gates
    ttw_p95_latency_ms: float = 5.0
    # SSA gates
    ssa_speedup_target: float = 3.0
    ssa_accuracy_delta_max: float = 0.02
    # FDT gates
    fdt_epr_cv_max: float = 0.15
    # Topology gates
    topo_wasserstein_gap_max: float = 0.02
    topo_cosine_min: float = 0.90


@dataclass
class AvatarConfig:
    """Avatar generation configuration."""
    device: str = "auto"
    output_fps: int = 25
    output_resolution: tuple = (512, 512)
    cache_enabled: bool = True
    cache_max_size_mb: int = 2000
    cache_ttl_hours: int = 24


@dataclass
class AgentConfig:
    """HRRL agent configuration."""
    obs_dim: int = 64
    action_dim: int = 8
    hidden_dims: tuple = (256, 128)
    hyperbolic_dim: int = 128
    need_dim: int = 8
    drive_dim: int = 16


@dataclass
class TGSFNConfig:
    """TGSFN substrate configuration."""
    n_neurons: int = 256
    n_excitatory: int = 204  # ~80% E:I ratio
    n_inhibitory: int = 52
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    threshold: float = 1.0
    branching_target: float = 1.0
    alpha_target: float = 2.0


@dataclass
class AraConfig:
    """
    Master ARA configuration combining all components.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tgsfn: TGSFNConfig = field(default_factory=TGSFNConfig)

    # Device
    device: str = "auto"

    # Paths
    checkpoint_dir: str = "./checkpoints"
    artifact_dir: str = "./artifacts"
    log_dir: str = "./logs"
    cache_dir: str = "./cache"


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML not installed")

    if not path.exists():
        return {}

    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_dir: Optional[Path] = None,
    model_config_path: Optional[Path] = None,
    training_config_path: Optional[Path] = None,
) -> AraConfig:
    """
    Load unified ARA configuration.

    Args:
        config_dir: Directory containing config files (default: ./config)
        model_config_path: Path to model config YAML
        training_config_path: Path to training config YAML

    Returns:
        AraConfig instance
    """
    config = AraConfig()

    if config_dir is None:
        config_dir = _root / "config"

    # Load avatar config
    avatar_config_path = config_dir / "avatar_config.yaml"
    if avatar_config_path.exists() and YAML_AVAILABLE:
        data = load_yaml_config(avatar_config_path)
        if 'performance' in data:
            config.device = data['performance'].get('device', 'auto')
        if 'cache' in data:
            config.avatar.cache_enabled = data['cache'].get('enabled', True)
            config.avatar.cache_max_size_mb = data['cache'].get('max_cache_size_mb', 2000)

    # Load base config
    base_config_path = config_dir / "base.yaml"
    if base_config_path.exists() and YAML_AVAILABLE:
        data = load_yaml_config(base_config_path)
        # Apply base config values
        if 'model' in data:
            for key, value in data['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

    # Load TGSFN config
    tgsfn_config_path = config_dir / "tgsfn8b.yaml"
    if tgsfn_config_path.exists() and YAML_AVAILABLE:
        data = load_yaml_config(tgsfn_config_path)
        if 'network' in data:
            for key, value in data['network'].items():
                if hasattr(config.tgsfn, key):
                    setattr(config.tgsfn, key, value)

    return config


# Default config instance
default_config = None


def get_config() -> AraConfig:
    """Get or create default configuration."""
    global default_config
    if default_config is None:
        default_config = load_config()
    return default_config


__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "SystemConfig",
    "AvatarConfig",
    "AgentConfig",
    "TGSFNConfig",
    "AraConfig",
    "load_config",
    "get_config",
]
