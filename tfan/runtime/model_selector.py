"""
Auto-Deploy Model Selector

Reads Pareto-optimized configurations from configs/auto/best.yaml and
provides runtime model selection with override capabilities.

Usage:
    # Load auto-selected config
    selector = ModelSelector()
    config = selector.get_config()

    # Override specific parameters
    selector = ModelSelector(overrides={"n_heads": 16, "d_model": 1024})
    config = selector.get_config()

    # Use CLI flag
    python train.py --config-override '{"n_heads": 16}'
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """TF-A-N model configuration from Pareto optimization."""

    # Architecture
    n_heads: int = 8
    d_model: int = 512
    n_layers: int = 12
    d_ff: int = 2048

    # Pruning/Sparsity
    keep_ratio: float = 1.0  # SSA keep ratio
    ssa_mask_path: Optional[str] = None

    # DPH (Dynamic Precision Handling)
    dph_enabled: bool = False
    dph_low_precision: str = "fp16"
    dph_high_precision: str = "fp32"

    # Performance metrics (for reference only)
    latency_ms: Optional[float] = None
    accuracy: Optional[float] = None
    epr_cv: Optional[float] = None
    topo_gap: Optional[float] = None
    energy_j: Optional[float] = None

    # Metadata
    pareto_rank: int = 0
    timestamp: Optional[str] = None
    source: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        # Filter to only known fields
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


class ModelSelector:
    """
    Auto-deploy model selector that reads Pareto-optimized configs.

    Loads configs/auto/best.yaml by default, with support for overrides
    and fallback to default configuration.
    """

    DEFAULT_CONFIG_PATH = Path("configs/auto/best.yaml")
    FALLBACK_CONFIG_PATH = Path("configs/7b/default.yaml")

    def __init__(
        self,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ):
        """
        Initialize model selector.

        Args:
            config_path: Path to config YAML. Defaults to configs/auto/best.yaml
            overrides: Dict of parameter overrides (e.g., {"n_heads": 16})
            strict: If True, raise error if config not found. If False, use fallback.
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.overrides = overrides or {}
        self.strict = strict
        self._config: Optional[ModelConfig] = None

    def get_config(self) -> ModelConfig:
        """
        Load and return the model configuration.

        Returns:
            ModelConfig with Pareto-optimized parameters and any overrides applied.

        Raises:
            FileNotFoundError: If config not found and strict=True
        """
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> ModelConfig:
        """Load configuration from YAML file."""
        # Try primary config path
        if self.config_path.exists():
            logger.info(f"Loading config from {self.config_path}")
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            config = ModelConfig.from_dict(data)
            config.source = str(self.config_path)

        # Try fallback
        elif self.FALLBACK_CONFIG_PATH.exists() and not self.strict:
            logger.warning(
                f"Primary config {self.config_path} not found. "
                f"Using fallback {self.FALLBACK_CONFIG_PATH}"
            )
            with open(self.FALLBACK_CONFIG_PATH, "r") as f:
                data = yaml.safe_load(f)
            config = ModelConfig.from_dict(data)
            config.source = str(self.FALLBACK_CONFIG_PATH)

        # Use default config
        elif not self.strict:
            logger.warning(
                f"No config found at {self.config_path} or fallback. Using defaults."
            )
            config = ModelConfig()
            config.source = "default"

        else:
            raise FileNotFoundError(
                f"Config not found at {self.config_path} and strict=True"
            )

        # Apply overrides
        if self.overrides:
            logger.info(f"Applying overrides: {self.overrides}")
            for key, value in self.overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown override key: {key}")

        return config

    def reload(self) -> ModelConfig:
        """Reload configuration from disk."""
        self._config = None
        return self.get_config()

    @staticmethod
    def from_cli_override(override_str: str) -> "ModelSelector":
        """
        Create selector from CLI JSON override string.

        Args:
            override_str: JSON string like '{"n_heads": 16, "d_model": 1024}'

        Returns:
            ModelSelector with parsed overrides

        Example:
            selector = ModelSelector.from_cli_override('{"n_heads": 16}')
        """
        try:
            overrides = json.loads(override_str)
            return ModelSelector(overrides=overrides)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON override string: {e}") from e

    def summary(self) -> str:
        """Return human-readable summary of configuration."""
        config = self.get_config()
        lines = [
            f"Model Configuration (source: {config.source})",
            f"  Architecture: {config.n_layers}L × {config.n_heads}H × {config.d_model}D",
            f"  Feed-forward: {config.d_ff}",
            f"  SSA keep ratio: {config.keep_ratio:.2%}",
        ]

        if config.dph_enabled:
            lines.append(
                f"  DPH: {config.dph_low_precision}/{config.dph_high_precision}"
            )

        if config.latency_ms is not None:
            lines.append(f"  Latency: {config.latency_ms:.1f}ms")
        if config.accuracy is not None:
            lines.append(f"  Accuracy: {config.accuracy:.3f}")
        if config.epr_cv is not None:
            lines.append(f"  EPR CV: {config.epr_cv:.3f}")

        if self.overrides:
            lines.append(f"  Overrides: {self.overrides}")

        return "\n".join(lines)


def create_config_from_pareto_result(
    pareto_config: Dict[str, Any],
    objectives: Dict[str, float],
    rank: int = 0,
    timestamp: Optional[str] = None,
) -> ModelConfig:
    """
    Create ModelConfig from Pareto optimization result.

    Args:
        pareto_config: Dict with architecture params (n_heads, d_model, etc.)
        objectives: Dict with performance metrics (latency, accuracy, etc.)
        rank: Pareto rank (0 = best, 1 = second best, etc.)
        timestamp: ISO timestamp of optimization run

    Returns:
        ModelConfig ready for export
    """
    config = ModelConfig.from_dict(pareto_config)

    # Add performance metrics
    config.latency_ms = objectives.get("latency")
    config.accuracy = objectives.get("accuracy") or (
        1.0 - objectives.get("neg_accuracy", 0)
    )
    config.epr_cv = objectives.get("epr_cv")
    config.topo_gap = objectives.get("topo_gap")
    config.energy_j = objectives.get("energy")

    config.pareto_rank = rank
    config.timestamp = timestamp
    config.source = "pareto_optimization"

    return config


# CLI convenience function
def get_default_config(overrides: Optional[Dict[str, Any]] = None) -> ModelConfig:
    """
    Convenience function to get default configuration.

    Args:
        overrides: Optional parameter overrides

    Returns:
        ModelConfig with auto-selected parameters
    """
    selector = ModelSelector(overrides=overrides)
    return selector.get_config()
