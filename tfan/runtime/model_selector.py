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


# =============================================================================
# Autonomous Model Selector with Semantic Optimization
# =============================================================================

try:
    from tfan.system.semantic_optimizer import (
        SemanticSystemOptimizer,
        PADState,
        ResourceFeatures,
        RoutingDecision,
        Backend,
    )
    SEMANTIC_OPTIMIZER_AVAILABLE = True
except ImportError:
    SEMANTIC_OPTIMIZER_AVAILABLE = False


class AutonomousModelSelector(ModelSelector):
    """
    Context-aware model selector with semantic optimization.

    Extends ModelSelector with:
    1. PAD-based backend routing (valence → safety preference)
    2. Resource-aware scheduling (FPGA/GPU utilization)
    3. PGU verification integration
    4. Persistent routing score learning

    Usage:
        selector = AutonomousModelSelector()

        # Get config with context-aware backend
        config, routing = selector.get_config_with_routing(
            valence=-0.3,
            arousal=0.8,
            stability_gap=0.1,
        )

        # Execute on recommended backend
        result = execute_on_backend(config, routing.backend)

        # Provide feedback for learning
        selector.record_outcome(routing.backend, success=True, latency_ms=5.2)
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
        strict: bool = False,
        scores_path: Optional[Path] = None,
    ):
        """
        Initialize autonomous selector.

        Args:
            config_path: Path to model config YAML
            overrides: Parameter overrides
            strict: If True, raise error if config not found
            scores_path: Path to routing scores persistence
        """
        super().__init__(config_path, overrides, strict)

        # Initialize semantic optimizer if available
        self.semantic_optimizer = None
        if SEMANTIC_OPTIMIZER_AVAILABLE:
            self.semantic_optimizer = SemanticSystemOptimizer(
                scores_path=scores_path,
                auto_persist=True,
                safety_first=True,
            )
            logger.info("AutonomousModelSelector initialized with semantic optimizer")
        else:
            logger.warning("Semantic optimizer not available, using basic selection")

        # Cache for resource features
        self._cached_resources: Optional[ResourceFeatures] = None
        self._resource_cache_time: float = 0.0
        self._resource_cache_ttl: float = 5.0  # seconds

    def get_config_with_routing(
        self,
        valence: float = 0.0,
        arousal: float = 0.5,
        dominance: float = 0.5,
        stability_gap: float = 0.0,
        workload_hint: Optional[str] = None,
        resource_features: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Get model config with context-aware routing decision.

        This is the main entry point for autonomous backend selection.

        Args:
            valence: PAD valence [-1, 1]
            arousal: PAD arousal [0, 1]
            dominance: PAD dominance [0, 1]
            stability_gap: Topological stability metric
            workload_hint: Optional hint ("latency_critical", "throughput", "safe")
            resource_features: Optional resource state override

        Returns:
            (ModelConfig, RoutingDecision) tuple
        """
        # Get base config
        config = self.get_config()

        # If no optimizer, return default routing
        if self.semantic_optimizer is None:
            return config, self._default_routing()

        # Build PAD state
        pad = PADState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            stability_gap=stability_gap,
        )

        # Get resource features
        resources = self._get_resource_features(resource_features)

        # Get routing decision
        routing = self.semantic_optimizer.recommend_route(
            pad_state=pad,
            resource_features=resources,
            workload_hint=workload_hint,
        )

        # Apply routing-specific config adjustments
        config = self._apply_routing_config(config, routing)

        logger.info(
            f"Autonomous selection: {routing.backend.value} "
            f"(v={valence:.2f}, a={arousal:.2f}, conf={routing.confidence:.2f})"
        )

        return config, routing

    def record_outcome(
        self,
        backend: str,
        success: bool,
        latency_ms: float,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
    ):
        """
        Record execution outcome for online learning.

        Args:
            backend: Backend that was used (string or Backend enum)
            success: Whether execution succeeded
            latency_ms: Actual execution latency
            valence: PAD valence when decision was made
            arousal: PAD arousal when decision was made
        """
        if self.semantic_optimizer is None:
            return

        # Convert string to Backend enum if needed
        if isinstance(backend, str):
            try:
                backend = Backend(backend)
            except ValueError:
                logger.warning(f"Unknown backend: {backend}")
                return

        # Build PAD state if provided
        pad = None
        if valence is not None:
            pad = PADState(
                valence=valence,
                arousal=arousal or 0.5,
            )

        self.semantic_optimizer.update_from_feedback(
            backend=backend,
            success=success,
            latency_ms=latency_ms,
            pad_state=pad,
        )

    def _get_resource_features(
        self,
        override: Optional[Dict[str, Any]] = None,
    ) -> "ResourceFeatures":
        """Get current resource features (with caching)."""
        import time

        # Use override if provided
        if override:
            return ResourceFeatures.from_dict(override)

        # Check cache
        now = time.time()
        if (
            self._cached_resources is not None
            and now - self._resource_cache_time < self._resource_cache_ttl
        ):
            return self._cached_resources

        # Probe actual resources
        resources = self._probe_resources()
        self._cached_resources = resources
        self._resource_cache_time = now

        return resources

    def _probe_resources(self) -> "ResourceFeatures":
        """Probe actual hardware resource state."""
        features = ResourceFeatures()

        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                features.gpu_available = True
                features.gpu_memory_free_gb = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated(0)
                ) / (1024 ** 3)
        except Exception:
            pass

        # Try to get FPGA info from synergy
        try:
            from synergy.fpga_device import MCP_AVAILABLE
            features.fpga_available = MCP_AVAILABLE
        except Exception:
            pass

        return features

    def _apply_routing_config(
        self,
        config: ModelConfig,
        routing: "RoutingDecision",
    ) -> ModelConfig:
        """Apply routing-specific configuration adjustments."""
        # Sparse backend → enable SSA
        if routing.backend == Backend.GPU_SPARSE:
            if config.keep_ratio == 1.0:
                config.keep_ratio = 0.5  # Enable sparse if not set

        # FPGA backend → adjust for hardware constraints
        if routing.backend == Backend.FPGA_SNN:
            config.dph_enabled = True
            config.dph_low_precision = "int8"

        # PGU verified → ensure constraints respected
        if routing.backend == Backend.PGU_VERIFIED:
            # Could adjust parameters to ensure PGU verification passes
            pass

        return config

    def _default_routing(self) -> "RoutingDecision":
        """Return default routing when optimizer unavailable."""
        return RoutingDecision(
            backend=Backend.GPU_DENSE,
            confidence=0.5,
            scores={},
            reasoning=["Default routing (optimizer unavailable)"],
            pgu_required=False,
            fallback_backend=Backend.CPU_FALLBACK,
        )

    def get_optimizer_status(self) -> Dict[str, Any]:
        """Get semantic optimizer status."""
        if self.semantic_optimizer is None:
            return {"available": False}

        status = self.semantic_optimizer.get_status()
        status["available"] = True
        return status


def create_autonomous_selector(
    config_path: Optional[Path] = None,
    scores_path: Optional[Path] = None,
) -> AutonomousModelSelector:
    """
    Factory function to create AutonomousModelSelector.

    Args:
        config_path: Path to model config
        scores_path: Path to routing scores

    Returns:
        Configured AutonomousModelSelector
    """
    return AutonomousModelSelector(
        config_path=config_path,
        scores_path=scores_path,
    )
