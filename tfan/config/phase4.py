"""
Phase 4 Configuration: Cognitive Autonomy Feature Toggles

This module provides configuration toggles for Phase 4 cognitive autonomy
components. Use these to enable/disable features for A/B testing and
performance comparison.

Usage:
    from tfan.config.phase4 import Phase4Config, get_phase4_config

    config = get_phase4_config()
    if config.l5_enabled:
        # Use L5 meta-learning
        ...

    # Or modify at runtime:
    config.l5_enabled = False
"""

import os
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("tfan.config.phase4")


@dataclass
class Phase4Config:
    """
    Configuration toggles for Phase 4 Cognitive Autonomy components.

    These flags allow enabling/disabling individual cognitive features
    for testing, comparison, and gradual rollout.
    """

    # L5 Meta-Learning: AEPO learns L3 control laws
    l5_enabled: bool = True

    # L6 Reasoning: PGU + Knowledge Graph + L3-aware retrieval
    l6_enabled: bool = True

    # Adaptive Geometry: Task-optimized hyperbolic curvature
    geometry_enabled: bool = True

    # Adaptive Entropy: CLV-modulated exploration
    entropy_enabled: bool = True

    # Metadata
    config_name: str = "default"
    description: str = ""

    def __post_init__(self):
        """Log configuration state."""
        enabled = []
        disabled = []
        for name in ["l5", "l6", "geometry", "entropy"]:
            if getattr(self, f"{name}_enabled"):
                enabled.append(name.upper())
            else:
                disabled.append(name.upper())

        if enabled:
            logger.info(f"Phase 4 enabled: {', '.join(enabled)}")
        if disabled:
            logger.info(f"Phase 4 disabled: {', '.join(disabled)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved Phase 4 config to {path}")

    @classmethod
    def load(cls, path: str) -> "Phase4Config":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Phase4Config":
        """
        Create configuration from environment variables.

        Environment variables:
            PHASE4_L5_ENABLED=1|0
            PHASE4_L6_ENABLED=1|0
            PHASE4_GEOMETRY_ENABLED=1|0
            PHASE4_ENTROPY_ENABLED=1|0
        """
        def env_bool(name: str, default: bool = True) -> bool:
            val = os.environ.get(name, "").lower()
            if val in ("0", "false", "no", "off"):
                return False
            if val in ("1", "true", "yes", "on"):
                return True
            return default

        return cls(
            l5_enabled=env_bool("PHASE4_L5_ENABLED"),
            l6_enabled=env_bool("PHASE4_L6_ENABLED"),
            geometry_enabled=env_bool("PHASE4_GEOMETRY_ENABLED"),
            entropy_enabled=env_bool("PHASE4_ENTROPY_ENABLED"),
            config_name="from_env",
        )

    def all_enabled(self) -> bool:
        """Check if all Phase 4 features are enabled."""
        return (
            self.l5_enabled and
            self.l6_enabled and
            self.geometry_enabled and
            self.entropy_enabled
        )

    def all_disabled(self) -> bool:
        """Check if all Phase 4 features are disabled."""
        return not (
            self.l5_enabled or
            self.l6_enabled or
            self.geometry_enabled or
            self.entropy_enabled
        )

    def enabled_features(self) -> list:
        """Get list of enabled feature names."""
        features = []
        if self.l5_enabled:
            features.append("L5")
        if self.l6_enabled:
            features.append("L6")
        if self.geometry_enabled:
            features.append("Geometry")
        if self.entropy_enabled:
            features.append("Entropy")
        return features

    def summary(self) -> str:
        """Get human-readable summary."""
        if self.all_enabled():
            return "Phase 4: ALL ENABLED"
        if self.all_disabled():
            return "Phase 4: ALL DISABLED (baseline)"
        return f"Phase 4: {', '.join(self.enabled_features())} enabled"


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# All Phase 4 features enabled (production)
PHASE4_FULL = Phase4Config(
    l5_enabled=True,
    l6_enabled=True,
    geometry_enabled=True,
    entropy_enabled=True,
    config_name="full",
    description="All Phase 4 cognitive autonomy features enabled",
)

# All Phase 4 features disabled (baseline for comparison)
PHASE4_BASELINE = Phase4Config(
    l5_enabled=False,
    l6_enabled=False,
    geometry_enabled=False,
    entropy_enabled=False,
    config_name="baseline",
    description="Phase 4 disabled - baseline for A/B comparison",
)

# Only L5 meta-learning enabled
PHASE4_L5_ONLY = Phase4Config(
    l5_enabled=True,
    l6_enabled=False,
    geometry_enabled=False,
    entropy_enabled=False,
    config_name="l5_only",
    description="Only L5 meta-learning enabled",
)

# Only L6 reasoning enabled
PHASE4_L6_ONLY = Phase4Config(
    l5_enabled=False,
    l6_enabled=True,
    geometry_enabled=False,
    entropy_enabled=False,
    config_name="l6_only",
    description="Only L6 reasoning enabled",
)

# Only adaptive geometry enabled
PHASE4_GEOMETRY_ONLY = Phase4Config(
    l5_enabled=False,
    l6_enabled=False,
    geometry_enabled=True,
    entropy_enabled=False,
    config_name="geometry_only",
    description="Only adaptive geometry enabled",
)

# Only adaptive entropy enabled
PHASE4_ENTROPY_ONLY = Phase4Config(
    l5_enabled=False,
    l6_enabled=False,
    geometry_enabled=False,
    entropy_enabled=True,
    config_name="entropy_only",
    description="Only adaptive entropy enabled",
)


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

_global_config: Optional[Phase4Config] = None


def get_phase4_config() -> Phase4Config:
    """Get the global Phase 4 configuration."""
    global _global_config
    if _global_config is None:
        # Check for environment-based config first
        if any(k.startswith("PHASE4_") for k in os.environ):
            _global_config = Phase4Config.from_env()
        else:
            _global_config = PHASE4_FULL  # Default to all enabled
    return _global_config


def set_phase4_config(config: Phase4Config):
    """Set the global Phase 4 configuration."""
    global _global_config
    _global_config = config
    logger.info(f"Phase 4 config set: {config.summary()}")


def reset_phase4_config():
    """Reset to default configuration."""
    global _global_config
    _global_config = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_l5_enabled() -> bool:
    """Check if L5 meta-learning is enabled."""
    return get_phase4_config().l5_enabled


def is_l6_enabled() -> bool:
    """Check if L6 reasoning is enabled."""
    return get_phase4_config().l6_enabled


def is_geometry_enabled() -> bool:
    """Check if adaptive geometry is enabled."""
    return get_phase4_config().geometry_enabled


def is_entropy_enabled() -> bool:
    """Check if adaptive entropy is enabled."""
    return get_phase4_config().entropy_enabled


__all__ = [
    "Phase4Config",
    "PHASE4_FULL",
    "PHASE4_BASELINE",
    "PHASE4_L5_ONLY",
    "PHASE4_L6_ONLY",
    "PHASE4_GEOMETRY_ONLY",
    "PHASE4_ENTROPY_ONLY",
    "get_phase4_config",
    "set_phase4_config",
    "reset_phase4_config",
    "is_l5_enabled",
    "is_l6_enabled",
    "is_geometry_enabled",
    "is_entropy_enabled",
]
