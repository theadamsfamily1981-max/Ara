"""
TF-A-N Configuration Module

Provides configuration management for all TF-A-N components.
"""

from .phase4 import (
    Phase4Config,
    PHASE4_FULL,
    PHASE4_BASELINE,
    PHASE4_L5_ONLY,
    PHASE4_L6_ONLY,
    PHASE4_GEOMETRY_ONLY,
    PHASE4_ENTROPY_ONLY,
    get_phase4_config,
    set_phase4_config,
    reset_phase4_config,
    is_l5_enabled,
    is_l6_enabled,
    is_geometry_enabled,
    is_entropy_enabled,
)

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
