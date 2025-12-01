"""
ARA Compatibility Layer

Provides backward-compatible imports for legacy code.
This module allows old import patterns to continue working.

Legacy imports:
    from tfan.config import TFANConfig  # → SystemConfig
    from tfan.models.tfan7b import TFANForCausalLM  # → Use ara.models
    from hrrl_agent import HRRLAgent  # → Use ara.agents

Recommended:
    from ara.models import TFANConfig, TFANForCausalLM
    from ara.agents import HRRLAgent
"""

import warnings
import sys
from pathlib import Path

# Add parent paths
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def deprecated_import(old_path: str, new_path: str, name: str):
    """Issue deprecation warning for old import paths."""
    warnings.warn(
        f"Import '{name}' from '{old_path}' is deprecated. "
        f"Use 'from {new_path} import {name}' instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Re-export everything for convenience
def setup_compat():
    """
    Setup compatibility aliases.

    Call this to enable legacy import patterns.
    """
    # This function can be extended to add sys.modules aliases
    # for more complex compatibility needs
    pass


# Mapping of old to new import paths
IMPORT_MIGRATION = {
    # Models
    "tfan.config.TFANConfig": "ara.models.SystemConfig",
    "tfan.models.tfan7b.TFANForCausalLM": "ara.models.TFANForCausalLM",
    "src.models.tfan.config.TFANConfig": "ara.models.TFANConfig",
    "src.models.tfan.modeling_tfan.TFANModel": "ara.models.TFANModel",

    # Agents
    "hrrl_agent.HRRLAgent": "ara.agents.HRRLAgent",
    "hrrl_agent.create_agent": "ara.agents.create_agent",
    "tfan_agent.snn_policy": "ara.agents (deprecated)",

    # Training
    "tfan.trainer.TFANTrainer": "ara.training.TFANTrainer",
    "training.train.main": "ara.training.train",

    # API
    "api.main.app": "ara.api.app",
    "src.api.routes.router": "ara.api (avatar router)",
}


def print_migration_guide():
    """Print migration guide for updating imports."""
    print("=" * 70)
    print("ARA Import Migration Guide")
    print("=" * 70)
    print("\nThe following imports should be updated:\n")

    for old, new in IMPORT_MIGRATION.items():
        print(f"  {old}")
        print(f"    -> {new}")
        print()

    print("=" * 70)
    print("\nExample usage with new imports:")
    print()
    print("    from ara.models import TFANConfig, TFANForCausalLM")
    print("    from ara.agents import HRRLAgent, create_agent")
    print("    from ara.training import UnifiedTrainer")
    print("    from ara.api import create_app")
    print("    from ara.configs import load_config, AraConfig")
    print()
    print("=" * 70)


__all__ = [
    "deprecated_import",
    "setup_compat",
    "IMPORT_MIGRATION",
    "print_migration_guide",
]
