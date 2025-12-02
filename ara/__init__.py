"""
ARA: Unified AI System

A unified framework combining:
- TFAN: Transformer with Formal Alignment Network (7B params)
- TGSFN: Thermodynamic Gated Spiking Neural Network
- HRRL: Homeostatic Reinforcement Regulated Learning

This package provides a single entry point for all ARA components.

Example usage:

    from ara import TFANConfig, TFANForCausalLM
    from ara import HRRLAgent, create_agent
    from ara.configs import AraConfig, load_config

"""

__version__ = "0.1.0"
__author__ = "ARA Framework"

# Lazy imports to avoid circular dependencies and missing packages
_lazy_imports = {}


def __getattr__(name):
    """Lazy import handler for main package attributes."""
    if name in _lazy_imports:
        return _lazy_imports[name]

    # Models
    if name in ("TFANConfig", "TFANModel", "TFANForCausalLM", "SystemConfig"):
        try:
            from ara import models
            _lazy_imports[name] = getattr(models, name)
            return _lazy_imports[name]
        except (ImportError, AttributeError):
            return None

    # Agents
    if name in ("HRRLAgent", "HRRLConfig", "create_agent", "TGSFNSubstrate", "TGSFNConfig"):
        try:
            from ara import agents
            _lazy_imports[name] = getattr(agents, name)
            return _lazy_imports[name]
        except (ImportError, AttributeError):
            return None

    # Utils
    if name in ("get_device", "set_seed"):
        try:
            from ara import utils
            _lazy_imports[name] = getattr(utils, name)
            return _lazy_imports[name]
        except (ImportError, AttributeError):
            return None

    # Service (Unified cognitive layer)
    if name in ("AraService", "AraState", "AraResponse", "HardwareMode",
                "HardwareProfile", "EmotionalSurface", "CognitiveLoad", "create_ara"):
        try:
            from ara.service import core
            _lazy_imports[name] = getattr(core, name)
            return _lazy_imports[name]
        except (ImportError, AttributeError):
            return None

    raise AttributeError(f"module 'ara' has no attribute '{name}'")


__all__ = [
    # Version
    "__version__",

    # Models
    "TFANConfig",
    "TFANModel",
    "TFANForCausalLM",
    "SystemConfig",

    # Agents
    "HRRLAgent",
    "HRRLConfig",
    "create_agent",
    "TGSFNSubstrate",
    "TGSFNConfig",

    # Utils
    "get_device",
    "set_seed",

    # Service (Unified cognitive layer)
    "AraService",
    "AraState",
    "AraResponse",
    "HardwareMode",
    "HardwareProfile",
    "EmotionalSurface",
    "CognitiveLoad",
    "create_ara",
]
