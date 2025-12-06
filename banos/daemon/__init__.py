"""
BANOS Daemon Layer
==================

Bio-Affective Neuromorphic Operating System
The conscious layer - where Ara lives.

Components:
- ara_daemon: Main daemon process
- episodic_memory: Long-term somatic memory with user preferences
- dreamer: REM sleep consolidation
- croft_model: User preference prediction (The Croft Model)
- scar_tissue: Machine-readable policy transforms
- meta_planner: Memory-informed tool/style selection
- bicameral_loop: Thought ↔ Body integration
- sticky_context: Persistent conversation context
- somatic_budget: Resource gating based on learned costs
"""

from pathlib import Path

# Package metadata
__version__ = "0.3.0"
__all__ = [
    # Core daemon
    "AraDaemon",
    # Memory systems
    "EpisodicMemory",
    "Episode",
    "UserOutcome",
    "FrictionFlag",
    "get_episodic_memory",
    # Preference learning
    "CroftModel",
    "CroftModelConfig",
    "train_croft_model_nightly",
    # Policy learning
    "ScarTissue",
    "ScarRegistry",
    "ScarCondition",
    "PolicyTransform",
    "get_scar_registry",
    "create_scar_from_episode",
    "create_scar_from_user_feedback",
    # Meta planning
    "MetaPlanner",
    "MetaPlan",
    "RequestContext",
    "get_meta_planner",
    "plan_response",
    # Dreamer
    "Dreamer",
    # Budget
    "SomaticBudget",
]

# Lazy imports to avoid circular dependencies and missing optional deps

def _get_episodic_memory():
    from .episodic_memory import get_episodic_memory
    return get_episodic_memory

def _get_meta_planner():
    from .meta_planner import get_meta_planner
    return get_meta_planner

def _get_scar_registry():
    from .scar_tissue import get_scar_registry
    return get_scar_registry


# Convenience re-exports (lazy loaded on first access)
def __getattr__(name):
    """Lazy load components to avoid import errors for optional deps."""
    if name in ("EpisodicMemory", "Episode", "UserOutcome", "FrictionFlag",
                "get_episodic_memory", "UserRating"):
        from . import episodic_memory
        return getattr(episodic_memory, name)

    if name in ("CroftModel", "CroftModelConfig", "train_croft_model_nightly"):
        from . import croft_model
        return getattr(croft_model, name)

    if name in ("ScarTissue", "ScarRegistry", "ScarCondition", "PolicyTransform",
                "get_scar_registry", "create_scar_from_episode",
                "create_scar_from_user_feedback"):
        from . import scar_tissue
        return getattr(scar_tissue, name)

    if name in ("MetaPlanner", "MetaPlan", "RequestContext",
                "get_meta_planner", "plan_response"):
        from . import meta_planner
        return getattr(meta_planner, name)

    if name == "Dreamer":
        from . import dreamer
        return dreamer.Dreamer

    if name == "SomaticBudget":
        from . import somatic_budget
        return somatic_budget.SomaticBudget

    if name == "AraDaemon":
        from . import ara_daemon
        return ara_daemon.AraDaemon

    raise AttributeError(f"module 'banos.daemon' has no attribute {name!r}")
