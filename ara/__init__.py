"""
ARA: Autonomous Research Agent

A unified AI system with self-awareness, meta-learning, and embodiment:

Core Components:
- TFAN: Transformer with Formal Alignment Network (7B params)
- TGSFN: Thermodynamic Gated Spiking Neural Network
- HRRL: Homeostatic Reinforcement Regulated Learning

Meta-Learning Layers:
- meta: Pattern detection, research tracking, self-improvement
- academy: Skill internalization from human demonstrations
- institute: Formal research with hypotheses and experiments
- embodied: Hardware awareness and body state
- user: User modeling and prediction

Example usage:

    # Core models and agents
    from ara import TFANConfig, TFANForCausalLM
    from ara import HRRLAgent, create_agent
    from ara.configs import AraConfig, load_config

    # Meta-learning
    from ara.meta import get_meta_brain
    from ara.academy import get_skill_registry
    from ara.institute import get_research_graph
    from ara.embodied import get_embodiment_core
    from ara.user import get_user_model

CLI Commands:
    ara status          # System status
    ara api             # Start API server
    ara chat            # Interactive conversation
    ara meta            # Meta-learning subsystem
    ara academy         # Skill management
    ara institute       # Research institute
    ara body            # Embodiment status
    ara user            # User model

"""

__version__ = "0.2.0"
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

    # Meta-learning layer
    if name in ("get_meta_brain", "MetaBrain", "PatternCard"):
        try:
            from ara import meta
            _lazy_imports[name] = getattr(meta, name)
            return _lazy_imports[name]
        except (ImportError, AttributeError):
            return None

    # Academy (Skill internalization)
    if name in ("get_skill_registry", "SkillRegistry", "get_curriculum_manager"):
        try:
            from ara import academy
            _lazy_imports[name] = getattr(academy, name)
            return _lazy_imports[name]
        except (ImportError, AttributeError):
            return None

    # Institute (Research)
    if name in ("get_research_graph", "ResearchGraph", "get_experiment_scheduler",
                "get_teacher_council", "get_safety_contract", "get_autonomy_manager"):
        try:
            from ara import institute
            _lazy_imports[name] = getattr(institute, name)
            return _lazy_imports[name]
        except (ImportError, AttributeError):
            return None

    # Embodied (Hardware awareness)
    if name in ("get_embodiment_core", "EmbodimentCore", "get_device_graph",
                "get_health_monitor", "wake_ara", "is_ara_awake"):
        try:
            from ara import embodied
            _lazy_imports[name] = getattr(embodied, name)
            return _lazy_imports[name]
        except (ImportError, AttributeError):
            return None

    # User model
    if name in ("get_user_model", "UserModel", "get_user_predictor"):
        try:
            from ara import user
            _lazy_imports[name] = getattr(user, name)
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

    # Meta-learning layer
    "get_meta_brain",
    "MetaBrain",
    "PatternCard",

    # Academy (Skill internalization)
    "get_skill_registry",
    "SkillRegistry",
    "get_curriculum_manager",

    # Institute (Research)
    "get_research_graph",
    "ResearchGraph",
    "get_experiment_scheduler",
    "get_teacher_council",
    "get_safety_contract",
    "get_autonomy_manager",

    # Embodied (Hardware awareness)
    "get_embodiment_core",
    "EmbodimentCore",
    "get_device_graph",
    "get_health_monitor",
    "wake_ara",
    "is_ara_awake",

    # User model
    "get_user_model",
    "UserModel",
    "get_user_predictor",
]
