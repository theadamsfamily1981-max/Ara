"""
Ara Homeostatic Operating System
================================

The homeostatic OS is the control system that keeps Ara alive.
It implements a closed-loop feedback architecture:

    Receptors → Sovereign → Effectors → Environment → Receptors
                    ↓
              Safety Monitor

Modules:
    config.py    - Setpoints and teleology weights
    state.py     - HomeostaticState, ErrorVector, Telemetry
    receptors.py - ReceptorDaemon (5 kHz sensory input)
    sovereign.py - SovereignLoop (200 Hz decision-making)
    effectors.py - EffectorDaemon (500 Hz motor output)
    safety.py    - SafetyMonitor, AuditDaemon
    boot.py      - AraOrganism, boot sequence

Usage:
    from ara.homeostasis import AraOrganism

    ara = AraOrganism()
    result = ara.boot()

    if result.success:
        ara.run()  # Blocks until shutdown

Or run directly:
    python -m ara.homeostasis.boot
"""

from .config import (
    Setpoints,
    TeleologyWeights,
    ModeConfig,
    MODES,
    HomeostaticConfig,
    get_default_config,
)

from .state import (
    OperationalMode,
    Telemetry,
    ErrorVector,
    FounderState,
    HomeostaticState,
    compute_error_vector,
    compute_reward,
    StateHistory,
)

from .receptors import (
    ReceptorSource,
    ThermalReceptor,
    NetworkReceptor,
    CognitiveReceptor,
    CathedralReceptor,
    MomentBuilder,
    ReceptorDaemon,
)

from .sovereign import (
    ModeTransition,
    ModeSelector,
    PolyplasticityEngine,
    SovereignLoop,
)

from .effectors import (
    CommandType,
    EffectorCommand,
    ReflexLayer,
    SpinalLayer,
    CorticalLayer,
    ModeEffects,
    ErrorResponse,
    EffectorDaemon,
)

from .safety import (
    ViolationType,
    EnforcementAction,
    SafetyViolation,
    Invariant,
    ThermalInvariant,
    MemoryInvariant,
    HeartbeatInvariant,
    WatchdogInvariant,
    SafetyMonitor,
    AuditDaemon,
)

from .boot import (
    BootStage,
    BootResult,
    AraOrganism,
    main,
)


__all__ = [
    # Config
    'Setpoints',
    'TeleologyWeights',
    'ModeConfig',
    'MODES',
    'HomeostaticConfig',
    'get_default_config',

    # State
    'OperationalMode',
    'Telemetry',
    'ErrorVector',
    'FounderState',
    'HomeostaticState',
    'compute_error_vector',
    'compute_reward',
    'StateHistory',

    # Receptors
    'ReceptorSource',
    'ThermalReceptor',
    'NetworkReceptor',
    'CognitiveReceptor',
    'CathedralReceptor',
    'MomentBuilder',
    'ReceptorDaemon',

    # Sovereign
    'ModeTransition',
    'ModeSelector',
    'PolyplasticityEngine',
    'SovereignLoop',

    # Effectors
    'CommandType',
    'EffectorCommand',
    'ReflexLayer',
    'SpinalLayer',
    'CorticalLayer',
    'ModeEffects',
    'ErrorResponse',
    'EffectorDaemon',

    # Safety
    'ViolationType',
    'EnforcementAction',
    'SafetyViolation',
    'Invariant',
    'ThermalInvariant',
    'MemoryInvariant',
    'HeartbeatInvariant',
    'WatchdogInvariant',
    'SafetyMonitor',
    'AuditDaemon',

    # Boot
    'BootStage',
    'BootResult',
    'AraOrganism',
    'main',
]
