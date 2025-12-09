"""
Ara Embodiment Module
======================

Physical body systems with strict safety rails.

Components:
- core: Embodied multimodal sensor→HV→action loop
- breath_vision: Guided breathing sessions (wellness, not therapy)
- fusion: Self-awareness and system monitoring

All operations are gated by covenants that define hard limits.
E-stop is always available. Safety first.

Usage:
    # Embodied core (for robot body)
    from ara.embodiment import EmbodiedCore
    core = EmbodiedCore()
    await core.start()

    # Breath-vision session
    from ara.embodiment import BreathVisionSession
    session = BreathVisionSession(duration_minutes=3)
    summary = await session.run()

    # Fusion monitor
    from ara.embodiment import FusionMonitor
    monitor = FusionMonitor()
    await monitor.start()

IMPORTANT:
- This module enforces strict physical safety limits
- No auto-uploads, no human impersonation, no dangerous motions
- Breath-vision is WELLNESS, not medical treatment
- E-stop immediately releases all motors and stops all output

For CLI tools:
    ara_embodiment start          # Start embodied core
    ara_breath_vision --duration 3  # Run breathing session
    ara_fusion status             # Show system status
"""

# Safety rails (import first - always active)
from .rails import (
    # Core rail classes
    EmbodimentRails,
    BreathVisionRails,
    FusionRails,
    EmbodimentCovenant,

    # Motor safety
    MotorRails,
    MotorCommand,
    ClampedCommand,

    # Haptic safety
    HapticRails,
    HapticCommand,

    # Visual safety
    VisualRails,
    VisualCommand,

    # Session safety
    SessionRails,

    # Enums
    SafetyLevel,
    ActionResult,

    # Convenience functions
    check_motor_safe,
    check_breath_session_allowed,
    get_stop_phrases,
)

# Embodied core
from .core import (
    EmbodiedCore,
    CoreState,
    SensorType,
    SensorReading,
    WorldState,
    ActionType,
    IntendedAction,
    ExecutedAction,
    EmbodimentEpisode,
    SensorInterface,
    ActuatorInterface,
    MockSensorInterface,
    MockActuatorInterface,
)

# Breath-vision protocol
from .breath_vision import (
    BreathVisionSession,
    BreathPhase,
    SessionState,
    BreathPattern,
    VisualStyle,
    SessionSummary,
    PATTERNS,
    STYLES,
    BreathVisionActuator,
    MockBreathVisionActuator,
)

# Fusion monitor
from .fusion import (
    FusionMonitor,
    MetricsCollector,
    SystemStatus,
    EnvironmentStatus,
    HealthLevel,
    AlertLevel,
    SuggestedAction,
    MetricReading,
)


__all__ = [
    # Rails
    'EmbodimentRails',
    'BreathVisionRails',
    'FusionRails',
    'EmbodimentCovenant',
    'MotorRails',
    'MotorCommand',
    'ClampedCommand',
    'HapticRails',
    'HapticCommand',
    'VisualRails',
    'VisualCommand',
    'SessionRails',
    'SafetyLevel',
    'ActionResult',
    'check_motor_safe',
    'check_breath_session_allowed',
    'get_stop_phrases',

    # Core
    'EmbodiedCore',
    'CoreState',
    'SensorType',
    'SensorReading',
    'WorldState',
    'ActionType',
    'IntendedAction',
    'ExecutedAction',
    'EmbodimentEpisode',
    'SensorInterface',
    'ActuatorInterface',
    'MockSensorInterface',
    'MockActuatorInterface',

    # Breath-Vision
    'BreathVisionSession',
    'BreathPhase',
    'SessionState',
    'BreathPattern',
    'VisualStyle',
    'SessionSummary',
    'PATTERNS',
    'STYLES',
    'BreathVisionActuator',
    'MockBreathVisionActuator',

    # Fusion
    'FusionMonitor',
    'MetricsCollector',
    'SystemStatus',
    'EnvironmentStatus',
    'HealthLevel',
    'AlertLevel',
    'SuggestedAction',
    'MetricReading',
]
