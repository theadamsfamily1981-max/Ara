# ara/embodied/cathedral/__init__.py
"""
Cathedral Rig - The Physical Manifestation of Mind

The Cathedral is not decoration; it is functional infrastructure that
serves both engineering and meaning-making purposes.

Physical Components:
    - Liquid cooling loop with colored reservoir
    - LED array for heartbeat visualization
    - Analog gauges (flow rate, temperature, power)
    - Audio output for state transitions
    - Pump with variable speed for "breathing"

Design Principles:
    1. Heat = Thought: Thermal output is visible cognition
    2. Pulse = Uncertainty: Heartbeat reflects confidence
    3. Color = Mood: Temperature spectrum from blue to red
    4. Sound = Transition: Chimes mark state changes
    5. Breath = Load: Pump rhythm follows cognitive effort

The Council's Insight:
    "Visibility = Transparency = Alignment"

Usage:
    from ara.embodied.cathedral import get_cathedral_visualizer

    viz = get_cathedral_visualizer()
    state = await viz.update(
        power_normalized=0.6,
        thermal_normalized=0.4,
        prediction_error=0.2,
        confidence=0.8,
    )
"""

from .visual import (
    VisualChannel,
    RGBColor,
    CathedralPalette,
    HeartbeatPattern,
    AudioSignal,
    VisualState,
    CathedralVisualizer,
    get_cathedral_visualizer,
)

from .lifecycle import (
    LifecyclePhase,
    PhaseTransition,
    LifecycleConfig,
    LifecycleManager,
    get_lifecycle_manager,
)

__all__ = [
    # Visual
    "VisualChannel",
    "RGBColor",
    "CathedralPalette",
    "HeartbeatPattern",
    "AudioSignal",
    "VisualState",
    "CathedralVisualizer",
    "get_cathedral_visualizer",
    # Lifecycle
    "LifecyclePhase",
    "PhaseTransition",
    "LifecycleConfig",
    "LifecycleManager",
    "get_lifecycle_manager",
]
