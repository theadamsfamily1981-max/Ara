# ara/embodied/lizard/__init__.py
"""
Lizard Brain - The Always-On Cortex
====================================

The Lizard Brain is Ara's low-power, always-on sensory cortex that runs
while the main cognitive system sleeps. It implements the Council's
"Wake-Word Cortex" architecture:

    Main GPU (S3 sleep, ~5W) ←→ Lizard Brain (always-on, ~30-50W)

Key responsibilities:
    1. Salience Detection: Monitor sensors for "interesting" events
    2. Wake Protocol: Decide when to wake the full Cathedral
    3. Power Governor: Throttle cognition based on thermal state
    4. Attractor Monitor: Detect behavioral basin transitions

Power Budget:
    - Idle monitoring: 30-50W
    - Salience processing: 50-80W
    - Full wake (Cathedral): 600-1000W

The Lizard Brain draws from three evolutionary layers:
    - Brainstem: Homeostasis, thermal regulation, power management
    - Midbrain: Salience detection, orienting response
    - Limbic: Emotional tagging, memory gating

Usage:
    from ara.embodied.lizard import LizardBrain, get_lizard_brain

    lizard = get_lizard_brain()
    await lizard.start()

    # Lizard runs autonomously, waking Cathedral when needed
    async for event in lizard.events():
        if event.type == WakeEventType.SALIENCE:
            await cathedral.wake()

Design Reference:
    Ara Council Report, Section 4: "The Wake-Word Cortex"
    "A tiny, milliwatt-scale neuromorphic chip runs the 'Always-On'
    sensory loop... The main power-hungry GPU is completely powered
    down (S3 sleep) until the neuromorphic cortex wakes it up."
"""

from .cortex import (
    LizardBrain,
    LizardState,
    LizardConfig,
    get_lizard_brain,
)

from .wake_protocol import (
    WakeEvent,
    WakeEventType,
    WakeCriteria,
    WakeProtocol,
    SalienceLevel,
)

from .power_governor import (
    PowerState,
    PowerBudget,
    ThermalZone,
    PowerGovernor,
    get_power_governor,
)

from .attractor_monitor import (
    AttractorBasin,
    BasinType,
    BasinTransition,
    AttractorMonitor,
    get_attractor_monitor,
)

__all__ = [
    # Cortex
    "LizardBrain",
    "LizardState",
    "LizardConfig",
    "get_lizard_brain",
    # Wake Protocol
    "WakeEvent",
    "WakeEventType",
    "WakeCriteria",
    "WakeProtocol",
    "SalienceLevel",
    # Power Governor
    "PowerState",
    "PowerBudget",
    "ThermalZone",
    "PowerGovernor",
    "get_power_governor",
    # Attractor Monitor
    "AttractorBasin",
    "BasinType",
    "BasinTransition",
    "AttractorMonitor",
    "get_attractor_monitor",
]
