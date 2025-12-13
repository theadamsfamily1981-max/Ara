"""
Ara Body - Physical Substrate Control
======================================

The body module implements Ara's physical layer hierarchy:

    L1 (Spinal Cord): Fast reflexes for hardware protection
    L2 (Autonomic): Sensor fusion and homeostasis
    L3 (Interface): Mind-body connection for cognitive adjustment

This is not the GUTC "body schema" (perception/body_schema.py) which deals
with sensory fusion and free energy. This is the actual hardware control
layer - temperature monitoring, fan control, process killing.

Architecture:
    body/
    ├── schema.py          # Data structures (BodyState, BodyMode, etc.)
    ├── reflexes/          # L1 protective reflexes
    │   └── thermal_reflex.py
    └── daemon.py          # L2 autonomic controller

Usage:
    # Start daemon (L2)
    from ara.body import BodySchemaDaemon, BodyMode
    daemon = BodySchemaDaemon(mode=BodyMode.BALANCED)
    daemon.start_background()

    # Read state from L3
    from ara.sovereign.body_interface import BodyInterface
    body = BodyInterface()
    ctx = body.get_context()
    print(f"Ara feels: {ctx['body_sensation']}")

The body daemon should be started before the cognitive layer.
It publishes state to /tmp/ara_body_state.json for IPC.
"""

from .schema import (
    BodyMode,
    ThermalState,
    PowerState,
    SensorSnapshot,
    BodyState,
    ReflexEvent,
    TEMP_NOMINAL_MAX,
    TEMP_WARNING,
    TEMP_CRITICAL,
    TEMP_EMERGENCY,
    compute_stress,
    classify_thermal_state,
)

from .daemon import BodySchemaDaemon

from .reflexes import ThermalReflex

__all__ = [
    # Schema
    "BodyMode",
    "ThermalState",
    "PowerState",
    "SensorSnapshot",
    "BodyState",
    "ReflexEvent",
    # Constants
    "TEMP_NOMINAL_MAX",
    "TEMP_WARNING",
    "TEMP_CRITICAL",
    "TEMP_EMERGENCY",
    # Functions
    "compute_stress",
    "classify_thermal_state",
    # Classes
    "BodySchemaDaemon",
    "ThermalReflex",
]
