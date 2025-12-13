"""
Ara HUD: Cognitive Dashboard Visualization
==========================================

Browser-based visualization for Ara's cognitive state.

Components:
    - TelemetryBridge: Connects GUTC monitors to the HUD
    - cognitive_core.html: Triple-ring gauge visualization

Usage:
    # Run the demo HUD
    python -m ara.hud.telemetry_bridge --demo --port 8080

    # Then open: http://localhost:8080/cognitive_core.html

    # In your code
    from ara.hud import TelemetryBridge

    bridge = TelemetryBridge()
    bridge.update(rho=0.85, delusion_index=1.0, mode="HEALTHY_CORRIDOR")
"""

from ara.hud.telemetry_bridge import (
    TelemetryBridge,
    TelemetryState,
    run_hud_server,
    run_demo,
)

__all__ = [
    "TelemetryBridge",
    "TelemetryState",
    "run_hud_server",
    "run_demo",
]
