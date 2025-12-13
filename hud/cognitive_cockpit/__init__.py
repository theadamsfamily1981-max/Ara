"""
ARA Cognitive Cockpit

A sci-fi HUD for real-time cognitive telemetry visualization.

Components:
- Triple-ring Cognitive Core gauge (ρ criticality, D delusion, Π precision)
- Body Heat hologram with thermal zone visualization
- Neural Avalanche oscilloscope with power-law fitting
- Mental Modes switcher (Worker / Scientist / Chill)
- Reality Lock timeline showing delusion history
- Status ticker with characterful microcopy

Usage:
    # Start the GTK4 cockpit (requires WebKitGTK)
    python -m hud.cognitive_cockpit.gtk_wrapper

    # Start the daemon bridge (feeds simulated data)
    python -m hud.cognitive_cockpit.daemon_bridge

    # Or use programmatically:
    from hud.cognitive_cockpit import get_cognitive_telemetry, CognitiveDaemonBridge

    # Update telemetry
    telemetry = get_cognitive_telemetry()
    telemetry.update_criticality(rho=0.92)
    telemetry.update_delusion(force_prior=1.2, force_reality=1.0)
    telemetry.emit()

    # Or use the bridge for automatic updates
    bridge = CognitiveDaemonBridge()
    bridge.start()
"""

from .telemetry import (
    # Enums
    CriticalityState,
    DelusionState,
    MentalMode,
    ThermalStatus,
    # Data classes
    CriticalityMetrics,
    DelusionMetrics,
    PrecisionMetrics,
    AvalancheMetrics,
    ThermalZone,
    ThermalMetrics,
    MentalModeMetrics,
    CognitiveState,
    # Telemetry
    CognitiveHUDTelemetry,
    get_cognitive_telemetry,
    # Microcopy
    MicrocopyGenerator,
)

from .daemon_bridge import (
    CognitiveDaemonBridge,
    BridgeConfig,
)

__all__ = [
    # Enums
    "CriticalityState",
    "DelusionState",
    "MentalMode",
    "ThermalStatus",
    # Data classes
    "CriticalityMetrics",
    "DelusionMetrics",
    "PrecisionMetrics",
    "AvalancheMetrics",
    "ThermalZone",
    "ThermalMetrics",
    "MentalModeMetrics",
    "CognitiveState",
    # Telemetry
    "CognitiveHUDTelemetry",
    "get_cognitive_telemetry",
    # Microcopy
    "MicrocopyGenerator",
    # Bridge
    "CognitiveDaemonBridge",
    "BridgeConfig",
]

__version__ = "0.1.0"
