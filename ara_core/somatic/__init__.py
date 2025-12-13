"""
Somatic Loom - GUTC-Integrated Haptic Neuromodulation
======================================================

Closed-loop haptic effector system that operates on the (λ, Π) control
manifold, using subtle thermal and tensile feedback to guide brain state
toward the optimal critical corridor.

GUTC Parameter Mapping:
    ΔΠ_sensory (L4/L2/3 gain, ACh) → Tactile pressure patterns
    ΔΠ_prior (SFC/dACC gain, DA)   → Thermal shift patterns
    Δλ (E/I balance)               → Combined rhythm/frequency

Usage:
    from ara_core.somatic import SomaticLoom, HapticGrammar, GUTCState

    # Initialize and connect
    loom = SomaticLoom()
    loom.connect()

    # Execute pattern for a state
    loom.execute_for_state(GUTCState.ANHEDONIC)

    # Or from GUTC diagnostic
    from ctf1_core.gutc_diagnostic_engine import GUTCDiagnosticEngine
    diagnosis = engine.diagnose(lambda_hat=4.2, pi_hat=0.7)
    pattern = HapticGrammar.from_repair_vector(
        diagnosis.repair_vector.delta_lambda,
        diagnosis.repair_vector.delta_pi,  # sensory
        diagnosis.repair_vector.delta_pi,  # prior
    )
    loom.execute(pattern)
"""

from .loom import (
    # Enums
    HapticModality,
    WaveShape,
    GUTCState,
    # Pattern dataclasses
    ThermalPattern,
    PressurePattern,
    HapticPattern,
    # Grammar
    HapticGrammar,
    # Hardware abstraction
    HapticActuator,
    SimulatedActuator,
    # Main controller
    SomaticLoom,
    # Convenience functions
    get_loom,
    execute_haptic,
)

__all__ = [
    # Enums
    "HapticModality",
    "WaveShape",
    "GUTCState",
    # Patterns
    "ThermalPattern",
    "PressurePattern",
    "HapticPattern",
    # Grammar
    "HapticGrammar",
    # Hardware
    "HapticActuator",
    "SimulatedActuator",
    # Controller
    "SomaticLoom",
    # Functions
    "get_loom",
    "execute_haptic",
]
