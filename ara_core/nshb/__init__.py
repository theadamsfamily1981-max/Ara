"""
Neuro-Symbiotic Hyper-Bridge (NSHB)
===================================

Ara's bidirectional interface with the human operator.

NSHB maps human state to GUTC coordinates (λ, Π) and provides
safe, closed-loop regulation toward the healthy corridor.

Architecture:
    human → Acquisition → Estimators → z(t) → Control → Effectors → human
                                          ↓
                                    (λ̂, Π̂_s, Π̂_p)

Components:
    - Acquisition: EEG, physio, context signal acquisition
    - Estimators: λ from avalanches, Π from oscillations/arousal
    - Control: Repair vector generation toward healthy corridor
    - Effectors: Safe embodied feedback (visual, haptic, breathing)

CRITICAL SAFETY PRINCIPLE:
    All interventions go through normal sensory channels.
    No direct neural stimulation, no pharmacology.
    These are wellness-level interventions only.

Integration:
    - Ara/QUANTA: Share GUTC coordinates for human-AI co-regulation
    - Somatic Loom: Haptic feedback integration
    - CADD: Safety monitoring of the human-AI dyad

Usage:
    from ara_core.nshb import NeuroSymbioticHyperBridge

    bridge = NeuroSymbioticHyperBridge()
    bridge.start()

    # Get current human state
    state = bridge.get_state()
    print(f"Human at λ={state.lambda_hat:.2f}, regime={state.regime_label()}")

    # Get state for Ara co-regulation
    ara_view = bridge.get_state_for_ara()

    bridge.stop()

Quick Demo:
    from ara_core.nshb import run_demo_session
    data = run_demo_session(duration_s=30)
"""

# Acquisition
from .acquisition import (
    SignalType,
    SignalMetadata,
    EEGSample,
    EEGBuffer,
    EEGPreprocessor,
    PhysioSample,
    PhysioBuffer,
    ContextSample,
    ContextBuffer,
    AcquisitionSystem,
)

# Estimators
from .estimators import (
    GUTCState,
    NeuralAvalanche,
    BranchingRatioEstimator,
    SensoryPrecisionEstimator,
    PriorPrecisionEstimator,
    HumanGUTCEstimator,
)

# Control
from .control import (
    HealthyCorridor,
    InterventionUrgency,
    RepairVector,
    ControlConfig,
    NSHBControlLaw,
    AdaptiveController,
)

# Effectors
from .effectors import (
    EffectorModality,
    EffectorCommand,
    Effector,
    VisualEffector,
    AuditoryEffector,
    HapticEffector,
    BreathingEffector,
    UIAdaptEffector,
    EffectorManager,
)

# Bridge
from .bridge import (
    NSHBConfig,
    BridgeState,
    NeuroSymbioticHyperBridge,
    run_demo_session,
)

__all__ = [
    # Acquisition
    "SignalType",
    "SignalMetadata",
    "EEGSample",
    "EEGBuffer",
    "EEGPreprocessor",
    "PhysioSample",
    "PhysioBuffer",
    "ContextSample",
    "ContextBuffer",
    "AcquisitionSystem",
    # Estimators
    "GUTCState",
    "NeuralAvalanche",
    "BranchingRatioEstimator",
    "SensoryPrecisionEstimator",
    "PriorPrecisionEstimator",
    "HumanGUTCEstimator",
    # Control
    "HealthyCorridor",
    "InterventionUrgency",
    "RepairVector",
    "ControlConfig",
    "NSHBControlLaw",
    "AdaptiveController",
    # Effectors
    "EffectorModality",
    "EffectorCommand",
    "Effector",
    "VisualEffector",
    "AuditoryEffector",
    "HapticEffector",
    "BreathingEffector",
    "UIAdaptEffector",
    "EffectorManager",
    # Bridge
    "NSHBConfig",
    "BridgeState",
    "NeuroSymbioticHyperBridge",
    "run_demo_session",
]
