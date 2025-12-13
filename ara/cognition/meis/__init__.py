"""
MEIS: Memory-Epistemic-Inference System
=======================================

GUTC-based criticality monitoring and adaptive control for Ara's cognitive systems.

This module implements runtime criticality monitoring based on the Grand Unified
Theory of Cognition (GUTC), ensuring that computational dynamics remain near the
critical point (λ ≈ 1) where capacity is maximized.

Components:
    CriticalityMonitor: Real-time branching ratio estimation and regime classification
    CriticalityController: PID-based closed-loop control to maintain criticality
    CriticalityState: Diagnostic state with action recommendations
    CriticalityRegime: Classification of dynamical regime
    ControlAction: Recommended control actions

Theory:
    At criticality (λ = 1, E(λ) = 0):
    - Computational capacity C ∝ 1/|E(λ)| is maximized
    - Avalanche statistics follow universal power laws (α ≈ 3/2, τ ≈ 2)
    - Information transmission and dynamic range are optimal
    - Epistemic value for exploration peaks

Usage:
    from ara.cognition.meis import CriticalityMonitor, CriticalityController

    monitor = CriticalityMonitor()
    controller = CriticalityController()

    for activity in activity_stream:
        state = monitor.update(activity)
        if state.requires_intervention:
            gain = controller.update(state)
            model.apply_gain(gain)

See Also:
    - ctf1-core/gutc_agency.py: GUTC agency functional J(π, λ, Π)
    - docs/theory/CRITICAL_CAPACITY_RUNTIME_POLICY.md: Full policy documentation
"""

from .criticality_monitor import (
    # Core monitor (activity-based)
    CriticalityMonitor,
    CriticalityController,

    # State and classification
    CriticalityState,
    CriticalityRegime,
    ControlAction,

    # Supporting components
    BranchingRatioEstimator,
    AvalancheCollector,
    AvalancheEvent,

    # Gradient-based monitor (for LM training)
    GradientCriticalityMonitor,
    GradientCriticalityState,
)

__all__ = [
    # Core (activity-based)
    "CriticalityMonitor",
    "CriticalityController",

    # State
    "CriticalityState",
    "CriticalityRegime",
    "ControlAction",

    # Components
    "BranchingRatioEstimator",
    "AvalancheCollector",
    "AvalancheEvent",

    # Gradient-based (LM training)
    "GradientCriticalityMonitor",
    "GradientCriticalityState",
]

__version__ = "1.0.0"
