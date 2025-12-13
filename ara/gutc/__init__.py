"""
GUTC - Grand Unified Theory of Cognition
=========================================

Runtime implementation of the GUTC framework for Ara.

This module provides the practical machinery for:
- Active Inference decision-making (EFE scoring)
- Precision diagnostics (Π_y vs Π_μ balance)
- Criticality monitoring (E(λ) tracking)
- Hierarchical memory with Γ coupling

Core Components:
    - active_inference: Policy scoring with extrinsic/intrinsic weights
    - precision_diagnostics: Clinical-style pathology detection
    - (future) hierarchical_memory: L1/L2/L3 memory with surprise propagation
    - (future) criticality: Real-time λ estimation

Pathology Detection:
    The precision_diagnostics module maps clinical phenomenology to
    control-theoretic parameter imbalances:

    SCHIZOPHRENIC (Π_μ >> Π_y): System ignores reality, hallucinating
    ASD (Π_y >> Π_μ): System overfits to details, obsessive looping
    HEALTHY (Π_y ≈ Π_μ): Balanced inference

Usage:
    from ara.gutc import (
        ActiveInferenceController,
        create_controller,
        PolicyEstimate,
        PrecisionMonitor,
        diagnose_from_config,
    )

    controller = create_controller("balanced")

    # Monitor precision balance
    monitor = PrecisionMonitor()
    monitor.update(controller.config.extrinsic_weight,
                   controller.config.intrinsic_weight)
    diagnosis = monitor.diagnose()

    if diagnosis.status != Pathology.HEALTHY:
        print(f"WARNING: {diagnosis.recommendation}")
"""

from .active_inference import (
    # Types
    PolicyType,
    # Config
    ActiveInferenceConfig,
    WORKER_MODE,
    SCIENTIST_MODE,
    BALANCED_MODE,
    CRISIS_MODE,
    # Data
    PolicyEstimate,
    ScoredPolicy,
    SystemState,
    # Controller
    ActiveInferenceController,
    # Factory
    create_controller,
)

from .precision_diagnostics import (
    # Types
    Pathology,
    Severity,
    # Data
    Diagnosis,
    DiagnosticThresholds,
    # Monitor
    PrecisionMonitor,
    # Convenience
    diagnose_from_config,
)

__all__ = [
    # Active Inference
    "PolicyType",
    "ActiveInferenceConfig",
    "WORKER_MODE",
    "SCIENTIST_MODE",
    "BALANCED_MODE",
    "CRISIS_MODE",
    "PolicyEstimate",
    "ScoredPolicy",
    "SystemState",
    "ActiveInferenceController",
    "create_controller",
    # Precision Diagnostics
    "Pathology",
    "Severity",
    "Diagnosis",
    "DiagnosticThresholds",
    "PrecisionMonitor",
    "diagnose_from_config",
]
