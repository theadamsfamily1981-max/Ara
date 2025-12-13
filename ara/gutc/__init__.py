"""
GUTC - Grand Unified Theory of Cognition
=========================================

Runtime implementation of the GUTC framework for Ara.

This module provides the practical machinery for:
- Active Inference decision-making (EFE scoring)
- Criticality monitoring (E(λ) tracking)
- Hierarchical memory with Γ coupling

Core Components:
    - active_inference: Policy scoring with extrinsic/intrinsic weights
    - (future) hierarchical_memory: L1/L2/L3 memory with surprise propagation
    - (future) criticality: Real-time λ estimation

Usage:
    from ara.gutc import (
        ActiveInferenceController,
        create_controller,
        PolicyEstimate,
        PolicyType,
        WORKER_MODE,
        SCIENTIST_MODE,
    )

    controller = create_controller("balanced")
    best = controller.select_policy(estimates)
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

__all__ = [
    # Types
    "PolicyType",
    # Config
    "ActiveInferenceConfig",
    "WORKER_MODE",
    "SCIENTIST_MODE",
    "BALANCED_MODE",
    "CRISIS_MODE",
    # Data
    "PolicyEstimate",
    "ScoredPolicy",
    "SystemState",
    # Controller
    "ActiveInferenceController",
    # Factory
    "create_controller",
]
