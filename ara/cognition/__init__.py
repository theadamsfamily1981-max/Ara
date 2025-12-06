"""
Ara Cognition - Higher-Level Cognitive Integration
===================================================

This package contains cognitive integration layers that bridge
Ara's various subsystems into coherent behavior.

Modules:
    teleology: HorizonEngine - Teleological integration layer
               Bridges Telos (purpose) with Institute (research) and
               CuriosityBridge (exploration). Provides alignment scoring
               and drift computation for the Morning Star ritual.

The cognition package sits above the core subsystems:
    - tfan.cognition.telos: Raw purpose/goal management
    - ara.institute: Research and experimentation
    - ara.curiosity: Novelty and exploration

And provides integration/translation services:
    - HorizonEngine: Purpose → Research agenda gating
    - (Future) AttentionManager: Priority allocation
    - (Future) BeliefIntegrator: Cross-system consistency
"""

from .teleology import (
    Horizon,
    HorizonEngine,
)

__all__ = [
    # Teleology
    'Horizon',
    'HorizonEngine',
]
