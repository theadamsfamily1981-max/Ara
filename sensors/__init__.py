"""
Ara Sensors Module
==================

Sensor interfaces for the Teleoplastic Cybernetic Organism.

Submodules:
    founder_state: Burnout/fatigue estimation via HDC
    thermal: Temperature sensors
    network: Network statistics
"""

from .founder_state import (
    FounderSensors,
    FounderStateEstimator,
    estimate_founder_state,
    get_founder_estimator,
)

__all__ = [
    'FounderSensors',
    'FounderStateEstimator',
    'estimate_founder_state',
    'get_founder_estimator',
]
