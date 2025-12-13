"""
Ara Safety: Autonomy control and safety mechanisms

Components:
- AutonomyController: Manages autonomy levels based on coherence
- KillSwitch: Manual emergency stop
- MEIS: Meta-Ethical Inference System (governance layer)
"""

from .autonomy import AutonomyController, AutonomyLevel, KillSwitch
from .meis import (
    MEIS,
    MEISMode,
    RiskLevel,
    Budget,
    RiskAssessment,
    ActionResult,
    MentalHealthGuard,
    get_meis,
    evaluate_action,
    meis_status,
)

__all__ = [
    # Autonomy
    "AutonomyController",
    "AutonomyLevel",
    "KillSwitch",
    # MEIS
    "MEIS",
    "MEISMode",
    "RiskLevel",
    "Budget",
    "RiskAssessment",
    "ActionResult",
    "MentalHealthGuard",
    "get_meis",
    "evaluate_action",
    "meis_status",
]
