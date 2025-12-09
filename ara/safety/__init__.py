"""
Ara Safety: Autonomy control and safety mechanisms

Components:
- AutonomyController: Manages autonomy levels based on coherence
- KillSwitch: Manual emergency stop
"""

from .autonomy import AutonomyController, AutonomyLevel, KillSwitch

__all__ = [
    "AutonomyController",
    "AutonomyLevel",
    "KillSwitch",
]
