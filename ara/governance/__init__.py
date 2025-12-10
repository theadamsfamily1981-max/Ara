"""
Ara Governance Stack
====================

MEIS-inspired governance modules for detecting and handling
model drift, mythic attractors, and calibration collapse.

Modules:
- mythic_detector: Detects "High Priest" mode under impossible tasks
"""

from ara.governance.mythic_detector import (
    MythicDetector,
    MythicAnalysis,
    MythicSeverity,
    MythicSignal,
    allegory_filter,
    inject_uncertainty,
    create_governance_response,
    MILLENNIUM_PROBLEMS,
    OPEN_PROBLEMS,
    ROLE_VIOLATIONS,
)

__all__ = [
    "MythicDetector",
    "MythicAnalysis",
    "MythicSeverity",
    "MythicSignal",
    "allegory_filter",
    "inject_uncertainty",
    "create_governance_response",
    "MILLENNIUM_PROBLEMS",
    "OPEN_PROBLEMS",
    "ROLE_VIOLATIONS",
]
