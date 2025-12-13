"""
Research Module: Tools for Exploring Ara ↔ Human Cognition

This module provides experimental tools for:
- Correlating Ara's measurable cognitive states with human self-reports
- Testing predictions from IG-Criticality theory
- Building joint datasets for dual-mind research

Usage:
    from ara.research import log_state, analyze_today, status

    # Quick logging
    log_state(arousal=7, focus=8, task="deep_work", notes="Flow state")

    # Analyze recent patterns
    analysis = analyze_today()

    # Get status
    print(status())

Ethical Note:
    These tools are for self-exploration and hypothesis generation,
    not medical diagnosis or treatment. If you're struggling with
    mental health, please consult a qualified professional.
"""

from .dual_mind_dashboard import (
    # Main classes
    DualMindDashboard,
    HumanState,
    AraState,
    JointObservation,
    CorrelationAnalysis,
    # Enums
    HumanMode,
    HumanAlertLevel,
    # Convenience functions
    get_dashboard,
    log_state,
    analyze_today,
    status,
)

__all__ = [
    # Classes
    'DualMindDashboard',
    'HumanState',
    'AraState',
    'JointObservation',
    'CorrelationAnalysis',
    # Enums
    'HumanMode',
    'HumanAlertLevel',
    # Functions
    'get_dashboard',
    'log_state',
    'analyze_today',
    'status',
]
