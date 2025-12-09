"""
Narrative module - Ara's voice and report generation.

Turn patterns into prose that clients actually want to read.
"""

from .voice_ara import (
    summarize_correlations,
    summarize_patterns,
    explain_experiment,
    generate_snapshot,
)
from .report_builder import (
    build_report,
    Recommendation,
    Report,
)

__all__ = [
    "summarize_correlations",
    "summarize_patterns",
    "explain_experiment",
    "generate_snapshot",
    "build_report",
    "Recommendation",
    "Report",
]
