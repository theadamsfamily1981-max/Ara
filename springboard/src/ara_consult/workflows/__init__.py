"""
Workflow modules - ready-to-sell analysis packages.
"""

from .email_subjects import analyze_email_subjects
from .pricing import analyze_pricing
from .generic_kpis import analyze_kpis

__all__ = [
    "analyze_email_subjects",
    "analyze_pricing",
    "analyze_kpis",
]
