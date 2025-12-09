"""
Analytics module - pattern finding for small data.

Simple, explainable correlations. No black boxes.
"""

from .correlations import (
    simple_correlations,
    categorical_correlations,
    find_strongest_patterns,
)
from .segments import (
    segment_by_performance,
    find_clusters,
)
from .experiments import (
    suggest_experiments,
    generate_ab_test,
)

__all__ = [
    "simple_correlations",
    "categorical_correlations",
    "find_strongest_patterns",
    "segment_by_performance",
    "find_clusters",
    "suggest_experiments",
    "generate_ab_test",
]
