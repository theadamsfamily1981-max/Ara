"""
Pareto v2 - Multi-Objective Optimization with EHVI

Enhanced Pareto optimization using Expected Hypervolume Improvement (EHVI)
for finding optimal trade-offs between accuracy, latency, stability, and energy.

Components:
- EHVI Optimizer: Bayesian optimization with GP models
- Pareto Runner: Orchestrates optimization for TF-A-N configs
- Metrics: Hypervolume, IGD, spread, spacing, coverage

Usage:
    from tfan.pareto_v2 import ParetoRunner

    runner = ParetoRunner()
    front = runner.run()
"""

from .ehvi import EHVIOptimizer, EHVIConfig, ParetoFront
from .runner import ParetoRunner, ParetoRunnerConfig, TFANConfigEvaluator
from .metrics import ParetoMetrics, compare_pareto_fronts

__all__ = [
    "EHVIOptimizer",
    "EHVIConfig",
    "ParetoFront",
    "ParetoRunner",
    "ParetoRunnerConfig",
    "TFANConfigEvaluator",
    "ParetoMetrics",
    "compare_pareto_fronts",
]
