"""
EXP-001: Avalanche Criticality Experiment

Studies neural avalanche statistics across criticality regimes:
- COLD (ρ ≈ 0.6): Subcritical
- TARGET (ρ ≈ 0.85): Near-critical  
- HOT (ρ ≈ 1.15): Supercritical

Key metrics:
- α: Size distribution exponent (P(S) ~ S^(-α))
- β: Duration distribution exponent
- Scaling relation: (α-1)/(β-1) ≈ 2 for mean-field

Usage:
    python -m experiments.avalanche_criticality.exp001_runner --condition all
"""

from .exp001_runner import (
    Condition,
    ConditionConfig,
    ExperimentConfig,
    SimulatedCognitionCore,
    AvalancheLogger,
    Avalanche,
    PowerLawFit,
    fit_powerlaw,
    ExperimentRunner,
)

__all__ = [
    "Condition",
    "ConditionConfig", 
    "ExperimentConfig",
    "SimulatedCognitionCore",
    "AvalancheLogger",
    "Avalanche",
    "PowerLawFit",
    "fit_powerlaw",
    "ExperimentRunner",
]
