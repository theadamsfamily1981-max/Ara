"""MIES Policy - Modality selection policies.

Two policy implementations:
1. HeuristicModalityPolicy: Rules-based baseline (works now)
2. ThermodynamicGovernor: EBM + AEPO-style learned policy (future training)

The policy takes a ModalityContext and returns a ModalityDecision.
"""

from .heuristic_baseline import HeuristicModalityPolicy
from .ebm_aepo_policy import (
    ThermodynamicGovernor,
    EnergyFunction,
    AEPOSampler,
)

__all__ = [
    "HeuristicModalityPolicy",
    "ThermodynamicGovernor",
    "EnergyFunction",
    "AEPOSampler",
]
