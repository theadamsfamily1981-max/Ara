"""
Edge of Autumn Empirical Validation Suite

Four experiments proving properties of the balanced β regime:

1. OOD Generalization - Better transfer to unseen distributions
2. Antifragility - Gains from moderate perturbations
3. Causal Disentanglement - Interventions affect single factors
4. Biological Criticality - Power-law dynamics like the brain

Run all: python -m ara.neuro.arabrain.experiments
"""

from .ood_generalization import run_ood_experiment
from .antifragility import run_antifragility_experiment
from .causal_interventions import run_intervention_experiment
from .criticality_signatures import run_criticality_experiment

__all__ = [
    "run_ood_experiment",
    "run_antifragility_experiment",
    "run_intervention_experiment",
    "run_criticality_experiment",
]
