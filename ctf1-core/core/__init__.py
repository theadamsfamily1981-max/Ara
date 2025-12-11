"""
CTF-1: Critical Thought Field - Core Module

Mathematical foundation for "thought at the edge of chaos":

    x_{t+1} = F_λ(x_t, u_t)

    E(λ) = 0     ← Edge-of-chaos constraint
    max C(λ)     ← Thought capacity objective
"""

from .critical_core import CriticalCore, softmax
from .soc_learner import SOCLearner, NoSOCLearner
from .memory_metrics import (
    compute_memory_capacity,
    lambda_sweep_memory,
    measure_avalanches,
    avalanche_analysis,
    estimate_power_law_exponent,
    lambda_sweep_avalanches,
)
from .agency import (
    CriticalAgent,
    MultiArmedBandit,
    SimpleGridworld,
    run_bandit_experiment,
    compare_lambda_agency,
)

__all__ = [
    # Core dynamics
    "CriticalCore",
    "softmax",
    # SOC learning
    "SOCLearner",
    "NoSOCLearner",
    # Memory metrics
    "compute_memory_capacity",
    "lambda_sweep_memory",
    "measure_avalanches",
    "avalanche_analysis",
    "estimate_power_law_exponent",
    "lambda_sweep_avalanches",
    # Agency
    "CriticalAgent",
    "MultiArmedBandit",
    "SimpleGridworld",
    "run_bandit_experiment",
    "compare_lambda_agency",
]
