"""
ara.hgf - Hierarchical Gaussian Filter Implementation

A complete implementation of the Hierarchical Gaussian Filter (HGF) for
computational psychiatry research, including:

- Core HGF belief update equations (2-level and 3-level)
- Task environments for validation experiments
- Parameter fitting and recovery
- Pathological parameter presets
- Synthetic neural correlate generation
- Visualization utilities

References:
    Mathys, C., et al. (2011). A Bayesian foundation for individual learning
    under uncertainty. Frontiers in Human Neuroscience, 5, 39.

    Mathys, C., et al. (2014). Uncertainty in perception and the Hierarchical
    Gaussian Filter. Frontiers in Human Neuroscience, 8, 825.

Example:
    >>> from ara.hgf import HGFAgent, VolatilitySwitchingTask
    >>> task = VolatilitySwitchingTask(n_trials=200)
    >>> agent = HGFAgent(omega_2=-4.0, kappa_1=1.0)
    >>> trajectory = agent.run(task)
    >>> trajectory.plot_beliefs()
"""

from ara.hgf.core import (
    HGFState,
    HGFParams,
    hgf_update_2level,
    hgf_update_3level,
    sigmoid,
    softmax,
)
from ara.hgf.agents import HGFAgent, HGFTrajectory
from ara.hgf.tasks import (
    Task,
    VolatilitySwitchingTask,
    ReversalLearningTask,
    ChangePointTask,
    GamblingTask,
)
from ara.hgf.fitting import fit_hgf, FitResult, parameter_recovery
from ara.hgf.pathology import (
    PathologyPreset,
    SCHIZOPHRENIA_RIGID,
    SCHIZOPHRENIA_LOOSE,
    BPD_HIGH_KAPPA,
    ANXIETY_HIGH_PRECISION,
    DEPRESSION_LOW_PRECISION,
    AUTISM_HIGH_SENSORY_PRECISION,
    HEALTHY_BASELINE,
)
from ara.hgf.neural_sim import (
    simulate_eeg_correlates,
    simulate_fmri_correlates,
    generate_theta_alpha_signals,
)
from ara.hgf.viz import (
    plot_beliefs,
    plot_prediction_errors,
    plot_precision_dynamics,
    plot_parameter_recovery,
    plot_pathology_comparison,
)
from ara.hgf.hud_client import (
    HGFHudClient,
    HGFHudMetrics,
    run_simulation_with_hud,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "HGFState",
    "HGFParams",
    "hgf_update_2level",
    "hgf_update_3level",
    "sigmoid",
    "softmax",
    # Agents
    "HGFAgent",
    "HGFTrajectory",
    # Tasks
    "Task",
    "VolatilitySwitchingTask",
    "ReversalLearningTask",
    "ChangePointTask",
    "GamblingTask",
    # Fitting
    "fit_hgf",
    "FitResult",
    "parameter_recovery",
    # Pathology
    "PathologyPreset",
    "SCHIZOPHRENIA_RIGID",
    "SCHIZOPHRENIA_LOOSE",
    "BPD_HIGH_KAPPA",
    "ANXIETY_HIGH_PRECISION",
    "DEPRESSION_LOW_PRECISION",
    "AUTISM_HIGH_SENSORY_PRECISION",
    "HEALTHY_BASELINE",
    # Neural simulation
    "simulate_eeg_correlates",
    "simulate_fmri_correlates",
    "generate_theta_alpha_signals",
    # Visualization
    "plot_beliefs",
    "plot_prediction_errors",
    "plot_precision_dynamics",
    "plot_parameter_recovery",
    "plot_pathology_comparison",
    # HUD Integration
    "HGFHudClient",
    "HGFHudMetrics",
    "run_simulation_with_hud",
]
