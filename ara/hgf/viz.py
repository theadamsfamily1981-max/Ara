"""
ara.hgf.viz - Visualization Utilities

Provides plotting functions for HGF trajectories, prediction errors,
precision dynamics, and parameter recovery analysis.

Uses matplotlib with optional seaborn for publication-quality figures.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ara.hgf.agents import HGFTrajectory
from ara.hgf.tasks import TaskData
from ara.hgf.fitting import FitResult


# Color palette matching the HUD theme
COLORS = {
    "cyan": "#00ffaa",
    "purple": "#aa66ff",
    "blue": "#00ddff",
    "orange": "#ffaa00",
    "red": "#ff4466",
    "green": "#00ff88",
    "background": "#0a0a12",
    "grid": "#333344",
}


def _setup_style(dark: bool = True):
    """Set up matplotlib style."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if dark:
        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["grid"],
            "grid.color": COLORS["grid"],
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        })
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def plot_beliefs(
    trajectory: HGFTrajectory,
    task_data: Optional[TaskData] = None,
    levels: List[int] = [2, 3],
    show_observations: bool = True,
    show_predictions: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    dark: bool = True,
) -> plt.Figure:
    """
    Plot belief trajectories across HGF levels.

    Args:
        trajectory: HGF trajectory
        task_data: Optional task data for true probabilities
        levels: Which levels to plot
        show_observations: Show observations as dots
        show_predictions: Show model predictions
        figsize: Figure size
        dark: Use dark theme

    Returns:
        matplotlib Figure
    """
    _setup_style(dark)

    n_plots = len(levels) + (1 if show_observations else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    trials = np.arange(trajectory.n_trials)
    ax_idx = 0

    # Observations and predictions
    if show_observations:
        ax = axes[ax_idx]
        observations = trajectory.get_observations()

        if show_predictions:
            predictions = trajectory.get_predictions()
            ax.plot(trials, predictions, color=COLORS["cyan"], lw=2,
                   label="Prediction", alpha=0.8)

        ax.scatter(trials, observations, c=COLORS["purple"], s=20,
                  alpha=0.6, label="Observations")

        if task_data is not None:
            true_probs = task_data.true_probabilities
            ax.plot(trials, true_probs, color=COLORS["orange"], lw=1.5,
                   linestyle="--", label="True P", alpha=0.7)

        ax.set_ylabel("Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right")
        ax.set_title("Observations & Predictions")
        ax_idx += 1

    # Level 2: Hidden State
    if 2 in levels:
        ax = axes[ax_idx]
        mu_2 = trajectory.get_beliefs(2)
        sigma_2 = trajectory.get_uncertainties(2)

        ax.plot(trials, mu_2, color=COLORS["blue"], lw=2, label="μ₂")
        ax.fill_between(trials, mu_2 - sigma_2, mu_2 + sigma_2,
                       color=COLORS["blue"], alpha=0.2)

        ax.axhline(0, color=COLORS["grid"], linestyle="--", alpha=0.5)
        ax.set_ylabel("μ₂ (logit)")
        ax.legend(loc="upper right")
        ax.set_title("Level 2: Hidden State (logit space)")
        ax_idx += 1

    # Level 3: Volatility
    if 3 in levels:
        ax = axes[ax_idx]
        mu_3 = trajectory.get_beliefs(3)
        sigma_3 = trajectory.get_uncertainties(3)

        ax.plot(trials, mu_3, color=COLORS["orange"], lw=2, label="μ₃")
        ax.fill_between(trials, mu_3 - sigma_3, mu_3 + sigma_3,
                       color=COLORS["orange"], alpha=0.2)

        ax.set_ylabel("μ₃ (log-volatility)")
        ax.legend(loc="upper right")
        ax.set_title("Level 3: Volatility")
        ax_idx += 1

    axes[-1].set_xlabel("Trial")
    plt.tight_layout()
    return fig


def plot_prediction_errors(
    trajectory: HGFTrajectory,
    levels: List[int] = [1, 2],
    figsize: Tuple[int, int] = (12, 6),
    dark: bool = True,
) -> plt.Figure:
    """
    Plot prediction error trajectories.

    Args:
        trajectory: HGF trajectory
        levels: Which PE levels to plot
        figsize: Figure size
        dark: Use dark theme

    Returns:
        matplotlib Figure
    """
    _setup_style(dark)

    fig, axes = plt.subplots(len(levels), 1, figsize=figsize, sharex=True)
    if len(levels) == 1:
        axes = [axes]

    trials = np.arange(trajectory.n_trials)
    colors = [COLORS["cyan"], COLORS["purple"], COLORS["orange"]]
    labels = ["δ₁ (sensory)", "δ₂ (volatility)", "δ₃ (meta)"]

    for ax, level in zip(axes, levels):
        delta = trajectory.get_prediction_errors(level)
        color = colors[level - 1]
        label = labels[level - 1]

        ax.bar(trials, delta, color=color, alpha=0.7, width=0.8)
        ax.axhline(0, color=COLORS["grid"], linestyle="-", alpha=0.5)
        ax.set_ylabel(label)
        ax.set_title(f"Prediction Error Level {level}")

    axes[-1].set_xlabel("Trial")
    plt.tight_layout()
    return fig


def plot_precision_dynamics(
    trajectory: HGFTrajectory,
    figsize: Tuple[int, int] = (12, 8),
    dark: bool = True,
) -> plt.Figure:
    """
    Plot precision dynamics (key for understanding Bayesian inference).

    Args:
        trajectory: HGF trajectory
        figsize: Figure size
        dark: Use dark theme

    Returns:
        matplotlib Figure
    """
    _setup_style(dark)

    precisions = trajectory.get_precisions()
    trials = np.arange(trajectory.n_trials)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Sensory precision
    ax = axes[0]
    ax.plot(trials, precisions["pi_1"], color=COLORS["cyan"], lw=2)
    ax.set_ylabel("π₁")
    ax.set_title("Sensory Precision (π₁)")

    # Prior precision
    ax = axes[1]
    ax.plot(trials, precisions["pi_hat_2"], color=COLORS["purple"], lw=2)
    ax.set_ylabel("π̂₂")
    ax.set_title("Prior Precision at Level 2 (π̂₂)")

    # Precision weight (Kalman-like gain)
    ax = axes[2]
    # Compute precision ratio
    pi_ratio = precisions["pi_1"] / (precisions["pi_1"] + precisions["pi_hat_2"] + 1e-10)
    ax.plot(trials, pi_ratio, color=COLORS["orange"], lw=2)
    ax.axhline(0.5, color=COLORS["grid"], linestyle="--", alpha=0.5)
    ax.set_ylabel("π₁ / (π₁ + π̂₂)")
    ax.set_ylim(0, 1)
    ax.set_title("Precision Weight (higher = more sensory-driven)")

    axes[-1].set_xlabel("Trial")
    plt.tight_layout()
    return fig


def plot_parameter_recovery(
    results: List[FitResult],
    figsize: Tuple[int, int] = (12, 4),
    dark: bool = True,
) -> plt.Figure:
    """
    Plot parameter recovery results.

    Args:
        results: List of FitResult from parameter recovery
        figsize: Figure size
        dark: Use dark theme

    Returns:
        matplotlib Figure
    """
    _setup_style(dark)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    params = ["omega_2", "kappa_1", "theta"]
    colors = [COLORS["cyan"], COLORS["purple"], COLORS["orange"]]
    labels = ["ω₂", "κ₁", "θ"]

    for ax, param, color, label in zip(axes, params, colors, labels):
        true_vals = [getattr(r, f"true_{param}") for r in results
                    if getattr(r, f"true_{param}") is not None]
        fitted_vals = [getattr(r, param) for r in results
                      if getattr(r, f"true_{param}") is not None]

        if not true_vals:
            continue

        ax.scatter(true_vals, fitted_vals, c=color, alpha=0.7, s=50)

        # Identity line
        lims = [min(min(true_vals), min(fitted_vals)),
                max(max(true_vals), max(fitted_vals))]
        ax.plot(lims, lims, color=COLORS["grid"], linestyle="--", lw=2)

        # Correlation
        if len(true_vals) > 1:
            r = np.corrcoef(true_vals, fitted_vals)[0, 1]
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                   fontsize=12, verticalalignment="top")

        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Fitted {label}")
        ax.set_title(f"Parameter: {label}")

    plt.tight_layout()
    return fig


def plot_pathology_comparison(
    trajectories: dict,
    metric: str = "beliefs",
    level: int = 2,
    figsize: Tuple[int, int] = (12, 6),
    dark: bool = True,
) -> plt.Figure:
    """
    Compare trajectories from different pathological presets.

    Args:
        trajectories: Dict of label -> HGFTrajectory
        metric: "beliefs", "prediction_errors", or "precisions"
        level: HGF level to plot
        figsize: Figure size
        dark: Use dark theme

    Returns:
        matplotlib Figure
    """
    _setup_style(dark)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))

    for (label, traj), color in zip(trajectories.items(), colors):
        trials = np.arange(traj.n_trials)

        if metric == "beliefs":
            data = traj.get_beliefs(level)
            ylabel = f"μ_{level}"
        elif metric == "prediction_errors":
            data = traj.get_prediction_errors(level)
            ylabel = f"δ_{level}"
        elif metric == "precisions":
            precisions = traj.get_precisions()
            data = precisions["pi_hat_2"] if level == 2 else precisions["pi_1"]
            ylabel = "Precision"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        ax.plot(trials, data, color=color, lw=2, label=label, alpha=0.8)

    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Pathology Comparison: {metric.replace('_', ' ').title()}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def plot_hgf_dashboard(
    trajectory: HGFTrajectory,
    task_data: Optional[TaskData] = None,
    figsize: Tuple[int, int] = (14, 10),
    dark: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive dashboard view of HGF dynamics.

    Args:
        trajectory: HGF trajectory
        task_data: Optional task data
        figsize: Figure size
        dark: Use dark theme

    Returns:
        matplotlib Figure
    """
    _setup_style(dark)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    trials = np.arange(trajectory.n_trials)
    observations = trajectory.get_observations()
    predictions = trajectory.get_predictions()

    # Top left: Observations and Predictions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(trials, observations, c=COLORS["purple"], s=15, alpha=0.5, label="Obs")
    ax1.plot(trials, predictions, color=COLORS["cyan"], lw=2, label="Pred")
    if task_data is not None:
        ax1.plot(trials, task_data.true_probabilities, color=COLORS["orange"],
                lw=1.5, linestyle="--", label="True", alpha=0.7)
    ax1.set_ylabel("P(outcome)")
    ax1.set_title("Observations & Predictions")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(-0.05, 1.05)

    # Top right: Beliefs (Level 2 and 3)
    ax2 = fig.add_subplot(gs[0, 1])
    mu_2 = trajectory.get_beliefs(2)
    mu_3 = trajectory.get_beliefs(3)
    ax2.plot(trials, mu_2, color=COLORS["blue"], lw=2, label="μ₂")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(trials, mu_3, color=COLORS["orange"], lw=2, label="μ₃")
    ax2.set_ylabel("μ₂ (logit)", color=COLORS["blue"])
    ax2_twin.set_ylabel("μ₃ (log-vol)", color=COLORS["orange"])
    ax2.set_title("Belief Dynamics")

    # Middle left: Prediction Errors
    ax3 = fig.add_subplot(gs[1, 0])
    delta_1 = trajectory.get_prediction_errors(1)
    delta_2 = trajectory.get_prediction_errors(2)
    ax3.bar(trials, delta_1, color=COLORS["cyan"], alpha=0.6, width=0.8, label="δ₁")
    ax3.plot(trials, delta_2 * 2, color=COLORS["purple"], lw=2, label="δ₂ (×2)")
    ax3.axhline(0, color=COLORS["grid"], linestyle="-", alpha=0.5)
    ax3.set_ylabel("Prediction Error")
    ax3.set_title("Prediction Errors")
    ax3.legend(loc="upper right", fontsize=8)

    # Middle right: Precision Dynamics
    ax4 = fig.add_subplot(gs[1, 1])
    precisions = trajectory.get_precisions()
    ax4.plot(trials, precisions["pi_1"], color=COLORS["cyan"], lw=2, label="π₁ (sensory)")
    ax4.plot(trials, precisions["pi_hat_2"], color=COLORS["purple"], lw=2, label="π̂₂ (prior)")
    ax4.set_ylabel("Precision")
    ax4.set_title("Precision Dynamics")
    ax4.legend(loc="upper right", fontsize=8)

    # Bottom: Uncertainties
    ax5 = fig.add_subplot(gs[2, :])
    sigma_2 = trajectory.get_uncertainties(2)
    sigma_3 = trajectory.get_uncertainties(3)
    ax5.fill_between(trials, 0, sigma_2, color=COLORS["blue"], alpha=0.4, label="σ₂")
    ax5.fill_between(trials, 0, sigma_3, color=COLORS["orange"], alpha=0.4, label="σ₃")
    ax5.set_xlabel("Trial")
    ax5.set_ylabel("Uncertainty (variance)")
    ax5.set_title("Uncertainty Dynamics")
    ax5.legend(loc="upper right", fontsize=8)

    # Add parameters as text
    if trajectory.params is not None:
        params_text = (
            f"ω₂={trajectory.params.omega_2:.2f}  "
            f"κ₁={trajectory.params.kappa_1:.2f}  "
            f"θ={trajectory.params.theta:.2f}"
        )
        fig.text(0.5, 0.02, params_text, ha="center", fontsize=10,
                color=COLORS["cyan"])

    plt.suptitle("HGF Dynamics Dashboard", fontsize=14, y=0.98)
    return fig


# =============================================================================
# Export for Jupyter
# =============================================================================

def show_in_notebook():
    """Configure matplotlib for Jupyter notebook display."""
    if HAS_MATPLOTLIB:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "inline")
