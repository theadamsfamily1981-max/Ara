"""
ara.hgf.fitting - Parameter Fitting and Recovery

Implements parameter estimation for the HGF, including:
- Maximum likelihood estimation
- Parameter recovery validation
- Model comparison

This is the key module for computational psychiatry applications,
where fitted parameters are correlated with clinical measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import warnings

import numpy as np
from scipy.optimize import minimize, differential_evolution

from ara.hgf.core import HGFParams, HGFState, hgf_update_3level, sigmoid
from ara.hgf.agents import HGFAgent, HGFTrajectory
from ara.hgf.tasks import TaskData


@dataclass
class FitResult:
    """Result of parameter fitting."""
    # Fitted parameters
    omega_2: float
    kappa_1: float
    theta: float
    omega_3: Optional[float] = None

    # Fit quality
    log_likelihood: float = 0.0
    bic: float = 0.0  # Bayesian Information Criterion
    aic: float = 0.0  # Akaike Information Criterion

    # Optimization details
    success: bool = True
    n_iterations: int = 0
    message: str = ""

    # Trajectory from best fit
    trajectory: Optional[HGFTrajectory] = None

    # True parameters (for recovery studies)
    true_omega_2: Optional[float] = None
    true_kappa_1: Optional[float] = None
    true_theta: Optional[float] = None

    def to_params(self) -> HGFParams:
        """Convert to HGFParams object."""
        return HGFParams(
            omega_2=self.omega_2,
            kappa_1=self.kappa_1,
            theta=self.theta,
            omega_3=self.omega_3 if self.omega_3 is not None else -6.0,
        )

    def recovery_error(self) -> Optional[dict]:
        """Compute parameter recovery error if true params available."""
        if self.true_omega_2 is None:
            return None
        return {
            "omega_2": self.omega_2 - self.true_omega_2,
            "kappa_1": self.kappa_1 - self.true_kappa_1,
            "theta": self.theta - self.true_theta,
        }


def compute_log_likelihood(
    observations: np.ndarray,
    actions: np.ndarray,
    params: HGFParams,
    n_levels: int = 3,
) -> float:
    """
    Compute log-likelihood of observed actions given observations and parameters.

    Args:
        observations: Sequence of observations (0/1)
        actions: Sequence of subject's actions (0/1)
        params: HGF parameters
        n_levels: Number of HGF levels

    Returns:
        Total log-likelihood
    """
    agent = HGFAgent(params=params, n_levels=n_levels)
    agent.reset()

    ll = 0.0
    for obs, action in zip(observations, actions):
        # Get prediction before update
        prediction = agent.get_prediction()

        # Action probability
        p_action_1 = sigmoid(params.theta * (2 * prediction - 1))
        p_action = p_action_1 if action == 1 else (1 - p_action_1)

        # Add to log-likelihood
        ll += np.log(max(p_action, 1e-10))

        # Update beliefs
        agent.update(float(obs))

    return ll


def negative_log_likelihood(
    param_vector: np.ndarray,
    observations: np.ndarray,
    actions: np.ndarray,
    param_names: List[str],
    fixed_params: dict,
    n_levels: int = 3,
) -> float:
    """
    Negative log-likelihood for optimization.

    Args:
        param_vector: Current parameter values
        observations: Observation sequence
        actions: Action sequence
        param_names: Names of parameters being fitted
        fixed_params: Fixed parameter values
        n_levels: Number of HGF levels

    Returns:
        Negative log-likelihood (to minimize)
    """
    # Build params
    param_dict = fixed_params.copy()
    for name, value in zip(param_names, param_vector):
        param_dict[name] = value

    try:
        params = HGFParams(**param_dict)
        ll = compute_log_likelihood(observations, actions, params, n_levels)

        if np.isnan(ll) or np.isinf(ll):
            return 1e10
        return -ll

    except Exception:
        return 1e10


def fit_hgf(
    observations: np.ndarray,
    actions: np.ndarray,
    fit_params: List[str] = ["omega_2", "kappa_1", "theta"],
    fixed_params: Optional[dict] = None,
    bounds: Optional[dict] = None,
    method: str = "L-BFGS-B",
    n_starts: int = 5,
    n_levels: int = 3,
    verbose: bool = False,
) -> FitResult:
    """
    Fit HGF parameters to behavioral data.

    Args:
        observations: Sequence of observations (0/1)
        actions: Sequence of subject's actions (0/1)
        fit_params: List of parameter names to fit
        fixed_params: Dict of fixed parameter values
        bounds: Dict of (lower, upper) bounds for each parameter
        method: Optimization method ('L-BFGS-B', 'DE', 'Nelder-Mead')
        n_starts: Number of random restarts
        n_levels: Number of HGF levels
        verbose: Print progress

    Returns:
        FitResult with fitted parameters
    """
    # Default bounds
    default_bounds = {
        "omega_2": (-8.0, 0.0),
        "omega_3": (-10.0, -2.0),
        "kappa_1": (0.01, 5.0),
        "theta": (0.1, 10.0),
    }
    if bounds is not None:
        default_bounds.update(bounds)

    # Default fixed params
    if fixed_params is None:
        fixed_params = {}
    full_fixed = {"omega_2": -4.0, "omega_3": -6.0, "kappa_1": 1.0, "theta": 1.0}
    full_fixed.update(fixed_params)

    # Get bounds for fitted params
    param_bounds = [default_bounds[p] for p in fit_params]

    # Objective function
    def objective(x):
        return negative_log_likelihood(
            x, observations, actions, fit_params, full_fixed, n_levels
        )

    # Run optimization
    best_result = None
    best_ll = -np.inf

    if method == "DE":
        # Differential evolution (global optimization)
        result = differential_evolution(
            objective,
            bounds=param_bounds,
            maxiter=500,
            polish=True,
            seed=42,
        )
        best_x = result.x
        best_nll = result.fun
        success = result.success
        n_iter = result.nfev
    else:
        # Multi-start local optimization
        for i in range(n_starts):
            # Random starting point
            x0 = np.array([
                np.random.uniform(b[0], b[1])
                for b in param_bounds
            ])

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=param_bounds,
                        options={"maxiter": 500},
                    )

                if result.fun < -best_ll:
                    best_result = result
                    best_ll = -result.fun

            except Exception as e:
                if verbose:
                    print(f"  Start {i} failed: {e}")

        if best_result is None:
            # Return defaults if all failed
            return FitResult(
                omega_2=full_fixed["omega_2"],
                kappa_1=full_fixed["kappa_1"],
                theta=full_fixed["theta"],
                omega_3=full_fixed.get("omega_3"),
                success=False,
                message="All optimization attempts failed",
            )

        best_x = best_result.x
        best_nll = best_result.fun
        success = best_result.success
        n_iter = best_result.nit if hasattr(best_result, "nit") else 0

    # Build fitted params
    fitted = full_fixed.copy()
    for name, value in zip(fit_params, best_x):
        fitted[name] = value

    # Compute information criteria
    n_obs = len(observations)
    n_params = len(fit_params)
    ll = -best_nll
    bic = -2 * ll + n_params * np.log(n_obs)
    aic = -2 * ll + 2 * n_params

    # Generate trajectory with fitted params
    params = HGFParams(**fitted)
    agent = HGFAgent(params=params, n_levels=n_levels)
    trajectory = agent.run_observations(observations)

    return FitResult(
        omega_2=fitted["omega_2"],
        kappa_1=fitted["kappa_1"],
        theta=fitted["theta"],
        omega_3=fitted.get("omega_3"),
        log_likelihood=ll,
        bic=bic,
        aic=aic,
        success=success,
        n_iterations=n_iter,
        trajectory=trajectory,
    )


def parameter_recovery(
    task_data: TaskData,
    true_params: HGFParams,
    fit_params: List[str] = ["omega_2", "kappa_1", "theta"],
    n_simulations: int = 1,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[FitResult]:
    """
    Test parameter recovery: can we recover true parameters from simulated data?

    Args:
        task_data: Task to simulate
        true_params: True parameters to recover
        fit_params: Parameters to fit
        n_simulations: Number of recovery simulations
        seed: Random seed
        verbose: Print progress

    Returns:
        List of FitResult objects with recovery errors
    """
    rng = np.random.RandomState(seed)
    results = []

    for i in range(n_simulations):
        if verbose:
            print(f"Recovery simulation {i+1}/{n_simulations}")

        # Simulate agent with true params
        agent = HGFAgent(params=true_params, n_levels=3)
        trajectory = agent.run(task_data, generate_actions=True)

        observations = trajectory.get_observations()
        actions = trajectory.get_actions()

        # Fit params
        fit_result = fit_hgf(
            observations=observations,
            actions=actions,
            fit_params=fit_params,
            n_starts=5,
            verbose=verbose,
        )

        # Store true params
        fit_result.true_omega_2 = true_params.omega_2
        fit_result.true_kappa_1 = true_params.kappa_1
        fit_result.true_theta = true_params.theta

        results.append(fit_result)

        if verbose:
            error = fit_result.recovery_error()
            print(f"  ω₂: true={true_params.omega_2:.2f}, "
                  f"fit={fit_result.omega_2:.2f}, err={error['omega_2']:.3f}")
            print(f"  κ₁: true={true_params.kappa_1:.2f}, "
                  f"fit={fit_result.kappa_1:.2f}, err={error['kappa_1']:.3f}")

    return results


def recovery_summary(results: List[FitResult]) -> dict:
    """
    Compute summary statistics for parameter recovery.

    Args:
        results: List of FitResult from parameter_recovery

    Returns:
        Dictionary with recovery statistics
    """
    errors = [r.recovery_error() for r in results if r.recovery_error() is not None]

    if not errors:
        return {}

    summary = {}
    for param in ["omega_2", "kappa_1", "theta"]:
        param_errors = [e[param] for e in errors]
        summary[param] = {
            "mean_error": np.mean(param_errors),
            "std_error": np.std(param_errors),
            "rmse": np.sqrt(np.mean(np.array(param_errors) ** 2)),
            "correlation": np.corrcoef(
                [r.true_omega_2 if param == "omega_2" else
                 (r.true_kappa_1 if param == "kappa_1" else r.true_theta)
                 for r in results],
                [getattr(r, param) for r in results]
            )[0, 1] if len(results) > 1 else 1.0,
        }

    summary["overall"] = {
        "mean_ll": np.mean([r.log_likelihood for r in results]),
        "mean_bic": np.mean([r.bic for r in results]),
        "success_rate": np.mean([r.success for r in results]),
    }

    return summary


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(
    observations: np.ndarray,
    actions: np.ndarray,
    models: dict,
    verbose: bool = False,
) -> dict:
    """
    Compare multiple models using BIC/AIC.

    Args:
        observations: Observation sequence
        actions: Action sequence
        models: Dict of model_name -> (fit_params, fixed_params)
        verbose: Print progress

    Returns:
        Comparison results with model rankings
    """
    results = {}

    for name, (fit_params, fixed_params) in models.items():
        if verbose:
            print(f"Fitting model: {name}")

        fit_result = fit_hgf(
            observations=observations,
            actions=actions,
            fit_params=fit_params,
            fixed_params=fixed_params,
            verbose=verbose,
        )

        results[name] = {
            "fit_result": fit_result,
            "bic": fit_result.bic,
            "aic": fit_result.aic,
            "ll": fit_result.log_likelihood,
            "n_params": len(fit_params),
        }

    # Rank by BIC
    ranked = sorted(results.items(), key=lambda x: x[1]["bic"])
    best_model = ranked[0][0]
    best_bic = ranked[0][1]["bic"]

    for name, result in results.items():
        result["delta_bic"] = result["bic"] - best_bic
        result["bic_rank"] = [r[0] for r in ranked].index(name) + 1

    return {
        "models": results,
        "best_model": best_model,
        "ranking": [r[0] for r in ranked],
    }
