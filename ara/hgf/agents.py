"""
ara.hgf.agents - HGF Agent Implementation

The HGF agent runs through experimental tasks, updating beliefs according
to the HGF equations and optionally generating behavioral responses.

The HGFTrajectory class stores the complete history of beliefs, prediction
errors, and precisions for analysis and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy as np

from ara.hgf.core import (
    HGFState,
    HGFParams,
    hgf_update_2level,
    hgf_update_3level,
    sigmoid,
    sample_action,
    response_model_softmax,
)
from ara.hgf.tasks import TaskData, TaskTrial


@dataclass
class TrajectoryStep:
    """A single step in the agent's trajectory."""
    trial: int
    observation: float
    state: HGFState
    prediction: float
    action: Optional[int] = None
    action_probability: Optional[float] = None
    reaction_time: Optional[float] = None


@dataclass
class HGFTrajectory:
    """
    Complete trajectory of an HGF agent through a task.

    Stores the full history of beliefs, prediction errors, and precisions
    for analysis, visualization, and model fitting.
    """
    steps: List[TrajectoryStep] = field(default_factory=list)
    params: Optional[HGFParams] = None
    task_type: str = ""
    n_levels: int = 3

    @property
    def n_trials(self) -> int:
        """Number of trials in trajectory."""
        return len(self.steps)

    def get_beliefs(self, level: int = 2) -> np.ndarray:
        """Get belief trajectory at a specific level."""
        if level == 1:
            return np.array([s.state.mu_1 for s in self.steps])
        elif level == 2:
            return np.array([s.state.mu_2 for s in self.steps])
        elif level == 3:
            return np.array([s.state.mu_3 for s in self.steps])
        else:
            raise ValueError(f"Invalid level: {level}")

    def get_predictions(self) -> np.ndarray:
        """Get prediction trajectory (in observation space [0,1])."""
        return np.array([s.prediction for s in self.steps])

    def get_uncertainties(self, level: int = 2) -> np.ndarray:
        """Get uncertainty (variance) trajectory at a specific level."""
        if level == 1:
            return np.array([s.state.sigma_1 for s in self.steps])
        elif level == 2:
            return np.array([s.state.sigma_2 for s in self.steps])
        elif level == 3:
            return np.array([s.state.sigma_3 for s in self.steps])
        else:
            raise ValueError(f"Invalid level: {level}")

    def get_prediction_errors(self, level: int = 1) -> np.ndarray:
        """Get prediction error trajectory at a specific level."""
        if level == 1:
            return np.array([s.state.delta_1 for s in self.steps])
        elif level == 2:
            return np.array([s.state.delta_2 for s in self.steps])
        elif level == 3:
            return np.array([s.state.delta_3 for s in self.steps])
        else:
            raise ValueError(f"Invalid level: {level}")

    def get_precisions(self) -> dict:
        """Get all precision trajectories."""
        return {
            "pi_1": np.array([s.state.pi_1 for s in self.steps]),
            "pi_hat_2": np.array([s.state.pi_hat_2 for s in self.steps]),
            "w_2": np.array([s.state.w_2 for s in self.steps]),
        }

    def get_observations(self) -> np.ndarray:
        """Get observation sequence."""
        return np.array([s.observation for s in self.steps])

    def get_actions(self) -> np.ndarray:
        """Get action sequence (if available)."""
        actions = [s.action for s in self.steps]
        if all(a is None for a in actions):
            return np.array([])
        return np.array([a if a is not None else np.nan for a in actions])

    def get_action_probabilities(self) -> np.ndarray:
        """Get action probability sequence."""
        probs = [s.action_probability for s in self.steps]
        if all(p is None for p in probs):
            return np.array([])
        return np.array([p if p is not None else np.nan for p in probs])

    def compute_log_likelihood(self) -> float:
        """
        Compute log-likelihood of observed choices given model predictions.

        Returns:
            Total log-likelihood (higher = better fit)
        """
        ll = 0.0
        for step in self.steps:
            if step.action is not None and step.action_probability is not None:
                p = step.action_probability if step.action == 1 else (1 - step.action_probability)
                ll += np.log(max(p, 1e-10))
        return ll

    def compute_accuracy(self) -> float:
        """
        Compute prediction accuracy (how well predictions match observations).

        Returns:
            Mean absolute prediction error
        """
        predictions = self.get_predictions()
        observations = self.get_observations()
        return 1.0 - np.mean(np.abs(predictions - observations))

    def to_dict(self) -> dict:
        """Convert trajectory to dictionary for serialization."""
        return {
            "n_trials": self.n_trials,
            "task_type": self.task_type,
            "n_levels": self.n_levels,
            "beliefs_2": self.get_beliefs(2).tolist(),
            "beliefs_3": self.get_beliefs(3).tolist(),
            "predictions": self.get_predictions().tolist(),
            "delta_1": self.get_prediction_errors(1).tolist(),
            "delta_2": self.get_prediction_errors(2).tolist(),
            "observations": self.get_observations().tolist(),
            "params": {
                "omega_2": self.params.omega_2 if self.params else None,
                "kappa_1": self.params.kappa_1 if self.params else None,
                "theta": self.params.theta if self.params else None,
            },
        }


class HGFAgent:
    """
    An agent that uses the Hierarchical Gaussian Filter for inference.

    The agent can:
    - Run through experimental tasks
    - Generate predictions and actions
    - Store complete belief trajectories for analysis
    """

    def __init__(
        self,
        # Key parameters (can also pass HGFParams directly)
        omega_2: float = -4.0,
        kappa_1: float = 1.0,
        theta: float = 1.0,
        omega_3: float = -6.0,
        n_levels: int = 3,
        params: Optional[HGFParams] = None,
    ):
        """
        Initialize HGF agent.

        Args:
            omega_2: Tonic log-volatility at level 2
            kappa_1: Coupling strength (level 3 → level 2)
            theta: Response model inverse temperature
            omega_3: Tonic log-volatility at level 3 (if 3-level)
            n_levels: Number of levels (2 or 3)
            params: Full parameter object (overrides individual params)
        """
        if params is not None:
            self.params = params
        else:
            self.params = HGFParams(
                omega_2=omega_2,
                kappa_1=kappa_1,
                theta=theta,
                omega_3=omega_3,
            )

        self.n_levels = n_levels
        self.state: Optional[HGFState] = None

    def reset(self) -> None:
        """Reset agent to initial state."""
        self.state = HGFState.from_params(self.params)

    def update(self, observation: float) -> HGFState:
        """
        Process one observation and update beliefs.

        Args:
            observation: New observation (0/1 for binary)

        Returns:
            Updated state
        """
        if self.state is None:
            self.reset()

        if self.n_levels == 2:
            self.state = hgf_update_2level(self.state, observation, self.params)
        else:
            self.state = hgf_update_3level(self.state, observation, self.params)

        return self.state

    def get_prediction(self) -> float:
        """Get current prediction in observation space [0,1]."""
        if self.state is None:
            return 0.5
        return sigmoid(self.state.mu_2)

    def get_action_probability(self) -> float:
        """Get probability of choosing action 1."""
        prediction = self.get_prediction()
        return response_model_softmax(prediction, self.params.theta)

    def sample_action(self) -> int:
        """Sample an action based on current beliefs."""
        prediction = self.get_prediction()
        return sample_action(prediction, self.params.theta)

    def run(
        self,
        task_data: TaskData,
        generate_actions: bool = False,
        callback: Optional[Callable[[int, HGFState], None]] = None,
    ) -> HGFTrajectory:
        """
        Run the agent through a complete task.

        Args:
            task_data: Task data with observations
            generate_actions: Whether to generate actions
            callback: Optional callback(trial, state) called each trial

        Returns:
            Complete trajectory
        """
        self.reset()
        trajectory = HGFTrajectory(
            params=self.params,
            task_type=task_data.task_type,
            n_levels=self.n_levels,
        )

        for trial in task_data.trials:
            # Get prediction before update
            prediction = self.get_prediction()

            # Generate action if requested
            action = None
            action_prob = None
            if generate_actions:
                action_prob = self.get_action_probability()
                action = self.sample_action()

            # Update beliefs with observation
            state = self.update(trial.observation)

            # Store step
            step = TrajectoryStep(
                trial=trial.trial,
                observation=trial.observation,
                state=state.clone(),
                prediction=prediction,
                action=action,
                action_probability=action_prob,
            )
            trajectory.steps.append(step)

            # Callback for real-time visualization
            if callback is not None:
                callback(trial.trial, state)

        return trajectory

    def run_observations(
        self,
        observations: np.ndarray,
        callback: Optional[Callable[[int, HGFState], None]] = None,
    ) -> HGFTrajectory:
        """
        Run the agent on a sequence of observations.

        Args:
            observations: Array of observations
            callback: Optional callback(trial, state) called each trial

        Returns:
            Complete trajectory
        """
        self.reset()
        trajectory = HGFTrajectory(
            params=self.params,
            task_type="observations",
            n_levels=self.n_levels,
        )

        for t, obs in enumerate(observations):
            prediction = self.get_prediction()
            state = self.update(float(obs))

            step = TrajectoryStep(
                trial=t,
                observation=float(obs),
                state=state.clone(),
                prediction=prediction,
            )
            trajectory.steps.append(step)

            if callback is not None:
                callback(t, state)

        return trajectory


# =============================================================================
# Utility Functions
# =============================================================================

def compare_agents(
    task_data: TaskData,
    agents: List[HGFAgent],
    labels: Optional[List[str]] = None,
) -> dict:
    """
    Run multiple agents on the same task and compare.

    Args:
        task_data: Task data
        agents: List of HGF agents with different parameters
        labels: Optional labels for each agent

    Returns:
        Dictionary with comparison results
    """
    if labels is None:
        labels = [f"Agent {i}" for i in range(len(agents))]

    results = {}
    for agent, label in zip(agents, labels):
        trajectory = agent.run(task_data, generate_actions=True)
        results[label] = {
            "trajectory": trajectory,
            "accuracy": trajectory.compute_accuracy(),
            "log_likelihood": trajectory.compute_log_likelihood(),
            "params": {
                "omega_2": agent.params.omega_2,
                "kappa_1": agent.params.kappa_1,
                "theta": agent.params.theta,
            },
        }

    return results


def simulate_population(
    task_data: TaskData,
    n_subjects: int = 50,
    param_distribution: Optional[dict] = None,
    seed: Optional[int] = None,
) -> List[HGFTrajectory]:
    """
    Simulate a population of subjects with varying parameters.

    Args:
        task_data: Task data
        n_subjects: Number of synthetic subjects
        param_distribution: Dict with param names and (mean, std)
        seed: Random seed

    Returns:
        List of trajectories, one per subject
    """
    rng = np.random.RandomState(seed)

    # Default parameter distribution
    if param_distribution is None:
        param_distribution = {
            "omega_2": (-4.0, 0.5),
            "kappa_1": (1.0, 0.2),
            "theta": (1.0, 0.3),
        }

    trajectories = []
    for i in range(n_subjects):
        # Sample parameters
        omega_2 = rng.normal(*param_distribution["omega_2"])
        kappa_1 = max(0.1, rng.normal(*param_distribution["kappa_1"]))
        theta = max(0.1, rng.normal(*param_distribution["theta"]))

        agent = HGFAgent(omega_2=omega_2, kappa_1=kappa_1, theta=theta)
        trajectory = agent.run(task_data, generate_actions=True)
        trajectories.append(trajectory)

    return trajectories
