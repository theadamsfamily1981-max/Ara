"""
ara.hgf.tasks - Task Environments for HGF Validation

Implements standard experimental paradigms used to validate hierarchical
Bayesian models in computational psychiatry:

1. Volatility Switching Task - Environment switches between stable and volatile
2. Reversal Learning Task - Reward contingencies flip periodically
3. Change Point Task - Hidden state changes at unpredictable times
4. Gambling Task - Iowa Gambling Task style with deck selection

These tasks generate sequences of observations that can be used to:
- Test HGF belief dynamics
- Validate parameter fitting procedures
- Compare pathological vs. healthy learning
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TaskTrial:
    """A single trial in a task."""
    trial: int
    observation: float  # The outcome (0/1 for binary, real for continuous)
    true_probability: float  # True underlying probability
    volatility_state: str  # "stable" or "volatile"
    context: dict = field(default_factory=dict)  # Additional trial info


@dataclass
class TaskData:
    """Complete dataset from running a task."""
    trials: List[TaskTrial]
    n_trials: int
    task_type: str
    params: dict

    @property
    def observations(self) -> np.ndarray:
        """Get array of observations."""
        return np.array([t.observation for t in self.trials])

    @property
    def true_probabilities(self) -> np.ndarray:
        """Get array of true probabilities."""
        return np.array([t.true_probability for t in self.trials])

    @property
    def volatility_states(self) -> List[str]:
        """Get list of volatility states."""
        return [t.volatility_state for t in self.trials]


class Task(ABC):
    """Abstract base class for experimental tasks."""

    @abstractmethod
    def generate(self, seed: Optional[int] = None) -> TaskData:
        """Generate a complete task dataset."""
        pass

    @abstractmethod
    def get_true_volatility(self, trial: int) -> float:
        """Get true volatility at a given trial."""
        pass


class VolatilitySwitchingTask(Task):
    """
    Volatility Switching Task

    The environment alternates between stable and volatile phases.
    In stable phases, the true probability is constant.
    In volatile phases, it changes trial-by-trial.

    This is a key paradigm for testing volatility learning (HGF level 3).
    """

    def __init__(
        self,
        n_trials: int = 200,
        stable_prob: float = 0.8,
        volatile_prob_range: Tuple[float, float] = (0.3, 0.7),
        phase_length_mean: int = 30,
        phase_length_std: int = 5,
        start_stable: bool = True,
    ):
        """
        Args:
            n_trials: Total number of trials
            stable_prob: Probability in stable phases
            volatile_prob_range: Range of probabilities in volatile phases
            phase_length_mean: Mean length of each phase
            phase_length_std: Std of phase length
            start_stable: Whether to start in stable phase
        """
        self.n_trials = n_trials
        self.stable_prob = stable_prob
        self.volatile_prob_range = volatile_prob_range
        self.phase_length_mean = phase_length_mean
        self.phase_length_std = phase_length_std
        self.start_stable = start_stable

        self._phase_boundaries: List[int] = []
        self._is_stable: List[bool] = []

    def generate(self, seed: Optional[int] = None) -> TaskData:
        """Generate task data."""
        rng = np.random.RandomState(seed)

        trials = []
        is_stable = self.start_stable
        current_trial = 0
        self._phase_boundaries = [0]
        self._is_stable = []

        while current_trial < self.n_trials:
            # Generate phase length
            phase_length = max(
                5,
                int(rng.normal(self.phase_length_mean, self.phase_length_std))
            )
            phase_end = min(current_trial + phase_length, self.n_trials)

            self._is_stable.append(is_stable)

            for t in range(current_trial, phase_end):
                if is_stable:
                    true_prob = self.stable_prob
                    vol_state = "stable"
                else:
                    # Volatile: probability changes each trial
                    true_prob = rng.uniform(*self.volatile_prob_range)
                    vol_state = "volatile"

                # Generate observation
                observation = float(rng.random() < true_prob)

                trials.append(TaskTrial(
                    trial=t,
                    observation=observation,
                    true_probability=true_prob,
                    volatility_state=vol_state,
                    context={"phase": len(self._phase_boundaries) - 1},
                ))

            current_trial = phase_end
            self._phase_boundaries.append(phase_end)
            is_stable = not is_stable

        return TaskData(
            trials=trials,
            n_trials=len(trials),
            task_type="volatility_switching",
            params={
                "stable_prob": self.stable_prob,
                "volatile_prob_range": self.volatile_prob_range,
                "phase_length_mean": self.phase_length_mean,
            },
        )

    def get_true_volatility(self, trial: int) -> float:
        """Get true volatility (high in volatile phases, low in stable)."""
        for i, boundary in enumerate(self._phase_boundaries[:-1]):
            if trial < self._phase_boundaries[i + 1]:
                return 0.1 if self._is_stable[i] else 0.9
        return 0.5


class ReversalLearningTask(Task):
    """
    Reversal Learning Task

    A simple paradigm where reward contingencies periodically reverse.
    Option A starts as rewarding (p=0.8), then reverses to B (p=0.2).

    Tests ability to detect and adapt to environmental changes.
    """

    def __init__(
        self,
        n_trials: int = 200,
        high_prob: float = 0.8,
        low_prob: float = 0.2,
        n_reversals: int = 4,
        min_trials_per_block: int = 20,
    ):
        """
        Args:
            n_trials: Total number of trials
            high_prob: Probability for the "good" option
            low_prob: Probability for the "bad" option
            n_reversals: Number of reversals
            min_trials_per_block: Minimum trials per block
        """
        self.n_trials = n_trials
        self.high_prob = high_prob
        self.low_prob = low_prob
        self.n_reversals = n_reversals
        self.min_trials_per_block = min_trials_per_block

        self._reversal_points: List[int] = []

    def generate(self, seed: Optional[int] = None) -> TaskData:
        """Generate task data."""
        rng = np.random.RandomState(seed)

        # Generate reversal points
        n_blocks = self.n_reversals + 1
        block_size = self.n_trials // n_blocks

        self._reversal_points = []
        current_point = 0
        for i in range(self.n_reversals):
            # Add some jitter to reversal timing
            jitter = rng.randint(-5, 6)
            reversal = (i + 1) * block_size + jitter
            reversal = max(
                self.min_trials_per_block + current_point,
                min(self.n_trials - self.min_trials_per_block, reversal)
            )
            self._reversal_points.append(reversal)
            current_point = reversal

        # Generate trials
        trials = []
        a_is_good = True

        for t in range(self.n_trials):
            # Check for reversal
            if t in self._reversal_points:
                a_is_good = not a_is_good

            # True probability for option A (which we track)
            true_prob = self.high_prob if a_is_good else self.low_prob

            # In this task, observation = reward from option A
            observation = float(rng.random() < true_prob)

            # Volatility is high right after reversals
            trials_since_reversal = min(
                t - max([0] + [r for r in self._reversal_points if r <= t])
            , 100)
            vol_state = "volatile" if trials_since_reversal < 5 else "stable"

            trials.append(TaskTrial(
                trial=t,
                observation=observation,
                true_probability=true_prob,
                volatility_state=vol_state,
                context={
                    "a_is_good": a_is_good,
                    "block": sum(1 for r in self._reversal_points if r <= t),
                },
            ))

        return TaskData(
            trials=trials,
            n_trials=len(trials),
            task_type="reversal_learning",
            params={
                "high_prob": self.high_prob,
                "low_prob": self.low_prob,
                "n_reversals": self.n_reversals,
                "reversal_points": self._reversal_points,
            },
        )

    def get_true_volatility(self, trial: int) -> float:
        """Volatility spikes around reversal points."""
        for rp in self._reversal_points:
            if abs(trial - rp) < 5:
                return 0.9
        return 0.1


class ChangePointTask(Task):
    """
    Change Point Detection Task

    The underlying probability changes at unpredictable times.
    The agent must detect these changes and update beliefs rapidly.

    This is a more challenging version of volatility learning.
    """

    def __init__(
        self,
        n_trials: int = 300,
        hazard_rate: float = 0.05,
        prob_values: List[float] = [0.2, 0.5, 0.8],
    ):
        """
        Args:
            n_trials: Total number of trials
            hazard_rate: Probability of change point at each trial
            prob_values: Possible probability values
        """
        self.n_trials = n_trials
        self.hazard_rate = hazard_rate
        self.prob_values = prob_values

        self._change_points: List[int] = []

    def generate(self, seed: Optional[int] = None) -> TaskData:
        """Generate task data."""
        rng = np.random.RandomState(seed)

        trials = []
        self._change_points = [0]
        current_prob = rng.choice(self.prob_values)

        for t in range(self.n_trials):
            # Check for change point
            if t > 0 and rng.random() < self.hazard_rate:
                # Change to a different probability
                other_probs = [p for p in self.prob_values if p != current_prob]
                current_prob = rng.choice(other_probs)
                self._change_points.append(t)

            # Generate observation
            observation = float(rng.random() < current_prob)

            # Volatility is high right after change points
            trials_since_cp = t - self._change_points[-1]
            vol_state = "volatile" if trials_since_cp < 3 else "stable"

            trials.append(TaskTrial(
                trial=t,
                observation=observation,
                true_probability=current_prob,
                volatility_state=vol_state,
                context={
                    "run_length": trials_since_cp,
                    "n_change_points": len(self._change_points),
                },
            ))

        return TaskData(
            trials=trials,
            n_trials=len(trials),
            task_type="change_point",
            params={
                "hazard_rate": self.hazard_rate,
                "prob_values": self.prob_values,
                "change_points": self._change_points,
            },
        )

    def get_true_volatility(self, trial: int) -> float:
        """Volatility based on proximity to change points."""
        for cp in self._change_points:
            if abs(trial - cp) < 3:
                return 0.95
        return 0.05


class GamblingTask(Task):
    """
    Iowa Gambling Task (IGT) Style

    Four decks with different reward/loss characteristics.
    Decks A and B are "bad" (net negative), C and D are "good" (net positive).

    This tests decision-making under ambiguity.
    """

    def __init__(
        self,
        n_trials: int = 100,
        deck_probs: Optional[dict] = None,
    ):
        """
        Args:
            n_trials: Total number of trials
            deck_probs: Dict with deck reward probabilities and magnitudes
        """
        self.n_trials = n_trials

        # Default deck structure (simplified IGT)
        self.deck_probs = deck_probs or {
            "A": {"win_prob": 0.5, "win_amount": 100, "loss_amount": -250},
            "B": {"win_prob": 0.9, "win_amount": 100, "loss_amount": -1250},
            "C": {"win_prob": 0.5, "win_amount": 50, "loss_amount": -50},
            "D": {"win_prob": 0.9, "win_amount": 50, "loss_amount": -250},
        }

    def generate(self, seed: Optional[int] = None) -> TaskData:
        """Generate task data with random deck selections."""
        rng = np.random.RandomState(seed)

        trials = []
        decks = list(self.deck_probs.keys())

        for t in range(self.n_trials):
            # Random deck selection (in real task, this comes from subject)
            deck = rng.choice(decks)
            deck_params = self.deck_probs[deck]

            # Generate outcome
            if rng.random() < deck_params["win_prob"]:
                outcome = deck_params["win_amount"]
            else:
                outcome = deck_params["loss_amount"]

            # Normalize to [0, 1] for HGF
            # Map [-1250, 100] to [0, 1]
            normalized = (outcome + 1250) / (100 + 1250)
            observation = max(0.0, min(1.0, normalized))

            trials.append(TaskTrial(
                trial=t,
                observation=observation,
                true_probability=deck_params["win_prob"],
                volatility_state="stable",  # IGT is typically stable
                context={
                    "deck": deck,
                    "raw_outcome": outcome,
                    "is_good_deck": deck in ["C", "D"],
                },
            ))

        return TaskData(
            trials=trials,
            n_trials=len(trials),
            task_type="gambling",
            params={"deck_probs": self.deck_probs},
        )

    def get_true_volatility(self, trial: int) -> float:
        """IGT is typically low volatility."""
        return 0.2


# =============================================================================
# Utility Functions
# =============================================================================

def generate_synthetic_dataset(
    task: Task,
    n_subjects: int = 50,
    seed: Optional[int] = None,
) -> List[TaskData]:
    """
    Generate a dataset with multiple synthetic subjects.

    Args:
        task: Task to generate data from
        n_subjects: Number of synthetic subjects
        seed: Random seed

    Returns:
        List of TaskData, one per subject
    """
    rng = np.random.RandomState(seed)
    datasets = []

    for i in range(n_subjects):
        subject_seed = rng.randint(0, 1000000)
        datasets.append(task.generate(seed=subject_seed))

    return datasets
