"""
CTF-1: Agency Module

Minimal RL agent using CriticalCore as the "brain".

Hypothesis: Agents at criticality (λ ≈ 1) should show optimal
exploration-exploitation tradeoff.

- λ < 1 (ordered): Too deterministic, poor exploration
- λ = 1 (critical): Balanced, good learning
- λ > 1 (chaotic): Too random, poor exploitation
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from .critical_core import CriticalCore, softmax


class CriticalAgent:
    """
    Simple RL agent with critical dynamics core.

    State: x_t ∈ ℝⁿ (reservoir state)
    Policy: π(a|x) = softmax(W_policy @ x)
    Learning: REINFORCE gradient
    """

    def __init__(
        self,
        core: CriticalCore,
        n_actions: int,
        learning_rate: float = 0.01,
    ):
        """
        Args:
            core: CriticalCore instance
            n_actions: Number of discrete actions
            learning_rate: Policy gradient step size
        """
        self.core = core
        self.n_actions = n_actions
        self.lr = learning_rate

        # Policy weights
        self.W_policy = np.random.randn(n_actions, core.n) * 0.01

        # Value baseline (running average)
        self.baseline = 0.0
        self.baseline_decay = 0.99

        # History
        self.reward_history: List[float] = []
        self.entropy_history: List[float] = []

    def get_action_probs(self) -> np.ndarray:
        """Compute action probabilities from current state."""
        logits = self.W_policy @ self.core.x
        return softmax(logits)

    def select_action(self) -> Tuple[int, np.ndarray]:
        """
        Sample action from policy.

        Returns:
            (action_index, probabilities)
        """
        probs = self.get_action_probs()
        action = np.random.choice(self.n_actions, p=probs)
        return action, probs

    def policy_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of policy distribution."""
        return -np.sum(probs * np.log(probs + 1e-10))

    def update(self, action: int, reward: float, probs: np.ndarray):
        """
        REINFORCE update with baseline.

        Args:
            action: Action taken
            reward: Reward received
            probs: Action probabilities when action was taken
        """
        # Update baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward

        # Advantage
        advantage = reward - self.baseline

        # Policy gradient
        # ∇ log π(a|x) = e_a - π (one-hot minus probs)
        one_hot = np.zeros(self.n_actions)
        one_hot[action] = 1.0
        grad_log_pi = one_hot - probs

        # Update: W += lr * advantage * grad_log_pi ⊗ x
        self.W_policy += self.lr * advantage * np.outer(grad_log_pi, self.core.x)

        # Record history
        self.reward_history.append(reward)
        self.entropy_history.append(self.policy_entropy(probs))

    def step(self, observation: Optional[np.ndarray] = None):
        """
        Update core state with observation.

        Args:
            observation: External input (optional)
        """
        self.core.step(observation)


# =============================================================================
# Environments
# =============================================================================

class MultiArmedBandit:
    """
    Simple k-armed bandit environment.

    Each arm has fixed (unknown) reward probability.
    """

    def __init__(self, n_arms: int = 4, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        self.n_arms = n_arms
        self.probs = np.random.rand(n_arms)  # True reward probabilities

    def pull(self, arm: int) -> float:
        """Pull an arm, get stochastic reward."""
        if np.random.rand() < self.probs[arm]:
            return 1.0
        return 0.0

    def optimal_arm(self) -> int:
        """Return index of best arm."""
        return int(np.argmax(self.probs))

    def optimal_prob(self) -> float:
        """Return probability of best arm."""
        return float(np.max(self.probs))


class SimpleGridworld:
    """
    Minimal 2D gridworld.

    Agent starts at (0,0), goal at (size-1, size-1).
    Actions: 0=up, 1=right, 2=down, 3=left
    """

    def __init__(self, size: int = 5):
        self.size = size
        self.pos = [0, 0]
        self.goal = [size - 1, size - 1]

    def reset(self) -> np.ndarray:
        """Reset to start."""
        self.pos = [0, 0]
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Get observation (normalized position)."""
        return np.array(self.pos, dtype=float) / (self.size - 1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take action.

        Args:
            action: 0=up, 1=right, 2=down, 3=left

        Returns:
            (observation, reward, done)
        """
        # Move
        if action == 0:  # up
            self.pos[1] = min(self.pos[1] + 1, self.size - 1)
        elif action == 1:  # right
            self.pos[0] = min(self.pos[0] + 1, self.size - 1)
        elif action == 2:  # down
            self.pos[1] = max(self.pos[1] - 1, 0)
        elif action == 3:  # left
            self.pos[0] = max(self.pos[0] - 1, 0)

        # Check goal
        done = (self.pos == self.goal)
        reward = 1.0 if done else -0.01

        return self._get_obs(), reward, done


# =============================================================================
# Experiment runner
# =============================================================================

def run_bandit_experiment(
    lambda_value: float,
    n_dims: int = 64,
    n_arms: int = 4,
    n_episodes: int = 1000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Run bandit experiment at given λ.

    Returns:
        Dictionary with reward_history, entropy_history, etc.
    """
    np.random.seed(seed)

    # Create agent
    core = CriticalCore(n_dims=n_dims, lambda_init=lambda_value, seed=seed)
    agent = CriticalAgent(core, n_actions=n_arms, learning_rate=0.05)

    # Create environment
    env = MultiArmedBandit(n_arms=n_arms, seed=seed + 1)

    # Run episodes
    for ep in range(n_episodes):
        # Update core state with noise input
        agent.step(np.random.randn(1) * 0.1)

        # Select action
        action, probs = agent.select_action()

        # Get reward
        reward = env.pull(action)

        # Update policy
        agent.update(action, reward, probs)

    # Compute metrics
    window = 100
    smoothed_rewards = np.convolve(
        agent.reward_history,
        np.ones(window) / window,
        mode='valid'
    )

    return {
        'lambda': lambda_value,
        'rewards': np.array(agent.reward_history),
        'smoothed_rewards': smoothed_rewards,
        'entropy': np.array(agent.entropy_history),
        'final_reward': np.mean(agent.reward_history[-100:]),
        'final_entropy': np.mean(agent.entropy_history[-100:]),
        'optimal_prob': env.optimal_prob(),
    }


def compare_lambda_agency(
    lambdas: List[float] = [0.7, 1.0, 1.3],
    n_episodes: int = 1000,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Compare agent performance at different λ values.

    Returns:
        Dictionary mapping metric names to lists (one per λ)
    """
    results = {
        'lambdas': lambdas,
        'final_rewards': [],
        'final_entropy': [],
        'regret': [],
    }

    for lam in lambdas:
        exp = run_bandit_experiment(lam, n_episodes=n_episodes, seed=seed)
        results['final_rewards'].append(exp['final_reward'])
        results['final_entropy'].append(exp['final_entropy'])
        results['regret'].append(exp['optimal_prob'] - exp['final_reward'])

    return results


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("CTF-1: Agency Test")
    print("=" * 50)

    print("\nBandit performance vs λ:")
    for lam in [0.7, 1.0, 1.3]:
        result = run_bandit_experiment(lam, n_episodes=500, seed=42)
        print(f"  λ={lam:.1f}: reward={result['final_reward']:.3f}, "
              f"entropy={result['final_entropy']:.3f}, "
              f"regret={result['optimal_prob'] - result['final_reward']:.3f}")

    print("\nExpected: λ=1.0 shows best exploration-exploitation balance")
