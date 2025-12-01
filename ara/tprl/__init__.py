"""
Topological Meta-Plasticity via Reinforcement Learning (TP-RL)

Uses AEPO agent to treat the network's sparse connectivity mask (M) and
core SNN parameters (τ, v_th) as the ACTION SPACE for Reinforcement Learning.

Key Features:
1. Dynamic Mask Evolution - RL agent adjusts connectivity every N steps
2. Hyperbolic Reward Signal - Pareto-optimal trade-off between Accuracy, Energy, Topology
3. Anti-fragile Policy - Uses L3 Metacontrol "Jerk" signal in reward

Control Law:
    R = α * Accuracy + β * (1 - Energy) + γ * TopologyStability - δ * Jerk

Usage:
    from ara.tprl import TPRLAgent, MaskEnvironment

    env = MaskEnvironment(snn_model, validator)
    agent = TPRLAgent(env)
    agent.train(num_episodes=1000)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
import json
import math

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("ara.tprl")

# Try numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class TPRLState:
    """State representation for TP-RL environment."""
    # Mask state
    mask_density: float = 0.03  # Target 97% sparsity
    active_connections: int = 0
    total_possible: int = 0

    # SNN parameters
    tau_mean: float = 10.0  # Mean time constant
    tau_std: float = 2.0
    v_th_mean: float = 1.0  # Mean threshold
    v_th_std: float = 0.1

    # Performance metrics
    accuracy: float = 0.0
    energy_proxy: float = 0.0  # Spike count / total
    topology_stability: float = 1.0  # L2 distance to previous

    # L3 Metacontrol signals
    jerk: float = 0.0  # Rate of change of arousal
    confidence: float = 1.0

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_vector(self) -> "np.ndarray":
        """Convert to numpy vector for policy network."""
        if not NUMPY_AVAILABLE:
            return [
                self.mask_density, self.tau_mean, self.tau_std,
                self.v_th_mean, self.v_th_std, self.accuracy,
                self.energy_proxy, self.topology_stability,
                self.jerk, self.confidence
            ]
        return np.array([
            self.mask_density, self.tau_mean, self.tau_std,
            self.v_th_mean, self.v_th_std, self.accuracy,
            self.energy_proxy, self.topology_stability,
            self.jerk, self.confidence
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_density": self.mask_density,
            "active_connections": self.active_connections,
            "tau_mean": self.tau_mean,
            "v_th_mean": self.v_th_mean,
            "accuracy": self.accuracy,
            "energy_proxy": self.energy_proxy,
            "topology_stability": self.topology_stability,
            "jerk": self.jerk,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class TPRLAction:
    """Action space for TP-RL agent."""
    # Mask modifications
    prune_ratio: float = 0.0  # Fraction of connections to prune [0, 0.1]
    grow_ratio: float = 0.0   # Fraction of connections to grow [0, 0.1]

    # Parameter modifications
    tau_delta: float = 0.0    # Change to time constants [-1, 1]
    v_th_delta: float = 0.0   # Change to thresholds [-0.1, 0.1]

    # Mask selection strategy
    strategy: str = "topological"  # topological, random, gradient

    def to_vector(self) -> "np.ndarray":
        """Convert to numpy vector."""
        strategy_idx = {"topological": 0, "random": 1, "gradient": 2}.get(self.strategy, 0)
        if not NUMPY_AVAILABLE:
            return [self.prune_ratio, self.grow_ratio, self.tau_delta, self.v_th_delta, strategy_idx]
        return np.array([
            self.prune_ratio, self.grow_ratio,
            self.tau_delta, self.v_th_delta,
            strategy_idx
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, vec) -> "TPRLAction":
        """Create from numpy vector."""
        strategies = ["topological", "random", "gradient"]
        return cls(
            prune_ratio=float(max(0, min(0.1, vec[0]))),
            grow_ratio=float(max(0, min(0.1, vec[1]))),
            tau_delta=float(max(-1, min(1, vec[2]))),
            v_th_delta=float(max(-0.1, min(0.1, vec[3]))),
            strategy=strategies[int(vec[4]) % 3],
        )


@dataclass
class TPRLReward:
    """Reward computation for TP-RL."""
    # Weights
    alpha_accuracy: float = 1.0
    beta_energy: float = 0.3
    gamma_topology: float = 0.2
    delta_jerk: float = 0.5  # Penalize rapid changes

    # Bounds for normalization
    accuracy_target: float = 0.95
    energy_target: float = 0.1  # Low spike rate
    stability_target: float = 0.9

    def compute(self, state: TPRLState, prev_state: Optional[TPRLState] = None) -> float:
        """
        Compute reward signal.

        R = α * Accuracy + β * (1 - Energy) + γ * Stability - δ * Jerk
        """
        # Accuracy reward (normalized)
        acc_reward = min(1.0, state.accuracy / self.accuracy_target)

        # Energy reward (lower is better)
        energy_reward = max(0, 1.0 - state.energy_proxy / self.energy_target)

        # Topology stability reward
        stability_reward = state.topology_stability

        # Jerk penalty (from L3 metacontrol)
        jerk_penalty = abs(state.jerk)

        # Compute total reward
        reward = (
            self.alpha_accuracy * acc_reward +
            self.beta_energy * energy_reward +
            self.gamma_topology * stability_reward -
            self.delta_jerk * jerk_penalty
        )

        # Bonus for meeting sparsity target (97-99%)
        if 0.01 <= state.mask_density <= 0.03:
            reward += 0.1

        return float(reward)


class MaskEnvironment:
    """
    RL Environment for Topological Meta-Plasticity.

    State: Current mask + SNN parameters + performance metrics
    Action: Prune/grow mask, adjust τ/v_th
    Reward: Pareto-optimal balance of accuracy/energy/stability
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        initial_density: float = 0.03,
        update_interval: int = 100,  # Steps between mask updates
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.initial_density = initial_density
        self.update_interval = update_interval

        # Initialize mask (sparse connectivity)
        self.total_connections = input_dim * hidden_dim + hidden_dim * output_dim
        self.active_connections = int(self.total_connections * initial_density)

        # SNN parameters (learnable)
        self.tau = initial_density * 10 + 5  # Time constant
        self.v_th = 1.0  # Threshold

        # Reward computer
        self.reward_fn = TPRLReward()

        # State tracking (initialize before _make_state)
        self._step_count = 0
        self._prev_state = None
        self._episode_rewards = []
        self._current_state = self._make_state()

        logger.info(f"MaskEnvironment initialized: {self.total_connections} total, {self.active_connections} active")

    def _make_state(self) -> TPRLState:
        """Create current state observation."""
        return TPRLState(
            mask_density=self.active_connections / self.total_connections,
            active_connections=self.active_connections,
            total_possible=self.total_connections,
            tau_mean=self.tau,
            v_th_mean=self.v_th,
            accuracy=0.5 + 0.1 * math.sin(self._step_count / 50),  # Simulated
            energy_proxy=self.active_connections / self.total_connections * 0.5,
            topology_stability=0.9,
            jerk=0.0,
            confidence=0.8,
        )

    def reset(self) -> TPRLState:
        """Reset environment to initial state."""
        self.active_connections = int(self.total_connections * self.initial_density)
        self.tau = 10.0
        self.v_th = 1.0
        self._step_count = 0
        self._prev_state = None
        self._current_state = self._make_state()
        self._episode_rewards = []
        return self._current_state

    def step(self, action: TPRLAction) -> Tuple[TPRLState, float, bool, Dict]:
        """
        Execute action and return (state, reward, done, info).
        """
        self._prev_state = self._current_state
        self._step_count += 1

        # Apply action to mask
        if action.prune_ratio > 0:
            prune_count = int(self.active_connections * action.prune_ratio)
            self.active_connections = max(1, self.active_connections - prune_count)

        if action.grow_ratio > 0:
            grow_count = int((self.total_connections - self.active_connections) * action.grow_ratio)
            self.active_connections = min(self.total_connections, self.active_connections + grow_count)

        # Apply parameter changes
        self.tau = max(1.0, min(50.0, self.tau + action.tau_delta))
        self.v_th = max(0.1, min(2.0, self.v_th + action.v_th_delta))

        # Get new state
        self._current_state = self._make_state()

        # Compute reward
        reward = self.reward_fn.compute(self._current_state, self._prev_state)
        self._episode_rewards.append(reward)

        # Check if done (max steps or constraints violated)
        done = self._step_count >= 1000 or self._current_state.mask_density < 0.001

        info = {
            "step": self._step_count,
            "density": self._current_state.mask_density,
            "tau": self.tau,
            "v_th": self.v_th,
            "episode_reward": sum(self._episode_rewards),
        }

        return self._current_state, reward, done, info

    def get_pareto_metrics(self) -> Dict[str, float]:
        """Get current Pareto metrics (Accuracy, Energy, Stability)."""
        return {
            "accuracy": self._current_state.accuracy,
            "energy": self._current_state.energy_proxy,
            "stability": self._current_state.topology_stability,
            "density": self._current_state.mask_density,
        }


class TPRLAgent:
    """
    Topological Meta-Plasticity RL Agent.

    Uses policy gradient to learn optimal mask evolution strategy.
    Integrates with L3 Metacontrol for jerk-aware adaptation.
    """

    def __init__(
        self,
        env: MaskEnvironment,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        exploration_rate: float = 0.3,
    ):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate

        # Policy parameters (simple linear for now)
        self.state_dim = 10  # TPRLState vector size
        self.action_dim = 5  # TPRLAction vector size

        # Simple policy weights
        if NUMPY_AVAILABLE:
            self.policy_weights = np.random.randn(self.state_dim, self.action_dim) * 0.1
        else:
            self.policy_weights = [[0.01] * self.action_dim for _ in range(self.state_dim)]

        # Episode buffer
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # Training history
        self.training_history = []

        logger.info("TPRLAgent initialized")

    def select_action(self, state: TPRLState, explore: bool = True) -> TPRLAction:
        """Select action using current policy."""
        state_vec = state.to_vector()

        if NUMPY_AVAILABLE:
            # Compute action logits
            action_vec = np.dot(state_vec, self.policy_weights)

            # Add exploration noise
            if explore and np.random.random() < self.exploration_rate:
                action_vec += np.random.randn(self.action_dim) * 0.1

            # Clip to valid range
            action_vec = np.clip(action_vec, -1, 1)
        else:
            # Simple fallback
            action_vec = [0.01, 0.01, 0.0, 0.0, 0]

        return TPRLAction.from_vector(action_vec)

    def update_policy(self):
        """Update policy using collected episode data."""
        if not self.episode_rewards or not NUMPY_AVAILABLE:
            return

        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Simple policy gradient update
        for i, (state, action, G) in enumerate(zip(
            self.episode_states, self.episode_actions, returns
        )):
            state_vec = state.to_vector()
            action_vec = action.to_vector()

            # Gradient: ∇log(π) * G
            gradient = np.outer(state_vec, action_vec) * G
            self.policy_weights += self.lr * gradient

        # Clear episode buffer
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def train_episode(self) -> Dict[str, float]:
        """Train for one episode."""
        state = self.env.reset()
        total_reward = 0
        steps = 0

        while True:
            # Select and execute action
            action = self.select_action(state, explore=True)
            next_state, reward, done, info = self.env.step(action)

            # Store transition
            self.episode_states.append(state)
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        # Update policy
        self.update_policy()

        # Get final metrics
        metrics = self.env.get_pareto_metrics()
        metrics["total_reward"] = total_reward
        metrics["steps"] = steps

        self.training_history.append(metrics)

        return metrics

    def train(self, num_episodes: int = 100) -> List[Dict]:
        """Train for multiple episodes."""
        logger.info(f"Starting TP-RL training for {num_episodes} episodes")

        for ep in range(num_episodes):
            metrics = self.train_episode()

            if (ep + 1) % 10 == 0:
                logger.info(
                    f"Episode {ep+1}: reward={metrics['total_reward']:.3f}, "
                    f"density={metrics['density']:.4f}, acc={metrics['accuracy']:.3f}"
                )

        return self.training_history

    def get_optimal_mask_config(self) -> Dict[str, Any]:
        """Get current optimal mask configuration."""
        return {
            "density": self.env._current_state.mask_density,
            "active_connections": self.env.active_connections,
            "tau": self.env.tau,
            "v_th": self.env.v_th,
            "pareto_metrics": self.env.get_pareto_metrics(),
        }


# Singleton for global access
_tprl_agent: Optional[TPRLAgent] = None
_tprl_env: Optional[MaskEnvironment] = None


def get_tprl_env() -> MaskEnvironment:
    """Get or create global TP-RL environment."""
    global _tprl_env
    if _tprl_env is None:
        _tprl_env = MaskEnvironment()
    return _tprl_env


def get_tprl_agent() -> TPRLAgent:
    """Get or create global TP-RL agent."""
    global _tprl_agent
    if _tprl_agent is None:
        _tprl_agent = TPRLAgent(get_tprl_env())
    return _tprl_agent


def evolve_mask(jerk_signal: float = 0.0, confidence: float = 1.0) -> Dict[str, Any]:
    """
    Evolve mask based on current state and L3 metacontrol signals.

    Called periodically from AraOrchestrator.

    Args:
        jerk_signal: Rate of change from L3 metacontrol
        confidence: Confidence in current state

    Returns:
        Updated mask configuration
    """
    agent = get_tprl_agent()
    env = get_tprl_env()

    # Update state with metacontrol signals
    env._current_state.jerk = jerk_signal
    env._current_state.confidence = confidence

    # Select and apply action
    action = agent.select_action(env._current_state, explore=True)
    state, reward, done, info = env.step(action)

    return {
        "action": {
            "prune_ratio": action.prune_ratio,
            "grow_ratio": action.grow_ratio,
            "tau_delta": action.tau_delta,
            "strategy": action.strategy,
        },
        "reward": reward,
        "config": agent.get_optimal_mask_config(),
        "info": info,
    }


__all__ = [
    "TPRLState",
    "TPRLAction",
    "TPRLReward",
    "MaskEnvironment",
    "TPRLAgent",
    "get_tprl_env",
    "get_tprl_agent",
    "evolve_mask",
]
