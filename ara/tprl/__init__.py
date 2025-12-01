"""
Topological Meta-Plasticity via Reinforcement Learning (TP-RL)

Uses AEPO agent to treat the network's sparse connectivity mask (M) and
core SNN parameters (τ, v_th) as the ACTION SPACE for Reinforcement Learning.

Key Features:
1. Dynamic Mask Evolution - RL agent adjusts connectivity every N steps
2. Hyperbolic Reward Signal - Pareto-optimal trade-off between Accuracy, Energy, Topology
3. Anti-fragile Policy - Uses L3 Metacontrol "Jerk" signal in reward
4. Autonomous Training Loop with validation and checkpointing
5. Pareto-optimal multi-objective optimization

Control Law:
    R = α * Accuracy + β * (1 - Energy) + γ * TopologyStability - δ * Jerk

Usage:
    from ara.tprl import TPRLAgent, MaskEnvironment, TrainingConfig, TPRLTrainer

    # Basic usage
    env = MaskEnvironment(snn_model, validator)
    agent = TPRLAgent(env)
    agent.train(num_episodes=1000)

    # Full autonomous training
    config = TrainingConfig(num_episodes=1000, validate_every=50)
    trainer = TPRLTrainer(config)
    results = trainer.train()
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime
import logging
import json
import math
import time
import pickle

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


@dataclass
class TrainingConfig:
    """Configuration for autonomous TP-RL training."""
    # Episode settings
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000

    # Validation settings
    validate_every: int = 50  # Episodes between validation
    validation_episodes: int = 10

    # Early stopping
    early_stopping_patience: int = 100  # Episodes without improvement
    early_stopping_threshold: float = 0.001  # Minimum improvement

    # Checkpointing
    checkpoint_dir: str = "./checkpoints/tprl"
    save_every: int = 100  # Episodes between saves
    keep_best: int = 3  # Number of best checkpoints to keep

    # Learning settings
    learning_rate: float = 0.001
    gamma: float = 0.99
    exploration_rate: float = 0.3
    exploration_decay: float = 0.995
    min_exploration: float = 0.05

    # Pareto optimization weights
    pareto_accuracy_weight: float = 1.0
    pareto_energy_weight: float = 0.3
    pareto_stability_weight: float = 0.2
    pareto_jerk_penalty: float = 0.5

    # Target metrics (for validation)
    target_accuracy: float = 0.95
    target_sparsity: float = 0.97  # 97% sparse = 3% dense
    target_energy: float = 0.1

    # Model dimensions
    input_dim: int = 1024
    hidden_dim: int = 512
    output_dim: int = 256
    initial_density: float = 0.03

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingResult:
    """Result of a training run."""
    episode: int
    total_reward: float
    accuracy: float
    energy: float
    stability: float
    density: float
    steps: int
    tau: float
    v_th: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def pareto_score(self, config: TrainingConfig) -> float:
        """Compute Pareto score for multi-objective optimization."""
        acc_score = min(1.0, self.accuracy / config.target_accuracy)
        energy_score = max(0, 1.0 - self.energy / config.target_energy)

        # Sparsity score (higher is better, target 97%)
        sparsity = 1.0 - self.density
        sparsity_score = min(1.0, sparsity / config.target_sparsity)

        return (
            config.pareto_accuracy_weight * acc_score +
            config.pareto_energy_weight * energy_score +
            config.pareto_stability_weight * self.stability +
            0.1 * sparsity_score
        )


class ParetoFront:
    """Maintains Pareto-optimal solutions."""

    def __init__(self, objectives: List[str] = None):
        self.objectives = objectives or ["accuracy", "energy", "stability"]
        self.solutions: List[TrainingResult] = []

    def dominates(self, a: TrainingResult, b: TrainingResult) -> bool:
        """Check if solution a dominates solution b."""
        # a dominates b if a is >= b on all objectives and > on at least one
        better_or_equal = 0
        strictly_better = 0

        for obj in self.objectives:
            val_a = getattr(a, obj)
            val_b = getattr(b, obj)

            # For energy, lower is better
            if obj == "energy":
                val_a, val_b = -val_a, -val_b

            if val_a >= val_b:
                better_or_equal += 1
            if val_a > val_b:
                strictly_better += 1

        return better_or_equal == len(self.objectives) and strictly_better > 0

    def add(self, result: TrainingResult) -> bool:
        """Add solution to front if non-dominated. Returns True if added."""
        # Check if any existing solution dominates the new one
        for sol in self.solutions:
            if self.dominates(sol, result):
                return False

        # Remove solutions dominated by the new one
        self.solutions = [s for s in self.solutions if not self.dominates(result, s)]
        self.solutions.append(result)
        return True

    def get_best(self, config: TrainingConfig) -> Optional[TrainingResult]:
        """Get best solution by Pareto score."""
        if not self.solutions:
            return None
        return max(self.solutions, key=lambda r: r.pareto_score(config))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectives": self.objectives,
            "solutions": [s.to_dict() for s in self.solutions],
        }


class TPRLTrainer:
    """
    Autonomous trainer for TP-RL.

    Handles:
    - Full episode training loops
    - Validation with early stopping
    - Checkpointing and model persistence
    - Pareto front tracking
    - Learning rate and exploration scheduling
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        snn_validator: Optional[Callable] = None,
    ):
        self.config = config or TrainingConfig()
        self.snn_validator = snn_validator

        # Create environment and agent
        self.env = MaskEnvironment(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            initial_density=self.config.initial_density,
        )

        self.agent = TPRLAgent(
            env=self.env,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            exploration_rate=self.config.exploration_rate,
        )

        # Update reward weights
        self.env.reward_fn.alpha_accuracy = self.config.pareto_accuracy_weight
        self.env.reward_fn.beta_energy = self.config.pareto_energy_weight
        self.env.reward_fn.gamma_topology = self.config.pareto_stability_weight
        self.env.reward_fn.delta_jerk = self.config.pareto_jerk_penalty

        # Training state
        self.current_episode = 0
        self.best_pareto_score = float('-inf')
        self.episodes_without_improvement = 0
        self.training_history: List[TrainingResult] = []
        self.validation_history: List[Dict] = []
        self.pareto_front = ParetoFront()

        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoints: List[Tuple[float, str]] = []  # (score, path)

        logger.info(f"TPRLTrainer initialized with config: {self.config.num_episodes} episodes")

    def _run_episode(self, explore: bool = True) -> TrainingResult:
        """Run a single training episode."""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0

        while steps < self.config.max_steps_per_episode:
            # Select and execute action
            action = self.agent.select_action(state, explore=explore)
            next_state, reward, done, info = self.env.step(action)

            # Store transition for learning
            self.agent.episode_states.append(state)
            self.agent.episode_actions.append(action)
            self.agent.episode_rewards.append(reward)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        # Update policy
        if explore:
            self.agent.update_policy()

        # Create result
        metrics = self.env.get_pareto_metrics()
        result = TrainingResult(
            episode=self.current_episode,
            total_reward=total_reward,
            accuracy=metrics["accuracy"],
            energy=metrics["energy"],
            stability=metrics["stability"],
            density=metrics["density"],
            steps=steps,
            tau=self.env.tau,
            v_th=self.env.v_th,
        )

        return result

    def _validate(self) -> Dict[str, float]:
        """Run validation episodes."""
        val_results = []

        for _ in range(self.config.validation_episodes):
            result = self._run_episode(explore=False)
            val_results.append(result)

        # Compute mean metrics
        mean_reward = sum(r.total_reward for r in val_results) / len(val_results)
        mean_accuracy = sum(r.accuracy for r in val_results) / len(val_results)
        mean_energy = sum(r.energy for r in val_results) / len(val_results)
        mean_stability = sum(r.stability for r in val_results) / len(val_results)
        mean_density = sum(r.density for r in val_results) / len(val_results)
        mean_pareto = sum(r.pareto_score(self.config) for r in val_results) / len(val_results)

        # Run SNN validator if provided
        snn_accuracy = None
        if self.snn_validator:
            try:
                snn_accuracy = self.snn_validator(
                    density=mean_density,
                    tau=self.env.tau,
                    v_th=self.env.v_th,
                )
            except Exception as e:
                logger.warning(f"SNN validation failed: {e}")

        return {
            "mean_reward": mean_reward,
            "mean_accuracy": mean_accuracy,
            "mean_energy": mean_energy,
            "mean_stability": mean_stability,
            "mean_density": mean_density,
            "mean_pareto_score": mean_pareto,
            "snn_accuracy": snn_accuracy,
            "episode": self.current_episode,
        }

    def _save_checkpoint(self, score: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{self.current_episode}.pkl"

        checkpoint = {
            "episode": self.current_episode,
            "config": self.config.to_dict(),
            "policy_weights": self.agent.policy_weights if NUMPY_AVAILABLE else None,
            "env_state": {
                "tau": self.env.tau,
                "v_th": self.env.v_th,
                "active_connections": self.env.active_connections,
            },
            "training_history": [r.to_dict() for r in self.training_history[-100:]],
            "validation_history": self.validation_history[-20:],
            "pareto_front": self.pareto_front.to_dict(),
            "best_pareto_score": self.best_pareto_score,
            "exploration_rate": self.agent.exploration_rate,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)

            # Track best checkpoints
            self.best_checkpoints.append((score, str(checkpoint_path)))
            self.best_checkpoints.sort(key=lambda x: -x[0])  # Sort by score desc

            # Remove old checkpoints beyond keep_best
            while len(self.best_checkpoints) > self.config.keep_best:
                _, old_path = self.best_checkpoints.pop()
                try:
                    Path(old_path).unlink()
                except:
                    pass

            if is_best:
                best_path = self.checkpoint_dir / "best_model.pkl"
                with open(best_path, "wb") as f:
                    pickle.dump(checkpoint, f)

            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, path: str) -> bool:
        """Load model from checkpoint."""
        try:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)

            self.current_episode = checkpoint["episode"]
            self.best_pareto_score = checkpoint.get("best_pareto_score", float('-inf'))
            self.agent.exploration_rate = checkpoint.get("exploration_rate", 0.3)

            if NUMPY_AVAILABLE and checkpoint.get("policy_weights") is not None:
                self.agent.policy_weights = checkpoint["policy_weights"]

            env_state = checkpoint.get("env_state", {})
            self.env.tau = env_state.get("tau", 10.0)
            self.env.v_th = env_state.get("v_th", 1.0)
            self.env.active_connections = env_state.get(
                "active_connections",
                int(self.env.total_connections * self.config.initial_density)
            )

            logger.info(f"Checkpoint loaded from {path}, episode {self.current_episode}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _decay_exploration(self):
        """Decay exploration rate."""
        self.agent.exploration_rate = max(
            self.config.min_exploration,
            self.agent.exploration_rate * self.config.exploration_decay
        )

    def train(
        self,
        resume_from: Optional[str] = None,
        callback: Optional[Callable[[int, TrainingResult], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Run full autonomous training loop.

        Args:
            resume_from: Path to checkpoint to resume from
            callback: Optional callback(episode, result) -> bool. Return False to stop.

        Returns:
            Training results dict
        """
        start_time = time.time()

        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(f"Starting TP-RL training from episode {self.current_episode}")

        try:
            while self.current_episode < self.config.num_episodes:
                self.current_episode += 1

                # Run episode
                result = self._run_episode(explore=True)
                self.training_history.append(result)

                # Update Pareto front
                self.pareto_front.add(result)

                # Decay exploration
                self._decay_exploration()

                # Logging
                if self.current_episode % 10 == 0:
                    logger.info(
                        f"Episode {self.current_episode}/{self.config.num_episodes}: "
                        f"reward={result.total_reward:.3f}, density={result.density:.4f}, "
                        f"acc={result.accuracy:.3f}, explore={self.agent.exploration_rate:.3f}"
                    )

                # Validation
                if self.current_episode % self.config.validate_every == 0:
                    val_metrics = self._validate()
                    self.validation_history.append(val_metrics)

                    pareto_score = val_metrics["mean_pareto_score"]

                    logger.info(
                        f"Validation @ ep {self.current_episode}: "
                        f"pareto={pareto_score:.4f}, acc={val_metrics['mean_accuracy']:.3f}, "
                        f"density={val_metrics['mean_density']:.4f}"
                    )

                    # Check for improvement
                    if pareto_score > self.best_pareto_score + self.config.early_stopping_threshold:
                        self.best_pareto_score = pareto_score
                        self.episodes_without_improvement = 0
                        self._save_checkpoint(pareto_score, is_best=True)
                    else:
                        self.episodes_without_improvement += self.config.validate_every

                    # Early stopping
                    if self.episodes_without_improvement >= self.config.early_stopping_patience:
                        logger.info(
                            f"Early stopping at episode {self.current_episode}: "
                            f"no improvement for {self.episodes_without_improvement} episodes"
                        )
                        break

                # Periodic checkpoint
                if self.current_episode % self.config.save_every == 0:
                    score = result.pareto_score(self.config)
                    self._save_checkpoint(score)

                # Callback
                if callback and not callback(self.current_episode, result):
                    logger.info(f"Training stopped by callback at episode {self.current_episode}")
                    break

        except KeyboardInterrupt:
            logger.info(f"Training interrupted at episode {self.current_episode}")

        # Final save
        final_score = self.training_history[-1].pareto_score(self.config) if self.training_history else 0
        self._save_checkpoint(final_score)

        elapsed = time.time() - start_time

        # Get best result
        best_result = self.pareto_front.get_best(self.config)

        return {
            "total_episodes": self.current_episode,
            "elapsed_seconds": elapsed,
            "best_pareto_score": self.best_pareto_score,
            "best_result": best_result.to_dict() if best_result else None,
            "final_exploration_rate": self.agent.exploration_rate,
            "pareto_front_size": len(self.pareto_front.solutions),
            "validation_history": self.validation_history,
            "checkpoint_dir": str(self.checkpoint_dir),
        }


def train_tprl(
    num_episodes: int = 1000,
    validate_every: int = 50,
    checkpoint_dir: str = "./checkpoints/tprl",
    resume_from: Optional[str] = None,
    snn_validator: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Convenience function for autonomous TP-RL training.

    Args:
        num_episodes: Number of episodes to train
        validate_every: Episodes between validation
        checkpoint_dir: Directory for checkpoints
        resume_from: Optional checkpoint path to resume from
        snn_validator: Optional callback for SNN validation

    Returns:
        Training results

    Example:
        results = train_tprl(num_episodes=500, validate_every=25)
        print(f"Best Pareto score: {results['best_pareto_score']}")
    """
    config = TrainingConfig(
        num_episodes=num_episodes,
        validate_every=validate_every,
        checkpoint_dir=checkpoint_dir,
    )

    trainer = TPRLTrainer(config=config, snn_validator=snn_validator)
    return trainer.train(resume_from=resume_from)


__all__ = [
    "TPRLState",
    "TPRLAction",
    "TPRLReward",
    "MaskEnvironment",
    "TPRLAgent",
    "TrainingConfig",
    "TrainingResult",
    "ParetoFront",
    "TPRLTrainer",
    "get_tprl_env",
    "get_tprl_agent",
    "evolve_mask",
    "train_tprl",
]
