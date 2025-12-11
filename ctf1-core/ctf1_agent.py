#!/usr/bin/env python3
"""
CTF-1: Critical Thought Field – Minimal Cognitive Agent

Implements a minimal live demonstration of the "thought at criticality" framework:

    x_{t+1} = F_{λ,θ}(x_t, o_t)

with:

- CriticalDynamics: internal critical system (reservoir-like RNN)
- SOCController: self-organized criticality (keeps spectral radius ρ(W) ≈ 1)
- WorldModel: simple linear predictor of next observation
- PolicyNetwork: π_φ(a | x) with REINFORCE-style updates
- CognitiveAgent: binds everything into perceive/act/learn loop
- BanditEnv: tiny K-armed bandit environment

The goal is conceptual:
- Show how a critical agent (λ → 1) learns a bandit faster / more stably
  than subcritical (λ < 1) and supercritical (λ > 1) variants.

This is NOT a production RL library – it's a reference blueprint of the math.
"""

from typing import Optional, List
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt


# ================================================================
# Utilities
# ================================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def one_hot(idx: int, n: int) -> np.ndarray:
    """One-hot vector of length n with 1 at idx."""
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


# ================================================================
# Critical core: F_{λ,θ}, E(λ), C(λ), T(λ)
# ================================================================

class CriticalDynamics:
    """
    Internal critical system:

        x_{t+1} = F_{λ,θ}(x_t, o_t) = φ(W x_t + W_in o_t)

    where:
        - λ is encoded as spectral radius ρ(W)
        - E(λ) ≈ ρ(W) - 1  (edge function)
        - C(λ) ≈ simple correlation-based capacity proxy

    This is a minimal "reservoir" that we tune toward the edge of chaos.
    """

    def __init__(
        self,
        n_dims: int = 64,
        n_obs: int = 8,
        lambda_init: float = 1.0,
        nonlinearity: str = "tanh",
    ):
        self.n = n_dims
        self.n_obs = n_obs
        self.lambda_param = lambda_init
        self.nonlinearity = nonlinearity

        # Internal state
        self.x = np.random.randn(self.n).astype(np.float32) * 0.1

        # Recurrent weights W and input weights W_in
        self.W = np.random.randn(self.n, self.n).astype(np.float32) / np.sqrt(self.n)
        self.W_in = np.random.randn(self.n, self.n_obs).astype(np.float32) / np.sqrt(
            self.n_obs
        )
        self._set_spectral_radius(self.lambda_param)

        # History for capacity estimation
        self.history: List[np.ndarray] = []

    # ----------------- internal helpers -----------------

    def _phi(self, z: np.ndarray) -> np.ndarray:
        if self.nonlinearity == "tanh":
            return np.tanh(z)
        elif self.nonlinearity == "relu":
            return np.maximum(0.0, z)
        else:
            return z

    def _set_spectral_radius(self, target_lambda: float) -> None:
        """Scale W so that spectral radius ρ(W) = target_lambda."""
        vals = eig(self.W, right=False)
        rho = np.max(np.abs(vals)).real
        if rho > 0:
            self.W *= (target_lambda / rho).astype(self.W.dtype)
        self.lambda_param = float(target_lambda)

    # ----------------- public API -----------------

    def spectral_radius(self) -> float:
        vals = eig(self.W, right=False)
        rho = np.max(np.abs(vals)).real
        return float(rho)

    def step(self, obs: np.ndarray) -> np.ndarray:
        """Single update: x_{t+1} = φ(W x_t + W_in o_t)."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.n_obs:
            raise ValueError(f"Expected obs dim {self.n_obs}, got {obs.shape[0]}")

        z = self.W @ self.x + self.W_in @ obs
        self.x = self._phi(z)
        self.history.append(self.x.copy())
        return self.x

    def reset_state(self) -> None:
        self.x = np.random.randn(self.n).astype(np.float32) * 0.1
        self.history = []

    # ----------------- edge function E(λ) -----------------

    def E_spectral(self) -> float:
        """
        Edge function E(λ) ≈ ρ(W) - 1.
        0 → critical, <0 subcritical, >0 supercritical.
        """
        rho = self.spectral_radius()
        return float(rho - 1.0)

    # ----------------- capacity C(λ) -----------------

    def C_capacity(self) -> float:
        """
        Simple proxy for capacity:
        Mean absolute lag-1 autocorrelation across state dimensions.
        """
        if len(self.history) < 3:
            return 0.0

        traj = np.stack(self.history, axis=0)  # (T, n)
        T = traj.shape[0]
        if T < 3:
            return 0.0

        corrs = []
        for i in range(self.n):
            series = traj[:, i]
            x0 = series[:-1] - np.mean(series[:-1])
            x1 = series[1:] - np.mean(series[1:])
            num = float(np.dot(x0, x1))
            den = float(np.linalg.norm(x0) * np.linalg.norm(x1))
            if den > 0:
                corrs.append(num / den)

        if not corrs:
            return 0.0

        return float(np.mean(np.abs(corrs)))

    def T_thought(self, sigma: float = 0.05) -> float:
        """
        Thought functional (for diagnostics):

            T(λ) = C(λ) · exp(-|E(λ)| / σ)

        so it's high only when capacity is high AND E ≈ 0.
        """
        C = self.C_capacity()
        E = self.E_spectral()
        crit = float(np.exp(-abs(E) / max(sigma, 1e-6)))
        return float(C * crit)


# ================================================================
# SOC controller: self-organized criticality
# ================================================================

class SOCController:
    """
    Maintains criticality by homeostatically adjusting spectral radius ρ(W).

    This is a simple "criticality thermostat":

        every `update_interval` steps:
            - measure ρ(W)
            - rescale W so ρ(W) = target_lambda
    """

    def __init__(self, target_lambda: float = 1.0, update_interval: int = 25):
        self.target_lambda = float(target_lambda)
        self.update_interval = int(update_interval)
        self._step_counter = 0

    def update(self, dynamics: CriticalDynamics) -> float:
        """
        Optionally rescale dynamics.W toward ρ(W) = target, returns E(λ).
        """
        self._step_counter += 1

        # Only occasionally "touch" the system to keep it near critical.
        if self._step_counter % self.update_interval == 0:
            rho = dynamics.spectral_radius()
            if rho > 0:
                scale = self.target_lambda / rho
                dynamics.W *= scale
                dynamics.lambda_param = self.target_lambda

        return dynamics.E_spectral()


# ================================================================
# World model: predict next observation (very simple)
# ================================================================

class WorldModel:
    """
    Tiny linear model predicting next observation:

        o_{t+1} ≈ W [x_t ; one_hot(a_t)] + b

    This is mainly here to show how a world model *would* plug in.
    We do simple online gradient steps on squared error.
    """

    def __init__(self, n_state: int, n_obs: int, n_actions: int, lr: float = 1e-3):
        self.n_state = n_state
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.lr = lr

        self.W = np.zeros((n_obs, n_state + n_actions), dtype=np.float32)
        self.b = np.zeros(n_obs, dtype=np.float32)

    def _make_input(self, x: np.ndarray, action: int) -> np.ndarray:
        a_one = one_hot(action, self.n_actions)
        return np.concatenate([x, a_one], axis=0)

    def predict(self, x: np.ndarray, action: int) -> np.ndarray:
        h = self._make_input(x, action)
        return self.W @ h + self.b

    def update(self, x: np.ndarray, action: int, obs_next: np.ndarray) -> None:
        h = self._make_input(x, action)
        y_hat = self.W @ h + self.b
        err = obs_next - y_hat
        # Gradient step to minimize 0.5 * ||err||^2
        self.W += self.lr * np.outer(err, h)
        self.b += self.lr * err


# ================================================================
# Policy network: π_φ(a | x)
# ================================================================

class PolicyNetwork:
    """
    Simple linear-softmax policy:

        π(a | x) = softmax(W x + b)

    Updated with a per-step REINFORCE-like rule using a scalar baseline.
    """

    def __init__(self, n_state: int, n_actions: int, lr: float = 5e-2):
        self.n_state = n_state
        self.n_actions = n_actions
        self.lr = lr

        self.W = (
            np.random.randn(n_actions, n_state).astype(np.float32) * 0.1 / np.sqrt(n_state)
        )
        self.b = np.zeros(n_actions, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits = self.W @ x + self.b
        return softmax(logits)

    def act(self, x: np.ndarray) -> tuple:
        pi = self.forward(x)
        action = int(np.random.choice(self.n_actions, p=pi))
        return action, pi

    def update(
        self,
        x: np.ndarray,
        action: int,
        pi: np.ndarray,
        reward: float,
        baseline: float,
    ) -> None:
        """REINFORCE-style update with baseline."""
        advantage = reward - baseline
        one = one_hot(action, self.n_actions)
        delta = one - pi  # (n_actions,)

        # Gradient ascent on J ≈ E[adv * log π(a|x)]
        self.W += self.lr * advantage * np.outer(delta, x)
        self.b += self.lr * advantage * delta


# ================================================================
# Environment: simple K-armed bandit
# ================================================================

class BanditEnv:
    """
    Simple stationary K-armed Bernoulli bandit.

    - Each arm i has reward probability p_i
    - Observation encodes: last chosen arm (one-hot) + last reward scalar

    obs_dim = n_arms + 1
    """

    def __init__(self, arm_probs: List[float]):
        self.arm_probs = np.asarray(arm_probs, dtype=np.float32)
        self.n_arms = len(self.arm_probs)
        self.obs_dim = self.n_arms + 1

        self.last_action: Optional[int] = None
        self.last_reward: float = 0.0

    def _get_obs(self) -> np.ndarray:
        o = np.zeros(self.obs_dim, dtype=np.float32)
        if self.last_action is not None:
            o[self.last_action] = 1.0
        o[-1] = self.last_reward
        return o

    def reset(self) -> np.ndarray:
        self.last_action = None
        self.last_reward = 0.0
        return self._get_obs()

    def step(self, action: int) -> tuple:
        p = float(self.arm_probs[action])
        reward = 1.0 if np.random.rand() < p else 0.0
        self.last_action = action
        self.last_reward = reward
        obs = self._get_obs()
        return obs, reward


# ================================================================
# Cognitive agent: CriticalDynamics + SOC + WorldModel + Policy
# ================================================================

class CognitiveAgent:
    """
    CTF-1 agent:

        - Perception: o_t
        - Thought: x_{t+1} = F_{λ,θ}(x_t, o_t)   (CriticalDynamics)
        - Action: a_t ~ π_φ(a | x_t)             (PolicyNetwork)
        - World model: ô_{t+1}                   (WorldModel, optional)
        - SOCController: keeps λ ≈ target        (homeostatic criticality)

    This is deliberately minimal: enough to show that changing λ changes
    learning quality in a simple bandit.
    """

    def __init__(
        self,
        env: BanditEnv,
        n_state: int = 64,
        lambda_init: float = 1.0,
        soc_target: Optional[float] = None,
        policy_lr: float = 5e-2,
        wm_lr: float = 1e-3,
        soc_update_interval: int = 25,
    ):
        self.env = env

        # Internal critical dynamics
        self.dynamics = CriticalDynamics(
            n_dims=n_state,
            n_obs=env.obs_dim,
            lambda_init=lambda_init,
            nonlinearity="tanh",
        )

        # World model (not needed for bandit performance, but included conceptually)
        self.world_model = WorldModel(
            n_state=n_state,
            n_obs=env.obs_dim,
            n_actions=env.n_arms,
            lr=wm_lr,
        )

        # Policy
        self.policy = PolicyNetwork(
            n_state=n_state,
            n_actions=env.n_arms,
            lr=policy_lr,
        )

        # Self-organized criticality controller (optional)
        self.soc = (
            SOCController(target_lambda=soc_target, update_interval=soc_update_interval)
            if soc_target is not None
            else None
        )

        # Baseline for REINFORCE
        self.baseline: float = 0.0
        self.baseline_momentum: float = 0.9

        # Logging
        self.rewards_history: List[float] = []
        self.E_history: List[float] = []
        self.C_history: List[float] = []
        self.T_history: List[float] = []

    def run_episode(self, n_steps: int = 20) -> float:
        obs = self.env.reset()
        self.dynamics.reset_state()

        ep_rewards = []

        for _ in range(n_steps):
            # Critical internal update (thought)
            x = self.dynamics.step(obs)

            # Policy chooses action
            action, pi = self.policy.act(x)

            # Environment transition
            obs_next, reward = self.env.step(action)

            # Policy update (REINFORCE with baseline)
            self.baseline = (
                self.baseline_momentum * self.baseline
                + (1.0 - self.baseline_momentum) * reward
            )
            self.policy.update(x, action, pi, reward, self.baseline)

            # World model update (not used by policy, just illustrative)
            self.world_model.update(x, action, obs_next)

            # Criticality maintenance
            if self.soc is not None:
                E = self.soc.update(self.dynamics)
            else:
                E = self.dynamics.E_spectral()

            C = self.dynamics.C_capacity()
            T = self.dynamics.T_thought()

            self.E_history.append(E)
            self.C_history.append(C)
            self.T_history.append(T)

            ep_rewards.append(reward)
            obs = obs_next

        mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        self.rewards_history.append(mean_r)
        return mean_r


# ================================================================
# Training / demo code
# ================================================================

def smooth(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average for prettier curves."""
    if window <= 1:
        return x
    if window > len(x):
        window = len(x)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid")


def train_agent(
    agent: CognitiveAgent,
    n_episodes: int = 200,
    steps_per_episode: int = 20,
    label: str = "",
) -> np.ndarray:
    rewards = []
    for ep in range(n_episodes):
        r = agent.run_episode(steps_per_episode)
        rewards.append(r)
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(
                f"[{label}] Episode {ep+1:4d}/{n_episodes}: "
                f"mean reward (last 10) = {np.mean(rewards[-10:]):.3f}"
            )
    return np.array(rewards, dtype=np.float32)


def main():
    np.random.seed(42)

    # ---------------------------------------------------------------
    # Environment: 3-armed bandit
    # Best arm is index 2 (p = 0.8)
    # ---------------------------------------------------------------
    arm_probs = [0.2, 0.5, 0.8]
    env = BanditEnv(arm_probs)

    n_state = 64
    n_episodes = 250
    steps_per_episode = 20

    # ---------------------------------------------------------------
    # Agents with different λ regimes
    # ---------------------------------------------------------------
    agents = []
    configs = []

    # Subcritical (λ ≈ 0.8, no SOC)
    agents.append(
        CognitiveAgent(
            env=env,
            n_state=n_state,
            lambda_init=0.8,
            soc_target=None,
            policy_lr=5e-2,
            wm_lr=1e-3,
        )
    )
    configs.append("Subcritical λ ≈ 0.8")

    # Critical (λ ≈ 1.0 via SOC)
    agents.append(
        CognitiveAgent(
            env=env,
            n_state=n_state,
            lambda_init=0.9,
            soc_target=1.0,  # SOC controller clamps ρ(W) → 1
            policy_lr=5e-2,
            wm_lr=1e-3,
            soc_update_interval=25,
        )
    )
    configs.append("Critical SOC λ → 1.0")

    # Supercritical (λ ≈ 1.2, no SOC)
    agents.append(
        CognitiveAgent(
            env=env,
            n_state=n_state,
            lambda_init=1.2,
            soc_target=None,
            policy_lr=5e-2,
            wm_lr=1e-3,
        )
    )
    configs.append("Supercritical λ ≈ 1.2")

    # ---------------------------------------------------------------
    # Train all agents
    # ---------------------------------------------------------------
    all_rewards = []
    for agent, cfg in zip(agents, configs):
        print("\n" + "=" * 70)
        print(f"Training agent: {cfg}")
        print("=" * 70)
        rewards = train_agent(
            agent,
            n_episodes=n_episodes,
            steps_per_episode=steps_per_episode,
            label=cfg,
        )
        all_rewards.append(rewards)

    # ---------------------------------------------------------------
    # Plot reward curves
    # ---------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for rewards, cfg in zip(all_rewards, configs):
        sm = smooth(rewards, window=10)
        plt.plot(
            np.arange(len(sm)),
            sm,
            label=cfg,
            linewidth=2.0,
        )
    plt.axhline(
        y=max(arm_probs),
        color="k",
        linestyle="--",
        alpha=0.4,
        label="Optimal arm reward prob",
    )
    plt.xlabel("Episode")
    plt.ylabel("Mean reward per episode (smoothed, window=10)")
    plt.title("CTF-1: Critical vs Subcritical vs Supercritical Agents on Bandit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ctf1_results.png", dpi=150)
    print("\nPlot saved to ctf1_results.png")
    plt.show()


if __name__ == "__main__":
    main()
