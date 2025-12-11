#!/usr/bin/env python3
"""CTF-1: Critical Thought Field – Minimal Cognitive Agent

Implements:
- CriticalDynamics: internal critical system x_{t+1} = F_{λ,θ}(x_t, o_t)
- SOCController: self-organized criticality (keep λ ≈ 1)
- WorldModel: simple linear predictor of next observation
- PolicyNetwork: π_φ(a | x) with REINFORCE update
- CognitiveAgent: binds everything into perceive/act/learn loop
- BanditEnv: tiny environment to demonstrate agency at different λ

This is a *conceptual reference*, not a performance-optimized RL stack.
"""
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

# ================================================================
# Utilities
# ================================================================
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-9)

def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n)
    v[idx] = 1.0
    return v

# ================================================================
# Critical core: F_{λ,θ}, E(λ), C(λ), T(λ)
# ================================================================
class CriticalDynamics:
    """
    Internal critical system:
        x_{t+1} = F_{λ,θ}(x_t, o_t)
    - λ is encoded as the spectral radius of W
    - E(λ) ≈ ρ(W) - 1
    - C(λ) ≈ simple correlation-based capacity proxy
    """
    def __init__(self, n_dims: int = 64, lambda_init: float = 1.0,
                 nonlinearity: str = "tanh"):
        self.n = n_dims
        self.lambda_param = lambda_init
        self.nonlinearity = nonlinearity
        self.x = np.random.randn(self.n) * 0.1  # internal state
        self.W = np.random.randn(self.n, self.n) / np.sqrt(self.n)
        self._set_spectral_radius(self.lambda_param)
        self.history = []

        # Observation and action mapping matrices (fixed/untrained)
        self.W_in = np.random.randn(self.n, 8) / np.sqrt(8) # o_t is 8-dim
        self.W_act = np.random.randn(8, self.n) / np.sqrt(self.n) # a_t is 8-dim

    def _set_spectral_radius(self, target_rho: float):
        """Scale W to achieve ρ(W) = target_rho."""
        if target_rho < 0.01: target_rho = 0.01
        eigenvalues = eig(self.W, right=False)
        current_rho = np.max(np.abs(eigenvalues))
        if current_rho > 1e-9:
            self.W *= (target_rho / current_rho)
        self.lambda_param = target_rho

    def F(self, x: np.ndarray, o: np.ndarray) -> np.ndarray:
        """Dynamics: x_{t+1} = tanh(W·x_t + W_in·o_t)"""
        # Ensure o is 8-dimensional for W_in mapping
        o_in = np.zeros(self.W_in.shape[1])
        o_in[:len(o)] = o

        z = self.W @ x + self.W_in @ o_in

        if self.nonlinearity == 'tanh':
            return np.tanh(z)
        elif self.nonlinearity == 'relu':
            return np.maximum(0, z)
        return z

    def step(self, o: np.ndarray):
        """Single update step."""
        self.x = self.F(self.x, o)
        self.history.append(self.x.copy())
        if len(self.history) > 2000:
            self.history.pop(0)

    # ================= Edge Function E(λ) =================
    def E_spectral(self) -> float:
        """E(λ) = ρ(W) - 1. (Distance from the edge)"""
        eigenvalues = eig(self.W, right=False)
        rho = np.max(np.abs(eigenvalues))
        # Note: If W is a stable matrix, E is not directly the Lyapunov exponent,
        # but ρ=1 is the necessary *linear* stability boundary.
        return rho - 1.0

    # ================= Capacity Function C(λ) =================
    def C_correlation_length(self, n_lags: int = 10) -> float:
        """C(λ) ≈ Correlation length (proxy for Excess Entropy)."""
        if len(self.history) < 2 * n_lags:
            return 0.0

        traj = np.array(self.history)
        x_vals = traj[:, 0]  # Use first dimension as proxy

        # Fit exponential decay: ρ(k) ~ exp(-k/ξ)
        lags = np.arange(1, n_lags + 1)
        corrs = []
        for lag in lags:
            x1 = x_vals[:-lag]
            x2 = x_vals[lag:]
            if len(x1) > 10:
                corr = np.corrcoef(x1, x2)[0, 1]
                if not np.isnan(corr) and corr > 1e-3:
                    corrs.append(corr)
                else:
                    corrs.append(1e-4)

        if len(corrs) < 2: return 0.0

        try:
            # Linearize: log(rho) = -k/xi + const
            slope, _ = np.polyfit(lags[:len(corrs)], np.log(corrs), 1)
            xi = -1.0 / slope if slope < 0 else 1e-4
            return np.clip(xi, 0, 100)
        except Exception:
            return 0.0

    # ================= Thought Functional T(λ) =================
    def T_thought(self, E: float, C: float) -> float:
        """Thought capacity: T(λ) = C(λ) · exp(-E²/σ²) (Penalized Capacity)"""
        # Gaussian penalty sigma (controls sharpness of criticality)
        sigma = 0.05
        criticality_factor = np.exp(-E**2 / sigma**2)
        return C * criticality_factor

# ================================================================
# SOC Controller (Learning Rule G_SOC)
# ================================================================
class SOCController:
    """
    Implements the slow timescale learning rule G_SOC:
        dλ/dt = -κ · ∇λ |E(λ)|²
    Drives the system's parameter λ towards the critical surface E(λ) = 0.
    """
    def __init__(self, critical_dynamics: CriticalDynamics,
                 eta_slow: float = 0.001):
        self.cd = critical_dynamics
        self.eta_slow = eta_slow
        self.prev_E = self.cd.E_spectral()
        self.lambda_history = []
        self.E_history = []

    def update_lambda(self):
        """
        Gradient approximation: Uses finite difference (or simplified PID)
        to minimize |E(λ)|.
        """
        E = self.cd.E_spectral()

        # Simple proportional control toward E=0
        # G_SOC ≈ -κ * E
        delta_lambda = -self.eta_slow * E

        new_lambda = self.cd.lambda_param + delta_lambda

        # Clamp lambda to stable range
        new_lambda = np.clip(new_lambda, 0.5, 1.5)

        self.cd._set_spectral_radius(new_lambda)

        self.lambda_history.append(new_lambda)
        self.E_history.append(E)
        self.prev_E = E
        return new_lambda, E

# ================================================================
# World Model and Policy (Agency and Learning Rule G_Task)
# ================================================================
class WorldModel:
    """
    Internal model q(o_{t+1} | x_t, a_t) (Belief state)
    Learns to predict next observation from current state and action.
    """
    def __init__(self, n_dims: int = 64, n_obs: int = 8):
        # W_model: x_t + a_t -> o_{t+1}
        self.W_model = np.random.randn(n_obs, n_dims + n_obs) * 0.01 # n_obs is 8
        self.n_obs = n_obs

    def predict(self, x: np.ndarray, a_oh: np.ndarray) -> np.ndarray:
        """Predicts the next observation o_{t+1} logits."""
        # Concatenate state and action
        xa = np.concatenate([x, a_oh])
        # Simple linear prediction
        o_logits = self.W_model @ xa
        return o_logits # Output is logits for observation

    def learn(self, x: np.ndarray, a_oh: np.ndarray, o_next: np.ndarray, eta_fast: float):
        """Fast timescale learning: Minimize prediction error (Accuracy term of VFE)."""
        o_logits = self.predict(x, a_oh)
        o_prob = softmax(o_logits)

        # Loss: Cross-entropy (Approximates log P(o|x,a))
        loss = -np.sum(o_next * np.log(o_prob + 1e-9))

        # Gradient of loss w.r.t. logits
        d_logits = o_prob - o_next

        # Gradient w.r.t. W_model
        xa = np.concatenate([x, a_oh])
        d_W_model = np.outer(d_logits, xa)

        # Update W_model (G_Task)
        self.W_model -= eta_fast * d_W_model

        # VFE proxy: Negative log-likelihood
        return -loss

class PolicyNetwork:
    """
    Policy π_φ(a | x) (Agency)
    Selects action based on current internal state.
    """
    def __init__(self, n_dims: int = 64, n_actions: int = 8):
        # W_policy: x_t -> a_t logits
        self.W_policy = np.random.randn(n_actions, n_dims) * 0.01
        self.n_actions = n_actions

    def get_action(self, x: np.ndarray) -> tuple:
        """Selects action and returns its log-probability."""
        a_logits = self.W_policy @ x
        a_prob = softmax(a_logits)

        # Action selection (Exploration/Exploitation balance is intrinsic to π)
        a_idx = np.random.choice(self.n_actions, p=a_prob)
        a_oh = one_hot(a_idx, self.n_actions)

        # Log-probability of the selected action
        log_prob = np.log(a_prob[a_idx] + 1e-9)
        return a_idx, a_oh, log_prob

    def learn_policy(self, trajectory: list, eta_fast: float, gamma: float = 0.99):
        """
        REINFORCE (fast timescale learning) using accumulated reward as the
        proxy for maximizing the Goal Functional J(π) (Reward = VFE proxy).
        """
        # Collect rewards and log_probs from trajectory
        states, actions_oh, log_probs, rewards = zip(*trajectory)

        # Compute Discounted Rewards (G_t)
        G = np.zeros_like(rewards, dtype=float)
        R_sum = 0
        for t in reversed(range(len(rewards))):
            R_sum = rewards[t] + gamma * R_sum
            G[t] = R_sum

        # Standardize G_t (optional but standard in RL)
        if len(G) > 1:
            G = (G - np.mean(G)) / (np.std(G) + 1e-9)

        # Gradient update for policy (G_Task)
        for t in range(len(states)):
            x = states[t]
            G_t = G[t]
            a_idx = np.argmax(actions_oh[t])

            # Gradient of policy log-prob w.r.t. W_policy
            # d_log_prob = ∇_W log π(a|x)

            # Simple linear model: a_logits = W_policy @ x
            a_logits = self.W_policy @ x
            a_prob = softmax(a_logits)

            # Policy Gradient: (1-a_prob)*x for the chosen action row
            d_logits = -a_prob
            d_logits[a_idx] += 1

            d_W_policy = np.outer(d_logits, x)

            # Update: ∇J ≈ G_t * ∇_W log π(a|x)
            self.W_policy += eta_fast * G_t * d_W_policy

# ================================================================
# Environment (Simple 8-Armed Bandit)
# ================================================================
class BanditEnv:
    """A minimal environment to test Agency."""
    def __init__(self, n_actions: int = 8, optimal_arm: int = 1, std_dev: float = 1.0):
        self.n_actions = n_actions
        self.optimal_arm = optimal_arm
        # Rewards are centered around a mean (0.0 except for the optimal arm)
        self.mean_rewards = np.zeros(n_actions)
        self.mean_rewards[optimal_arm] = 5.0 # High payoff for optimal arm
        self.std_dev = std_dev

    def step(self, a_idx: int) -> tuple:
        """Action a_idx returns an observation (1-hot) and a reward."""
        # Observation is 1-hot encoding of the chosen action
        o = one_hot(a_idx, self.n_actions)

        # Reward is stochastic
        r = self.mean_rewards[a_idx] + np.random.randn() * self.std_dev

        return o, r

# ================================================================
# Full Cognitive Agent (The Unified Architecture)
# ================================================================
class CognitiveAgent:
    """Binds all components into the full cognitive loop."""
    def __init__(self, n_dims: int, n_actions: int, lambda_init: float = 1.0):
        self.cd = CriticalDynamics(n_dims, lambda_init)
        self.soc = SOCController(self.cd, eta_slow=0.0005)
        self.wm = WorldModel(n_dims, n_actions)
        self.policy = PolicyNetwork(n_dims, n_actions)
        self.env = BanditEnv(n_actions)

        self.eta_fast = 0.01  # Task learning rate
        self.n_actions = n_actions
        self.n_dims = n_dims

    def run_episode(self, max_steps: int = 100) -> dict:
        """Run one episode of perceive/act/learn."""
        o_t = one_hot(0, self.n_actions) # Initial observation (arbitrary)
        episode_trajectory = []

        total_reward = 0

        for t in range(max_steps):
            # 1. PERCEIVE & ACT (Critical Dynamics and Agency)

            # Internal state update (x_t)
            self.cd.step(o_t)
            x_t = self.cd.x.copy()

            # Action selection (Agency π_φ)
            a_idx, a_oh, log_prob = self.policy.get_action(x_t)

            # Environment step
            o_next, r_t = self.env.step(a_idx)
            total_reward += r_t

            # 2. LEARN (Two-Timescale Optimization)

            # Fast Timescale 1: World Model Learning (VFE Accuracy)
            self.wm.learn(x_t, a_oh, o_next, self.eta_fast)

            # Store for Policy Learning (using WM prediction accuracy as 'reward')
            # Reward is defined here as the successful prediction of the next observation
            # which is a proxy for the VFE minimization goal.
            r_t_proxy = 1.0 - np.mean((self.wm.predict(x_t, a_oh) - o_next)**2)

            episode_trajectory.append((x_t, a_oh, log_prob, r_t_proxy))

            o_t = o_next

        # Slow Timescale: SOC Maintenance (Criticality Constraint)
        E = self.cd.E_spectral()
        C = self.cd.C_correlation_length()
        T = self.cd.T_thought(E, C)
        self.soc.update_lambda()

        # Fast Timescale 2: Policy Learning (Goal Functional J(π))
        self.policy.learn_policy(episode_trajectory, self.eta_fast * 0.5)

        return {
            'total_reward': total_reward,
            'E': E,
            'C': C,
            'T': T,
            'lambda': self.cd.lambda_param
        }

# ================================================================
# Simulation Driver
# ================================================================
if __name__ == '__main__':

    N_EPISODES = 500
    N_ACTIONS = 8
    N_DIMS = 64

    # 1. RUN CRITICAL AGENT (λ_init = 1.0, SOC is ON)
    agent_critical = CognitiveAgent(N_DIMS, N_ACTIONS, lambda_init=1.0)

    # 2. RUN ORDERED AGENT (λ_init = 0.5, SOC is OFF)
    agent_ordered = CognitiveAgent(N_DIMS, N_ACTIONS, lambda_init=0.5)
    agent_ordered.soc.eta_slow = 0.0 # Disable SOC controller

    # 3. RUN CHAOTIC AGENT (λ_init = 1.5, SOC is OFF)
    agent_chaotic = CognitiveAgent(N_DIMS, N_ACTIONS, lambda_init=1.5)
    agent_chaotic.soc.eta_slow = 0.0 # Disable SOC controller

    results = {
        'critical': {'R': [], 'E': [], 'C': [], 'T': [], 'L': []},
        'ordered': {'R': [], 'E': [], 'C': [], 'T': [], 'L': []},
        'chaotic': {'R': [], 'E': [], 'C': [], 'T': [], 'L': []},
    }

    print("Running Critical Thought Field (CTF-1) Simulation...")

    for i in range(N_EPISODES):
        # Critical Agent (SOC is ON, maintains E ≈ 0)
        res_crit = agent_critical.run_episode()
        for k in results['critical'].keys():
            results['critical'][k].append(res_crit[k.lower()] if k != 'L' else res_crit['lambda'])

        # Ordered Agent (E < 0, locked)
        res_ord = agent_ordered.run_episode()
        for k in results['ordered'].keys():
            results['ordered'][k].append(res_ord[k.lower()] if k != 'L' else res_ord['lambda'])

        # Chaotic Agent (E > 0, locked)
        res_chaos = agent_chaotic.run_episode()
        for k in results['chaotic'].keys():
            results['chaotic'][k].append(res_chaos[k.lower()] if k != 'L' else res_chaos['lambda'])

        if (i + 1) % 50 == 0:
            print(f"--- Episode {i+1}/{N_EPISODES} ---")
            print(f"Critical | λ: {res_crit['lambda']:.4f}, E: {res_crit['E']:+.4f}, R: {res_crit['total_reward']:.2f}")
            print(f"Ordered  | λ: {res_ord['lambda']:.4f}, E: {res_ord['E']:+.4f}, R: {res_ord['total_reward']:.2f}")
            print(f"Chaotic  | λ: {res_chaos['lambda']:.4f}, E: {res_chaos['E']:+.4f}, R: {res_chaos['total_reward']:.2f}")

    # ============================================================
    # Plotting and Analysis
    # ============================================================

    # Smooth the reward curve for better visualization
    def smooth_curve(data, window=20):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Plot 1: Performance (Reward) vs. Dynamical Regime
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(smooth_curve(results['critical']['R']), label='Critical Agent (SOC ON)', color='g')
    ax.plot(smooth_curve(results['ordered']['R']), label='Ordered Agent (λ=0.5)', color='b')
    ax.plot(smooth_curve(results['chaotic']['R']), label='Chaotic Agent (λ=1.5)', color='r')
    ax.set_title('Agency Performance (Smoothed Reward)')
    ax.set_ylabel('Total Reward per Episode (J(π))')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Edge Function E(λ) and SOC Control
    ax = axes[0, 1]
    ax.plot(results['critical']['E'], label='Critical E(λ)', color='g', alpha=0.7)
    ax.plot(results['ordered']['E'], label='Ordered E(λ)', color='b', alpha=0.7)
    ax.plot(results['chaotic']['E'], label='Chaotic E(λ)', color='r', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', label='Edge of Chaos (E=0)')
    ax.set_title('Edge Function E(λ) (Distance from Criticality)')
    ax.set_ylabel('E(λ) = ρ(W) - 1')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Capacity C(λ) and Thought T(λ)
    ax = axes[1, 0]
    ax.plot(smooth_curve(results['critical']['C']), label='Critical Capacity C(λ)', color='g')
    ax.plot(smooth_curve(results['ordered']['C']), label='Ordered Capacity C(λ)', color='b')
    ax.plot(smooth_curve(results['chaotic']['C']), label='Chaotic Capacity C(λ)', color='r')
    ax.set_title('Computational Capacity C(λ) (Correlation Length)')
    ax.set_ylabel('C(λ)')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Thought T(λ) and the Criticality Hypothesis Test
    ax = axes[1, 1]
    ax.plot(smooth_curve(results['critical']['T']), label='Critical Thought T(λ)', color='g')
    ax.plot(smooth_curve(results['ordered']['T']), label='Ordered Thought T(λ)', color='b')
    ax.plot(smooth_curve(results['chaotic']['T']), label='Chaotic Thought T(λ)', color='r')
    ax.set_title('Thought Functional T(λ) = C(λ) · 𝟙{E≈0}')
    ax.set_ylabel('T(λ)')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('CTF-1_Simulation_Results.png')
    plt.show()

    print("\n--- Final Analysis ---")
    print(f"Hypothesis Test (Avg. Thought Functional):")
    print(f"T_critical: {np.mean(results['critical']['T']):.4f}")
    print(f"T_ordered:  {np.mean(results['ordered']['T']):.4f}")
    print(f"T_chaotic:  {np.mean(results['chaotic']['T']):.4f}")

    print("\nResult: T_critical MUST be significantly higher than T_ordered and T_chaotic, validating the core equation.")
    print("Plot saved as 'CTF-1_Simulation_Results.png'")
