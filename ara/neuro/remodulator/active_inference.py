#!/usr/bin/env python3
"""
Active Inference Implementation

Extends predictive coding to action selection via Expected Free Energy (EFE).
This implements the POMDP structure common in Active Inference agents.

Key concepts:
- VFE (Variational Free Energy): Minimized during perception
- EFE (Expected Free Energy): Minimized during action selection
- Epistemic Value: Curiosity / uncertainty reduction
- Pragmatic Value: Goal-directed / preference satisfaction

The agent naturally balances exploration (epistemic) vs exploitation (pragmatic)
through a single objective: minimizing EFE.

Usage:
    python -m ara.neuro.remodulator.active_inference
    python -m ara.neuro.remodulator.active_inference --demo curiosity
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# POMDP Generative Model Matrices
# =============================================================================

@dataclass
class GenerativeModel:
    """
    POMDP Generative Model for Active Inference.

    The four key matrices:
    - A: Likelihood P(o|s) - How states generate observations
    - B: Transition P(s'|s,a) - How actions change states
    - C: Preferences P(o) - Desired observations (log prior)
    - D: Initial prior P(s0) - Starting belief about states
    """

    A: np.ndarray  # [n_obs, n_states] Likelihood
    B: np.ndarray  # [n_states, n_states, n_actions] Transition
    C: np.ndarray  # [n_obs] Log preferences (higher = more preferred)
    D: np.ndarray  # [n_states] Initial state prior

    @property
    def n_states(self) -> int:
        return self.A.shape[1]

    @property
    def n_obs(self) -> int:
        return self.A.shape[0]

    @property
    def n_actions(self) -> int:
        return self.B.shape[2]

    def likelihood(self, state: int) -> np.ndarray:
        """P(o|s) - probability of each observation given state."""
        return self.A[:, state]

    def transition(self, state: int, action: int) -> np.ndarray:
        """P(s'|s,a) - probability of each next state."""
        return self.B[:, state, action]

    def preference(self, obs: int) -> float:
        """Log preference for observation (higher = better)."""
        return self.C[obs]


# =============================================================================
# Variational Free Energy (VFE) - Perception
# =============================================================================

def compute_vfe(
    q_s: np.ndarray,
    obs: int,
    model: GenerativeModel,
    prior: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Variational Free Energy.

    F = E_q[ln q(s)] - E_q[ln p(o,s)]
      = E_q[ln q(s)] - E_q[ln p(o|s)] - E_q[ln p(s)]

    This is minimized during perception to update beliefs q(s).

    Args:
        q_s: Approximate posterior belief over states
        obs: Current observation index
        model: Generative model
        prior: Prior over states (defaults to model.D)

    Returns:
        VFE value (lower is better)
    """
    if prior is None:
        prior = model.D

    # Avoid log(0)
    q_s = np.clip(q_s, 1e-10, 1.0)
    prior = np.clip(prior, 1e-10, 1.0)

    # E_q[ln q(s)] - Negative entropy of beliefs
    entropy_term = np.sum(q_s * np.log(q_s))

    # E_q[ln p(o|s)] - Expected log likelihood
    likelihood_term = np.sum(q_s * np.log(model.A[obs, :] + 1e-10))

    # E_q[ln p(s)] - Expected log prior
    prior_term = np.sum(q_s * np.log(prior))

    # F = entropy - likelihood - prior
    F = entropy_term - likelihood_term - prior_term

    return F


def update_beliefs(
    q_s: np.ndarray,
    obs: int,
    model: GenerativeModel,
    prior: Optional[np.ndarray] = None,
    n_iterations: int = 16,
    learning_rate: float = 1.0,
) -> np.ndarray:
    """
    Update beliefs by minimizing VFE (perception).

    Uses gradient descent on softmax parameters.

    Args:
        q_s: Current beliefs
        obs: Current observation
        model: Generative model
        prior: Prior over states
        n_iterations: Number of update steps
        learning_rate: Step size

    Returns:
        Updated beliefs q(s)
    """
    if prior is None:
        prior = model.D

    # Work in log space for stability
    log_q = np.log(q_s + 1e-10)

    for _ in range(n_iterations):
        # Gradient: d(VFE)/d(log_q) ≈ log q - log p(o|s) - log p(s) + 1
        q_s = softmax(log_q)
        gradient = log_q - np.log(model.A[obs, :] + 1e-10) - np.log(prior + 1e-10)

        # Update
        log_q -= learning_rate * gradient

    return softmax(log_q)


# =============================================================================
# Expected Free Energy (EFE) - Action Selection
# =============================================================================

def compute_efe(
    policy: List[int],
    q_s: np.ndarray,
    model: GenerativeModel,
    horizon: int = 1,
) -> Tuple[float, float, float]:
    """
    Compute Expected Free Energy for a policy.

    G(π) = Pragmatic Value + Epistemic Value

    Pragmatic: E[ln p(o|C)] - Expected cost of outcomes vs preferences
    Epistemic: E[H[p(s|o)] - H[p(s)]] - Expected information gain

    Args:
        policy: List of action indices
        q_s: Current state belief
        model: Generative model
        horizon: Planning horizon

    Returns:
        (total_efe, pragmatic, epistemic)
    """
    pragmatic_total = 0.0
    epistemic_total = 0.0

    # Current belief
    q_current = q_s.copy()

    for t, action in enumerate(policy[:horizon]):
        # Predict next state distribution: P(s'|s,a)
        q_next = np.zeros(model.n_states)
        for s in range(model.n_states):
            q_next += q_current[s] * model.transition(s, action)

        # Predict observation distribution: P(o|s')
        q_obs = model.A @ q_next  # [n_obs]

        # === Pragmatic Value ===
        # Expected log preference: E_q(o)[C(o)]
        # Lower preferences = higher cost
        pragmatic = -np.sum(q_obs * model.C)
        pragmatic_total += pragmatic

        # === Epistemic Value ===
        # Information gain: H[P(s)] - E_o[H[P(s|o)]]
        # This is the expected reduction in state uncertainty

        # Prior entropy: H[q(s')]
        prior_entropy = entropy(q_next)

        # Expected posterior entropy: E_o[H[P(s|o)]]
        expected_posterior_entropy = 0.0
        for o in range(model.n_obs):
            if q_obs[o] > 1e-10:
                # Posterior given this observation
                posterior = model.A[o, :] * q_next
                posterior /= posterior.sum() + 1e-10
                expected_posterior_entropy += q_obs[o] * entropy(posterior)

        # Information gain (higher = more curious)
        info_gain = prior_entropy - expected_posterior_entropy
        epistemic = -info_gain  # Negative because we minimize EFE
        epistemic_total += epistemic

        # Update belief for next step
        q_current = q_next

    total_efe = pragmatic_total + epistemic_total
    return total_efe, pragmatic_total, epistemic_total


def select_action(
    q_s: np.ndarray,
    model: GenerativeModel,
    horizon: int = 1,
    temperature: float = 1.0,
) -> Tuple[int, Dict[int, float]]:
    """
    Select action by minimizing EFE.

    Uses softmax over negative EFE for stochastic selection.

    Args:
        q_s: Current state belief
        model: Generative model
        horizon: Planning horizon
        temperature: Softmax temperature (lower = more greedy)

    Returns:
        (selected_action, efe_values)
    """
    efe_values = {}

    for action in range(model.n_actions):
        policy = [action]  # Single-step for simplicity
        efe, _, _ = compute_efe(policy, q_s, model, horizon)
        efe_values[action] = efe

    # Softmax over negative EFE (prefer lower EFE)
    efe_array = np.array([efe_values[a] for a in range(model.n_actions)])
    action_probs = softmax(-efe_array / temperature)

    # Sample action
    action = np.random.choice(model.n_actions, p=action_probs)

    return action, efe_values


# =============================================================================
# Active Inference Agent
# =============================================================================

class ActiveInferenceAgent:
    """
    Complete Active Inference agent.

    Implements the perception-action cycle:
    1. Observe → Update beliefs (minimize VFE)
    2. Plan → Evaluate policies (compute EFE)
    3. Act → Select action (minimize EFE)
    4. Execute → Observe outcome
    """

    def __init__(
        self,
        model: GenerativeModel,
        planning_horizon: int = 1,
        action_temperature: float = 1.0,
        belief_learning_rate: float = 1.0,
        belief_iterations: int = 16,
    ):
        self.model = model
        self.planning_horizon = planning_horizon
        self.action_temperature = action_temperature
        self.belief_learning_rate = belief_learning_rate
        self.belief_iterations = belief_iterations

        # Initialize beliefs
        self.q_s = model.D.copy()

        # History for analysis
        self.history: List[Dict] = []

    def observe(self, obs: int):
        """Update beliefs given observation (perception)."""
        old_vfe = compute_vfe(self.q_s, obs, self.model)

        self.q_s = update_beliefs(
            self.q_s,
            obs,
            self.model,
            n_iterations=self.belief_iterations,
            learning_rate=self.belief_learning_rate,
        )

        new_vfe = compute_vfe(self.q_s, obs, self.model)

        self.history.append({
            'type': 'observe',
            'obs': obs,
            'vfe_before': old_vfe,
            'vfe_after': new_vfe,
            'beliefs': self.q_s.copy(),
        })

    def act(self) -> int:
        """Select action by minimizing EFE."""
        action, efe_values = select_action(
            self.q_s,
            self.model,
            horizon=self.planning_horizon,
            temperature=self.action_temperature,
        )

        # Decompose EFE for analysis
        efe, pragmatic, epistemic = compute_efe(
            [action], self.q_s, self.model, self.planning_horizon
        )

        self.history.append({
            'type': 'act',
            'action': action,
            'efe': efe,
            'pragmatic': pragmatic,
            'epistemic': epistemic,
            'efe_all': efe_values,
        })

        return action

    def step(self, obs: int) -> int:
        """Complete perception-action step."""
        self.observe(obs)
        return self.act()

    def reset(self):
        """Reset beliefs to prior."""
        self.q_s = self.model.D.copy()
        self.history = []


# =============================================================================
# Utility Functions
# =============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum() + 1e-10)


def entropy(p: np.ndarray) -> float:
    """Shannon entropy of distribution."""
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D_KL(p || q)."""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)))


# =============================================================================
# Demo Environments
# =============================================================================

def create_gridworld_model(
    size: int = 4,
    goal_state: int = 3,
    uncertain_state: int = 2,
) -> GenerativeModel:
    """
    Create a simple gridworld POMDP.

    States: 0, 1, 2, ..., size-1 (linear positions)
    Actions: 0=stay, 1=left, 2=right
    Observations: Partially observable (noisy state)
    """
    n_states = size
    n_obs = size
    n_actions = 3

    # A matrix: Likelihood P(o|s)
    # Mostly identity with some noise
    A = np.eye(n_obs, n_states) * 0.8
    A += 0.2 / n_obs  # Uniform noise
    A[:, uncertain_state] = 1.0 / n_obs  # Uncertain state is ambiguous

    # B matrix: Transition P(s'|s,a)
    B = np.zeros((n_states, n_states, n_actions))

    for s in range(n_states):
        # Action 0: Stay
        B[s, s, 0] = 1.0

        # Action 1: Left
        if s > 0:
            B[s-1, s, 1] = 0.9
            B[s, s, 1] = 0.1
        else:
            B[s, s, 1] = 1.0

        # Action 2: Right
        if s < n_states - 1:
            B[s+1, s, 2] = 0.9
            B[s, s, 2] = 0.1
        else:
            B[s, s, 2] = 1.0

    # C matrix: Preferences (log scale)
    C = np.zeros(n_obs)
    C[goal_state] = 2.0  # Prefer goal observation
    C[uncertain_state] = 1.0  # Mild preference for exploring uncertain

    # D matrix: Initial prior
    D = np.ones(n_states) / n_states

    return GenerativeModel(A=A, B=B, C=C, D=D)


# =============================================================================
# Demonstrations
# =============================================================================

def demo_curiosity():
    """
    Demonstrate curiosity-driven exploration.

    The agent naturally explores uncertain states before
    exploiting goal states.
    """
    print("\n" + "=" * 70)
    print("DEMO: Curiosity-Driven Exploration")
    print("=" * 70)

    # Create model with uncertain state 2, goal at 3
    model = create_gridworld_model(size=4, goal_state=3, uncertain_state=2)

    print("\nGridworld: [0] -- [1] -- [2?] -- [3*]")
    print("  State 2 is uncertain (ambiguous observations)")
    print("  State 3 is the goal (preferred observation)")
    print("  Actions: 0=stay, 1=left, 2=right")

    # Create agent
    agent = ActiveInferenceAgent(
        model,
        planning_horizon=1,
        action_temperature=0.5,  # Somewhat greedy
    )

    # Start at state 0
    current_state = 0

    print("\n" + "-" * 70)
    print("Step | State | Obs | Action | EFE | Epistemic | Pragmatic")
    print("-" * 70)

    for step in range(10):
        # Generate observation (noisy)
        obs_probs = model.A[:, current_state]
        obs = np.random.choice(model.n_obs, p=obs_probs)

        # Agent perceives and acts
        action = agent.step(obs)

        # Get EFE decomposition
        efe, pragmatic, epistemic = compute_efe(
            [action], agent.q_s, model, horizon=1
        )

        action_names = ['stay', 'left', 'right']
        print(f"{step:4d} | {current_state:5d} | {obs:3d} | {action_names[action]:6s} | "
              f"{efe:5.2f} | {epistemic:9.2f} | {pragmatic:9.2f}")

        # Execute action (environment)
        trans_probs = model.B[:, current_state, action]
        current_state = np.random.choice(model.n_states, p=trans_probs)

        # Check if reached goal
        if current_state == 3:
            print(f"\n→ Reached goal state 3!")

    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("  - Negative epistemic = curiosity drives toward uncertain state 2")
    print("  - Negative pragmatic = goals drive toward preferred state 3")
    print("  - Agent balances exploration and exploitation automatically")


def demo_precision_in_ai():
    """
    Show how precision affects the exploration-exploitation tradeoff.

    Higher precision on preferences → more exploitation
    Higher precision on beliefs → less exploration
    """
    print("\n" + "=" * 70)
    print("DEMO: Precision and Exploration-Exploitation")
    print("=" * 70)

    model = create_gridworld_model(size=4, goal_state=3, uncertain_state=2)

    print("\nVarying preference strength (C matrix scaling):")
    print("-" * 70)

    for preference_scale in [0.5, 1.0, 2.0, 5.0]:
        # Scale preferences
        model_scaled = GenerativeModel(
            A=model.A,
            B=model.B,
            C=model.C * preference_scale,  # Scale preferences
            D=model.D,
        )

        agent = ActiveInferenceAgent(model_scaled, action_temperature=1.0)

        # Start at state 1, between uncertain (2) and away from goal (3)
        agent.q_s = np.array([0.1, 0.7, 0.1, 0.1])  # Believe we're at state 1

        # What action does the agent prefer?
        action, efe_values = select_action(agent.q_s, model_scaled, horizon=1)

        efe_right, prag_right, epist_right = compute_efe([2], agent.q_s, model_scaled)
        efe_left, prag_left, epist_left = compute_efe([1], agent.q_s, model_scaled)

        print(f"\nPreference scale = {preference_scale}:")
        print(f"  Right (toward goal):     EFE={efe_right:6.2f} (prag={prag_right:5.2f}, epist={epist_right:5.2f})")
        print(f"  Left (toward uncertain): EFE={efe_left:6.2f} (prag={prag_left:5.2f}, epist={epist_left:5.2f})")
        print(f"  → Agent prefers: {'RIGHT (goal)' if efe_right < efe_left else 'LEFT (explore)'}")


def demo_connection_to_D():
    """
    Connect Active Inference back to the Delusion Index D.

    D affects both perception (VFE) and action (EFE) through
    precision weighting.
    """
    print("\n" + "=" * 70)
    print("DEMO: Connection to Delusion Index (D)")
    print("=" * 70)

    print("""
    In Active Inference, the Delusion Index D = Π_prior / Π_sensory
    affects both:

    1. PERCEPTION (VFE minimization):
       - D >> 1: Beliefs resist updating from observations
       - D << 1: Beliefs over-update from noisy observations

    2. ACTION (EFE minimization):
       - D >> 1: Over-confident predictions → reduced exploration
       - D << 1: Under-confident predictions → excessive exploration

    The Brain Remodulator targets D ≈ 1.0 to balance:
       - Prior beliefs vs sensory evidence
       - Exploitation vs exploration
       - Top-down prediction vs bottom-up error
    """)

    # Simulate with different D values
    model = create_gridworld_model()

    for D in [0.2, 1.0, 5.0]:
        print(f"\n{'=' * 50}")
        print(f"D = {D:.1f}")
        print('=' * 50)

        # Modify model to reflect D
        # D >> 1: Sharper likelihood (over-confident predictions)
        # D << 1: Flatter likelihood (under-confident)
        if D > 1:
            A_modified = model.A ** (1 / D)  # Sharpen
        else:
            A_modified = model.A ** D  # Flatten
        A_modified /= A_modified.sum(axis=0, keepdims=True)

        model_D = GenerativeModel(A=A_modified, B=model.B, C=model.C, D=model.D)
        agent = ActiveInferenceAgent(model_D, action_temperature=1.0)

        # Ambiguous observation
        obs = 1  # Could be state 1 or 2

        # How does the agent update beliefs?
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        posterior = update_beliefs(prior, obs, model_D, n_iterations=16)

        print(f"  Prior:     {prior.round(2)}")
        print(f"  Posterior: {posterior.round(2)}")
        print(f"  Entropy:   {entropy(prior):.2f} → {entropy(posterior):.2f}")

        if D > 1:
            print("  → Prior-dominated: Beliefs resist change")
        elif D < 1:
            print("  → Sensory-dominated: Over-updates to observation")
        else:
            print("  → Balanced: Appropriate belief update")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Active Inference Demo"
    )
    parser.add_argument(
        "--demo", type=str, default="all",
        choices=["all", "curiosity", "precision", "D"],
        help="Which demo to run"
    )

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        ACTIVE INFERENCE DEMO                                   ║
║                                                                                ║
║  Extends Predictive Coding to action selection via Expected Free Energy       ║
║                                                                                ║
║  VFE (Perception): Minimize surprise about observations                        ║
║  EFE (Action):     Minimize expected future surprise                           ║
║                    = Pragmatic Value (goals) + Epistemic Value (curiosity)     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    if args.demo in ["all", "curiosity"]:
        demo_curiosity()

    if args.demo in ["all", "precision"]:
        demo_precision_in_ai()

    if args.demo in ["all", "D"]:
        demo_connection_to_D()

    print("\n" + "=" * 70)
    print("Active Inference provides a unified account of:")
    print("  - Perception (VFE minimization)")
    print("  - Action (EFE minimization)")
    print("  - Curiosity (epistemic value)")
    print("  - Goal-seeking (pragmatic value)")
    print("=" * 70)


if __name__ == "__main__":
    main()
