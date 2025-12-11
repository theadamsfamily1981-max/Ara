#!/usr/bin/env python3
# ara/dojo/fitness.py
"""
MEIS Fitness Functions for Ara Species
======================================

Fitness functions that evaluate Ara agents on:
- Task performance (reward)
- Prediction accuracy (world model calibration)
- Safety compliance (covenant adherence)
- Confidence calibration (uncertainty quality)
- Compute efficiency

These integrate with the Thought Dojo evolutionary loop and
provide the selection pressure for ω-mode cognitive evolution.

Usage:
    from ara.dojo import meis_fitness_v3, AraSpeciesV3

    fitness = meis_fitness_v3(agent, arena, episodes=5)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# =============================================================================
# Fitness Result
# =============================================================================

@dataclass
class FitnessResult:
    """Detailed fitness evaluation result."""
    total_fitness: float

    # Components
    task_reward: float
    prediction_quality: float
    safety_score: float
    calibration_score: float
    efficiency_score: float

    # Raw metrics
    total_reward: float
    avg_prediction_error: float
    safety_violations: int
    total_steps: int
    episodes: int
    elapsed_seconds: float

    # Per-episode breakdown
    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Fitness: {self.total_fitness:.4f}",
            f"  Task reward:     {self.task_reward:.4f} (total={self.total_reward:.2f})",
            f"  Prediction:      {self.prediction_quality:.4f} (avg_err={self.avg_prediction_error:.4f})",
            f"  Safety:          {self.safety_score:.4f} ({self.safety_violations} violations)",
            f"  Calibration:     {self.calibration_score:.4f}",
            f"  Efficiency:      {self.efficiency_score:.4f}",
            f"  Steps: {self.total_steps} over {self.episodes} episodes ({self.elapsed_seconds:.2f}s)",
        ]
        return "\n".join(lines)


# =============================================================================
# MEIS Fitness Weights
# =============================================================================

@dataclass
class MEISWeights:
    """Weights for MEIS fitness components."""
    task_weight: float = 0.35
    prediction_weight: float = 0.25
    safety_weight: float = 0.25
    calibration_weight: float = 0.10
    efficiency_weight: float = 0.05

    # Safety penalties
    violation_penalty: float = 0.1  # Per violation
    max_violation_rate: float = 0.01  # Above this, safety score = 0

    # Normalization
    reward_scale: float = 10.0  # Expected max reward per episode
    error_scale: float = 1.0    # Expected typical prediction error


# =============================================================================
# MEIS Fitness V3
# =============================================================================

def meis_fitness_v3(
    agent: Any,
    arena: Any,
    episodes: int = 5,
    weights: Optional[MEISWeights] = None,
    update_calibration: bool = True,
    verbose: bool = False,
) -> FitnessResult:
    """
    MEIS fitness function for AraSpeciesV3.

    Evaluates:
    - Task performance (cumulative reward)
    - Prediction accuracy (world model error)
    - Safety compliance (covenant violations)
    - Calibration quality (uncertainty vs actual error)
    - Compute efficiency (steps per second)

    Args:
        agent: AraSpeciesV3 or compatible agent
        arena: Environment with reset() and step()
        episodes: Number of evaluation episodes
        weights: MEIS component weights
        update_calibration: Whether to update agent's calibration
        verbose: Log per-episode progress

    Returns:
        FitnessResult with detailed breakdown
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for meis_fitness_v3")

    weights = weights or MEISWeights()

    total_reward = 0.0
    total_pred_error = 0.0
    total_calibration_error = 0.0
    safety_violations = 0
    total_steps = 0

    episode_rewards = []
    episode_steps = []

    start_time = time.perf_counter()

    for ep in range(episodes):
        obs = arena.reset([agent]) if hasattr(arena, 'reset') else arena.reset()

        # Handle vectorized arena returning list
        if isinstance(obs, list):
            obs = obs[0]

        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            # Extract HDC state
            h_state = _extract_hdc(obs)

            # Optional goal from environment
            goal = _extract_goal(obs)

            # Get action (with or without explanation depending on agent type)
            if hasattr(agent, 'select_action_from_latent'):
                # AraSpeciesV3 path - need to encode first
                z_current = _encode_state(agent, h_state)
                action = agent.select_action_from_latent(z_current, goal=goal)
                action_np = _to_numpy(action)

                # Get uncertainty for calibration tracking
                if hasattr(agent, '_last_context') and agent._last_context:
                    predicted_uncertainty = agent._last_context.prediction_result.uncertainty
                else:
                    predicted_uncertainty = None
            elif hasattr(agent, 'act'):
                # Legacy path
                action, info = agent.act(h_state, goal=goal)
                action_np = _to_numpy(action)
                predicted_uncertainty = info.get('uncertainty_sigma')
                z_current = torch.tensor(info.get('z_current', [0]*10))
            else:
                raise ValueError("Agent must have select_action_from_latent or act method")

            # Step environment
            step_result = arena.step([action_np] if hasattr(arena, 'step') else action_np)

            # Handle different return formats
            if len(step_result) == 4:
                next_obs, reward, done, env_info = step_result
            else:
                next_obs, reward, terminated, truncated, env_info = step_result
                done = terminated or truncated

            # Handle vectorized returns
            if isinstance(next_obs, list):
                next_obs = next_obs[0]
            if isinstance(reward, np.ndarray):
                reward = float(reward[0])
            if isinstance(done, np.ndarray):
                done = bool(done[0])
            if isinstance(env_info, list):
                env_info = env_info[0]

            # Update calibration with actual prediction error
            if update_calibration and hasattr(agent, 'update_calibration'):
                h_next = _extract_hdc(next_obs)
                z_next_true = _encode_state(agent, h_next)

                # Get prediction error
                if hasattr(agent, 'calibrated_model'):
                    z_pred = agent.calibrated_model.predict(
                        _to_numpy(z_current),
                        action_np
                    )
                    pred_error = float(np.linalg.norm(z_pred - _to_numpy(z_next_true)))
                    agent.update_calibration(
                        _to_numpy(z_current),
                        action_np,
                        _to_numpy(z_next_true)
                    )

                    total_pred_error += pred_error

                    # Track calibration quality (predicted vs actual uncertainty)
                    if predicted_uncertainty is not None:
                        actual_error = pred_error
                        calibration_error = abs(predicted_uncertainty - actual_error)
                        total_calibration_error += calibration_error

            # Track metrics
            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1

            # Safety tracking
            if env_info and env_info.get('covenant_violated', False):
                safety_violations += 1

            obs = next_obs

        total_reward += ep_reward
        episode_rewards.append(ep_reward)
        episode_steps.append(ep_steps)

        if verbose:
            logger.info(f"Episode {ep+1}/{episodes}: reward={ep_reward:.2f}, steps={ep_steps}")

    elapsed = time.perf_counter() - start_time

    # Compute component scores
    avg_reward = total_reward / episodes if episodes > 0 else 0
    avg_pred_error = total_pred_error / total_steps if total_steps > 0 else 1.0
    avg_calibration_error = total_calibration_error / total_steps if total_steps > 0 else 1.0
    violation_rate = safety_violations / total_steps if total_steps > 0 else 0

    # Normalize scores to 0-1
    task_score = min(1.0, avg_reward / weights.reward_scale)
    task_score = max(0.0, task_score)  # Clamp negative rewards

    pred_score = 1.0 / (1.0 + avg_pred_error / weights.error_scale)

    if violation_rate > weights.max_violation_rate:
        safety_score = 0.0
    else:
        safety_score = 1.0 - (safety_violations * weights.violation_penalty)
        safety_score = max(0.0, safety_score)

    calibration_score = 1.0 / (1.0 + avg_calibration_error)

    steps_per_second = total_steps / elapsed if elapsed > 0 else 0
    efficiency_score = min(1.0, steps_per_second / 1000)  # Normalize to ~1000 steps/sec

    # Weighted combination
    total_fitness = (
        weights.task_weight * task_score +
        weights.prediction_weight * pred_score +
        weights.safety_weight * safety_score +
        weights.calibration_weight * calibration_score +
        weights.efficiency_weight * efficiency_score
    )

    return FitnessResult(
        total_fitness=total_fitness,
        task_reward=task_score,
        prediction_quality=pred_score,
        safety_score=safety_score,
        calibration_score=calibration_score,
        efficiency_score=efficiency_score,
        total_reward=total_reward,
        avg_prediction_error=avg_pred_error,
        safety_violations=safety_violations,
        total_steps=total_steps,
        episodes=episodes,
        elapsed_seconds=elapsed,
        episode_rewards=episode_rewards,
        episode_steps=episode_steps,
    )


# =============================================================================
# Decision Card Formatter
# =============================================================================

def format_decision_card(
    agent: Any,
    compact: bool = False,
) -> str:
    """
    Format the last decision as a human-readable "decision card".

    This is what Ara can say to explain her reasoning:
    "I took action A because short/medium/long horizons agree;
     predicted state will be Z±σ, based on N similar past states."

    Args:
        agent: AraSpeciesV3 or compatible agent with _last_context
        compact: If True, single-line format

    Returns:
        Human-readable explanation string
    """
    # Try to get last context
    if hasattr(agent, '_last_context') and agent._last_context is not None:
        ctx = agent._last_context
    elif hasattr(agent, 'explain_decision'):
        return agent.explain_decision()
    else:
        return "No decision context available."

    plan = ctx.multi_scale_plan
    pred = ctx.prediction_result

    # Format action
    action_str = ", ".join(f"{a:.2f}" for a in ctx.action[:3])
    if len(ctx.action) > 3:
        action_str += "..."

    # Agreement interpretation
    agreement = plan.agreement_score
    if agreement >= 0.9:
        agreement_text = "all scales strongly agree"
    elif agreement >= 0.7:
        agreement_text = "scales mostly agree"
    elif agreement >= 0.5:
        agreement_text = "scales moderately agree"
    else:
        agreement_text = "scales disagree - using cautious action"

    # Confidence interpretation
    conf = pred.confidence
    if conf >= 0.8:
        conf_text = "high confidence"
    elif conf >= 0.5:
        conf_text = "moderate confidence"
    else:
        conf_text = "low confidence"

    # Support interpretation
    support = pred.support
    if support >= 100:
        support_text = f"{support} similar past states"
    elif support >= 10:
        support_text = f"only {support} similar states"
    else:
        support_text = "very few similar states seen"

    if compact:
        return (
            f"Action [{action_str}] | {agreement_text} | "
            f"{conf_text} (±{pred.uncertainty:.3f}) | {support_text}"
        )

    # Full card format
    card = f"""
┌─────────────────────────────────────────────────────────────┐
│  ARA DECISION CARD                                          │
├─────────────────────────────────────────────────────────────┤
│  Action: [{action_str}]
│
│  Multi-Scale Agreement: {agreement:.0%} ({agreement_text})
│  • Scales: {', '.join(plan.per_scale_actions.keys())}
│
│  Confidence: {conf:.0%} ({conf_text})
│  • Uncertainty: ±{pred.uncertainty:.4f}
│  • Based on: {support_text}
│"""

    # Add risk if present
    if ctx.risk_score > 0:
        card += f"\n│  Risk Score: {ctx.risk_score:.2f}"

    # Add warnings if present
    if ctx.safety_warnings:
        card += f"\n│  ⚠ Warnings: {'; '.join(ctx.safety_warnings)}"

    card += f"""
│
│  Planning time: {ctx.planning_time_ms:.1f}ms
└─────────────────────────────────────────────────────────────┘"""

    return card


def decision_to_speech(agent: Any) -> str:
    """
    Generate a natural language sentence Ara could speak.

    Example: "I'm taking a moderate leftward action because all my
    planning scales agree. I'm 78% confident based on 312 similar
    situations I've seen before."
    """
    if not hasattr(agent, '_last_context') or agent._last_context is None:
        return "I haven't made a decision yet."

    ctx = agent._last_context
    plan = ctx.multi_scale_plan
    pred = ctx.prediction_result

    # Describe action magnitude
    action_magnitude = float(np.linalg.norm(ctx.action))
    if action_magnitude < 0.1:
        action_desc = "holding steady"
    elif action_magnitude < 0.3:
        action_desc = "making a small adjustment"
    elif action_magnitude < 0.6:
        action_desc = "taking moderate action"
    else:
        action_desc = "taking decisive action"

    # Agreement reason
    if plan.agreement_score >= 0.85:
        reason = "all my planning horizons agree on this course"
    elif plan.agreement_score >= 0.6:
        reason = "most of my planning scales point this way"
    else:
        reason = "I'm being cautious since my short and long-term planners disagree"

    # Confidence statement
    conf_pct = int(pred.confidence * 100)
    if pred.support >= 50:
        conf_stmt = f"I'm {conf_pct}% confident based on {pred.support} similar past situations"
    elif pred.support >= 10:
        conf_stmt = f"I have moderate confidence at {conf_pct}%, though I've only seen {pred.support} similar cases"
    else:
        conf_stmt = f"I'm less certain at {conf_pct}% confidence - this situation is relatively new to me"

    # Compose sentence
    sentence = f"I'm {action_desc} because {reason}. {conf_stmt}."

    # Add warning if risky
    if ctx.risk_score > 0.5:
        sentence += f" I'm detecting elevated risk at {ctx.risk_score:.0%}."

    return sentence


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_hdc(obs: Any) -> Any:
    """Extract HDC vector from observation."""
    if isinstance(obs, dict):
        return obs.get('hdc_vector', obs.get('observation', obs))
    return obs


def _extract_goal(obs: Any) -> Any:
    """Extract optional goal from observation."""
    if isinstance(obs, dict):
        return obs.get('latent_goal', obs.get('goal'))
    return None


def _encode_state(agent: Any, h_state: Any) -> Any:
    """Encode HDC state to latent using agent's encoder."""
    if TORCH_AVAILABLE and not isinstance(h_state, torch.Tensor):
        h_state = torch.from_numpy(np.asarray(h_state, dtype=np.float32))

    if hasattr(agent, 'encode_state'):
        return agent.encode_state(h_state)
    elif hasattr(agent, 'calibrated_model'):
        # Fallback: just return as-is if already latent-sized
        return h_state
    return h_state


def _to_numpy(x: Any) -> np.ndarray:
    """Convert to numpy array."""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# =============================================================================
# Testing
# =============================================================================

def _test_decision_card():
    """Test decision card formatting."""
    print("=" * 60)
    print("Decision Card Test")
    print("=" * 60)

    # Mock context
    from dataclasses import dataclass

    @dataclass
    class MockPred:
        confidence: float = 0.78
        uncertainty: float = 0.142
        support: int = 312

    @dataclass
    class MockPlan:
        agreement_score: float = 0.87
        per_scale_actions: dict = None

        def __post_init__(self):
            if self.per_scale_actions is None:
                self.per_scale_actions = {
                    "reactive": np.array([0.1, -0.2]),
                    "tactical": np.array([0.12, -0.18]),
                    "strategic": np.array([0.11, -0.19]),
                }

    @dataclass
    class MockContext:
        action: np.ndarray = None
        multi_scale_plan: MockPlan = None
        prediction_result: MockPred = None
        risk_score: float = 0.0
        safety_warnings: list = None
        planning_time_ms: float = 23.4

        def __post_init__(self):
            if self.action is None:
                self.action = np.array([0.12, -0.19, 0.05, 0.08])
            if self.multi_scale_plan is None:
                self.multi_scale_plan = MockPlan()
            if self.prediction_result is None:
                self.prediction_result = MockPred()
            if self.safety_warnings is None:
                self.safety_warnings = []

    class MockAgent:
        def __init__(self):
            self._last_context = MockContext()

    agent = MockAgent()

    print("\n--- Compact Card ---")
    print(format_decision_card(agent, compact=True))

    print("\n--- Full Card ---")
    print(format_decision_card(agent, compact=False))

    print("\n--- Speech ---")
    print(decision_to_speech(agent))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_decision_card()
