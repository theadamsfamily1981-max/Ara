"""Policy Evaluation - Test policy changes before applying.

Before applying a workflow change globally, replay past episodes:
1. "If we had used the new policy, would outcomes have been better?"
2. Use shadow models to approximate new outcomes
3. Only if expected improvement, present to user for approval

This is like CI for Ara's decision-making policies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .predictor import ShadowPredictor, get_predictor
from .planner import WorkflowPlanner, get_planner

logger = logging.getLogger(__name__)


@dataclass
class PolicyChange:
    """A proposed change to workflow policy."""

    id: str
    description: str
    intent: str  # Intent this applies to

    # Old and new workflows
    old_workflow: List[str]  # e.g., ["nova"]
    new_workflow: List[str]  # e.g., ["claude", "nova"]

    # Rule for when this applies
    condition: str = "always"  # "always", "complex_only", "has_stacktrace", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "intent": self.intent,
            "old_workflow": self.old_workflow,
            "new_workflow": self.new_workflow,
            "condition": self.condition,
        }


@dataclass
class EpisodeReplay:
    """Result of replaying a past episode with a new policy."""

    episode_id: str
    intent: str

    # Original outcome
    original_workflow: List[str]
    original_reward: float
    original_latency: float

    # Simulated outcome with new policy
    simulated_workflow: List[str]
    simulated_reward: float
    simulated_latency: float

    # Difference
    reward_delta: float = 0.0
    latency_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "intent": self.intent,
            "original_workflow": self.original_workflow,
            "original_reward": round(self.original_reward, 3),
            "original_latency": round(self.original_latency, 1),
            "simulated_workflow": self.simulated_workflow,
            "simulated_reward": round(self.simulated_reward, 3),
            "simulated_latency": round(self.simulated_latency, 1),
            "reward_delta": round(self.reward_delta, 3),
            "latency_delta": round(self.latency_delta, 1),
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a policy change."""

    policy_change: PolicyChange
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    # Episodes analyzed
    episodes_analyzed: int = 0
    episodes_improved: int = 0
    episodes_degraded: int = 0
    episodes_unchanged: int = 0

    # Aggregate metrics
    avg_reward_before: float = 0.0
    avg_reward_after: float = 0.0
    avg_latency_before: float = 0.0
    avg_latency_after: float = 0.0

    # Net change
    reward_improvement: float = 0.0
    latency_change: float = 0.0

    # Replay details
    replays: List[EpisodeReplay] = field(default_factory=list)

    # Recommendation
    recommended: bool = False
    confidence: float = 0.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_change": self.policy_change.to_dict(),
            "evaluated_at": self.evaluated_at.isoformat(),
            "episodes_analyzed": self.episodes_analyzed,
            "episodes_improved": self.episodes_improved,
            "episodes_degraded": self.episodes_degraded,
            "avg_reward_before": round(self.avg_reward_before, 3),
            "avg_reward_after": round(self.avg_reward_after, 3),
            "avg_latency_before": round(self.avg_latency_before, 1),
            "avg_latency_after": round(self.avg_latency_after, 1),
            "reward_improvement": round(self.reward_improvement, 3),
            "latency_change": round(self.latency_change, 1),
            "recommended": self.recommended,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
        }

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        lines = [
            f"PROPOSED POLICY UPDATE: {self.policy_change.id}",
            "",
            f"Offline replay over {self.episodes_analyzed} episodes:",
            "",
            "Current policy:",
            f"  avg_reward: {self.avg_reward_before:.0%}",
            f"  avg_latency: {self.avg_latency_before:.1f}s",
            "",
            f"Proposed policy ({' → '.join(self.policy_change.new_workflow)}):",
            f"  est_reward: {self.avg_reward_after:.0%}",
            f"  est_latency: {self.avg_latency_after:.1f}s",
            "",
            "Net change:",
            f"  {'+' if self.reward_improvement >= 0 else ''}{self.reward_improvement:.0%} reward",
            f"  {'+' if self.latency_change >= 0 else ''}{self.latency_change:.1f}s latency",
            "",
            f"Episodes improved: {self.episodes_improved}/{self.episodes_analyzed}",
            f"Episodes degraded: {self.episodes_degraded}/{self.episodes_analyzed}",
            "",
            f"Recommendation: {'APPLY' if self.recommended else 'DO NOT APPLY'}",
            f"Confidence: {self.confidence:.0%}",
            f"Rationale: {self.rationale}",
        ]
        return "\n".join(lines)


@dataclass
class PolicyProposal:
    """A policy proposal with evaluation results."""

    change: PolicyChange
    evaluation: EvaluationResult

    # Status
    status: str = "pending"  # "pending", "approved", "rejected"
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change": self.change.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "status": self.status,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
        }


class PolicyEvaluator:
    """Evaluates policy changes through offline replay.

    Tests proposed workflow changes by replaying past episodes
    and estimating what would have happened with the new policy.
    """

    def __init__(
        self,
        predictor: Optional[ShadowPredictor] = None,
        min_improvement: float = 0.05,
        max_latency_increase: float = 5.0,
    ):
        """Initialize the evaluator.

        Args:
            predictor: Shadow predictor to use
            min_improvement: Minimum reward improvement to recommend
            max_latency_increase: Maximum latency increase to tolerate
        """
        self.predictor = predictor or get_predictor()
        self.min_improvement = min_improvement
        self.max_latency_increase = max_latency_increase

        # Past episodes (loaded from meta logger)
        self._episodes: List[Dict[str, Any]] = []

    def load_episodes_from_logger(self, days: int = 30) -> int:
        """Load past episodes from meta logger.

        Args:
            days: Days of history to load

        Returns:
            Number of episodes loaded
        """
        from ..meta_logger import get_meta_logger

        logger_instance = get_meta_logger()
        since = datetime.utcnow() - timedelta(days=days)
        records = logger_instance.query(since=since, limit=1000)

        self._episodes = []
        for record in records:
            if record.outcome_quality is not None and record.teachers:
                self._episodes.append({
                    "id": record.id,
                    "intent": record.user_intent or "general",
                    "workflow": record.teachers,
                    "reward": record.outcome_quality,
                    "latency": record.latency_sec or 15.0,
                    "features": None,  # Could extract from record
                })

        return len(self._episodes)

    def add_synthetic_episodes(self, episodes: List[Dict[str, Any]]) -> None:
        """Add synthetic episodes for testing.

        Args:
            episodes: List of episode dicts
        """
        self._episodes.extend(episodes)

    def evaluate_change(
        self,
        change: PolicyChange,
        episodes: Optional[List[Dict[str, Any]]] = None,
    ) -> EvaluationResult:
        """Evaluate a policy change through offline replay.

        Args:
            change: The proposed policy change
            episodes: Episodes to replay (uses loaded if None)

        Returns:
            Evaluation result
        """
        if episodes is None:
            episodes = self._episodes

        # Filter to relevant episodes
        relevant = [
            ep for ep in episodes
            if ep["intent"] == change.intent
        ]

        if not relevant:
            return EvaluationResult(
                policy_change=change,
                episodes_analyzed=0,
                recommended=False,
                confidence=0.0,
                rationale="No relevant episodes to evaluate",
            )

        replays = []
        total_reward_before = 0.0
        total_reward_after = 0.0
        total_latency_before = 0.0
        total_latency_after = 0.0
        improved = 0
        degraded = 0
        unchanged = 0

        for ep in relevant:
            # Simulate with new policy
            sim_pred = self.predictor.predict_workflow(
                change.new_workflow, ep["intent"]
            )

            replay = EpisodeReplay(
                episode_id=ep["id"],
                intent=ep["intent"],
                original_workflow=ep["workflow"],
                original_reward=ep["reward"],
                original_latency=ep["latency"],
                simulated_workflow=change.new_workflow,
                simulated_reward=sim_pred["expected_reward"],
                simulated_latency=sim_pred["expected_latency_sec"],
            )

            replay.reward_delta = replay.simulated_reward - replay.original_reward
            replay.latency_delta = replay.simulated_latency - replay.original_latency

            replays.append(replay)

            total_reward_before += ep["reward"]
            total_reward_after += sim_pred["expected_reward"]
            total_latency_before += ep["latency"]
            total_latency_after += sim_pred["expected_latency_sec"]

            if replay.reward_delta > 0.05:
                improved += 1
            elif replay.reward_delta < -0.05:
                degraded += 1
            else:
                unchanged += 1

        n = len(relevant)
        avg_reward_before = total_reward_before / n
        avg_reward_after = total_reward_after / n
        avg_latency_before = total_latency_before / n
        avg_latency_after = total_latency_after / n

        reward_improvement = avg_reward_after - avg_reward_before
        latency_change = avg_latency_after - avg_latency_before

        # Determine recommendation
        recommended = (
            reward_improvement >= self.min_improvement
            and latency_change <= self.max_latency_increase
            and improved > degraded
        )

        # Compute confidence
        if n >= 20:
            confidence = 0.9
        elif n >= 10:
            confidence = 0.7
        elif n >= 5:
            confidence = 0.5
        else:
            confidence = 0.3

        # Adjust confidence based on consistency
        if improved > 0 and degraded == 0:
            confidence = min(1.0, confidence * 1.1)
        elif degraded > improved:
            confidence *= 0.5

        # Generate rationale
        if recommended:
            rationale = (
                f"Higher success rate on {change.intent}; "
                f"{'small' if latency_change <= 2 else 'acceptable'} latency increase."
            )
        elif reward_improvement < self.min_improvement:
            rationale = f"Improvement ({reward_improvement:.0%}) below threshold ({self.min_improvement:.0%})"
        elif latency_change > self.max_latency_increase:
            rationale = f"Latency increase ({latency_change:.1f}s) exceeds limit ({self.max_latency_increase:.1f}s)"
        else:
            rationale = f"Too many degraded episodes ({degraded}/{n})"

        return EvaluationResult(
            policy_change=change,
            episodes_analyzed=n,
            episodes_improved=improved,
            episodes_degraded=degraded,
            episodes_unchanged=unchanged,
            avg_reward_before=avg_reward_before,
            avg_reward_after=avg_reward_after,
            avg_latency_before=avg_latency_before,
            avg_latency_after=avg_latency_after,
            reward_improvement=reward_improvement,
            latency_change=latency_change,
            replays=replays[:10],  # Keep top 10
            recommended=recommended,
            confidence=confidence,
            rationale=rationale,
        )

    def propose_policy_change(
        self,
        intent: str,
        old_workflow: List[str],
        new_workflow: List[str],
        description: str = "",
    ) -> PolicyProposal:
        """Create and evaluate a policy proposal.

        Args:
            intent: Intent this applies to
            old_workflow: Current workflow
            new_workflow: Proposed workflow
            description: Change description

        Returns:
            Policy proposal with evaluation
        """
        change_id = f"POL-{intent}-{len(self._episodes):03d}"

        change = PolicyChange(
            id=change_id,
            description=description or f"Change {intent} workflow from {old_workflow} to {new_workflow}",
            intent=intent,
            old_workflow=old_workflow,
            new_workflow=new_workflow,
        )

        evaluation = self.evaluate_change(change)

        return PolicyProposal(
            change=change,
            evaluation=evaluation,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

_default_evaluator: Optional[PolicyEvaluator] = None


def get_evaluator() -> PolicyEvaluator:
    """Get the default policy evaluator."""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = PolicyEvaluator()
    return _default_evaluator


def evaluate_policy_change(
    intent: str,
    old_workflow: List[str],
    new_workflow: List[str],
    description: str = "",
) -> PolicyProposal:
    """Evaluate a proposed policy change.

    Args:
        intent: Intent this applies to
        old_workflow: Current workflow
        new_workflow: Proposed workflow
        description: Change description

    Returns:
        Policy proposal with evaluation
    """
    evaluator = get_evaluator()
    evaluator.load_episodes_from_logger()
    return evaluator.propose_policy_change(intent, old_workflow, new_workflow, description)
