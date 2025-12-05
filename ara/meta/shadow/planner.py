"""Workflow Planner - Simulate and pick the optimal workflow.

Before pinging any real API, Ara does a "what-if planning" step:
1. Enumerate 2-3 candidate workflows
2. For each, run a shadow rollout
3. Pick the best under the objective: maximize(reward - α*latency - β*cost)

This is Ara doing mental simulation of conversations with her teachers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .profiles import TeacherFeatures, get_profile_manager
from .predictor import ShadowPredictor, Prediction, get_predictor


@dataclass
class SimulatedRollout:
    """Result of simulating a workflow."""

    workflow_id: str
    teachers: List[str]
    intent: str

    # Simulated outcomes
    expected_reward: float
    expected_latency_sec: float
    success_probability: float
    confidence: float

    # Per-teacher predictions
    step_predictions: List[Dict[str, Any]] = field(default_factory=list)

    # Computed utility
    utility: float = 0.0

    # Notes
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "teachers": self.teachers,
            "intent": self.intent,
            "expected_reward": round(self.expected_reward, 3),
            "expected_latency_sec": round(self.expected_latency_sec, 1),
            "success_probability": round(self.success_probability, 3),
            "confidence": round(self.confidence, 3),
            "utility": round(self.utility, 3),
            "notes": self.notes,
        }


@dataclass
class WorkflowPlan:
    """A plan for executing a workflow."""

    chosen_workflow: str
    teachers: List[str]
    intent: str
    features: Optional[TeacherFeatures]

    # Why this plan was chosen
    expected_reward: float
    expected_latency_sec: float
    utility: float
    confidence: float

    # Alternatives considered
    alternatives: List[SimulatedRollout] = field(default_factory=list)

    # Planning metadata
    planned_at: datetime = field(default_factory=datetime.utcnow)
    policy_version: str = "v1.0"
    planning_mode: str = "standard"  # "standard", "cheap", "thorough"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_workflow": self.chosen_workflow,
            "teachers": self.teachers,
            "intent": self.intent,
            "expected_reward": round(self.expected_reward, 3),
            "expected_latency_sec": round(self.expected_latency_sec, 1),
            "utility": round(self.utility, 3),
            "confidence": round(self.confidence, 3),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "planned_at": self.planned_at.isoformat(),
            "policy_version": self.policy_version,
            "planning_mode": self.planning_mode,
        }

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        lines = [
            f"Chosen: {' → '.join(self.teachers)}",
            f"  Expected reward: {self.expected_reward:.0%}",
            f"  Expected latency: {self.expected_latency_sec:.1f}s",
            f"  Confidence: {self.confidence:.0%}",
        ]

        if self.alternatives:
            lines.append("")
            lines.append("Alternatives considered:")
            for alt in self.alternatives[:2]:
                lines.append(f"  {' → '.join(alt.teachers)}: "
                            f"reward={alt.expected_reward:.0%}, "
                            f"latency={alt.expected_latency_sec:.1f}s")

        return "\n".join(lines)


class WorkflowPlanner:
    """Plans workflows by simulating different options.

    Enumerates candidate workflows, simulates each using shadow
    predictions, and picks the best one.
    """

    def __init__(
        self,
        predictor: Optional[ShadowPredictor] = None,
        alpha_latency: float = 0.01,
        beta_cost: float = 0.0,
    ):
        """Initialize the planner.

        Args:
            predictor: Shadow predictor to use
            alpha_latency: Penalty weight for latency
            beta_cost: Penalty weight for cost (not used yet)
        """
        self.predictor = predictor or get_predictor()
        self.alpha_latency = alpha_latency
        self.beta_cost = beta_cost

        # Policy version
        self.policy_version = "v1.0"

        # Default workflow templates by intent
        self.workflow_templates = {
            "debug_code": [
                ["claude"],
                ["claude", "nova"],
                ["nova", "claude"],
            ],
            "design_arch": [
                ["nova"],
                ["nova", "gemini"],
                ["gemini", "nova"],
            ],
            "research": [
                ["gemini"],
                ["gemini", "claude"],
                ["claude", "gemini"],
            ],
            "implement": [
                ["claude"],
                ["claude", "nova"],
            ],
            "refactor": [
                ["claude"],
                ["claude", "nova"],
                ["nova", "claude"],
            ],
            "review": [
                ["nova"],
                ["claude", "nova"],
            ],
            "general": [
                ["claude"],
                ["nova"],
                ["claude", "nova"],
            ],
        }

    def enumerate_workflows(
        self,
        intent: str,
        available_teachers: Optional[List[str]] = None,
        max_workflows: int = 5,
    ) -> List[List[str]]:
        """Enumerate candidate workflows for an intent.

        Args:
            intent: Intent classification
            available_teachers: Available teachers
            max_workflows: Maximum workflows to enumerate

        Returns:
            List of teacher sequences
        """
        if available_teachers is None:
            available_teachers = ["claude", "nova", "gemini"]

        # Get templates for this intent
        templates = self.workflow_templates.get(
            intent,
            self.workflow_templates["general"]
        )

        # Filter to available teachers
        workflows = []
        for template in templates:
            if all(t in available_teachers for t in template):
                workflows.append(template)

        # Add single-teacher options if not already present
        for teacher in available_teachers:
            if [teacher] not in workflows:
                workflows.append([teacher])

        return workflows[:max_workflows]

    def simulate_rollout(
        self,
        teachers: List[str],
        intent: str,
        features: Optional[TeacherFeatures] = None,
    ) -> SimulatedRollout:
        """Simulate a workflow rollout.

        Args:
            teachers: Teacher sequence
            intent: Intent classification
            features: Optional features

        Returns:
            Simulated rollout result
        """
        workflow_id = f"{intent}.{'->'.join(teachers)}"

        # Get predictions for each teacher
        step_predictions = []
        total_latency = 0.0
        combined_success = 1.0

        for teacher in teachers:
            pred = self.predictor.predict(teacher, intent, features)
            step_predictions.append(pred.to_dict())
            total_latency += pred.expected_latency_sec
            combined_success *= pred.success_probability

        # Combine rewards (weighted by position)
        if step_predictions:
            # Primary teacher gets full weight, subsequent get 30%
            weights = [1.0] + [0.3] * (len(step_predictions) - 1)
            total_weight = sum(weights)
            combined_reward = sum(
                w * p["expected_reward"]
                for w, p in zip(weights, step_predictions)
            ) / total_weight

            # Multi-teacher bonus
            if len(teachers) > 1:
                combined_reward = min(1.0, combined_reward * 1.05)

            # Average confidence
            combined_confidence = sum(
                p["confidence"] for p in step_predictions
            ) / len(step_predictions)
        else:
            combined_reward = 0.0
            combined_confidence = 0.0

        # Compute utility
        utility = combined_reward - self.alpha_latency * total_latency

        # Generate notes
        notes = []
        if len(teachers) == 1:
            notes.append("Single teacher - faster but less rigorous")
        elif len(teachers) == 2:
            notes.append("Two-teacher workflow - balanced speed/rigor")
        else:
            notes.append("Multi-teacher workflow - thorough but slow")

        if combined_confidence < 0.5:
            notes.append("Low confidence - limited historical data")

        return SimulatedRollout(
            workflow_id=workflow_id,
            teachers=teachers,
            intent=intent,
            expected_reward=combined_reward,
            expected_latency_sec=total_latency,
            success_probability=combined_success,
            confidence=combined_confidence,
            step_predictions=step_predictions,
            utility=utility,
            notes=notes,
        )

    def plan(
        self,
        intent: str,
        features: Optional[TeacherFeatures] = None,
        available_teachers: Optional[List[str]] = None,
        mode: str = "standard",
    ) -> WorkflowPlan:
        """Plan the optimal workflow.

        Args:
            intent: Intent classification
            features: Optional features
            available_teachers: Available teachers
            mode: Planning mode ("cheap", "standard", "thorough")

        Returns:
            The chosen plan
        """
        # Enumerate candidates
        workflows = self.enumerate_workflows(intent, available_teachers)

        # Adjust for mode
        if mode == "cheap":
            # Prefer single-teacher workflows
            workflows = [w for w in workflows if len(w) == 1] + workflows
        elif mode == "thorough":
            # Prefer multi-teacher workflows
            workflows = [w for w in workflows if len(w) > 1] + workflows

        # Simulate each
        rollouts = [
            self.simulate_rollout(teachers, intent, features)
            for teachers in workflows
        ]

        # Sort by utility
        rollouts.sort(key=lambda r: r.utility, reverse=True)

        # Choose the best
        chosen = rollouts[0]
        alternatives = rollouts[1:]

        return WorkflowPlan(
            chosen_workflow=chosen.workflow_id,
            teachers=chosen.teachers,
            intent=intent,
            features=features,
            expected_reward=chosen.expected_reward,
            expected_latency_sec=chosen.expected_latency_sec,
            utility=chosen.utility,
            confidence=chosen.confidence,
            alternatives=alternatives,
            policy_version=self.policy_version,
            planning_mode=mode,
        )

    def plan_from_query(
        self,
        query: str,
        intent: Optional[str] = None,
        mode: str = "standard",
    ) -> WorkflowPlan:
        """Plan workflow from a raw query.

        Args:
            query: User query
            intent: Pre-classified intent (auto-detected if None)
            mode: Planning mode

        Returns:
            The chosen plan
        """
        # Extract features
        features = TeacherFeatures.extract_from_query(query)

        # Auto-detect intent if not provided
        if not intent:
            from ..reflection import classify_intent
            intent = classify_intent(query)

        return self.plan(intent, features, mode=mode)

    def get_plan_explanation(self, plan: WorkflowPlan) -> str:
        """Generate a natural language explanation of the plan.

        Args:
            plan: The plan

        Returns:
            Explanation text
        """
        lines = []

        teachers_str = " → ".join(plan.teachers)
        lines.append(f"I'd go with: {teachers_str}")
        lines.append(f"(confidence {plan.confidence:.0%})")
        lines.append("")

        # Explain why
        reasons = []
        if plan.expected_reward >= 0.8:
            reasons.append("high expected success rate")
        if plan.expected_latency_sec < 15:
            reasons.append("relatively fast")
        if len(plan.teachers) == 1:
            reasons.append("simple and direct")
        elif len(plan.teachers) == 2:
            reasons.append("good balance of speed and rigor")

        if reasons:
            lines.append(f"Because: {', '.join(reasons)}.")

        # Mention alternatives
        if plan.alternatives:
            alt = plan.alternatives[0]
            alt_teachers = " → ".join(alt.teachers)
            lines.append("")
            lines.append(f"Alternative: {alt_teachers}")
            lines.append(f"  (reward {alt.expected_reward:.0%}, "
                        f"latency {alt.expected_latency_sec:.1f}s)")

            # Note trade-off
            if alt.expected_latency_sec < plan.expected_latency_sec:
                lines.append("  Trade-off: faster but potentially less thorough")
            elif alt.expected_reward > plan.expected_reward:
                lines.append("  Trade-off: higher reward but slower")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_planner: Optional[WorkflowPlanner] = None


def get_planner(
    alpha_latency: float = 0.01,
    beta_cost: float = 0.0,
) -> WorkflowPlanner:
    """Get the default workflow planner."""
    global _default_planner
    if _default_planner is None:
        _default_planner = WorkflowPlanner(
            alpha_latency=alpha_latency,
            beta_cost=beta_cost,
        )
    return _default_planner


def plan_workflow(
    intent: str,
    features: Optional[TeacherFeatures] = None,
    mode: str = "standard",
) -> WorkflowPlan:
    """Plan the optimal workflow for an intent.

    Args:
        intent: Intent classification
        features: Optional features
        mode: Planning mode

    Returns:
        The chosen plan
    """
    return get_planner().plan(intent, features, mode=mode)


def plan_from_query(query: str, mode: str = "standard") -> WorkflowPlan:
    """Plan workflow from a raw query.

    Args:
        query: User query
        mode: Planning mode

    Returns:
        The chosen plan
    """
    return get_planner().plan_from_query(query, mode=mode)
