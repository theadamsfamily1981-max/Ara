"""Experiment Engine - A/B testing with user consent.

Ara can intentionally randomize her own behavior to learn faster.
For intents flagged as "under study", she'll:
- Randomly choose between competing policies
- Keep the user in the loop (never spooky)
- Log outcomes and update hypotheses

Example experiment notice:
    [ARA-EXPERIMENT NOTICE]
    I'm currently testing H1:
      "Claude→Nova vs Gemini→Claude→Nova for hardware debugging."
    For this request, I'd like to try: gemini→claude→nova
    You okay with that? [y/N]
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .programs import get_program_manager, ResearchProgram, Hypothesis

logger = logging.getLogger(__name__)


@dataclass
class ExperimentArm:
    """One arm of an experiment (control or treatment)."""

    name: str
    workflow: List[str]  # Teacher sequence
    weight: float = 0.5  # Allocation weight

    # Stats
    assignments: int = 0
    completions: int = 0
    total_reward: float = 0.0

    @property
    def avg_reward(self) -> Optional[float]:
        if self.completions == 0:
            return None
        return self.total_reward / self.completions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "workflow": self.workflow,
            "weight": self.weight,
            "assignments": self.assignments,
            "completions": self.completions,
            "avg_reward": self.avg_reward,
        }


@dataclass
class ExperimentDesign:
    """Design for an A/B (or A/B/C) experiment."""

    id: str
    program_id: str
    hypothesis_id: str
    description: str

    # Arms
    arms: List[ExperimentArm] = field(default_factory=list)

    # What triggers this experiment
    trigger_intent: str = ""
    trigger_condition: str = "always"  # "always", "complex_only", etc.

    # Status
    status: str = "active"  # "active", "paused", "concluded"
    created_at: datetime = field(default_factory=datetime.utcnow)
    min_samples_per_arm: int = 10

    def get_random_arm(self) -> ExperimentArm:
        """Randomly select an arm based on weights."""
        total_weight = sum(arm.weight for arm in self.arms)
        r = random.random() * total_weight
        cumulative = 0.0
        for arm in self.arms:
            cumulative += arm.weight
            if r <= cumulative:
                return arm
        return self.arms[-1]  # Fallback

    def record_assignment(self, arm_name: str) -> None:
        """Record that an arm was assigned."""
        for arm in self.arms:
            if arm.name == arm_name:
                arm.assignments += 1
                break

    def record_completion(self, arm_name: str, reward: float) -> None:
        """Record completion with reward."""
        for arm in self.arms:
            if arm.name == arm_name:
                arm.completions += 1
                arm.total_reward += reward
                break

    def is_ready_for_conclusion(self) -> bool:
        """Check if we have enough data to conclude."""
        return all(
            arm.completions >= self.min_samples_per_arm
            for arm in self.arms
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "program_id": self.program_id,
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "arms": [a.to_dict() for a in self.arms],
            "trigger_intent": self.trigger_intent,
            "trigger_condition": self.trigger_condition,
            "status": self.status,
            "min_samples_per_arm": self.min_samples_per_arm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentDesign":
        exp = cls(
            id=data["id"],
            program_id=data.get("program_id", ""),
            hypothesis_id=data.get("hypothesis_id", ""),
            description=data.get("description", ""),
            trigger_intent=data.get("trigger_intent", ""),
            trigger_condition=data.get("trigger_condition", "always"),
            status=data.get("status", "active"),
            min_samples_per_arm=data.get("min_samples_per_arm", 10),
        )
        for arm_data in data.get("arms", []):
            exp.arms.append(ExperimentArm(
                name=arm_data["name"],
                workflow=arm_data["workflow"],
                weight=arm_data.get("weight", 0.5),
                assignments=arm_data.get("assignments", 0),
                completions=arm_data.get("completions", 0),
                total_reward=arm_data.get("total_reward", 0.0),
            ))
        return exp


@dataclass
class ExperimentAssignment:
    """An assignment to an experiment arm for one episode."""

    experiment_id: str
    arm_name: str
    workflow: List[str]
    hypothesis_id: str
    program_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    requires_consent: bool = True

    def format_consent_message(self, hypothesis_statement: str = "") -> str:
        """Format a consent message for the user."""
        workflow_str = " → ".join(self.workflow)

        lines = [
            "[ARA-EXPERIMENT NOTICE]",
            "",
            f"I'm currently testing hypothesis {self.hypothesis_id}:",
        ]

        if hypothesis_statement:
            lines.append(f'  "{hypothesis_statement}"')

        lines.extend([
            "",
            f"For this request, I'd like to try: {workflow_str}",
            "",
            "I'll log:",
            "  - success/quality rating",
            "  - time to completion",
            "  - your satisfaction",
            "",
            "You okay with that? [y/N]",
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "arm_name": self.arm_name,
            "workflow": self.workflow,
            "hypothesis_id": self.hypothesis_id,
            "program_id": self.program_id,
            "assigned_at": self.assigned_at.isoformat(),
        }


class ExperimentController:
    """Controls experiment execution and tracking."""

    def __init__(self, experiments_path: Optional[Path] = None):
        """Initialize the experiment controller.

        Args:
            experiments_path: Path to experiments JSON file
        """
        self.experiments_path = experiments_path or (
            Path.home() / ".ara" / "meta" / "research" / "experiments.json"
        )
        self.experiments_path.parent.mkdir(parents=True, exist_ok=True)

        self._experiments: Dict[str, ExperimentDesign] = {}
        self._loaded = False

        # Current assignment (for tracking)
        self._current_assignment: Optional[ExperimentAssignment] = None

    def _load(self, force: bool = False) -> None:
        """Load experiments from disk."""
        if self._loaded and not force:
            return

        self._experiments.clear()

        if self.experiments_path.exists():
            try:
                with open(self.experiments_path) as f:
                    data = json.load(f)
                for exp_data in data.get("experiments", []):
                    exp = ExperimentDesign.from_dict(exp_data)
                    self._experiments[exp.id] = exp
            except Exception as e:
                logger.warning(f"Failed to load experiments: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save experiments to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "experiments": [e.to_dict() for e in self._experiments.values()],
        }
        with open(self.experiments_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentDesign]:
        """Get an experiment by ID."""
        self._load()
        return self._experiments.get(experiment_id)

    def get_active_experiments(self) -> List[ExperimentDesign]:
        """Get all active experiments."""
        self._load()
        return [e for e in self._experiments.values() if e.status == "active"]

    def create_experiment(
        self,
        experiment_id: str,
        program_id: str,
        hypothesis_id: str,
        description: str,
        trigger_intent: str,
        arms: List[Dict[str, Any]],
    ) -> ExperimentDesign:
        """Create a new experiment.

        Args:
            experiment_id: Unique ID
            program_id: Associated program
            hypothesis_id: Hypothesis being tested
            description: What we're testing
            trigger_intent: Intent that triggers this
            arms: List of arm configurations

        Returns:
            The new experiment
        """
        self._load()

        exp = ExperimentDesign(
            id=experiment_id,
            program_id=program_id,
            hypothesis_id=hypothesis_id,
            description=description,
            trigger_intent=trigger_intent,
        )

        for arm_data in arms:
            exp.arms.append(ExperimentArm(
                name=arm_data["name"],
                workflow=arm_data["workflow"],
                weight=arm_data.get("weight", 0.5),
            ))

        self._experiments[experiment_id] = exp
        self._save()
        logger.info(f"Created experiment: {experiment_id}")
        return exp

    def should_run_experiment(self, intent: str) -> Optional[ExperimentDesign]:
        """Check if an experiment should run for this intent.

        Args:
            intent: The intent classification

        Returns:
            Experiment to run, or None
        """
        self._load()

        for exp in self._experiments.values():
            if exp.status != "active":
                continue
            if exp.trigger_intent == intent:
                return exp

        return None

    def get_assignment(
        self,
        experiment: ExperimentDesign,
        auto_assign: bool = True,
    ) -> ExperimentAssignment:
        """Get an assignment for an experiment.

        Args:
            experiment: The experiment
            auto_assign: Whether to auto-select arm

        Returns:
            Assignment object
        """
        arm = experiment.get_random_arm()

        # Get hypothesis statement for consent message
        pm = get_program_manager()
        prog = pm.get_program(experiment.program_id)
        hypothesis_statement = ""
        if prog:
            hyp = prog.get_hypothesis(experiment.hypothesis_id)
            if hyp:
                hypothesis_statement = hyp.statement

        assignment = ExperimentAssignment(
            experiment_id=experiment.id,
            arm_name=arm.name,
            workflow=arm.workflow,
            hypothesis_id=experiment.hypothesis_id,
            program_id=experiment.program_id,
        )

        if auto_assign:
            experiment.record_assignment(arm.name)
            self._save()

        self._current_assignment = assignment
        return assignment

    def record_result(
        self,
        experiment_id: str,
        arm_name: str,
        reward: float,
        is_treatment: bool = True,
    ) -> None:
        """Record experiment result.

        Args:
            experiment_id: The experiment
            arm_name: Which arm was used
            reward: Outcome quality [0, 1]
            is_treatment: Whether this was treatment arm
        """
        self._load()

        exp = self._experiments.get(experiment_id)
        if not exp:
            return

        exp.record_completion(arm_name, reward)

        # Also update the hypothesis in the program
        pm = get_program_manager()
        pm.record_hypothesis_observation(
            program_id=exp.program_id,
            hypothesis_id=exp.hypothesis_id,
            is_treatment=is_treatment,
            metric_value=reward,
        )

        # Check if we can conclude
        if exp.is_ready_for_conclusion():
            self._check_experiment_conclusion(exp)

        self._save()

    def _check_experiment_conclusion(self, exp: ExperimentDesign) -> None:
        """Check if experiment can be concluded."""
        if len(exp.arms) < 2:
            return

        # Simple comparison of top two arms
        sorted_arms = sorted(
            exp.arms,
            key=lambda a: a.avg_reward or 0,
            reverse=True,
        )

        best = sorted_arms[0]
        second = sorted_arms[1]

        if best.avg_reward and second.avg_reward:
            diff = best.avg_reward - second.avg_reward
            if diff > 0.1:  # Significant difference
                exp.status = "concluded"
                logger.info(
                    f"Experiment {exp.id} concluded: {best.name} wins "
                    f"({best.avg_reward:.0%} vs {second.avg_reward:.0%})"
                )

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        self._load()

        active = [e for e in self._experiments.values() if e.status == "active"]
        concluded = [e for e in self._experiments.values() if e.status == "concluded"]

        return {
            "total_experiments": len(self._experiments),
            "active": len(active),
            "concluded": len(concluded),
            "experiments": [
                {
                    "id": e.id,
                    "description": e.description,
                    "status": e.status,
                    "arms": len(e.arms),
                    "total_completions": sum(a.completions for a in e.arms),
                }
                for e in self._experiments.values()
            ],
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_controller: Optional[ExperimentController] = None


def get_experiment_controller() -> ExperimentController:
    """Get the default experiment controller."""
    global _default_controller
    if _default_controller is None:
        _default_controller = ExperimentController()
    return _default_controller


def should_run_experiment(intent: str) -> Optional[ExperimentDesign]:
    """Check if an experiment should run for this intent."""
    return get_experiment_controller().should_run_experiment(intent)


def get_experiment_assignment(experiment: ExperimentDesign) -> ExperimentAssignment:
    """Get an assignment for an experiment."""
    return get_experiment_controller().get_assignment(experiment)


def record_experiment_result(
    experiment_id: str,
    arm_name: str,
    reward: float,
    is_treatment: bool = True,
) -> None:
    """Record experiment result."""
    get_experiment_controller().record_result(
        experiment_id, arm_name, reward, is_treatment
    )
