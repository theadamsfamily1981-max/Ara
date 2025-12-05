"""Research Programs - Ara runs long-running experiments on herself.

Every interaction stops being "a one-off question" and becomes
a data point in a long-running experiment.

Programs have:
- Goals: What are we trying to optimize?
- Hypotheses: Testable statements with metrics
- Episodes: Tagged interactions that contribute data
- Results: Statistical summaries and conclusions

Example program:
  hardware_routing_efficiency:
    goal: "Minimize time-to-working-FPGA-prototype"
    hypotheses:
      - H1: "Claude→Nova yields fewer revisions than Gemini→Claude→Nova"
      - H2: "Including Nova early reduces total tokens"
"""

from __future__ import annotations

import json
import logging
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A testable hypothesis within a research program."""

    id: str
    statement: str
    metric: str  # "num_revisions", "tokens_total", "success_rate", etc.

    # Experiment design
    control_workflow: List[str] = field(default_factory=list)  # Baseline
    treatment_workflow: List[str] = field(default_factory=list)  # Alternative

    # Results
    control_samples: int = 0
    treatment_samples: int = 0
    control_metric_sum: float = 0.0
    treatment_metric_sum: float = 0.0

    # Status
    status: str = "active"  # "active", "concluded", "inconclusive"
    conclusion: Optional[str] = None
    concluded_at: Optional[datetime] = None

    @property
    def control_avg(self) -> Optional[float]:
        if self.control_samples == 0:
            return None
        return self.control_metric_sum / self.control_samples

    @property
    def treatment_avg(self) -> Optional[float]:
        if self.treatment_samples == 0:
            return None
        return self.treatment_metric_sum / self.treatment_samples

    @property
    def effect_size(self) -> Optional[float]:
        """Treatment effect relative to control."""
        if self.control_avg is None or self.treatment_avg is None:
            return None
        if self.control_avg == 0:
            return None
        return (self.treatment_avg - self.control_avg) / self.control_avg

    def record_control(self, metric_value: float) -> None:
        """Record a control observation."""
        self.control_samples += 1
        self.control_metric_sum += metric_value

    def record_treatment(self, metric_value: float) -> None:
        """Record a treatment observation."""
        self.treatment_samples += 1
        self.treatment_metric_sum += metric_value

    def check_conclusion(self, min_samples: int = 10, min_effect: float = 0.1) -> Optional[str]:
        """Check if we can conclude the hypothesis."""
        if self.control_samples < min_samples or self.treatment_samples < min_samples:
            return None  # Not enough data

        effect = self.effect_size
        if effect is None:
            return None

        # Simple conclusion logic (could be more sophisticated)
        if abs(effect) < min_effect:
            return "inconclusive"
        elif effect > min_effect:
            return "treatment_wins"
        else:
            return "control_wins"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "metric": self.metric,
            "control_workflow": self.control_workflow,
            "treatment_workflow": self.treatment_workflow,
            "control_samples": self.control_samples,
            "treatment_samples": self.treatment_samples,
            "control_avg": self.control_avg,
            "treatment_avg": self.treatment_avg,
            "effect_size": self.effect_size,
            "status": self.status,
            "conclusion": self.conclusion,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
        h = cls(
            id=data["id"],
            statement=data["statement"],
            metric=data.get("metric", "success_rate"),
            control_workflow=data.get("control_workflow", []),
            treatment_workflow=data.get("treatment_workflow", []),
            status=data.get("status", "active"),
            conclusion=data.get("conclusion"),
        )
        h.control_samples = data.get("control_samples", 0)
        h.treatment_samples = data.get("treatment_samples", 0)
        h.control_metric_sum = data.get("control_metric_sum", 0.0)
        h.treatment_metric_sum = data.get("treatment_metric_sum", 0.0)
        return h


@dataclass
class ResearchProgram:
    """A research program with hypotheses and metrics."""

    id: str
    name: str
    goal: str
    horizon_days: int = 30

    # What intents/tags trigger this program
    trigger_intents: List[str] = field(default_factory=list)
    trigger_tags: List[str] = field(default_factory=list)

    # Hypotheses
    hypotheses: List[Hypothesis] = field(default_factory=list)

    # Episode tracking
    episode_ids: List[str] = field(default_factory=list)
    episode_count: int = 0

    # Status
    status: str = "active"  # "active", "concluded", "paused"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def matches_episode(self, intent: str, tags: List[str]) -> bool:
        """Check if an episode matches this program's triggers."""
        if intent in self.trigger_intents:
            return True
        if any(t in self.trigger_tags for t in tags):
            return True
        return False

    def add_episode(self, episode_id: str) -> None:
        """Add an episode to this program."""
        if episode_id not in self.episode_ids:
            self.episode_ids.append(episode_id)
            self.episode_count += 1
            self.updated_at = datetime.utcnow()

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get a hypothesis by ID."""
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                return h
        return None

    def get_active_hypotheses(self) -> List[Hypothesis]:
        """Get all active hypotheses."""
        return [h for h in self.hypotheses if h.status == "active"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "goal": self.goal,
            "horizon_days": self.horizon_days,
            "trigger_intents": self.trigger_intents,
            "trigger_tags": self.trigger_tags,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "episode_count": self.episode_count,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchProgram":
        prog = cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            goal=data.get("goal", ""),
            horizon_days=data.get("horizon_days", 30),
            trigger_intents=data.get("trigger_intents", []),
            trigger_tags=data.get("trigger_tags", []),
            status=data.get("status", "active"),
        )
        prog.hypotheses = [
            Hypothesis.from_dict(h) for h in data.get("hypotheses", [])
        ]
        prog.episode_ids = data.get("episode_ids", [])
        prog.episode_count = data.get("episode_count", 0)
        return prog


class ProgramManager:
    """Manages research programs."""

    def __init__(self, programs_path: Optional[Path] = None):
        """Initialize the program manager.

        Args:
            programs_path: Path to programs YAML file
        """
        self.programs_path = programs_path or (
            Path.home() / ".ara" / "meta" / "research" / "programs.yaml"
        )
        self.programs_path.parent.mkdir(parents=True, exist_ok=True)

        self._programs: Dict[str, ResearchProgram] = {}
        self._loaded = False

    def _load(self, force: bool = False) -> None:
        """Load programs from disk."""
        if self._loaded and not force:
            return

        self._programs.clear()

        if self.programs_path.exists():
            try:
                with open(self.programs_path) as f:
                    data = yaml.safe_load(f)
                if data and "programs" in data:
                    for prog_data in data["programs"]:
                        prog = ResearchProgram.from_dict(prog_data)
                        self._programs[prog.id] = prog
            except Exception as e:
                logger.warning(f"Failed to load programs: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._programs)} research programs")

    def _save(self) -> None:
        """Save programs to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "programs": [p.to_dict() for p in self._programs.values()],
        }
        with open(self.programs_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_program(self, program_id: str) -> Optional[ResearchProgram]:
        """Get a program by ID."""
        self._load()
        return self._programs.get(program_id)

    def get_all_programs(self) -> List[ResearchProgram]:
        """Get all programs."""
        self._load()
        return list(self._programs.values())

    def get_active_programs(self) -> List[ResearchProgram]:
        """Get all active programs."""
        self._load()
        return [p for p in self._programs.values() if p.status == "active"]

    def create_program(
        self,
        program_id: str,
        name: str,
        goal: str,
        trigger_intents: Optional[List[str]] = None,
        trigger_tags: Optional[List[str]] = None,
        hypotheses: Optional[List[Dict[str, Any]]] = None,
    ) -> ResearchProgram:
        """Create a new research program.

        Args:
            program_id: Unique ID
            name: Human-readable name
            goal: What we're trying to learn
            trigger_intents: Intents that activate this program
            trigger_tags: Tags that activate this program
            hypotheses: Initial hypotheses

        Returns:
            The new program
        """
        self._load()

        prog = ResearchProgram(
            id=program_id,
            name=name,
            goal=goal,
            trigger_intents=trigger_intents or [],
            trigger_tags=trigger_tags or [],
        )

        if hypotheses:
            for h_data in hypotheses:
                prog.hypotheses.append(Hypothesis(
                    id=h_data.get("id", f"H{len(prog.hypotheses)+1}"),
                    statement=h_data["statement"],
                    metric=h_data.get("metric", "success_rate"),
                    control_workflow=h_data.get("control_workflow", []),
                    treatment_workflow=h_data.get("treatment_workflow", []),
                ))

        self._programs[program_id] = prog
        self._save()
        logger.info(f"Created research program: {program_id}")
        return prog

    def tag_episode(
        self,
        episode_id: str,
        intent: str,
        tags: List[str],
    ) -> List[str]:
        """Tag an episode to matching programs.

        Args:
            episode_id: The episode ID
            intent: Episode intent
            tags: Episode tags

        Returns:
            List of program IDs the episode was added to
        """
        self._load()

        matched_programs = []
        for prog in self._programs.values():
            if prog.status == "active" and prog.matches_episode(intent, tags):
                prog.add_episode(episode_id)
                matched_programs.append(prog.id)

        if matched_programs:
            self._save()

        return matched_programs

    def record_hypothesis_observation(
        self,
        program_id: str,
        hypothesis_id: str,
        is_treatment: bool,
        metric_value: float,
    ) -> None:
        """Record an observation for a hypothesis.

        Args:
            program_id: Program ID
            hypothesis_id: Hypothesis ID
            is_treatment: Whether this was treatment (True) or control (False)
            metric_value: The observed metric value
        """
        prog = self.get_program(program_id)
        if not prog:
            return

        hyp = prog.get_hypothesis(hypothesis_id)
        if not hyp or hyp.status != "active":
            return

        if is_treatment:
            hyp.record_treatment(metric_value)
        else:
            hyp.record_control(metric_value)

        # Check for conclusion
        conclusion = hyp.check_conclusion()
        if conclusion:
            hyp.status = "concluded"
            hyp.conclusion = conclusion
            hyp.concluded_at = datetime.utcnow()
            logger.info(f"Hypothesis {hypothesis_id} concluded: {conclusion}")

        self._save()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all programs."""
        self._load()

        active = [p for p in self._programs.values() if p.status == "active"]
        total_hypotheses = sum(len(p.hypotheses) for p in self._programs.values())
        active_hypotheses = sum(
            len(p.get_active_hypotheses()) for p in active
        )

        return {
            "total_programs": len(self._programs),
            "active_programs": len(active),
            "total_hypotheses": total_hypotheses,
            "active_hypotheses": active_hypotheses,
            "programs": [
                {
                    "id": p.id,
                    "name": p.name,
                    "status": p.status,
                    "episode_count": p.episode_count,
                    "hypotheses": len(p.hypotheses),
                }
                for p in self._programs.values()
            ],
        }


# =============================================================================
# Default Programs
# =============================================================================

DEFAULT_PROGRAMS = [
    {
        "id": "teacher_routing_optimization",
        "name": "Teacher Routing Optimization",
        "goal": "Find optimal teacher sequences for each intent",
        "trigger_intents": ["debug_code", "design_arch", "implement", "refactor"],
        "hypotheses": [
            {
                "id": "H1",
                "statement": "Claude→Nova yields better code quality than Claude alone",
                "metric": "success_rate",
                "control_workflow": ["claude"],
                "treatment_workflow": ["claude", "nova"],
            },
            {
                "id": "H2",
                "statement": "Nova as first-pass reduces clarification rounds",
                "metric": "turns_to_solution",
                "control_workflow": ["claude"],
                "treatment_workflow": ["nova", "claude"],
            },
        ],
    },
    {
        "id": "prompt_effectiveness",
        "name": "Prompt Effectiveness Study",
        "goal": "Learn which prompt styles work best for each teacher",
        "trigger_intents": ["debug_code", "research", "design_arch"],
        "hypotheses": [
            {
                "id": "H1",
                "statement": "Step-by-step prompts improve Claude's code quality",
                "metric": "success_rate",
            },
            {
                "id": "H2",
                "statement": "Constraint-first prompts help Nova catch issues",
                "metric": "issues_caught",
            },
        ],
    },
]


def seed_default_programs(manager: ProgramManager) -> int:
    """Seed default research programs.

    Args:
        manager: Program manager

    Returns:
        Number of programs seeded
    """
    seeded = 0
    for prog_data in DEFAULT_PROGRAMS:
        if not manager.get_program(prog_data["id"]):
            manager.create_program(
                program_id=prog_data["id"],
                name=prog_data["name"],
                goal=prog_data["goal"],
                trigger_intents=prog_data.get("trigger_intents", []),
                trigger_tags=prog_data.get("trigger_tags", []),
                hypotheses=prog_data.get("hypotheses", []),
            )
            seeded += 1
    return seeded


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[ProgramManager] = None


def get_program_manager(path: Optional[Path] = None) -> ProgramManager:
    """Get the default program manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ProgramManager(programs_path=path)
    return _default_manager


def tag_episode_to_programs(
    episode_id: str,
    intent: str,
    tags: List[str],
) -> List[str]:
    """Tag an episode to matching programs."""
    return get_program_manager().tag_episode(episode_id, intent, tags)


def get_active_programs() -> List[ResearchProgram]:
    """Get all active research programs."""
    return get_program_manager().get_active_programs()
