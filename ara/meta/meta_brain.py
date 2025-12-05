"""Meta Brain - Ara's self-improvement coordination layer.

This is where Ara runs a research lab on Ara:
- Tracks research questions
- Runs experiments
- Generates suggestions
- Coordinates the meta-learning loop
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .schemas import (
    InteractionRecord,
    PatternSuggestion,
    ResearchQuestion,
    Experiment,
    ResearchAgenda,
)
from .meta_logger import MetaLogger, get_meta_logger
from .pattern_miner import PatternMiner, get_miner

logger = logging.getLogger(__name__)


class MetaBrain:
    """Ara's meta-learning coordination layer.

    Manages:
    - Research agenda
    - Pattern suggestions
    - Experiment tracking
    - Self-improvement loop
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        meta_logger: Optional[MetaLogger] = None,
        pattern_miner: Optional[PatternMiner] = None,
    ):
        """Initialize the meta brain.

        Args:
            data_dir: Directory for persisting state
            meta_logger: Logger to use
            pattern_miner: Miner to use
        """
        self.data_dir = data_dir or Path.home() / ".ara" / "meta"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.meta_logger = meta_logger or get_meta_logger()
        self.pattern_miner = pattern_miner or get_miner()

        # Paths
        self.agenda_path = self.data_dir / "research_agenda.json"
        self.suggestions_path = self.data_dir / "pending_suggestions.json"

        # State
        self.agenda = self._load_agenda()
        self.pending_suggestions: List[PatternSuggestion] = self._load_suggestions()

    def _load_agenda(self) -> ResearchAgenda:
        """Load research agenda from disk."""
        if self.agenda_path.exists():
            try:
                with open(self.agenda_path) as f:
                    data = json.load(f)
                return ResearchAgenda(**data)
            except Exception as e:
                logger.warning(f"Failed to load agenda: {e}")

        # Default agenda
        return ResearchAgenda(
            high_level_goal="Become a better collaborator and learner",
            open_questions=[
                ResearchQuestion(
                    id="RQ-001",
                    title="Which teacher is best for code vs architecture?",
                    hypothesis="Claude excels at code, Nova at architecture",
                    status="active",
                    metrics=["success_rate_by_task_type", "quality_by_tool"],
                ),
                ResearchQuestion(
                    id="RQ-002",
                    title="When should I ask for a second opinion?",
                    hypothesis="High-stakes or low-confidence situations benefit from consensus",
                    status="active",
                    metrics=["backtrack_rate", "multi_teacher_success_rate"],
                ),
                ResearchQuestion(
                    id="RQ-003",
                    title="How can I reduce round-trips?",
                    hypothesis="Better first prompts reduce clarification needs",
                    status="active",
                    metrics=["turns_to_solution", "prompt_exemplar_score"],
                ),
            ],
        )

    def _save_agenda(self) -> None:
        """Save research agenda to disk."""
        with open(self.agenda_path, "w") as f:
            json.dump(self.agenda.model_dump(), f, indent=2, default=str)

    def _load_suggestions(self) -> List[PatternSuggestion]:
        """Load pending suggestions from disk."""
        if self.suggestions_path.exists():
            try:
                with open(self.suggestions_path) as f:
                    data = json.load(f)
                return [PatternSuggestion(**s) for s in data]
            except Exception as e:
                logger.warning(f"Failed to load suggestions: {e}")
        return []

    def _save_suggestions(self) -> None:
        """Save pending suggestions to disk."""
        with open(self.suggestions_path, "w") as f:
            json.dump([s.model_dump() for s in self.pending_suggestions], f, indent=2, default=str)

    def get_status(self) -> Dict[str, Any]:
        """Get current meta-learning status.

        Returns:
            Status dictionary
        """
        # Analyze patterns
        analysis = self.pattern_miner.analyze()

        # Count various states
        active_questions = len(self.agenda.get_active_questions())
        running_experiments = len(self.agenda.get_running_experiments())
        pending_suggestions = len([s for s in self.pending_suggestions if s.status == "pending"])

        return {
            "goal": self.agenda.high_level_goal,
            "active_questions": active_questions,
            "running_experiments": running_experiments,
            "pending_suggestions": pending_suggestions,
            "pattern_analysis": analysis,
            "last_updated": self.agenda.last_updated.isoformat(),
        }

    def refresh_suggestions(self) -> List[PatternSuggestion]:
        """Refresh pattern-based suggestions.

        Returns:
            New suggestions
        """
        new_suggestions = self.pattern_miner.suggest_patterns()

        # Deduplicate against existing
        existing_ids = {s.id for s in self.pending_suggestions}
        for suggestion in new_suggestions:
            if suggestion.id not in existing_ids:
                self.pending_suggestions.append(suggestion)

        self._save_suggestions()
        return new_suggestions

    def get_pending_suggestions(
        self,
        scope: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[PatternSuggestion]:
        """Get pending suggestions.

        Args:
            scope: Filter by scope
            min_confidence: Minimum confidence

        Returns:
            Filtered suggestions
        """
        results = []
        for s in self.pending_suggestions:
            if s.status != "pending":
                continue
            if scope and s.scope != scope:
                continue
            if s.confidence < min_confidence:
                continue
            results.append(s)
        return results

    def apply_suggestion(self, suggestion_id: str) -> bool:
        """Mark a suggestion as applied.

        Args:
            suggestion_id: The suggestion ID

        Returns:
            True if found and updated
        """
        for s in self.pending_suggestions:
            if s.id == suggestion_id:
                s.status = "applied"
                s.applied_at = datetime.utcnow()
                self._save_suggestions()
                logger.info(f"Applied suggestion: {suggestion_id}")
                return True
        return False

    def reject_suggestion(self, suggestion_id: str, feedback: str = "") -> bool:
        """Reject a suggestion.

        Args:
            suggestion_id: The suggestion ID
            feedback: Optional feedback

        Returns:
            True if found and updated
        """
        for s in self.pending_suggestions:
            if s.id == suggestion_id:
                s.status = "rejected"
                s.user_feedback = feedback
                self._save_suggestions()
                logger.info(f"Rejected suggestion: {suggestion_id}")
                return True
        return False

    def add_research_question(
        self,
        title: str,
        hypothesis: str,
        metrics: Optional[List[str]] = None,
    ) -> ResearchQuestion:
        """Add a new research question.

        Args:
            title: Question title
            hypothesis: Initial hypothesis
            metrics: Metrics to track

        Returns:
            The new question
        """
        question = ResearchQuestion(
            id=f"RQ-{len(self.agenda.open_questions) + 1:03d}",
            title=title,
            hypothesis=hypothesis,
            metrics=metrics or [],
        )
        self.agenda.open_questions.append(question)
        self.agenda.last_updated = datetime.utcnow()
        self._save_agenda()
        return question

    def resolve_question(self, question_id: str, resolution: str) -> bool:
        """Resolve a research question.

        Args:
            question_id: The question ID
            resolution: What was learned

        Returns:
            True if found and resolved
        """
        question = self.agenda.get_question_by_id(question_id)
        if question:
            question.status = "resolved"
            question.resolved_at = datetime.utcnow()
            question.resolution = resolution
            self.agenda.last_updated = datetime.utcnow()
            self._save_agenda()
            return True
        return False

    def start_experiment(
        self,
        question_id: str,
        description: str,
    ) -> Experiment:
        """Start a new experiment.

        Args:
            question_id: Related research question
            description: Experiment description

        Returns:
            The new experiment
        """
        experiment = Experiment(
            id=f"EXP-{len(self.agenda.experiments) + 1:03d}",
            question_id=question_id,
            description=description,
            status="running",
            started_at=datetime.utcnow(),
        )
        self.agenda.experiments.append(experiment)
        self.agenda.last_updated = datetime.utcnow()
        self._save_agenda()
        return experiment

    def complete_experiment(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        conclusion: str,
    ) -> bool:
        """Complete an experiment.

        Args:
            experiment_id: The experiment ID
            results: Experiment results
            conclusion: What was learned

        Returns:
            True if found and completed
        """
        experiment = self.agenda.get_experiment_by_id(experiment_id)
        if experiment:
            experiment.status = "completed"
            experiment.completed_at = datetime.utcnow()
            experiment.results = results
            experiment.conclusion = conclusion
            self.agenda.last_updated = datetime.utcnow()
            self._save_agenda()
            return True
        return False

    def record_interaction(
        self,
        user_query: str,
        strategy: str,
        tools_used: List[Dict[str, Any]],
        outcome_quality: Optional[float] = None,
        context_tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        session_id: Optional[str] = None,
        issue_id: Optional[str] = None,
    ) -> InteractionRecord:
        """Record an interaction for meta-learning.

        Convenience method that creates and logs an interaction.

        Args:
            user_query: What was asked
            strategy: Strategy used
            tools_used: Tools called (list of dicts)
            outcome_quality: Quality score [0, 1]
            context_tags: Classification tags
            notes: Any notes
            session_id: Session ID
            issue_id: Issue ID

        Returns:
            The logged record
        """
        from .schemas import ToolCall

        record = InteractionRecord(
            user_query=user_query,
            chosen_strategy=strategy,
            context_tags=context_tags or [],
            outcome_quality=outcome_quality,
            notes=notes,
            session_id=session_id,
            issue_id=issue_id,
        )

        # Add tool calls
        for tc_data in tools_used:
            record.add_tool_call(ToolCall(**tc_data))

        self.meta_logger.log_interaction(record)
        return record

    def get_recommendations(self) -> Dict[str, Any]:
        """Get current recommendations based on analysis.

        Returns:
            Recommendations dictionary
        """
        self.pattern_miner.analyze()

        recommendations = {
            "tool_preferences": {},
            "strategies_to_try": [],
            "workflows_to_use": [],
            "warnings": [],
        }

        # Tool preferences
        rankings = self.pattern_miner.get_tool_ranking()
        if rankings:
            recommendations["tool_preferences"]["default"] = rankings[0][0]

        # Context-specific preferences
        for context, tools in self.pattern_miner._tool_by_context.items():
            if tools:
                best = max(tools.values(), key=lambda t: t.success_rate)
                if best.success_rate >= 0.7:
                    recommendations["tool_preferences"][context] = best.tool_name

        # Golden paths
        for pattern in self.pattern_miner.get_golden_paths():
            recommendations["workflows_to_use"].append({
                "pattern": pattern.pattern_id,
                "success_rate": pattern.success_rate,
            })

        # Failure warnings
        for failure in self.pattern_miner.get_failure_modes():
            recommendations["warnings"].append(failure["recommendation"])

        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

_default_brain: Optional[MetaBrain] = None


def get_meta_brain(data_dir: Optional[Path] = None) -> MetaBrain:
    """Get the default meta brain."""
    global _default_brain
    if _default_brain is None:
        _default_brain = MetaBrain(data_dir=data_dir)
    return _default_brain


def get_meta_status() -> Dict[str, Any]:
    """Get meta-learning status.

    Returns:
        Status dictionary
    """
    return get_meta_brain().get_status()


def refresh_meta_suggestions() -> List[PatternSuggestion]:
    """Refresh suggestions from pattern mining.

    Returns:
        New suggestions
    """
    return get_meta_brain().refresh_suggestions()


def get_meta_recommendations() -> Dict[str, Any]:
    """Get current recommendations.

    Returns:
        Recommendations dictionary
    """
    return get_meta_brain().get_recommendations()
