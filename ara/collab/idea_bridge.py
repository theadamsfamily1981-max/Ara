"""Bridge between Idea Board and Collaboration Layer.

This connects Ara's Idea Board with her ability to have dev-idea sessions.
Ideas can trigger collaborative refinement before execution.

Flow:
    1. Curiosity Core spawns an idea
    2. Idea enters Inbox for review
    3. When reviewing, Croft can say "Ara, let's refine this"
    4. Triggers a dev-idea session focused on that idea
    5. Session results update the idea with more detail
    6. Better-informed approval/rejection decision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from .models import (
    DevMode,
    DevSession,
    SessionSummary,
    SuggestedAction,
    RiskLevel,
)
from .session import DevIdeaSession

if TYPE_CHECKING:
    from ara.ideas import Idea, IdeaBoard

logger = logging.getLogger(__name__)


# =============================================================================
# Idea → DevMode Mapping
# =============================================================================

def mode_for_idea_category(category_name: str) -> DevMode:
    """Map idea category to appropriate dev mode.

    Args:
        category_name: The idea's category name

    Returns:
        Best DevMode for refining this type of idea
    """
    mapping = {
        "PERFORMANCE": DevMode.ENGINEER,
        "STABILITY": DevMode.POSTMORTEM,
        "UX": DevMode.BRAINSTORM,
        "SAFETY": DevMode.REVIEW,
        "RESEARCH": DevMode.RESEARCH,
        "WEIRD_IDEA": DevMode.BRAINSTORM,
        "MAINTENANCE": DevMode.ENGINEER,
        "INTEGRATION": DevMode.ARCHITECT,
    }
    return mapping.get(category_name, DevMode.ARCHITECT)


# =============================================================================
# IdeaSessionBridge
# =============================================================================

@dataclass
class RefinementResult:
    """Result of refining an idea through a dev session."""

    idea_id: str
    session_id: str
    summary: SessionSummary

    # Updates for the idea
    updated_hypothesis: Optional[str] = None
    updated_plan: Optional[List[str]] = None
    updated_risk: Optional[str] = None

    # New actions discovered
    actions: List[SuggestedAction] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []


class IdeaSessionBridge:
    """Bridges ideas with dev-idea sessions for collaborative refinement.

    When an idea needs more thought, this creates a dev session
    focused on that idea's hypothesis/plan and updates the idea
    with insights from collaborators.
    """

    def __init__(
        self,
        session_manager: DevIdeaSession,
        board: Optional["IdeaBoard"] = None,
    ):
        """Initialize the bridge.

        Args:
            session_manager: DevIdeaSession for running sessions
            board: Optional IdeaBoard to update ideas
        """
        self.session_manager = session_manager
        self.board = board

        # Track which ideas have been refined
        self._refinement_history: Dict[str, List[str]] = {}  # idea_id → session_ids

    def refine_idea(
        self,
        idea: "Idea",
        additional_context: str = "",
    ) -> RefinementResult:
        """Refine an idea through a dev session.

        This creates a collaborative session focused on the idea,
        then extracts insights to improve the idea.

        Args:
            idea: The idea to refine
            additional_context: Extra context from Croft

        Returns:
            RefinementResult with session findings
        """
        # Formulate the refinement request
        request = self._build_refinement_request(idea, additional_context)

        # Determine appropriate mode
        mode = mode_for_idea_category(idea.category.name if hasattr(idea.category, 'name') else str(idea.category))

        # Start session linked to this idea
        session = self.session_manager.start_session(
            user_request=request,
            mode=mode,
            idea_id=idea.id,
        )

        # Run the session
        summary = self.session_manager.run_session(session)

        # Track refinement
        if idea.id not in self._refinement_history:
            self._refinement_history[idea.id] = []
        self._refinement_history[idea.id].append(session.session_id)

        # Extract updates for the idea
        result = self._extract_idea_updates(idea, summary, session.session_id)

        # Update the idea if board is connected
        if self.board and result.updated_plan:
            self._apply_updates_to_idea(idea, result)

        return result

    def _build_refinement_request(
        self,
        idea: "Idea",
        additional_context: str,
    ) -> str:
        """Build the request text for refining an idea.

        Args:
            idea: The idea to refine
            additional_context: Extra context

        Returns:
            Request string for the dev session
        """
        parts = [
            f"I'm working on an idea: **{idea.title}**",
            "",
            f"Category: {idea.category.name if hasattr(idea.category, 'name') else idea.category}",
            f"Current risk assessment: {idea.risk.name if hasattr(idea.risk, 'name') else idea.risk}",
            "",
            "Hypothesis:",
            idea.hypothesis or "(none yet)",
            "",
            "Current plan:",
        ]

        if idea.plan:
            for i, step in enumerate(idea.plan, 1):
                parts.append(f"  {i}. {step}")
        else:
            parts.append("  (no plan yet)")

        parts.append("")
        parts.append("I want to refine this idea:")
        parts.append("- Is the hypothesis sound?")
        parts.append("- Are there better approaches?")
        parts.append("- What risks am I missing?")
        parts.append("- What's the concrete implementation plan?")

        if additional_context:
            parts.append("")
            parts.append(f"Additional context: {additional_context}")

        return "\n".join(parts)

    def _extract_idea_updates(
        self,
        idea: "Idea",
        summary: SessionSummary,
        session_id: str,
    ) -> RefinementResult:
        """Extract updates for the idea from session summary.

        Args:
            idea: The original idea
            summary: Session summary
            session_id: The session's ID

        Returns:
            RefinementResult with extracted updates
        """
        result = RefinementResult(
            idea_id=idea.id,
            session_id=session_id,
            summary=summary,
            actions=summary.actions,
        )

        # Update hypothesis if we have better insight
        if summary.consensus:
            result.updated_hypothesis = f"{idea.hypothesis or ''} [Refined: {summary.consensus}]".strip()

        # Update plan from options and actions
        if summary.options:
            result.updated_plan = [
                f"Consider: {opt}" for opt in summary.options[:3]
            ]
            # Add concrete actions
            for action in summary.actions[:3]:
                if action.action_type != "general":
                    result.updated_plan.append(f"Action: {action.description}")

        # Update risk if we found concerns
        high_risk_actions = [
            a for a in summary.actions
            if a.risk_level.value >= RiskLevel.HIGH.value
        ]
        if high_risk_actions:
            result.updated_risk = "HIGH"
        elif summary.trade_offs:
            result.updated_risk = "MEDIUM"

        return result

    def _apply_updates_to_idea(
        self,
        idea: "Idea",
        result: RefinementResult,
    ) -> None:
        """Apply refinement updates to the idea.

        Args:
            idea: The idea to update
            result: Refinement result with updates
        """
        try:
            updates = {}

            if result.updated_hypothesis:
                updates["hypothesis"] = result.updated_hypothesis

            if result.updated_plan:
                # Merge with existing plan
                existing_plan = idea.plan or []
                updates["plan"] = existing_plan + result.updated_plan

            if result.updated_risk and hasattr(idea, 'risk'):
                # Import here to avoid circular dependency
                try:
                    from ara.ideas import IdeaRisk
                    updates["risk"] = IdeaRisk[result.updated_risk]
                except (ImportError, KeyError):
                    pass

            if updates:
                self.board.update(idea.id, **updates)
                logger.info(f"Applied refinement updates to idea {idea.id}")

        except Exception as e:
            logger.error(f"Failed to apply updates to idea {idea.id}: {e}")

    def get_refinement_history(self, idea_id: str) -> List[str]:
        """Get session IDs for all refinements of an idea.

        Args:
            idea_id: The idea's ID

        Returns:
            List of session IDs
        """
        return self._refinement_history.get(idea_id, [])


# =============================================================================
# Convenience Functions
# =============================================================================

def refine_idea_with_session(
    idea: "Idea",
    session_manager: DevIdeaSession,
    context: str = "",
) -> RefinementResult:
    """Quick way to refine an idea through a dev session.

    Args:
        idea: The idea to refine
        session_manager: Session manager to use
        context: Additional context

    Returns:
        RefinementResult
    """
    bridge = IdeaSessionBridge(session_manager)
    return bridge.refine_idea(idea, context)
