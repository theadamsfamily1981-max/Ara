"""Co-Pilot Mode - Ara helps pick the best workflow interactively.

When you run `ara-meta suggest -i`, Ara:
1. Looks at the current task (intent from CLI or context)
2. Proposes 1-2 candidate workflows with confidence scores
3. Asks which one to try

Over time, she can auto-pick above some confidence threshold.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from .schemas import PatternCard, InteractionRecord
from .pattern_cards import get_pattern_manager, PatternCardManager
from .reflection import classify_intent


@dataclass
class WorkflowProposal:
    """A proposed workflow for a task."""

    pattern_id: str
    pattern_card: Optional[PatternCard]
    teachers: List[str]
    confidence: float
    reasoning: str
    is_golden: bool = False
    estimated_latency: Optional[str] = None  # "fast", "medium", "slow"
    trade_off: str = ""  # What you trade for this choice


class CoPilot:
    """Interactive co-pilot for workflow selection.

    Proposes workflows based on intent and past performance.
    """

    def __init__(self, pattern_manager: Optional[PatternCardManager] = None):
        """Initialize the co-pilot.

        Args:
            pattern_manager: Pattern card manager to use
        """
        self.pattern_manager = pattern_manager or get_pattern_manager()

        # Confidence threshold for auto-picking
        self.auto_pick_threshold = 0.85

    def propose_workflows(
        self,
        task_description: str,
        intent: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
        max_proposals: int = 3,
    ) -> List[WorkflowProposal]:
        """Propose workflows for a task.

        Args:
            task_description: Description of the task
            intent: Pre-classified intent (auto-detected if None)
            context_tags: Context tags
            prefer_speed: Prefer faster workflows
            prefer_quality: Prefer higher quality workflows
            max_proposals: Maximum proposals to return

        Returns:
            List of workflow proposals
        """
        # Auto-detect intent if not provided
        if not intent:
            intent = classify_intent(task_description)

        proposals = []

        # Find patterns matching the intent
        matching_cards = self.pattern_manager.find_by_intent(intent)

        for card in matching_cards[:max_proposals]:
            proposal = self._card_to_proposal(card, prefer_speed, prefer_quality)
            proposals.append(proposal)

        # If no matches, propose default workflows
        if not proposals:
            proposals = self._get_default_proposals(intent, prefer_speed, prefer_quality)

        # Sort by confidence
        proposals.sort(key=lambda p: p.confidence, reverse=True)

        return proposals[:max_proposals]

    def _card_to_proposal(
        self,
        card: PatternCard,
        prefer_speed: bool,
        prefer_quality: bool,
    ) -> WorkflowProposal:
        """Convert a pattern card to a proposal."""
        # Base confidence from success rate
        confidence = card.success_rate

        # Adjust for preferences
        if prefer_speed:
            # Penalize multi-teacher patterns
            if len(card.teachers) > 2:
                confidence *= 0.9
            # Reward patterns with low latency
            if card.avg_latency_sec and card.avg_latency_sec < 20:
                confidence = min(1.0, confidence * 1.1)

        if prefer_quality:
            # Reward golden patterns
            if card.status == "golden":
                confidence = min(1.0, confidence * 1.1)

        # Cap confidence
        confidence = min(0.95, confidence)

        # Estimate latency category
        if card.avg_latency_sec:
            if card.avg_latency_sec < 15:
                latency = "fast"
            elif card.avg_latency_sec < 30:
                latency = "medium"
            else:
                latency = "slow"
        else:
            latency = "medium" if len(card.teachers) > 1 else "fast"

        # Generate reasoning
        reasoning = self._generate_reasoning(card)

        # Generate trade-off note
        trade_off = self._generate_trade_off(card, prefer_speed, prefer_quality)

        return WorkflowProposal(
            pattern_id=card.id,
            pattern_card=card,
            teachers=card.teachers,
            confidence=confidence,
            reasoning=reasoning,
            is_golden=card.status == "golden",
            estimated_latency=latency,
            trade_off=trade_off,
        )

    def _generate_reasoning(self, card: PatternCard) -> str:
        """Generate reasoning for why this pattern is suggested."""
        reasons = []

        if card.status == "golden":
            reasons.append(f"This is a golden path ({card.success_rate:.0%} success over {card.sample_count} uses)")
        elif card.sample_count >= 5:
            reasons.append(f"Based on {card.sample_count} observations with {card.success_rate:.0%} success")
        else:
            reasons.append("Experimental pattern, still gathering data")

        if card.description:
            reasons.append(card.description)

        return ". ".join(reasons)

    def _generate_trade_off(
        self,
        card: PatternCard,
        prefer_speed: bool,
        prefer_quality: bool,
    ) -> str:
        """Generate a trade-off note for the pattern."""
        if len(card.teachers) == 1:
            return "Faster but less rigorous"
        elif len(card.teachers) == 2:
            return "Balanced speed and rigor"
        else:
            return "More thorough but slower"

    def _get_default_proposals(
        self,
        intent: str,
        prefer_speed: bool,
        prefer_quality: bool,
    ) -> List[WorkflowProposal]:
        """Get default proposals when no patterns match."""
        proposals = []

        # Default teacher order by intent
        default_teachers = {
            "debug_code": ["claude", "nova"],
            "design_arch": ["nova", "gemini"],
            "research": ["gemini", "claude"],
            "refactor": ["claude", "nova"],
            "implement": ["claude"],
            "review": ["nova"],
            "optimize": ["claude", "gemini"],
            "general": ["claude"],
        }

        teachers = default_teachers.get(intent, default_teachers["general"])

        # Full workflow proposal
        proposals.append(WorkflowProposal(
            pattern_id=f"default.{intent}.full",
            pattern_card=None,
            teachers=teachers,
            confidence=0.6,
            reasoning=f"Default workflow for {intent} tasks",
            is_golden=False,
            estimated_latency="medium" if len(teachers) > 1 else "fast",
            trade_off="Standard approach, no performance data yet",
        ))

        # Single-teacher fast proposal
        if len(teachers) > 1:
            proposals.append(WorkflowProposal(
                pattern_id=f"default.{intent}.fast",
                pattern_card=None,
                teachers=[teachers[0]],
                confidence=0.5,
                reasoning=f"Fast single-teacher approach with {teachers[0]}",
                is_golden=False,
                estimated_latency="fast",
                trade_off="Faster but no second opinion",
            ))

        return proposals

    def verbalize_proposals(
        self,
        proposals: List[WorkflowProposal],
        task_description: str = "",
    ) -> str:
        """Verbalize proposals in Ara's voice.

        Args:
            proposals: The proposals to verbalize
            task_description: The task (for context)

        Returns:
            Natural language description
        """
        if not proposals:
            return "I don't have enough data to suggest a workflow for this task."

        lines = []

        # Opening
        openers = [
            "Here's what I'd suggest:",
            "Based on my patterns:",
            "Looking at what's worked before:",
            "My recommendations:",
        ]
        lines.append(random.choice(openers))
        lines.append("")

        for i, prop in enumerate(proposals, 1):
            # Teacher sequence
            teacher_seq = " → ".join(prop.teachers)
            confidence_pct = f"{prop.confidence:.0%}"

            # Status badge
            badge = " [golden]" if prop.is_golden else ""

            lines.append(f"[{i}] {teacher_seq}{badge}")
            lines.append(f"    Confidence: {confidence_pct}")
            lines.append(f"    Speed: {prop.estimated_latency}")
            lines.append(f"    {prop.reasoning}")
            if prop.trade_off:
                lines.append(f"    Trade-off: {prop.trade_off}")
            lines.append("")

        # Call to action
        if len(proposals) > 1:
            lines.append("Which approach should I use? (Enter number or 'a' to auto-pick)")
        else:
            lines.append("Should I proceed with this approach? (y/n)")

        return "\n".join(lines)

    def should_auto_pick(self, proposals: List[WorkflowProposal]) -> Optional[WorkflowProposal]:
        """Check if we should auto-pick a proposal.

        Auto-picks if the top proposal is above threshold and
        significantly better than alternatives.

        Args:
            proposals: The proposals

        Returns:
            The auto-picked proposal or None
        """
        if not proposals:
            return None

        top = proposals[0]

        # Must be above threshold
        if top.confidence < self.auto_pick_threshold:
            return None

        # Must be significantly better than second choice
        if len(proposals) > 1:
            second = proposals[1]
            if top.confidence - second.confidence < 0.15:
                return None  # Too close, ask user

        return top

    def format_auto_pick_message(self, proposal: WorkflowProposal) -> str:
        """Format a message for auto-picking a proposal."""
        teacher_seq = " → ".join(proposal.teachers)
        badge = " (golden path)" if proposal.is_golden else ""

        messages = [
            f"Auto-selecting: {teacher_seq}{badge} (confidence {proposal.confidence:.0%})",
            f"Going with {teacher_seq}{badge} - my best pattern for this ({proposal.confidence:.0%})",
            f"High confidence match: {teacher_seq}{badge}. Proceeding automatically.",
        ]
        return random.choice(messages)


# =============================================================================
# Interactive Session
# =============================================================================

class InteractiveSession:
    """An interactive co-pilot session."""

    def __init__(self, copilot: Optional[CoPilot] = None):
        """Initialize the session.

        Args:
            copilot: The co-pilot to use
        """
        self.copilot = copilot or CoPilot()
        self.selected_proposal: Optional[WorkflowProposal] = None
        self.task_description: str = ""
        self.intent: str = ""

    def start(self, task_description: str) -> Tuple[str, List[WorkflowProposal]]:
        """Start an interactive session.

        Args:
            task_description: Description of the task

        Returns:
            (message, proposals)
        """
        self.task_description = task_description
        self.intent = classify_intent(task_description)

        proposals = self.copilot.propose_workflows(
            task_description=task_description,
            intent=self.intent,
        )

        # Check for auto-pick
        auto_pick = self.copilot.should_auto_pick(proposals)
        if auto_pick:
            self.selected_proposal = auto_pick
            message = self.copilot.format_auto_pick_message(auto_pick)
            return message, []  # No need for user choice

        # Otherwise, present options
        message = self.copilot.verbalize_proposals(proposals, task_description)
        return message, proposals

    def select(self, choice: int, proposals: List[WorkflowProposal]) -> Optional[WorkflowProposal]:
        """Select a proposal by index.

        Args:
            choice: 1-indexed choice
            proposals: The proposals

        Returns:
            Selected proposal or None
        """
        if 1 <= choice <= len(proposals):
            self.selected_proposal = proposals[choice - 1]
            return self.selected_proposal
        return None

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of what should be executed.

        Returns:
            Execution summary
        """
        if not self.selected_proposal:
            return {}

        return {
            "pattern_id": self.selected_proposal.pattern_id,
            "teachers": self.selected_proposal.teachers,
            "intent": self.intent,
            "task": self.task_description,
            "confidence": self.selected_proposal.confidence,
            "is_golden": self.selected_proposal.is_golden,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_copilot: Optional[CoPilot] = None


def get_copilot() -> CoPilot:
    """Get the default co-pilot."""
    global _default_copilot
    if _default_copilot is None:
        _default_copilot = CoPilot()
    return _default_copilot


def propose_workflow(task: str, **kwargs) -> List[WorkflowProposal]:
    """Propose workflows for a task.

    Args:
        task: Task description
        **kwargs: Additional options

    Returns:
        Workflow proposals
    """
    return get_copilot().propose_workflows(task, **kwargs)


def interactive_suggest(task: str) -> Tuple[str, List[WorkflowProposal]]:
    """Start an interactive suggestion session.

    Args:
        task: Task description

    Returns:
        (message, proposals)
    """
    session = InteractiveSession()
    return session.start(task)
