"""Curiosity-to-Ideas Bridge - Transforms discoveries into proposals.

This module bridges the Curiosity Core and the Idea Board, allowing
discoveries and investigations to spawn structured ideas.

Flow:
    CuriosityAgent.tick() → CuriosityReport
                              ↓
    CuriosityBridge.process_report()
                              ↓
    Idea (if report is interesting enough)
                              ↓
    IdeaBoard.create()
"""

from __future__ import annotations

import logging
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ara.curiosity import CuriosityAgent, CuriosityReport, WorldObject
    from ara.cognition.vision import VisionCore
    from .board import IdeaBoard
    from .models import Idea

from .models import (
    Idea,
    IdeaCategory,
    IdeaRisk,
    IdeaStatus,
    Signal,
)

logger = logging.getLogger(__name__)


# Mapping from WorldObject categories to IdeaCategories
CATEGORY_MAP = {
    "PCIE_DEVICE": IdeaCategory.PERFORMANCE,
    "MEMORY_REGION": IdeaCategory.PERFORMANCE,
    "THERMAL_ZONE": IdeaCategory.STABILITY,
    "POWER_RAIL": IdeaCategory.STABILITY,
    "NETWORK_IFACE": IdeaCategory.INTEGRATION,
    "STORAGE_DEVICE": IdeaCategory.PERFORMANCE,
    "FPGA_REGION": IdeaCategory.RESEARCH,
    "KERNEL_MODULE": IdeaCategory.STABILITY,
    "PROCESS": IdeaCategory.SAFETY,
    "SENSOR": IdeaCategory.RESEARCH,
    "CXL_DEVICE": IdeaCategory.PERFORMANCE,
    "SNN_REGION": IdeaCategory.RESEARCH,
}


class CuriosityBridge:
    """Bridges curiosity discoveries to idea proposals.

    This class monitors CuriosityAgent output and spawns ideas
    when interesting discoveries are made.

    With VisionCore integration, discoveries that align with active
    Dreams get a curiosity boost - strategic exploration, not random.
    """

    def __init__(
        self,
        board: "IdeaBoard",
        min_curiosity_score: float = 0.5,
        auto_submit: bool = True,
        vision: Optional["VisionCore"] = None,
    ):
        """Initialize the bridge.

        Args:
            board: IdeaBoard to create ideas on
            min_curiosity_score: Minimum score to spawn an idea
            auto_submit: Automatically submit ideas to inbox
            vision: Optional VisionCore for Dream alignment boost
        """
        self.board = board
        self.min_score = min_curiosity_score
        self.auto_submit = auto_submit
        self.vision = vision

        # Track what we've already proposed
        self._proposed_objects: set = set()

    def process_report(self, report: "CuriosityReport") -> List[Idea]:
        """Process a curiosity report and spawn ideas.

        Args:
            report: CuriosityReport from CuriosityAgent

        Returns:
            List of created Ideas
        """
        ideas = []

        # Check if report has interesting content
        if not report.related_objects:
            return ideas

        # For each related object, consider spawning an idea
        for obj_id in report.related_objects:
            if obj_id in self._proposed_objects:
                continue

            idea = self._create_idea_from_report(report, obj_id)
            if idea:
                if self.board.create(idea):
                    self._proposed_objects.add(obj_id)
                    ideas.append(idea)

                    if self.auto_submit:
                        self.board.submit_to_inbox(idea)

        return ideas

    def _create_idea_from_report(
        self,
        report: "CuriosityReport",
        obj_id: str
    ) -> Optional[Idea]:
        """Create an idea from a curiosity report about an object."""
        # Extract object category from obj_id
        # Format: "category:identifier" e.g. "pcie:0000:01:00.0"
        parts = obj_id.split(":", 1)
        obj_type = parts[0].upper() if parts else "UNKNOWN"

        # Map to idea category
        category = CATEGORY_MAP.get(obj_type, IdeaCategory.RESEARCH)

        # Create title from report subject and object
        title = f"Investigate {obj_id}: {report.subject}"
        if len(title) > 80:
            title = title[:77] + "..."

        # Build hypothesis from report body
        hypothesis = report.body[:500] if report.body else "Needs investigation."

        # Create signals from report metrics if available
        signals = []
        if hasattr(report, 'metrics') and report.metrics:
            for name, value in report.metrics.items():
                signals.append(Signal(
                    name=name,
                    value=float(value) if isinstance(value, (int, float)) else 0,
                    source="curiosity"
                ))

        # Determine risk based on object type
        risk = IdeaRisk.NONE  # Default to observation only
        if obj_type in ("KERNEL_MODULE", "FPGA_REGION", "PROCESS"):
            risk = IdeaRisk.LOW  # These might need more care

        # Create the idea
        idea = Idea(
            title=title,
            category=category,
            risk=risk,
            status=IdeaStatus.DRAFT,
            hypothesis=hypothesis,
            plan=[
                f"Investigate {obj_id} further",
                "Collect relevant metrics",
                "Determine if action is needed",
            ],
            signals=signals,
            related_objects=[obj_id],
            tags=[obj_type.lower(), "curiosity", report.emotion],
        )

        return idea

    def process_discovery(
        self,
        obj: "WorldObject",
        curiosity_score: float
    ) -> Optional[Idea]:
        """Process a single discovery and maybe spawn an idea.

        Args:
            obj: Discovered WorldObject
            curiosity_score: Current curiosity score for this object

        Returns:
            Created Idea if interesting enough, None otherwise
        """
        # === VISION ALIGNMENT BOOST ===
        # Discoveries that match active Dreams get a curiosity boost
        # This turns random exploration into strategic exploration
        matched_dreams = []
        if self.vision is not None:
            obj_desc = f"{obj.name} {obj.category.name}"
            matched_dreams = self.vision.find_matching_dreams(obj_desc)
            if matched_dreams:
                boost = self.vision.get_alignment_boost(obj_desc)
                curiosity_score *= boost
                logger.info(
                    f"🌟 DISCOVERY ALIGNS WITH DREAM(S): {[d.id for d in matched_dreams]} "
                    f"(boost={boost:.1f}x)"
                )

        # Check score threshold (after boost)
        if curiosity_score < self.min_score:
            return None

        # Check if already proposed
        if obj.obj_id in self._proposed_objects:
            return None

        # Map category
        category = CATEGORY_MAP.get(obj.category.name, IdeaCategory.RESEARCH)

        # Determine risk
        risk = IdeaRisk.NONE
        if obj.category.name in ("KERNEL_MODULE", "FPGA_REGION", "PROCESS"):
            risk = IdeaRisk.LOW

        # Build title
        title = f"Discovered: {obj.name}"

        # Build hypothesis based on uncertainty
        if obj.uncertainty > 0.7:
            hypothesis = f"I found {obj.name} but I'm not sure what it does. " \
                        f"Uncertainty is high ({obj.uncertainty:.0%}). " \
                        f"I'd like to investigate further."
        else:
            hypothesis = f"I noticed {obj.name} ({obj.category.name}). " \
                        f"It seems important (importance: {obj.importance:.0%}). " \
                        f"Worth looking into."

        # Build signals from properties
        signals = []
        for key, value in list(obj.properties.items())[:5]:
            if isinstance(value, (int, float)):
                signals.append(Signal(
                    name=key,
                    value=float(value),
                    source=obj.obj_id
                ))

        # Create idea
        idea = Idea(
            title=title,
            category=category,
            risk=risk,
            status=IdeaStatus.DRAFT,
            hypothesis=hypothesis,
            plan=[
                f"Learn more about {obj.name}",
                f"Check {obj.category.name} status",
                "Update world model with findings",
            ],
            signals=signals,
            related_objects=[obj.obj_id],
            tags=self._build_discovery_tags(obj, matched_dreams),
        )

        # Create and submit
        if self.board.create(idea):
            self._proposed_objects.add(obj.obj_id)
            if self.auto_submit:
                self.board.submit_to_inbox(idea)
            return idea

        return None

    def _build_discovery_tags(
        self,
        obj: "WorldObject",
        matched_dreams: List = None,
    ) -> List[str]:
        """Build tags for a discovery idea."""
        tags = [obj.category.name.lower(), "discovery"]
        if matched_dreams:
            tags.append("strategic")
            for dream in matched_dreams:
                tags.append(dream.id)
        return tags

    def suggest_improvement(
        self,
        observation: str,
        component: str,
        signals: List[Signal],
        category: IdeaCategory = IdeaCategory.PERFORMANCE,
        risk: IdeaRisk = IdeaRisk.LOW
    ) -> Optional[Idea]:
        """Create an improvement suggestion idea.

        This is for when Ara notices something could be better,
        not just a discovery.

        Args:
            observation: What Ara observed
            component: What component it's about
            signals: Supporting metrics
            category: Idea category
            risk: Risk level

        Returns:
            Created Idea
        """
        title = f"I think {component} could be improved"

        # Build hypothesis
        if signals:
            sig_summary = ", ".join(f"{s.name}={s.value}{s.unit}" for s in signals[:3])
            hypothesis = f"I noticed {observation}. The metrics show: {sig_summary}. " \
                        f"I think we could make this better."
        else:
            hypothesis = f"I noticed {observation}. I have an idea for improvement."

        idea = Idea(
            title=title,
            category=category,
            risk=risk,
            status=IdeaStatus.DRAFT,
            hypothesis=hypothesis,
            plan=[
                "Analyze current behavior",
                "Propose specific change",
                "Test in sandbox",
                "Compare before/after metrics",
            ],
            rollback_plan=[
                "Revert configuration",
                "Verify original behavior restored",
            ],
            signals=signals,
            tags=[component.lower(), "improvement"],
        )

        if self.board.create(idea):
            if self.auto_submit:
                self.board.submit_to_inbox(idea)
            return idea

        return None

    def reset_proposed(self) -> None:
        """Clear the proposed objects set (for testing)."""
        self._proposed_objects.clear()


def create_bridge(board: "IdeaBoard", **kwargs) -> CuriosityBridge:
    """Factory function to create a CuriosityBridge.

    Args:
        board: IdeaBoard to use
        **kwargs: Additional arguments for CuriosityBridge

    Returns:
        Configured CuriosityBridge
    """
    return CuriosityBridge(board, **kwargs)
