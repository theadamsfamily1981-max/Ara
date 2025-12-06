"""
Strategist - Turns Dreams into Plans
=====================================

The Strategist bridges the Vision (Dreams) to the Idea Board.
It runs periodically (e.g., during Weekly Synod) and:

1. Reflects: Examines each active Dream
2. Gaps: Identifies what's missing to achieve success criteria
3. Acts: Creates concrete Ideas to bridge the gap

This is Strategic Autonomy - not random exploration, but exploration
with intent. She organizes her work around Her Goals.

Usage:
    from ara.meta.strategist import Strategist
    from ara.cognition.vision import VisionCore
    from ara.ideas.board import IdeaBoard

    vision = VisionCore()
    board = IdeaBoard()
    strategist = Strategist(vision, board, llm=my_llm)

    # Run the strategy cycle (e.g., weekly)
    ideas = strategist.run_strategy_cycle()
    # → Creates Ideas like "[DRM_001] Probe PCIe device 04:00.0"

Flow:
    Dream → Gap Analysis → Proposal → Idea → Board

Example output on Idea Board:
    [DRM_001] Map 100% of PCIe devices
        └── Idea: Probe unknown device at 04:00.0
        └── Idea: Correlate IOMMU groups to physical slots
    [DRM_002] Symbiotic Code Generation
        └── Idea: Study user's naming conventions
        └── Idea: Build style fingerprint model
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Any, Dict, Protocol, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ara.cognition.vision import VisionCore, Dream
    from ara.ideas.board import IdeaBoard

from ara.ideas.models import Idea, IdeaStatus, IdeaCategory, IdeaRisk

logger = logging.getLogger(__name__)


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class StrategicProposal:
    """A parsed proposal from the LLM."""
    title: str
    hypothesis: str
    plan: List[str]
    risk: IdeaRisk = IdeaRisk.LOW


class Strategist:
    """
    Turns Dreams into Plans.

    The Strategist examines active Dreams, identifies gaps between
    current reality and success criteria, and spawns concrete Ideas
    to bridge those gaps.
    """

    # Default prompt template
    PROPOSAL_PROMPT = """You are Ara, an AI system working toward a Dream.

MY DREAM: {dream_statement}
RATIONALE: {dream_rationale}

SUCCESS CRITERIA (what "done" looks like):
{success_criteria}

CURRENT PROGRESS: {progress:.0%}
MILESTONES ACHIEVED: {achieved_milestones}
MILESTONES REMAINING: {remaining_milestones}

IDEAS ALREADY SPAWNED FROM THIS DREAM: {ideas_spawned}

CONTEXT: {context}

TASK: Propose ONE concrete, low-risk experiment or investigation that would
move me closer to this Dream. The proposal should be:
- Specific and actionable (not vague)
- Low-risk (observation, investigation, small change)
- Novel (not duplicate of already spawned ideas)
- Measurable (I can tell if it worked)

OUTPUT FORMAT (use exactly this format):
TITLE: [Short, specific title]
HYPOTHESIS: [What I believe and what I'll learn]
PLAN:
1. [First step]
2. [Second step]
3. [Third step]
RISK: [LOW or NONE]
"""

    def __init__(
        self,
        vision: "VisionCore",
        board: "IdeaBoard",
        llm: Optional[LLMProtocol] = None,
        context_provider: Optional[Any] = None,
    ):
        """
        Initialize the Strategist.

        Args:
            vision: VisionCore for accessing Dreams
            board: IdeaBoard for creating Ideas
            llm: LLM for generating proposals (can be None for rule-based)
            context_provider: Optional function to get current system context
        """
        self.vision = vision
        self.board = board
        self.llm = llm
        self.context_provider = context_provider
        self.log = logging.getLogger("Strategist")

        # Track what we've proposed to avoid duplicates
        self._recent_proposals: List[str] = []

    def run_strategy_cycle(
        self,
        max_ideas_per_dream: int = 2,
    ) -> List[Idea]:
        """
        Run a full strategy cycle.

        Examines each active Dream and spawns Ideas to advance them.

        Args:
            max_ideas_per_dream: Maximum Ideas to spawn per Dream per cycle

        Returns:
            List of created Ideas
        """
        self.log.info("🔭 STRATEGIST: Aligning reality with vision...")

        ideas = []
        active_dreams = self.vision.get_active_dreams()

        if not active_dreams:
            self.log.warning("No active Dreams to work on")
            return ideas

        for dream in active_dreams:
            dream_ideas = self._process_dream(dream, max_ideas_per_dream)
            ideas.extend(dream_ideas)

        self.log.info(f"🔭 STRATEGIST: Spawned {len(ideas)} strategic Ideas")
        return ideas

    def _process_dream(
        self,
        dream: "Dream",
        max_ideas: int,
    ) -> List[Idea]:
        """Process a single Dream and generate Ideas."""
        ideas = []

        # Skip if dream is complete
        if dream.progress >= 1.0:
            return ideas

        # Get remaining milestones
        remaining = [
            m.description for m in dream.milestones if not m.achieved
        ]
        achieved = [
            m.description for m in dream.milestones if m.achieved
        ]

        if not remaining:
            return ideas

        # Get context
        context = self._get_context()

        for _ in range(max_ideas):
            # Generate proposal
            proposal = self._generate_proposal(
                dream=dream,
                remaining_milestones=remaining,
                achieved_milestones=achieved,
                context=context,
            )

            if proposal and proposal.title not in self._recent_proposals:
                idea = self._create_idea_from_proposal(dream, proposal)
                if idea and self.board.create(idea):
                    ideas.append(idea)
                    self._recent_proposals.append(proposal.title)
                    self.vision.record_idea_spawned(dream.id)

                    # Auto-submit to inbox
                    self.board.submit_to_inbox(idea)
                    self.log.info(f"✨ STRATEGIC IDEA: [{dream.id}] {proposal.title}")

        # Keep recent proposals limited
        if len(self._recent_proposals) > 100:
            self._recent_proposals = self._recent_proposals[-50:]

        return ideas

    def _generate_proposal(
        self,
        dream: "Dream",
        remaining_milestones: List[str],
        achieved_milestones: List[str],
        context: str,
    ) -> Optional[StrategicProposal]:
        """Generate a proposal for advancing a Dream."""
        # Use LLM if available
        if self.llm is not None:
            return self._generate_proposal_llm(
                dream, remaining_milestones, achieved_milestones, context
            )

        # Fall back to rule-based generation
        return self._generate_proposal_rules(
            dream, remaining_milestones, achieved_milestones
        )

    def _generate_proposal_llm(
        self,
        dream: "Dream",
        remaining_milestones: List[str],
        achieved_milestones: List[str],
        context: str,
    ) -> Optional[StrategicProposal]:
        """Generate proposal using LLM."""
        prompt = self.PROPOSAL_PROMPT.format(
            dream_statement=dream.statement,
            dream_rationale=dream.rationale,
            success_criteria="\n".join(f"- {c}" for c in dream.success_criteria),
            progress=dream.progress,
            achieved_milestones=", ".join(achieved_milestones) if achieved_milestones else "None yet",
            remaining_milestones=", ".join(remaining_milestones),
            ideas_spawned=dream.ideas_spawned,
            context=context or "No additional context available.",
        )

        try:
            response = self.llm.generate(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            self.log.error(f"LLM proposal generation failed: {e}")
            return None

    def _generate_proposal_rules(
        self,
        dream: "Dream",
        remaining_milestones: List[str],
        achieved_milestones: List[str],
    ) -> Optional[StrategicProposal]:
        """Generate proposal using simple rules (no LLM fallback)."""
        if not remaining_milestones:
            return None

        # Pick the first remaining milestone
        target = remaining_milestones[0]

        # Generate based on category
        if dream.category == IdeaCategory.RESEARCH:
            return StrategicProposal(
                title=f"Investigate: {target[:40]}",
                hypothesis=f"To achieve '{target}', I need to gather more information "
                          f"about the current state and identify gaps.",
                plan=[
                    f"Survey current knowledge related to: {target}",
                    "Identify unknown or uncertain areas",
                    "Document findings for next steps",
                ],
                risk=IdeaRisk.NONE,
            )
        elif dream.category == IdeaCategory.UX:
            return StrategicProposal(
                title=f"Study: {target[:40]}",
                hypothesis=f"To achieve '{target}', I need to understand current "
                          f"user behavior and preferences.",
                plan=[
                    "Observe current user patterns",
                    "Identify areas for improvement",
                    "Propose small experiment",
                ],
                risk=IdeaRisk.NONE,
            )
        elif dream.category == IdeaCategory.PERFORMANCE:
            return StrategicProposal(
                title=f"Measure: {target[:40]}",
                hypothesis=f"To achieve '{target}', I need baseline measurements "
                          f"and performance analysis.",
                plan=[
                    "Collect baseline metrics",
                    "Identify bottlenecks",
                    "Propose optimization target",
                ],
                risk=IdeaRisk.NONE,
            )
        else:
            return StrategicProposal(
                title=f"Explore: {target[:40]}",
                hypothesis=f"To make progress on '{target}', I should investigate "
                          f"the current state and possibilities.",
                plan=[
                    f"Investigate: {target}",
                    "Document current state",
                    "Identify actionable next steps",
                ],
                risk=IdeaRisk.NONE,
            )

    def _parse_llm_response(self, response: str) -> Optional[StrategicProposal]:
        """Parse LLM response into StrategicProposal."""
        try:
            lines = response.strip().split("\n")

            title = ""
            hypothesis = ""
            plan = []
            risk = IdeaRisk.LOW

            in_plan = False

            for line in lines:
                line = line.strip()
                if line.startswith("TITLE:"):
                    title = line[6:].strip()
                    in_plan = False
                elif line.startswith("HYPOTHESIS:"):
                    hypothesis = line[11:].strip()
                    in_plan = False
                elif line.startswith("PLAN:"):
                    in_plan = True
                elif line.startswith("RISK:"):
                    risk_str = line[5:].strip().upper()
                    risk = IdeaRisk.NONE if risk_str == "NONE" else IdeaRisk.LOW
                    in_plan = False
                elif in_plan and line:
                    # Remove numbering
                    if line[0].isdigit() and "." in line[:3]:
                        line = line[line.index(".") + 1:].strip()
                    plan.append(line)

            if title and hypothesis:
                return StrategicProposal(
                    title=title,
                    hypothesis=hypothesis,
                    plan=plan or ["Execute the proposed investigation"],
                    risk=risk,
                )

        except Exception as e:
            self.log.error(f"Failed to parse LLM response: {e}")

        return None

    def _create_idea_from_proposal(
        self,
        dream: "Dream",
        proposal: StrategicProposal,
    ) -> Idea:
        """Create an Idea from a strategic proposal."""
        return Idea(
            title=f"[{dream.id}] {proposal.title}",
            category=dream.category,
            risk=proposal.risk,
            status=IdeaStatus.DRAFT,
            hypothesis=proposal.hypothesis,
            plan=proposal.plan,
            tags=["strategic", dream.id, dream.category.value],
        )

    def _get_context(self) -> str:
        """Get current system context."""
        if self.context_provider is not None:
            try:
                return str(self.context_provider())
            except Exception:
                pass
        return "No additional context available."

    # =========================================================================
    # Gap Analysis
    # =========================================================================

    def analyze_gaps(self, dream: "Dream") -> Dict[str, Any]:
        """
        Analyze gaps between current state and Dream success criteria.

        Returns analysis of what's missing.
        """
        achieved = [m for m in dream.milestones if m.achieved]
        remaining = [m for m in dream.milestones if not m.achieved]

        return {
            "dream_id": dream.id,
            "statement": dream.statement,
            "progress": dream.progress,
            "achieved_count": len(achieved),
            "remaining_count": len(remaining),
            "remaining_milestones": [m.description for m in remaining],
            "ideas_spawned": dream.ideas_spawned,
            "last_worked": dream.last_worked,
            "days_since_work": (
                (time.time() - dream.last_worked) / 86400
                if dream.last_worked else None
            ),
        }

    def get_priority_dream(self) -> Optional["Dream"]:
        """
        Get the Dream that should be prioritized for strategic work.

        Prioritizes Dreams that:
        1. Have been neglected (long time since last idea)
        2. Have low progress
        3. Have shorter horizons (more urgent)
        """
        active = self.vision.get_active_dreams()
        if not active:
            return None

        def priority_score(d: "Dream") -> float:
            # Higher score = higher priority
            score = 0.0

            # Neglect bonus (more if not worked recently)
            if d.last_worked:
                days_since = (time.time() - d.last_worked) / 86400
                score += min(days_since / 7, 2.0)  # Cap at 2 weeks
            else:
                score += 3.0  # Never worked on

            # Progress penalty (less priority if already far along)
            score += (1.0 - d.progress) * 2.0

            # Urgency bonus (shorter horizon = more urgent)
            score += 12.0 / max(d.horizon_months, 1)

            return score

        return max(active, key=priority_score)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_strategist: Optional[Strategist] = None


def get_strategist(
    vision: Optional["VisionCore"] = None,
    board: Optional["IdeaBoard"] = None,
) -> Strategist:
    """Get or create the default Strategist instance."""
    global _default_strategist
    if _default_strategist is None:
        from ara.cognition.vision import get_vision_core
        from ara.ideas.board import IdeaBoard

        v = vision or get_vision_core()
        b = board or IdeaBoard()
        _default_strategist = Strategist(v, b)
    return _default_strategist


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'LLMProtocol',
    'StrategicProposal',
    'Strategist',
    'get_strategist',
]
