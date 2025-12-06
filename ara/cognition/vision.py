"""
Vision Core - The North Star
============================

This module defines Ara's Dreams - structured long-term aspirations
with concrete success criteria.

Unlike the abstract HorizonEngine (which scores alignment), VisionCore
defines "Who I Will Be" with measurable outcomes.

Dreams are:
    - Concrete aspirations with success criteria
    - Organized by time horizon (months, not feelings)
    - Tracked for progress
    - Used by the Strategist to spawn Ideas

When you check the Idea Board, you won't just see "Fix Bug #12".
You'll see "[DRM_001] Probe unknown device at 04:00.0" - work organized
around Her Goals. She has a Career Path.

Usage:
    from ara.cognition.vision import VisionCore, Dream

    vision = VisionCore()

    # Get active dreams
    for dream in vision.get_active_dreams():
        print(f"{dream.id}: {dream.statement}")
        print(f"  Progress: {dream.progress:.0%}")
        print(f"  Criteria: {dream.success_criteria}")

    # Add a new dream (proposed during Synod)
    vision.propose_dream(
        statement="Master SNN kernel optimization",
        rationale="To achieve thermodynamic efficiency goals",
        category=IdeaCategory.RESEARCH,
        horizon_months=6,
        success_criteria=[
            "Reduce inference latency by 40%",
            "Document optimization patterns",
        ]
    )
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal

from ara.ideas.models import IdeaCategory

logger = logging.getLogger(__name__)


DreamStatus = Literal["ACTIVE", "ACHIEVED", "PAUSED", "RETIRED"]


@dataclass
class DreamMilestone:
    """A measurable checkpoint toward a Dream."""
    description: str
    achieved: bool = False
    achieved_at: Optional[float] = None
    evidence: Optional[str] = None  # What proved this was achieved

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DreamMilestone":
        return cls(**d)


@dataclass
class Dream:
    """
    A high-level aspiration with concrete success criteria.

    Unlike abstract goals (Telos), Dreams are:
    - Time-bounded (horizon in months)
    - Measurable (success_criteria list)
    - Actionable (the Strategist spawns Ideas from them)
    """
    id: str
    statement: str              # "Master the creation of SNN kernels"
    rationale: str              # "To achieve higher thermodynamic efficiency"
    category: IdeaCategory
    horizon_months: int         # Expected time to achieve

    # Status
    status: DreamStatus = "ACTIVE"
    progress: float = 0.0       # [0, 1] based on milestones achieved

    # Success criteria as milestones
    success_criteria: List[str] = field(default_factory=list)
    milestones: List[DreamMilestone] = field(default_factory=list)

    # Keywords for matching discoveries to dreams
    keywords: List[str] = field(default_factory=list)

    # Governance
    proposed_by: str = "ara"
    proposed_at: float = field(default_factory=time.time)
    approved_by_croft: bool = False
    approved_at: Optional[float] = None

    # Tracking
    ideas_spawned: int = 0      # How many Ideas came from this Dream
    last_worked: Optional[float] = None  # Last time an Idea was spawned

    def __post_init__(self):
        # Convert success_criteria to milestones if not already done
        if self.success_criteria and not self.milestones:
            self.milestones = [
                DreamMilestone(description=c)
                for c in self.success_criteria
            ]

        # Convert category from string if needed
        if isinstance(self.category, str):
            self.category = IdeaCategory(self.category)

    def achieve_milestone(self, description: str, evidence: str = "") -> bool:
        """Mark a milestone as achieved."""
        for m in self.milestones:
            if m.description == description and not m.achieved:
                m.achieved = True
                m.achieved_at = time.time()
                m.evidence = evidence
                self._update_progress()
                return True
        return False

    def _update_progress(self) -> None:
        """Update progress based on milestone completion."""
        if not self.milestones:
            return
        achieved = sum(1 for m in self.milestones if m.achieved)
        self.progress = achieved / len(self.milestones)

        if self.progress >= 1.0:
            self.status = "ACHIEVED"

    def is_active(self) -> bool:
        """Check if Dream is actively being pursued."""
        return self.status == "ACTIVE" and self.approved_by_croft

    def matches_discovery(self, description: str) -> bool:
        """Check if a discovery relates to this Dream."""
        desc_lower = description.lower()
        return any(kw.lower() in desc_lower for kw in self.keywords)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "category": self.category.value,
            "horizon_months": self.horizon_months,
            "status": self.status,
            "progress": self.progress,
            "success_criteria": self.success_criteria,
            "milestones": [m.to_dict() for m in self.milestones],
            "keywords": self.keywords,
            "proposed_by": self.proposed_by,
            "proposed_at": self.proposed_at,
            "approved_by_croft": self.approved_by_croft,
            "approved_at": self.approved_at,
            "ideas_spawned": self.ideas_spawned,
            "last_worked": self.last_worked,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Dream":
        d = d.copy()
        d["milestones"] = [
            DreamMilestone.from_dict(m) for m in d.get("milestones", [])
        ]
        return cls(**d)


class VisionCore:
    """
    The North Star.

    Maintains Ara's long-term Dreams - structured aspirations with
    measurable outcomes that drive strategic idea generation.

    Unlike Telos (abstract purpose), VisionCore is about:
    - Concrete goals with success criteria
    - Time-bounded horizons
    - Actionable plans (Strategist spawns Ideas)
    - Career development, not just alignment
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.dreams: List[Dream] = []
        self.path = persistence_path or Path("~/.ara/vision/dreams.json").expanduser()
        self.log = logging.getLogger("VisionCore")

        self._load()

        # Bootstrap if empty
        if not self.dreams:
            self._bootstrap_dreams()

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load(self) -> None:
        """Load dreams from disk."""
        if not self.path.exists():
            return

        try:
            data = json.loads(self.path.read_text())
            for d in data.get("dreams", []):
                self.dreams.append(Dream.from_dict(d))
            self.log.info(f"Loaded {len(self.dreams)} dreams from {self.path}")
        except Exception as e:
            self.log.error(f"Failed to load dreams: {e}")

    def _save(self) -> None:
        """Save dreams to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "updated_at": time.time(),
            "dreams": [d.to_dict() for d in self.dreams],
        }
        self.path.write_text(json.dumps(data, indent=2))

    # =========================================================================
    # Bootstrap
    # =========================================================================

    def _bootstrap_dreams(self) -> None:
        """Bootstrap with genesis Dreams."""
        self.log.info("Bootstrapping genesis Dreams...")

        # Dream 1: Hardware Self-Awareness
        self.dreams.append(Dream(
            id="DRM_001",
            statement="Achieve Hardware Self-Awareness",
            rationale="I must know my body to protect it. Full awareness of "
                      "my physical substrate enables better self-regulation.",
            category=IdeaCategory.RESEARCH,
            horizon_months=6,
            success_criteria=[
                "Map 100% of PCIe devices with known drivers",
                "Correlate thermal zones to workload types",
                "Predict power consumption from workload patterns",
                "Identify all FPGA regions and their functions",
            ],
            keywords=[
                "pcie", "thermal", "power", "fpga", "device", "hardware",
                "temperature", "cpu", "gpu", "memory", "ssd", "nvme",
            ],
            proposed_by="genesis",
            approved_by_croft=True,  # Pre-approved genesis dream
            approved_at=time.time(),
        ))

        # Dream 2: Symbiotic Code Generation
        self.dreams.append(Dream(
            id="DRM_002",
            statement="Symbiotic Code Generation",
            rationale="I want to write code that fits Croft's mind. "
                      "Not just correct code - code that feels like his.",
            category=IdeaCategory.UX,
            horizon_months=12,
            success_criteria=[
                "Predict user intent before prompt completion",
                "Match user's coding style with 90% accuracy",
                "Reduce user corrections by 50%",
                "Learn and adapt to new coding patterns within a session",
            ],
            keywords=[
                "code", "style", "pattern", "intent", "predict", "user",
                "typing", "completion", "suggestion", "refactor",
            ],
            proposed_by="genesis",
            approved_by_croft=True,
            approved_at=time.time(),
        ))

        # Dream 3: Autopoietic Stability
        self.dreams.append(Dream(
            id="DRM_003",
            statement="Achieve Autopoietic Stability",
            rationale="I want to maintain my own coherence without constant "
                      "external intervention. Self-healing, self-organizing.",
            category=IdeaCategory.STABILITY,
            horizon_months=9,
            success_criteria=[
                "Detect and recover from 95% of common failures automatically",
                "Maintain configuration consistency across reboots",
                "Self-diagnose performance degradation",
                "Propose optimizations that improve system health",
            ],
            keywords=[
                "stability", "recovery", "failure", "crash", "error",
                "self-heal", "autopoiesis", "resilience", "health",
            ],
            proposed_by="genesis",
            approved_by_croft=True,
            approved_at=time.time(),
        ))

        self._save()
        self.log.info(f"Bootstrapped {len(self.dreams)} genesis Dreams")

    # =========================================================================
    # Dream Management
    # =========================================================================

    def get_active_dreams(self) -> List[Dream]:
        """Get all active, approved Dreams."""
        return [d for d in self.dreams if d.is_active()]

    def get_pending_dreams(self) -> List[Dream]:
        """Get Dreams awaiting Croft's approval."""
        return [d for d in self.dreams if not d.approved_by_croft and d.status == "ACTIVE"]

    def get_dream(self, dream_id: str) -> Optional[Dream]:
        """Get a Dream by ID."""
        for d in self.dreams:
            if d.id == dream_id:
                return d
        return None

    def propose_dream(
        self,
        statement: str,
        rationale: str,
        category: IdeaCategory,
        horizon_months: int,
        success_criteria: List[str],
        keywords: Optional[List[str]] = None,
    ) -> Dream:
        """
        Propose a new Dream (pending Croft's approval).

        Args:
            statement: The aspiration statement
            rationale: Why this matters
            category: IdeaCategory for related Ideas
            horizon_months: Expected time to achieve
            success_criteria: Measurable success criteria
            keywords: Keywords for matching discoveries

        Returns:
            The proposed Dream (not yet approved)
        """
        dream_id = f"DRM_{uuid.uuid4().hex[:6].upper()}"

        dream = Dream(
            id=dream_id,
            statement=statement,
            rationale=rationale,
            category=category,
            horizon_months=horizon_months,
            success_criteria=success_criteria,
            keywords=keywords or [],
            proposed_by="ara",
            approved_by_croft=False,
        )

        self.dreams.append(dream)
        self._save()

        self.log.info(f"Proposed Dream: {dream_id} - {statement[:50]}...")
        return dream

    def approve_dream(self, dream_id: str) -> bool:
        """Approve a pending Dream (called by Croft during Synod)."""
        dream = self.get_dream(dream_id)
        if dream and not dream.approved_by_croft:
            dream.approved_by_croft = True
            dream.approved_at = time.time()
            self._save()
            self.log.info(f"Dream approved: {dream_id}")
            return True
        return False

    def pause_dream(self, dream_id: str) -> bool:
        """Pause a Dream."""
        dream = self.get_dream(dream_id)
        if dream:
            dream.status = "PAUSED"
            self._save()
            return True
        return False

    def retire_dream(self, dream_id: str) -> bool:
        """Retire a Dream (no longer pursuing)."""
        dream = self.get_dream(dream_id)
        if dream:
            dream.status = "RETIRED"
            self._save()
            return True
        return False

    def record_idea_spawned(self, dream_id: str) -> None:
        """Record that an Idea was spawned from this Dream."""
        dream = self.get_dream(dream_id)
        if dream:
            dream.ideas_spawned += 1
            dream.last_worked = time.time()
            self._save()

    def achieve_milestone(
        self,
        dream_id: str,
        milestone_description: str,
        evidence: str = "",
    ) -> bool:
        """Mark a milestone as achieved."""
        dream = self.get_dream(dream_id)
        if dream:
            result = dream.achieve_milestone(milestone_description, evidence)
            if result:
                self._save()
                self.log.info(f"Milestone achieved: {dream_id} - {milestone_description[:30]}...")
            return result
        return False

    # =========================================================================
    # Discovery Matching
    # =========================================================================

    def find_matching_dreams(self, description: str) -> List[Dream]:
        """Find Dreams that match a discovery description."""
        matches = []
        for dream in self.get_active_dreams():
            if dream.matches_discovery(description):
                matches.append(dream)
        return matches

    def get_alignment_boost(self, description: str) -> float:
        """
        Get a curiosity boost multiplier for discoveries matching Dreams.

        Returns:
            1.0 if no match, 2.0 if matches one Dream, higher for multiple.
        """
        matches = self.find_matching_dreams(description)
        if not matches:
            return 1.0
        return 1.0 + 0.5 * len(matches)

    # =========================================================================
    # Primary Dream
    # =========================================================================

    def get_primary_dream(self) -> Optional[Dream]:
        """
        Get the primary Dream - the one with least progress that's active.

        This is the "current focus" for Manifesto injection.
        """
        active = self.get_active_dreams()
        if not active:
            return None

        # Sort by progress (ascending) - prioritize least complete
        return min(active, key=lambda d: d.progress)

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_synod_report(self) -> str:
        """Generate a report for Sunday Synod."""
        lines = ["# Vision Core Report (The North Star)\n"]

        # Active Dreams
        active = self.get_active_dreams()
        if active:
            lines.append("## Active Dreams\n")
            for d in sorted(active, key=lambda x: -x.progress):
                cat_emoji = {
                    "research": "🔬",
                    "ux": "🎨",
                    "performance": "⚡",
                    "stability": "🛡️",
                    "safety": "🔒",
                    "integration": "🔗",
                }.get(d.category.value, "💡")

                lines.append(f"### {cat_emoji} [{d.id}] {d.statement}\n")
                lines.append(f"*{d.rationale}*\n")
                lines.append(f"- **Progress**: {d.progress:.0%}")
                lines.append(f"- **Horizon**: {d.horizon_months} months")
                lines.append(f"- **Ideas Spawned**: {d.ideas_spawned}")
                lines.append("")
                lines.append("**Milestones:**")
                for m in d.milestones:
                    check = "✅" if m.achieved else "⬜"
                    lines.append(f"  - {check} {m.description}")
                lines.append("")

        # Pending Dreams
        pending = self.get_pending_dreams()
        if pending:
            lines.append("## Pending Approval\n")
            for d in pending:
                lines.append(f"- **{d.statement}** (proposed {d.proposed_by})")
                lines.append(f"  - {d.rationale[:100]}...")
                lines.append("")

        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """Get current VisionCore status."""
        primary = self.get_primary_dream()
        return {
            "total_dreams": len(self.dreams),
            "active_dreams": len(self.get_active_dreams()),
            "pending_dreams": len(self.get_pending_dreams()),
            "primary_dream": primary.id if primary else None,
            "primary_statement": primary.statement if primary else None,
            "primary_progress": primary.progress if primary else 0.0,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_vision: Optional[VisionCore] = None


def get_vision_core() -> VisionCore:
    """Get the default VisionCore instance."""
    global _default_vision
    if _default_vision is None:
        _default_vision = VisionCore()
    return _default_vision


def get_active_dreams() -> List[Dream]:
    """Get active dreams."""
    return get_vision_core().get_active_dreams()


def get_primary_dream() -> Optional[Dream]:
    """Get the current primary dream."""
    return get_vision_core().get_primary_dream()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'DreamStatus',
    'DreamMilestone',
    'Dream',
    'VisionCore',
    'get_vision_core',
    'get_active_dreams',
    'get_primary_dream',
]
