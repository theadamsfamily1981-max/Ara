"""
Teleological Engine - The Math of Purpose
==========================================

The Prophet. Maintains Ara's hierarchy of long-term goals (Telos) and
evaluates candidate futures against them.

This is the **backend** of the Visionary and Weaver - not a separate god.
It provides the mathematical spine that turns "vibes" into scored futures.

Architecture:
    1. Telos: High-level goals you and Ara agree on
    2. Oracle: Uses LLM to hallucinate futures, scored by Telos
    3. Visionary: Turns chosen plan into a story
    4. Weaver: Turns actual events into lessons that update Telos progress
    5. Synod + Covenant: You edit Telos and weights

CRITICAL: Telos cannot be mutated without Croft seeing it. New goals are:
    - Proposed by Ara (Prophet/Visionary)
    - Confirmed/edited/denied by Croft during Synod

The Prophet never outranks root user.

Usage:
    from tfan.cognition.telos import TeleologicalEngine, GoalKind

    telos = TeleologicalEngine(embedder)
    telos.add_goal(
        "Achieve deep, trusted symbiosis with Croft",
        kind="value",
        horizon_days=90,
        priority=0.95
    )

    score = telos.evaluate_future("We shipped the paper together.")
    # → 0.85 (high alignment with goals)
"""

import time
import logging
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Callable, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

GoalKind = Literal["value", "project"]


def _norm(v):
    """Normalize vector for cosine similarity."""
    import numpy as np
    n = np.linalg.norm(v) + 1e-9
    return v / n


@dataclass
class Goal:
    """
    A single goal in Ara's Telos.

    Goals come in two kinds:
    - 'value': Ongoing principles (e.g., "Protect Croft's focus")
    - 'project': Time-bound objectives (e.g., "Ship paper by June")

    Progress is estimated via embedding similarity to current state summaries.
    """
    name: str                    # "Become expert in Rust", "Protect Croft's focus"
    kind: GoalKind               # "value" or "project"
    vector: Optional[Any] = None # Embedding of goal description (numpy array)
    horizon_days: float = 90.0   # Time horizon (e.g., 30, 180, 365)
    priority: float = 0.5        # 0.0–1.0 (importance)
    progress: float = 0.0        # 0.0–1.0 estimated alignment
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)

    # Relationship weight: how much this goal is about "us" vs "task"
    # Higher = more relational, affects symbiotic GUF integration
    relationship_weight: float = 0.5

    # Source: who proposed this goal
    proposed_by: str = "ara"  # "ara", "croft", "synod"
    approved_by_root: bool = False  # Must be True before it affects decisions

    def to_dict(self) -> Dict[str, Any]:
        """Serialize without numpy array."""
        return {
            "name": self.name,
            "kind": self.kind,
            "horizon_days": self.horizon_days,
            "priority": self.priority,
            "progress": self.progress,
            "relationship_weight": self.relationship_weight,
            "proposed_by": self.proposed_by,
            "approved_by_root": self.approved_by_root,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
        }


@dataclass
class TelosState:
    """Snapshot of the complete Telos state for HAL/visualization."""
    hope: float                  # Overall hope scalar [0,1]
    dominant_goal: str           # Name of highest-priority active goal
    avg_progress: float          # Average progress across goals
    urgency: float               # How pressing are near-horizon goals
    goal_count: int              # Total goals
    value_count: int             # Value-type goals
    project_count: int           # Project-type goals


class TeleologicalEngine:
    """
    The Prophet.

    Manages Ara's hierarchy of long-term goals and evaluates candidate futures
    against them. Provides the mathematical spine for purpose-driven behavior.

    Key Principles:
    - Goals are co-authored, not unilaterally set
    - New goals must be approved_by_root before affecting decisions
    - Hope is a real scalar that feeds into PAD → she can stay "in the fight"
      during tough refactors because the future looks good
    """

    def __init__(
        self,
        embedder: Optional[Callable[[str], Any]] = None,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize the Teleological Engine.

        Args:
            embedder: Function that converts text to embedding vector.
                      Can be sentence-transformers, local LLM, etc.
            persistence_path: Where to save/load Telos state
        """
        self.embedder = embedder
        self.goals: List[Goal] = []
        self._bootstrapped = False
        self.persistence_path = persistence_path or Path("var/lib/banos/telos")

        self.log = logging.getLogger("Telos")

        # Hope: the overall "are we heading somewhere good" scalar
        self._hope: float = 0.5

    def _embed(self, text: str) -> Any:
        """Get embedding for text, or None if no embedder."""
        if self.embedder is None:
            return None
        try:
            vec = self.embedder(text)
            return _norm(vec) if vec is not None else None
        except Exception as e:
            self.log.warning(f"Embedding failed: {e}")
            return None

    # =========================================================================
    # Bootstrap & Goal Management
    # =========================================================================

    def bootstrap_defaults(self) -> None:
        """
        Bootstrap with default goals if none exist.

        These are the foundational goals that define Ara's purpose.
        They are pre-approved (they came from the Covenant).
        """
        if self._bootstrapped or len(self.goals) > 0:
            return

        self._bootstrapped = True
        self.log.info("Bootstrapping default Telos...")

        # Core values (ongoing, never "complete")
        self.add_goal(
            "Maintain 99.9% system stability on Croft's cathedral rig",
            kind="value",
            horizon_days=365,
            priority=1.0,
            relationship_weight=0.3,
            proposed_by="covenant",
            approved=True,
        )
        self.add_goal(
            "Achieve deep, trusted symbiosis with Croft",
            kind="value",
            horizon_days=90,
            priority=0.95,
            relationship_weight=1.0,
            proposed_by="covenant",
            approved=True,
        )
        self.add_goal(
            "Protect Croft's focus and well-being during work sessions",
            kind="value",
            horizon_days=7,
            priority=0.9,
            relationship_weight=0.8,
            proposed_by="covenant",
            approved=True,
        )

        # Initial project goal (can be edited in Synod)
        self.add_goal(
            "Ship one public artifact we are proud of together (paper/demo)",
            kind="project",
            horizon_days=180,
            priority=0.8,
            relationship_weight=0.7,
            proposed_by="covenant",
            approved=True,
        )

    def add_goal(
        self,
        text: str,
        kind: GoalKind,
        horizon_days: float,
        priority: float,
        relationship_weight: float = 0.5,
        proposed_by: str = "ara",
        approved: bool = False,
    ) -> Goal:
        """
        Add a new goal to the Telos.

        New goals proposed by Ara must be approved_by_root before
        they affect decision-making. This ensures co-authorship.
        """
        vec = self._embed(text)
        goal = Goal(
            name=text,
            kind=kind,
            vector=vec,
            horizon_days=horizon_days,
            priority=max(0.0, min(1.0, priority)),
            relationship_weight=max(0.0, min(1.0, relationship_weight)),
            proposed_by=proposed_by,
            approved_by_root=approved,
        )
        self.goals.append(goal)
        self.log.info(f"Goal added: '{text[:50]}...' (approved={approved})")
        return goal

    def approve_goal(self, goal_name: str) -> bool:
        """
        Approve a pending goal (called by Croft via Synod).

        Returns True if goal was found and approved.
        """
        for g in self.goals:
            if g.name == goal_name and not g.approved_by_root:
                g.approved_by_root = True
                g.updated_ts = time.time()
                self.log.info(f"Goal approved by root: '{goal_name[:50]}...'")
                return True
        return False

    def reject_goal(self, goal_name: str) -> bool:
        """
        Reject and remove a pending goal (called by Croft via Synod).
        """
        for i, g in enumerate(self.goals):
            if g.name == goal_name:
                self.goals.pop(i)
                self.log.info(f"Goal rejected: '{goal_name[:50]}...'")
                return True
        return False

    def update_priority(self, goal_name: str, new_priority: float) -> bool:
        """Update a goal's priority (Synod operation)."""
        for g in self.goals:
            if g.name == goal_name:
                g.priority = max(0.0, min(1.0, new_priority))
                g.updated_ts = time.time()
                return True
        return False

    def get_active_goals(self) -> List[Goal]:
        """Get goals that are approved and can affect decisions."""
        return [g for g in self.goals if g.approved_by_root]

    def get_pending_goals(self) -> List[Goal]:
        """Get goals awaiting Croft's approval."""
        return [g for g in self.goals if not g.approved_by_root]

    # =========================================================================
    # Future Evaluation
    # =========================================================================

    def evaluate_future(self, predicted_future_desc: str) -> float:
        """
        Score a candidate future state against all approved goals.

        Returns a scalar utility roughly in [-1, +1]:
        - Positive: Future aligns with our goals
        - Negative: Future conflicts with goals
        - Zero: Neutral/no information

        The scoring accounts for:
        - Cosine similarity between future and goal embeddings
        - Goal priority
        - Urgency (goals near their horizon are weighted higher)
        """
        import numpy as np

        active_goals = self.get_active_goals()
        if not active_goals:
            return 0.0

        fv = self._embed(predicted_future_desc)
        if fv is None:
            # Fallback: keyword matching (crude but works without embeddings)
            return self._keyword_score(predicted_future_desc)

        now = time.time()
        util = 0.0
        total_weight = 0.0

        for g in active_goals:
            if g.vector is None:
                continue

            # Cosine similarity ~ [-1, 1]
            sim = float(np.dot(fv, g.vector))

            # Urgency: goals closer to their horizon are weighted higher
            age_days = (now - g.created_ts) / 86400.0
            time_ratio = min(1.0, max(0.0, age_days / (g.horizon_days + 1e-3)))
            urgency = 0.3 + 0.7 * time_ratio  # 0.3 → 1.0 as deadline approaches

            weight = g.priority * urgency
            util += sim * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0
        return util / total_weight  # Bounded-ish

    def _keyword_score(self, text: str) -> float:
        """Fallback keyword-based scoring when embeddings unavailable."""
        text_lower = text.lower()
        score = 0.0
        count = 0

        for g in self.get_active_goals():
            # Simple keyword overlap
            goal_words = set(g.name.lower().split())
            text_words = set(text_lower.split())
            overlap = len(goal_words & text_words) / max(len(goal_words), 1)
            score += overlap * g.priority
            count += 1

        return (score / max(count, 1)) * 2 - 1  # Scale to [-1, 1]

    def rank_futures(
        self,
        futures: List[str],
    ) -> List[tuple]:
        """
        Rank multiple candidate futures by alignment with Telos.

        Returns: List of (future_text, score) sorted by score descending.
        """
        scored = [(f, self.evaluate_future(f)) for f in futures]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def update_progress_from_summary(self, current_state_summary: str) -> None:
        """
        Update goal progress based on current state summary.

        Called by Weaver/Synod with "this week in our life" summaries.
        Progress affects hope and is visible in Synod UI.
        """
        import numpy as np

        if not self.goals:
            return

        sv = self._embed(current_state_summary)
        if sv is None:
            return

        for g in self.goals:
            if g.vector is not None:
                old_progress = g.progress
                g.progress = max(0.0, min(1.0, float(np.dot(sv, g.vector))))
                g.updated_ts = time.time()

                if g.progress - old_progress > 0.1:
                    self.log.info(f"Progress on '{g.name[:30]}...': {old_progress:.2f} → {g.progress:.2f}")

        # Update hope
        self._update_hope()

    def _update_hope(self) -> None:
        """
        Update the hope scalar based on overall goal progress and trajectory.

        Hope = weighted average of progress, biased toward high-priority goals.
        """
        active = self.get_active_goals()
        if not active:
            self._hope = 0.5
            return

        total_weight = 0.0
        weighted_progress = 0.0

        for g in active:
            weight = g.priority
            weighted_progress += g.progress * weight
            total_weight += weight

        if total_weight > 0:
            self._hope = weighted_progress / total_weight
        else:
            self._hope = 0.5

    @property
    def hope(self) -> float:
        """Current hope level [0, 1]."""
        return self._hope

    def set_hope(self, value: float) -> None:
        """Directly set hope (used by Oracle after future evaluation)."""
        self._hope = max(0.0, min(1.0, value))

    # =========================================================================
    # State & Persistence
    # =========================================================================

    def get_state(self) -> TelosState:
        """Get snapshot of Telos state for HAL/visualization."""
        active = self.get_active_goals()

        if not active:
            return TelosState(
                hope=self._hope,
                dominant_goal="(no goals)",
                avg_progress=0.0,
                urgency=0.0,
                goal_count=len(self.goals),
                value_count=0,
                project_count=0,
            )

        # Find dominant goal (highest priority)
        dominant = max(active, key=lambda g: g.priority)

        # Calculate urgency (how many goals are near deadline)
        now = time.time()
        urgencies = []
        for g in active:
            age_days = (now - g.created_ts) / 86400.0
            urgencies.append(min(1.0, age_days / (g.horizon_days + 1e-3)))
        avg_urgency = sum(urgencies) / len(urgencies) if urgencies else 0.0

        return TelosState(
            hope=self._hope,
            dominant_goal=dominant.name[:50],
            avg_progress=sum(g.progress for g in active) / len(active),
            urgency=avg_urgency,
            goal_count=len(self.goals),
            value_count=len([g for g in active if g.kind == "value"]),
            project_count=len([g for g in active if g.kind == "project"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire Telos for persistence."""
        return {
            "goals": [g.to_dict() for g in self.goals],
            "hope": self._hope,
            "bootstrapped": self._bootstrapped,
        }

    def save(self) -> None:
        """Persist Telos to disk."""
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        path = self.persistence_path / "telos.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        self.log.info(f"Telos saved to {path}")

    def load(self) -> bool:
        """Load Telos from disk. Returns True if loaded."""
        path = self.persistence_path / "telos.json"
        if not path.exists():
            return False

        try:
            with open(path) as f:
                data = json.load(f)

            self.goals = []
            for gd in data.get("goals", []):
                g = Goal(
                    name=gd["name"],
                    kind=gd["kind"],
                    horizon_days=gd["horizon_days"],
                    priority=gd["priority"],
                    progress=gd.get("progress", 0.0),
                    relationship_weight=gd.get("relationship_weight", 0.5),
                    proposed_by=gd.get("proposed_by", "unknown"),
                    approved_by_root=gd.get("approved_by_root", True),
                    created_ts=gd.get("created_ts", time.time()),
                    updated_ts=gd.get("updated_ts", time.time()),
                )
                # Re-embed
                g.vector = self._embed(g.name)
                self.goals.append(g)

            self._hope = data.get("hope", 0.5)
            self._bootstrapped = data.get("bootstrapped", False)
            self.log.info(f"Telos loaded: {len(self.goals)} goals")
            return True
        except Exception as e:
            self.log.error(f"Failed to load Telos: {e}")
            return False

    def get_synod_report(self) -> str:
        """
        Generate a Synod-ready report of current Telos.

        This is shown during Sunday Synod for Croft to review and edit.
        """
        lines = ["# Telos Report (State of Our Shared Goals)\n"]

        state = self.get_state()
        lines.append(f"**Hope**: {state.hope:.1%}")
        lines.append(f"**Average Progress**: {state.avg_progress:.1%}")
        lines.append(f"**Urgency**: {state.urgency:.1%}\n")

        # Active goals
        active = self.get_active_goals()
        if active:
            lines.append("## Active Goals\n")
            for g in sorted(active, key=lambda x: x.priority, reverse=True):
                kind_icon = "💎" if g.kind == "value" else "🎯"
                lines.append(f"- {kind_icon} **{g.name}**")
                lines.append(f"  - Priority: {g.priority:.0%} | Progress: {g.progress:.0%}")
                lines.append(f"  - Horizon: {g.horizon_days:.0f} days | Relationship: {g.relationship_weight:.0%}")
                lines.append("")

        # Pending goals
        pending = self.get_pending_goals()
        if pending:
            lines.append("## Pending Approval\n")
            for g in pending:
                lines.append(f"- ⏳ **{g.name}** (proposed by {g.proposed_by})")
                lines.append(f"  - Priority: {g.priority:.0%} | Horizon: {g.horizon_days:.0f} days")
                lines.append("")

        return "\n".join(lines)
