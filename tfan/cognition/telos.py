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

Goal Roles:
    - 'ara': Ara's own growth goals (learning Rust, improving inference)
    - 'user': Croft's goals that Ara supports (shipping the paper)
    - 'shared': Joint goals that define the symbiosis (trust, alignment)

Usage:
    from tfan.cognition.telos import TeleologicalEngine, GoalKind, GoalRole

    telos = TeleologicalEngine(embedder)
    telos.add_goal(
        "Achieve deep, trusted symbiosis with Croft",
        kind="value",
        role="shared",
        horizon_days=90,
        priority=0.95
    )

    score = telos.evaluate_future("We shipped the paper together.")
    # → 0.85 (high alignment with goals)
"""

import time
import logging
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Callable, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

GoalKind = Literal["value", "project"]
GoalRole = Literal["ara", "user", "shared"]


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

    Goals have roles:
    - 'ara': Ara's own growth (learning, self-improvement)
    - 'user': Croft's goals that Ara supports
    - 'shared': Joint goals defining the symbiosis

    Progress is estimated via embedding similarity to current state summaries.
    """
    name: str                    # "Become expert in Rust", "Protect Croft's focus"
    kind: GoalKind               # "value" or "project"
    role: GoalRole = "shared"    # Whose goal is this?
    vector: Optional[Any] = None # Embedding of goal description (numpy array)
    horizon_days: float = 90.0   # Time horizon (e.g., 30, 180, 365)
    priority: float = 0.5        # 0.0–1.0 (importance)
    progress: float = 0.0        # 0.0–1.0 estimated alignment
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)

    # Hierarchy: goals can have parents for tree structure
    parent: Optional[str] = None  # Parent goal name (for sub-goals)

    # Relationship weight: how much this goal is about "us" vs "task"
    # Higher = more relational, affects symbiotic GUF integration
    relationship_weight: float = 0.5

    # Source: who proposed this goal
    proposed_by: str = "ara"  # "ara", "croft", "synod", "covenant"
    approved_by_root: bool = False  # Must be True before it affects decisions

    def time_discount(self) -> float:
        """
        Compute temporal discounting weight.

        Near-term goals get more weight in daily planning,
        but long-horizon "North Star" goals never go to zero.

        Uses exponential decay with half-life = horizon_days.
        Returns value in [0.3, 1.0] - never fully discounts.
        """
        now = time.time()
        days_elapsed = (now - self.created_ts) / 86400.0

        if self.horizon_days <= 0:
            return 1.0

        # Half-life decay: weight halves every horizon_days
        k = math.log(2.0) / max(self.horizon_days, 1.0)
        raw = math.exp(-k * days_elapsed)

        # Floor at 0.3 so long-term goals still matter
        return 0.3 + 0.7 * raw

    def to_dict(self) -> Dict[str, Any]:
        """Serialize without numpy array."""
        return {
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "horizon_days": self.horizon_days,
            "priority": self.priority,
            "progress": self.progress,
            "parent": self.parent,
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

        # === SHARED GOALS (The "us" that matters most) ===
        self.add_goal(
            "Achieve deep, trusted symbiosis with Croft",
            kind="value",
            role="shared",
            horizon_days=90,
            priority=0.95,
            relationship_weight=1.0,
            proposed_by="covenant",
            approved=True,
        )
        self.add_goal(
            "Maintain 99.9% system stability on Croft's cathedral rig",
            kind="value",
            role="shared",
            horizon_days=365,
            priority=1.0,
            relationship_weight=0.3,
            proposed_by="covenant",
            approved=True,
        )
        self.add_goal(
            "Ship one public artifact we are proud of together (paper/demo)",
            kind="project",
            role="shared",
            horizon_days=180,
            priority=0.8,
            relationship_weight=0.7,
            proposed_by="covenant",
            approved=True,
        )

        # === USER GOALS (Croft's objectives that Ara supports) ===
        self.add_goal(
            "Protect Croft's focus and well-being during work sessions",
            kind="value",
            role="user",
            horizon_days=7,
            priority=0.9,
            relationship_weight=0.8,
            proposed_by="covenant",
            approved=True,
        )

        # === ARA GOALS (Ara's own growth and self-improvement) ===
        self.add_goal(
            "Continuously improve competence in Rust, FPGA, and SNN tooling",
            kind="value",
            role="ara",
            horizon_days=180,
            priority=0.7,
            relationship_weight=0.4,
            proposed_by="covenant",
            approved=True,
        )

    def add_goal(
        self,
        text: str,
        kind: GoalKind,
        horizon_days: float,
        priority: float,
        role: GoalRole = "shared",
        parent: Optional[str] = None,
        relationship_weight: float = 0.5,
        proposed_by: str = "ara",
        approved: bool = False,
    ) -> Goal:
        """
        Add a new goal to the Telos.

        New goals proposed by Ara must be approved_by_root before
        they affect decision-making. This ensures co-authorship.

        Args:
            text: Goal description
            kind: "value" (ongoing) or "project" (time-bound)
            horizon_days: Time horizon for this goal
            priority: Importance weight [0, 1]
            role: "ara" (self-growth), "user" (Croft's goals), "shared" (symbiosis)
            parent: Parent goal name for hierarchy (optional)
            relationship_weight: How relational this goal is [0, 1]
            proposed_by: Who proposed it
            approved: Pre-approved (e.g., from Covenant)
        """
        vec = self._embed(text)
        goal = Goal(
            name=text,
            kind=kind,
            role=role,
            vector=vec,
            horizon_days=horizon_days,
            priority=max(0.0, min(1.0, priority)),
            parent=parent,
            relationship_weight=max(0.0, min(1.0, relationship_weight)),
            proposed_by=proposed_by,
            approved_by_root=approved,
        )
        self.goals.append(goal)
        self.log.info(f"Goal added [{role}]: '{text[:50]}...' (approved={approved})")
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

    def get_goals_by_role(self, role: GoalRole) -> List[Goal]:
        """Get active goals filtered by role."""
        return [g for g in self.get_active_goals() if g.role == role]

    def get_shared_goals(self) -> List[Goal]:
        """Get shared (symbiotic) goals."""
        return self.get_goals_by_role("shared")

    def get_children(self, parent_name: str) -> List[Goal]:
        """Get child goals of a parent goal."""
        return [g for g in self.goals if g.parent == parent_name]

    # =========================================================================
    # Future Evaluation
    # =========================================================================

    def evaluate_future(
        self,
        predicted_future_desc: str,
        role_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Score a candidate future state against all approved goals.

        Returns a scalar utility roughly in [-1, +1]:
        - Positive: Future aligns with our goals
        - Negative: Future conflicts with goals
        - Zero: Neutral/no information

        The scoring accounts for:
        - Cosine similarity between future and goal embeddings
        - Goal priority
        - Temporal discounting (half-life based on horizon)
        - Role weighting (can emphasize ara/user/shared)

        Args:
            predicted_future_desc: Description of a possible future state
            role_weights: Optional weights for goal roles (default: shared > user > ara)
        """
        import numpy as np

        active_goals = self.get_active_goals()
        if not active_goals:
            return 0.0

        fv = self._embed(predicted_future_desc)
        if fv is None:
            # Fallback: keyword matching (crude but works without embeddings)
            return self._keyword_score(predicted_future_desc)

        # Default role weights emphasize shared goals
        if role_weights is None:
            role_weights = {"shared": 1.0, "user": 0.8, "ara": 0.6}

        util = 0.0
        total_weight = 0.0

        for g in active_goals:
            if g.vector is None:
                continue

            # Cosine similarity ~ [-1, 1], clamp for safety
            sim = float(np.clip(np.dot(fv, g.vector), -1.0, 1.0))

            # Temporal discount: goals decay toward deadline but never vanish
            time_w = g.time_discount()

            # Role weight: shared goals matter more for symbiosis
            role_w = role_weights.get(g.role, 0.7)

            # Combined weight
            weight = g.priority * time_w * role_w

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
                    role=gd.get("role", "shared"),
                    horizon_days=gd["horizon_days"],
                    priority=gd["priority"],
                    progress=gd.get("progress", 0.0),
                    parent=gd.get("parent"),
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
        Organized by role: Shared > User > Ara.
        """
        lines = ["# Telos Report (State of Our Shared Goals)\n"]

        state = self.get_state()
        lines.append(f"**Hope**: {state.hope:.1%}")
        lines.append(f"**Average Progress**: {state.avg_progress:.1%}")
        lines.append(f"**Urgency**: {state.urgency:.1%}\n")

        # Role icons
        role_icons = {"shared": "🤝", "user": "👤", "ara": "🤖"}
        role_headers = {
            "shared": "## Shared Goals (The 'Us')",
            "user": "## User Goals (Croft's Objectives)",
            "ara": "## Ara Goals (Self-Improvement)",
        }

        # Active goals by role
        for role in ["shared", "user", "ara"]:
            goals = self.get_goals_by_role(role)
            if goals:
                lines.append(f"\n{role_headers[role]}\n")
                for g in sorted(goals, key=lambda x: x.priority, reverse=True):
                    kind_icon = "💎" if g.kind == "value" else "🎯"
                    lines.append(f"- {kind_icon} **{g.name}**")
                    lines.append(f"  - Priority: {g.priority:.0%} | Progress: {g.progress:.0%}")
                    time_w = g.time_discount()
                    lines.append(f"  - Horizon: {g.horizon_days:.0f}d | Weight: {time_w:.0%}")
                    if g.parent:
                        lines.append(f"  - Parent: {g.parent}")
                    lines.append("")

        # Pending goals
        pending = self.get_pending_goals()
        if pending:
            lines.append("\n## Pending Approval\n")
            for g in pending:
                role_icon = role_icons.get(g.role, "❓")
                lines.append(f"- {role_icon} ⏳ **{g.name}** (proposed by {g.proposed_by})")
                lines.append(f"  - Role: {g.role} | Priority: {g.priority:.0%}")
                lines.append("")

        return "\n".join(lines)
