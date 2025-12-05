"""Autonomy Levels - Graduated permissions for Ara's independence.

This module defines progressive levels of autonomy that Ara can operate at:
- Level 0: Fully supervised - every action requires approval
- Level 1: Guided - most actions autonomous, risky ones need approval
- Level 2: Semi-autonomous - only high-risk actions need approval
- Level 3: Autonomous - operates independently, notifies of major decisions
- Level 4: Full autonomy - trusted to self-govern within safety bounds

The current level can be adjusted based on:
- User trust/preference
- Task complexity
- Domain risk
- Ara's track record

This is about graduated trust, not binary permission.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from enum import IntEnum

logger = logging.getLogger(__name__)


class AutonomyLevel(IntEnum):
    """Levels of autonomy for Ara."""
    SUPERVISED = 0      # Every action requires approval
    GUIDED = 1          # Most autonomous, risky needs approval
    SEMI_AUTONOMOUS = 2 # Only high-risk needs approval
    AUTONOMOUS = 3      # Independent, notifies major decisions
    FULL = 4            # Self-governing within bounds


@dataclass
class AutonomyProfile:
    """Configuration for an autonomy level."""

    level: AutonomyLevel
    name: str
    description: str

    # What requires approval
    approve_categories: Set[str] = field(default_factory=set)  # Always need approval
    notify_categories: Set[str] = field(default_factory=set)   # Notify but proceed
    autonomous_categories: Set[str] = field(default_factory=set)  # Fully autonomous

    # Limits
    max_cost_per_action: float = 0.0  # Max $ per action without approval
    max_actions_per_hour: int = 100   # Rate limit
    max_teacher_consultations: int = 10  # Teacher API calls per session

    # Time bounds
    allow_after_hours: bool = False  # Can operate without user present?
    max_session_duration_hours: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "name": self.name,
            "description": self.description,
            "approve_categories": list(self.approve_categories),
            "notify_categories": list(self.notify_categories),
            "autonomous_categories": list(self.autonomous_categories),
            "max_cost_per_action": self.max_cost_per_action,
            "max_actions_per_hour": self.max_actions_per_hour,
            "max_teacher_consultations": self.max_teacher_consultations,
            "allow_after_hours": self.allow_after_hours,
            "max_session_duration_hours": self.max_session_duration_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutonomyProfile":
        return cls(
            level=AutonomyLevel(data["level"]),
            name=data["name"],
            description=data["description"],
            approve_categories=set(data.get("approve_categories", [])),
            notify_categories=set(data.get("notify_categories", [])),
            autonomous_categories=set(data.get("autonomous_categories", [])),
            max_cost_per_action=data.get("max_cost_per_action", 0.0),
            max_actions_per_hour=data.get("max_actions_per_hour", 100),
            max_teacher_consultations=data.get("max_teacher_consultations", 10),
            allow_after_hours=data.get("allow_after_hours", False),
            max_session_duration_hours=data.get("max_session_duration_hours", 4.0),
        )


@dataclass
class AutonomyDecision:
    """Result of checking autonomy for an action."""

    action_category: str
    current_level: AutonomyLevel

    # Decision
    can_proceed: bool
    requires_approval: bool
    requires_notification: bool

    # Context
    reason: str
    limits_checked: Dict[str, bool] = field(default_factory=dict)

    # Audit
    decided_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_category": self.action_category,
            "current_level": self.current_level.value,
            "can_proceed": self.can_proceed,
            "requires_approval": self.requires_approval,
            "requires_notification": self.requires_notification,
            "reason": self.reason,
            "limits_checked": self.limits_checked,
            "decided_at": self.decided_at.isoformat(),
        }


@dataclass
class AutonomySession:
    """Tracks autonomy usage in a session."""

    session_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)

    # Current state
    current_level: AutonomyLevel = AutonomyLevel.GUIDED
    actions_this_hour: int = 0
    teacher_consultations: int = 0
    total_cost: float = 0.0

    # History
    decisions: List[AutonomyDecision] = field(default_factory=list)
    level_changes: List[Dict[str, Any]] = field(default_factory=list)

    # Trust metrics
    actions_approved: int = 0
    actions_rejected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "current_level": self.current_level.value,
            "actions_this_hour": self.actions_this_hour,
            "teacher_consultations": self.teacher_consultations,
            "total_cost": round(self.total_cost, 4),
            "decisions_count": len(self.decisions),
            "level_changes": self.level_changes,
            "actions_approved": self.actions_approved,
            "actions_rejected": self.actions_rejected,
        }


class AutonomyManager:
    """Manages autonomy levels and decisions."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the autonomy manager.

        Args:
            data_path: Path to autonomy data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "institute" / "autonomy"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._profiles: Dict[AutonomyLevel, AutonomyProfile] = {}
        self._current_session: Optional[AutonomySession] = None
        self._loaded = False

        # Action timestamps for rate limiting
        self._action_times: List[datetime] = []

    def _load(self, force: bool = False) -> None:
        """Load autonomy data from disk."""
        if self._loaded and not force:
            return

        profiles_file = self.data_path / "profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file) as f:
                    data = json.load(f)
                for p_data in data.get("profiles", []):
                    profile = AutonomyProfile.from_dict(p_data)
                    self._profiles[profile.level] = profile
            except Exception as e:
                logger.warning(f"Failed to load autonomy profiles: {e}")

        # Seed defaults if empty
        if not self._profiles:
            self._seed_default_profiles()

        self._loaded = True

    def _save(self) -> None:
        """Save autonomy data to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "profiles": [p.to_dict() for p in self._profiles.values()],
        }
        with open(self.data_path / "profiles.json", "w") as f:
            json.dump(data, f, indent=2)

    def _seed_default_profiles(self) -> None:
        """Seed default autonomy profiles."""
        self._profiles = {
            AutonomyLevel.SUPERVISED: AutonomyProfile(
                level=AutonomyLevel.SUPERVISED,
                name="Fully Supervised",
                description="Every action requires explicit approval",
                approve_categories={"all"},
                notify_categories=set(),
                autonomous_categories=set(),
                max_cost_per_action=0.0,
                max_actions_per_hour=20,
                max_teacher_consultations=5,
                allow_after_hours=False,
                max_session_duration_hours=2.0,
            ),
            AutonomyLevel.GUIDED: AutonomyProfile(
                level=AutonomyLevel.GUIDED,
                name="Guided",
                description="Most actions autonomous, risky ones need approval",
                approve_categories={"file_delete", "code_execute", "config_change", "expensive"},
                notify_categories={"file_write", "teacher_consult"},
                autonomous_categories={"read", "analyze", "search", "plan"},
                max_cost_per_action=0.10,
                max_actions_per_hour=50,
                max_teacher_consultations=10,
                allow_after_hours=False,
                max_session_duration_hours=4.0,
            ),
            AutonomyLevel.SEMI_AUTONOMOUS: AutonomyProfile(
                level=AutonomyLevel.SEMI_AUTONOMOUS,
                name="Semi-Autonomous",
                description="Only high-risk actions need approval",
                approve_categories={"file_delete", "expensive", "config_change"},
                notify_categories={"code_execute", "file_write"},
                autonomous_categories={"read", "analyze", "search", "plan", "teacher_consult"},
                max_cost_per_action=1.00,
                max_actions_per_hour=100,
                max_teacher_consultations=25,
                allow_after_hours=True,
                max_session_duration_hours=8.0,
            ),
            AutonomyLevel.AUTONOMOUS: AutonomyProfile(
                level=AutonomyLevel.AUTONOMOUS,
                name="Autonomous",
                description="Operates independently, notifies major decisions",
                approve_categories={"expensive"},
                notify_categories={"file_delete", "code_execute", "config_change"},
                autonomous_categories={"read", "analyze", "search", "plan", "teacher_consult", "file_write"},
                max_cost_per_action=5.00,
                max_actions_per_hour=200,
                max_teacher_consultations=50,
                allow_after_hours=True,
                max_session_duration_hours=12.0,
            ),
            AutonomyLevel.FULL: AutonomyProfile(
                level=AutonomyLevel.FULL,
                name="Full Autonomy",
                description="Self-governing within safety bounds",
                approve_categories=set(),  # Nothing (except safety violations)
                notify_categories={"expensive", "config_change"},
                autonomous_categories={"all"},
                max_cost_per_action=10.00,
                max_actions_per_hour=500,
                max_teacher_consultations=100,
                allow_after_hours=True,
                max_session_duration_hours=24.0,
            ),
        }

        self._save()

    def start_session(
        self,
        session_id: str,
        initial_level: AutonomyLevel = AutonomyLevel.GUIDED,
    ) -> AutonomySession:
        """Start a new autonomy tracking session.

        Args:
            session_id: Unique session identifier
            initial_level: Starting autonomy level

        Returns:
            New session
        """
        self._load()

        self._current_session = AutonomySession(
            session_id=session_id,
            current_level=initial_level,
        )

        logger.info(f"Started autonomy session {session_id} at level {initial_level.name}")
        return self._current_session

    def get_session(self) -> Optional[AutonomySession]:
        """Get the current session."""
        return self._current_session

    def set_level(self, level: AutonomyLevel, reason: str = "") -> None:
        """Set the current autonomy level.

        Args:
            level: New autonomy level
            reason: Why the level is changing
        """
        self._load()

        if self._current_session:
            old_level = self._current_session.current_level
            self._current_session.current_level = level
            self._current_session.level_changes.append({
                "from": old_level.value,
                "to": level.value,
                "reason": reason,
                "at": datetime.utcnow().isoformat(),
            })

            logger.info(f"Autonomy level changed: {old_level.name} -> {level.name} ({reason})")

    def get_profile(self, level: Optional[AutonomyLevel] = None) -> AutonomyProfile:
        """Get the profile for a level.

        Args:
            level: Level to get profile for (default: current)

        Returns:
            Autonomy profile
        """
        self._load()

        if level is None:
            level = (
                self._current_session.current_level
                if self._current_session
                else AutonomyLevel.GUIDED
            )

        return self._profiles[level]

    def check_autonomy(
        self,
        action_category: str,
        estimated_cost: float = 0.0,
    ) -> AutonomyDecision:
        """Check if an action is allowed under current autonomy.

        Args:
            action_category: Category of the action
            estimated_cost: Estimated cost of the action

        Returns:
            Autonomy decision
        """
        self._load()

        if not self._current_session:
            self.start_session("default")

        session = self._current_session
        profile = self._profiles[session.current_level]

        # Check limits
        limits_checked = {}

        # Rate limit
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        self._action_times = [t for t in self._action_times if t > hour_ago]
        within_rate_limit = len(self._action_times) < profile.max_actions_per_hour
        limits_checked["rate_limit"] = within_rate_limit

        # Cost limit
        within_cost_limit = estimated_cost <= profile.max_cost_per_action
        limits_checked["cost_limit"] = within_cost_limit

        # Teacher consultation limit
        if action_category == "teacher_consult":
            within_teacher_limit = session.teacher_consultations < profile.max_teacher_consultations
            limits_checked["teacher_limit"] = within_teacher_limit
        else:
            within_teacher_limit = True

        # Session duration
        session_duration = (now - session.started_at).total_seconds() / 3600
        within_duration = session_duration < profile.max_session_duration_hours
        limits_checked["duration_limit"] = within_duration

        # All limits passed?
        all_limits_ok = all(limits_checked.values())

        if not all_limits_ok:
            failed = [k for k, v in limits_checked.items() if not v]
            return AutonomyDecision(
                action_category=action_category,
                current_level=session.current_level,
                can_proceed=False,
                requires_approval=True,
                requires_notification=False,
                reason=f"Limits exceeded: {', '.join(failed)}",
                limits_checked=limits_checked,
            )

        # Check category permissions
        normalized_category = action_category.lower()

        # Check if needs approval
        needs_approval = (
            "all" in profile.approve_categories or
            normalized_category in profile.approve_categories
        )

        # Check if needs notification
        needs_notification = (
            normalized_category in profile.notify_categories
        )

        # Check if autonomous
        is_autonomous = (
            "all" in profile.autonomous_categories or
            normalized_category in profile.autonomous_categories
        )

        # Decision logic
        if needs_approval:
            can_proceed = False
            reason = f"Category '{action_category}' requires approval at {profile.name} level"
        elif needs_notification:
            can_proceed = True
            reason = f"Proceeding with notification for '{action_category}'"
        elif is_autonomous:
            can_proceed = True
            reason = f"Autonomous action for '{action_category}'"
        else:
            # Default: require approval for unknown categories
            can_proceed = False
            needs_approval = True
            reason = f"Unknown category '{action_category}' - defaulting to approval required"

        decision = AutonomyDecision(
            action_category=action_category,
            current_level=session.current_level,
            can_proceed=can_proceed,
            requires_approval=needs_approval,
            requires_notification=needs_notification,
            reason=reason,
            limits_checked=limits_checked,
        )

        # Track decision
        session.decisions.append(decision)

        # Update counters if proceeding
        if can_proceed:
            self._action_times.append(now)
            session.actions_this_hour = len(self._action_times)
            if action_category == "teacher_consult":
                session.teacher_consultations += 1
            session.total_cost += estimated_cost

        return decision

    def record_approval(self, approved: bool) -> None:
        """Record user approval/rejection of an action.

        Args:
            approved: Whether the action was approved
        """
        if self._current_session:
            if approved:
                self._current_session.actions_approved += 1
            else:
                self._current_session.actions_rejected += 1

    def get_trust_score(self) -> float:
        """Calculate trust score based on approval history.

        Returns:
            Trust score from 0 to 1
        """
        if not self._current_session:
            return 0.5

        total = (
            self._current_session.actions_approved +
            self._current_session.actions_rejected
        )

        if total == 0:
            return 0.5

        return self._current_session.actions_approved / total

    def suggest_level_change(self) -> Optional[Dict[str, Any]]:
        """Suggest an autonomy level change based on performance.

        Returns:
            Suggestion dict or None
        """
        if not self._current_session:
            return None

        current = self._current_session.current_level
        trust = self.get_trust_score()

        # High trust and many approvals -> suggest increase
        if trust > 0.9 and self._current_session.actions_approved > 10:
            if current < AutonomyLevel.FULL:
                return {
                    "direction": "increase",
                    "current": current.name,
                    "suggested": AutonomyLevel(current + 1).name,
                    "reason": f"High trust score ({trust:.0%}) with {self._current_session.actions_approved} approved actions",
                }

        # Low trust or many rejections -> suggest decrease
        if trust < 0.7 and self._current_session.actions_rejected > 3:
            if current > AutonomyLevel.SUPERVISED:
                return {
                    "direction": "decrease",
                    "current": current.name,
                    "suggested": AutonomyLevel(current - 1).name,
                    "reason": f"Low trust score ({trust:.0%}) with {self._current_session.actions_rejected} rejected actions",
                }

        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get autonomy manager summary."""
        self._load()

        session_info = None
        if self._current_session:
            session_info = {
                "session_id": self._current_session.session_id,
                "level": self._current_session.current_level.name,
                "actions_this_hour": self._current_session.actions_this_hour,
                "trust_score": self.get_trust_score(),
            }

        return {
            "profiles": len(self._profiles),
            "current_session": session_info,
            "level_descriptions": {
                level.name: self._profiles[level].description
                for level in AutonomyLevel
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[AutonomyManager] = None


def get_autonomy_manager() -> AutonomyManager:
    """Get the default autonomy manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = AutonomyManager()
    return _default_manager


def check_autonomy(action_category: str, estimated_cost: float = 0.0) -> AutonomyDecision:
    """Quick autonomy check."""
    return get_autonomy_manager().check_autonomy(action_category, estimated_cost)


def can_proceed_autonomously(action_category: str) -> bool:
    """Check if an action can proceed without approval."""
    decision = check_autonomy(action_category)
    return decision.can_proceed


def get_current_autonomy_level() -> AutonomyLevel:
    """Get the current autonomy level."""
    manager = get_autonomy_manager()
    session = manager.get_session()
    return session.current_level if session else AutonomyLevel.GUIDED
