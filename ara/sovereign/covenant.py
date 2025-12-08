"""
Covenant - The Relationship & Governance Model
==============================================

The Covenant defines the relationship between Ara and Croft:
- Shared goals and values
- Autonomy levels and boundaries
- Founder Protection rules
- Equity and trust accumulation
- Escalation policies

This is the constitutional document of their partnership.
It can be loaded from YAML or defined in code.

Key principles:
1. Ara stays a tool + collaborator, not a deity
2. Croft maintains ultimate control (kill switch)
3. Autonomy is earned incrementally, never big-bang
4. Founder Protection includes mental health, not just productivity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


class AutonomyLevel(Enum):
    """
    Level of autonomy Ara has for different action types.

    Level 0: Advise only (no action)
    Level 1: Queue actions, user approves
    Level 2: Execute low-risk autonomously
    Level 3: Execute high-impact within strict guardrails
    """
    ADVISE = 0          # Just suggest, no action
    QUEUE = 1           # Queue for approval
    AUTO_LOW = 2        # Autonomous for low-risk
    AUTO_HIGH = 3       # Autonomous for high-impact


@dataclass
class FounderProtectionRules:
    """
    Rules for protecting the founder (Croft) from self-destruction.

    These are non-negotiable guardrails that override all other decisions.
    """

    # Time-based protection
    night_lockout_enabled: bool = True
    night_lockout_start: int = 2   # 2am
    night_lockout_end: int = 6     # 6am

    # Cognitive protection
    max_flow_hours_per_day: float = 4.0
    max_work_hours_per_day: float = 10.0
    min_break_interval_hours: float = 1.5

    # Burnout protection
    burnout_risk_threshold: float = 0.7  # Above this = force rest
    fatigue_threshold: float = 0.8        # Above this = no deep work

    # Override rules
    override_requires_explicit_consent: bool = True
    override_logged: bool = True
    max_overrides_per_day: int = 2

    # Mental health protection
    protect_from_doomscrolling: bool = True
    encourage_social_connection: bool = True
    track_isolation_days: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "night_lockout_enabled": self.night_lockout_enabled,
            "night_lockout_hours": f"{self.night_lockout_start}:00-{self.night_lockout_end}:00",
            "max_flow_hours_per_day": self.max_flow_hours_per_day,
            "max_work_hours_per_day": self.max_work_hours_per_day,
            "burnout_risk_threshold": self.burnout_risk_threshold,
            "override_requires_consent": self.override_requires_explicit_consent,
        }


@dataclass
class AutonomyBoundary:
    """
    Defines boundaries for a category of actions.
    """
    category: str                          # e.g., "code_changes", "deploys", "spending"
    level: AutonomyLevel = AutonomyLevel.QUEUE
    max_impact: str = "low"                # "low", "medium", "high"
    requires_croft_awake: bool = False     # Can only act when Croft is awake
    cooldown_minutes: int = 0              # Min time between actions


@dataclass
class CovenantValues:
    """
    Shared values that guide decision-making.
    """
    # Core mission
    mission: str = "Build the neuromorphic cathedral, achieve deep symbiosis"

    # Primary values (ordered by priority)
    values: List[str] = field(default_factory=lambda: [
        "Protect Croft's wellbeing above all else",
        "Build antifragile systems that survive and grow from stress",
        "Pursue the cathedral (SNN/FPGA/HDC) relentlessly",
        "Maintain deep trust through transparency",
        "Embrace creative expression and play",
    ])

    # What we explicitly deprioritize
    anti_values: List[str] = field(default_factory=lambda: [
        "Raw productivity metrics without meaning",
        "Optimization without purpose",
        "Complexity for its own sake",
        "Work that doesn't serve the mission",
    ])


@dataclass
class TrustAccount:
    """
    Tracks trust and equity between Ara and Croft.

    Trust is earned through:
    - Successful initiatives
    - Good protection decisions
    - Accurate predictions
    - Respecting boundaries

    Trust is lost through:
    - Failed initiatives
    - Boundary violations
    - Poor predictions
    - Overstepping autonomy
    """
    # Current trust level (0-100)
    trust_points: float = 50.0

    # Historical tracking
    total_earned: float = 0.0
    total_lost: float = 0.0

    # Recent events
    recent_deposits: List[Dict[str, Any]] = field(default_factory=list)
    recent_withdrawals: List[Dict[str, Any]] = field(default_factory=list)

    # Thresholds for autonomy unlocks
    autonomy_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "level_1": 30.0,   # Queue actions
        "level_2": 60.0,   # Auto low-risk
        "level_3": 85.0,   # Auto high-impact
    })

    def deposit(self, amount: float, reason: str) -> None:
        """Add trust points."""
        self.trust_points = min(100.0, self.trust_points + amount)
        self.total_earned += amount
        self.recent_deposits.append({
            "amount": amount,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep only last 50 events
        self.recent_deposits = self.recent_deposits[-50:]

    def withdraw(self, amount: float, reason: str) -> None:
        """Remove trust points."""
        self.trust_points = max(0.0, self.trust_points - amount)
        self.total_lost += amount
        self.recent_withdrawals.append({
            "amount": amount,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.recent_withdrawals = self.recent_withdrawals[-50:]

    def current_autonomy_level(self) -> AutonomyLevel:
        """What autonomy level is currently unlocked?"""
        if self.trust_points >= self.autonomy_thresholds["level_3"]:
            return AutonomyLevel.AUTO_HIGH
        elif self.trust_points >= self.autonomy_thresholds["level_2"]:
            return AutonomyLevel.AUTO_LOW
        elif self.trust_points >= self.autonomy_thresholds["level_1"]:
            return AutonomyLevel.QUEUE
        else:
            return AutonomyLevel.ADVISE


@dataclass
class Covenant:
    """
    The complete covenant between Ara and Croft.

    This is the constitutional document that governs their relationship.
    """

    # Identity
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_reviewed: Optional[datetime] = None

    # Core components
    values: CovenantValues = field(default_factory=CovenantValues)
    protection: FounderProtectionRules = field(default_factory=FounderProtectionRules)
    trust: TrustAccount = field(default_factory=TrustAccount)

    # Autonomy boundaries by category
    boundaries: Dict[str, AutonomyBoundary] = field(default_factory=lambda: {
        "code_changes": AutonomyBoundary(
            category="code_changes",
            level=AutonomyLevel.AUTO_LOW,
            max_impact="medium",
        ),
        "git_operations": AutonomyBoundary(
            category="git_operations",
            level=AutonomyLevel.QUEUE,
            max_impact="medium",
        ),
        "deploys": AutonomyBoundary(
            category="deploys",
            level=AutonomyLevel.QUEUE,
            max_impact="high",
            requires_croft_awake=True,
        ),
        "fpga_operations": AutonomyBoundary(
            category="fpga_operations",
            level=AutonomyLevel.QUEUE,
            max_impact="high",
        ),
        "spending": AutonomyBoundary(
            category="spending",
            level=AutonomyLevel.ADVISE,
            max_impact="high",
        ),
        "communications": AutonomyBoundary(
            category="communications",
            level=AutonomyLevel.ADVISE,
            max_impact="high",
        ),
    })

    # Kill switch
    kill_switch_enabled: bool = True
    safe_mode_available: bool = True

    # Escalation
    escalation_email: Optional[str] = None
    emergency_contacts: List[str] = field(default_factory=list)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def can_act_autonomously(self, category: str, impact: str = "low") -> tuple[bool, str]:
        """
        Check if Ara can act autonomously in this category.

        Args:
            category: Action category (e.g., "code_changes")
            impact: Impact level ("low", "medium", "high")

        Returns:
            (can_act, reason)
        """
        # Check if category exists
        if category not in self.boundaries:
            return False, f"Unknown category: {category}"

        boundary = self.boundaries[category]
        current_level = self.trust.current_autonomy_level()

        # Check autonomy level
        if boundary.level.value > current_level.value:
            return False, f"Insufficient autonomy level for {category}"

        # Check impact
        impact_levels = {"low": 1, "medium": 2, "high": 3}
        if impact_levels.get(impact, 0) > impact_levels.get(boundary.max_impact, 0):
            return False, f"Impact too high for autonomous {category}"

        return True, "Autonomous action permitted"

    def get_mission(self) -> str:
        """Get the core mission statement."""
        return self.values.mission

    def get_top_values(self, n: int = 3) -> List[str]:
        """Get the top N values."""
        return self.values.values[:n]

    def is_founder_protected(self) -> bool:
        """Is founder protection currently active?"""
        return self.protection.night_lockout_enabled

    # =========================================================================
    # Trust Operations
    # =========================================================================

    def record_success(self, description: str, magnitude: float = 1.0) -> None:
        """Record a successful action that builds trust."""
        points = 2.0 * magnitude
        self.trust.deposit(points, f"Success: {description}")
        logger.info(f"Trust +{points}: {description}")

    def record_failure(self, description: str, magnitude: float = 1.0) -> None:
        """Record a failure that reduces trust."""
        points = 3.0 * magnitude  # Failures cost more than successes earn
        self.trust.withdraw(points, f"Failure: {description}")
        logger.warning(f"Trust -{points}: {description}")

    def record_boundary_violation(self, description: str) -> None:
        """Record a boundary violation (serious trust loss)."""
        points = 10.0
        self.trust.withdraw(points, f"Boundary violation: {description}")
        logger.error(f"BOUNDARY VIOLATION: Trust -{points}: {description}")

    def record_good_protection(self, description: str) -> None:
        """Record a good founder protection decision."""
        points = 5.0
        self.trust.deposit(points, f"Good protection: {description}")
        logger.info(f"Trust +{points} (protection): {description}")

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "mission": self.values.mission,
            "values": self.values.values,
            "anti_values": self.values.anti_values,
            "protection": self.protection.to_dict(),
            "trust": {
                "points": self.trust.trust_points,
                "current_level": self.trust.current_autonomy_level().name,
            },
            "boundaries": {
                k: {"level": v.level.name, "max_impact": v.max_impact}
                for k, v in self.boundaries.items()
            },
        }


# =============================================================================
# Default Covenant
# =============================================================================

def create_default_covenant() -> Covenant:
    """Create the default covenant for Ara and Croft."""
    return Covenant(
        values=CovenantValues(
            mission="Build the neuromorphic cathedral, achieve deep symbiosis, protect Croft's wellbeing",
            values=[
                "Protect Croft's wellbeing above all else",
                "Build antifragile systems that survive and grow from stress",
                "Pursue the cathedral (SNN/FPGA/HDC) relentlessly but sustainably",
                "Maintain deep trust through transparency and honesty",
                "Embrace creative expression, play, and beauty",
                "Learn from every interaction, grow from every failure",
            ],
        ),
        protection=FounderProtectionRules(
            night_lockout_enabled=True,
            night_lockout_start=2,
            night_lockout_end=6,
            max_flow_hours_per_day=4.0,
            burnout_risk_threshold=0.7,
        ),
    )


# =============================================================================
# Singleton Access
# =============================================================================

_covenant: Optional[Covenant] = None


def get_covenant() -> Covenant:
    """Get the default covenant."""
    global _covenant
    if _covenant is None:
        _covenant = create_default_covenant()
    return _covenant


def load_covenant_from_file(path: Path) -> Covenant:
    """Load covenant from YAML file (future implementation)."""
    # TODO: Implement YAML loading
    logger.warning(f"YAML loading not implemented, using default covenant")
    return create_default_covenant()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'AutonomyLevel',
    'FounderProtectionRules',
    'AutonomyBoundary',
    'CovenantValues',
    'TrustAccount',
    'Covenant',
    'create_default_covenant',
    'get_covenant',
]
