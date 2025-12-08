"""
Initiative Schema - The Universal Work Unit
============================================

Everything that flows through Ara becomes an Initiative:
- A10 board bring-up
- SIGGRAPH paper
- Fixing K10
- Even "take a break"

The ChiefOfStaff evaluates each Initiative and decides:
- EXECUTE (Ara does it)
- DELEGATE (background agents)
- DEFER (protected future slot)
- KILL (distraction, ruthlessly cut)

This is the lingua franca of the sovereign loop.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class InitiativeStatus(Enum):
    """Status of an initiative in the pipeline."""
    PROPOSED = "proposed"      # Just created, awaiting evaluation
    APPROVED = "approved"      # CEO approved, not yet started
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"        # Waiting on something
    DEFERRED = "deferred"      # Pushed to future slot
    COMPLETED = "completed"
    KILLED = "killed"          # Ruthlessly cut as distraction
    FAILED = "failed"


class InitiativeType(Enum):
    """Type/category of initiative."""
    CATHEDRAL = "cathedral"      # Core mission: FPGA, SNN, soul
    RESEARCH = "research"        # Hypothesis testing, papers
    INFRASTRUCTURE = "infrastructure"  # Systems, tooling
    CREATIVE = "creative"        # Art, music, shaders
    MAINTENANCE = "maintenance"  # Admin, cleanup
    RECOVERY = "recovery"        # Rest, decompress, health
    EMERGENCY = "emergency"      # Critical failures


class CEODecision(Enum):
    """What the ChiefOfStaff decides to do with an initiative."""
    EXECUTE = "execute"          # Ara handles it now
    DELEGATE = "delegate"        # Background agents
    DEFER = "defer"              # Protected future slot
    KILL = "kill"                # Distraction, cut it
    ESCALATE = "escalate"        # Needs Croft's attention
    PROTECT = "protect"          # Founder protection triggered


@dataclass
class InitiativeMetrics:
    """Metrics for tracking initiative progress and impact."""
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    cognitive_burn: float = 0.0      # How much it drains Croft
    strategic_value: float = 0.0     # Teleology alignment score
    risk_level: float = 0.0          # 0-1, how risky
    compute_cost: float = 0.0        # GPU/FPGA hours
    money_cost: float = 0.0          # Actual dollars if any


@dataclass
class Initiative:
    """
    A unit of work in Ara's world.

    Everything becomes an Initiative - from "bring up A10 board"
    to "take a break because you're fried".
    """

    id: str = field(default_factory=lambda: f"init_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""

    # Classification
    type: InitiativeType = InitiativeType.MAINTENANCE
    tags: Dict[str, float] = field(default_factory=dict)  # For teleology scoring

    # Status tracking
    status: InitiativeStatus = InitiativeStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # CEO decision
    ceo_decision: Optional[CEODecision] = None
    ceo_reasoning: str = ""
    decision_timestamp: Optional[datetime] = None

    # Metrics
    metrics: InitiativeMetrics = field(default_factory=InitiativeMetrics)

    # Dependencies and context
    parent_id: Optional[str] = None  # If this is a sub-initiative
    blocked_by: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    # Execution details
    assigned_to: str = "ara"  # "ara", "background_agent", "croft"
    execution_plan: List[str] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)

    # Protection flags
    requires_croft_approval: bool = False
    blocked_by_founder_protection: bool = False
    protection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "tags": self.tags,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "ceo_decision": self.ceo_decision.value if self.ceo_decision else None,
            "ceo_reasoning": self.ceo_reasoning,
            "metrics": {
                "estimated_hours": self.metrics.estimated_hours,
                "actual_hours": self.metrics.actual_hours,
                "cognitive_burn": self.metrics.cognitive_burn,
                "strategic_value": self.metrics.strategic_value,
                "risk_level": self.metrics.risk_level,
            },
            "assigned_to": self.assigned_to,
            "blocked_by_founder_protection": self.blocked_by_founder_protection,
            "protection_reason": self.protection_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Initiative":
        """Deserialize from dict."""
        init = cls(
            id=data.get("id", f"init_{uuid.uuid4().hex[:8]}"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            type=InitiativeType(data.get("type", "maintenance")),
            tags=data.get("tags", {}),
            status=InitiativeStatus(data.get("status", "proposed")),
        )
        if data.get("ceo_decision"):
            init.ceo_decision = CEODecision(data["ceo_decision"])
        init.ceo_reasoning = data.get("ceo_reasoning", "")

        metrics = data.get("metrics", {})
        init.metrics = InitiativeMetrics(
            estimated_hours=metrics.get("estimated_hours", 0.0),
            cognitive_burn=metrics.get("cognitive_burn", 0.0),
            strategic_value=metrics.get("strategic_value", 0.0),
            risk_level=metrics.get("risk_level", 0.0),
        )

        return init


# =============================================================================
# Initiative Factory - Common Patterns
# =============================================================================

def create_cathedral_initiative(
    name: str,
    description: str,
    tags: Optional[Dict[str, float]] = None,
    estimated_hours: float = 0.0,
) -> Initiative:
    """Create a cathedral/core-mission initiative."""
    base_tags = {
        "cathedral": 1.0,
        "neuromorphic": 0.8,
        "fpga": 0.7,
        "soul": 0.6,
    }
    if tags:
        base_tags.update(tags)

    return Initiative(
        name=name,
        description=description,
        type=InitiativeType.CATHEDRAL,
        tags=base_tags,
        metrics=InitiativeMetrics(
            estimated_hours=estimated_hours,
            cognitive_burn=0.6,  # Cathedral work is engaging but draining
        ),
    )


def create_recovery_initiative(
    name: str = "Take a break",
    description: str = "Rest and recover",
) -> Initiative:
    """Create a recovery/rest initiative."""
    return Initiative(
        name=name,
        description=description,
        type=InitiativeType.RECOVERY,
        tags={"recovery": 1.0, "health": 0.9, "decompress": 0.8},
        metrics=InitiativeMetrics(
            cognitive_burn=-0.3,  # Negative = restorative
        ),
    )


def create_emergency_initiative(
    name: str,
    description: str,
    tags: Optional[Dict[str, float]] = None,
) -> Initiative:
    """Create an emergency/critical initiative."""
    base_tags = {
        "emergency": 1.0,
        "antifragility": 0.9,
        "recovery": 0.8,
    }
    if tags:
        base_tags.update(tags)

    return Initiative(
        name=name,
        description=description,
        type=InitiativeType.EMERGENCY,
        tags=base_tags,
        metrics=InitiativeMetrics(
            risk_level=0.8,
            cognitive_burn=0.9,
        ),
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'InitiativeStatus',
    'InitiativeType',
    'CEODecision',
    'InitiativeMetrics',
    'Initiative',
    'create_cathedral_initiative',
    'create_recovery_initiative',
    'create_emergency_initiative',
]
