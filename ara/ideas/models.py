"""Idea Data Models - The structure of Ara's proposals.

Each idea represents a hypothesis Ara wants to test or a change she proposes.
Ideas flow through a lifecycle from draft to completion or rejection.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, List, Any


class IdeaCategory(Enum):
    """Categories of ideas Ara can propose."""

    PERFORMANCE = "performance"      # Speed, latency, throughput
    STABILITY = "stability"          # Reliability, error handling
    UX = "ux"                        # User experience improvements
    SAFETY = "safety"                # Security, safety rails
    RESEARCH = "research"            # Curiosity-driven exploration
    WEIRD_IDEA = "weird_idea"        # Unconventional proposals
    MAINTENANCE = "maintenance"      # Housekeeping, cleanup
    INTEGRATION = "integration"      # Connecting systems


class IdeaRisk(Enum):
    """Risk level of an idea."""

    NONE = "none"        # Pure observation, no changes
    LOW = "low"          # Reversible, isolated changes
    MEDIUM = "medium"    # Requires careful rollback plan
    HIGH = "high"        # Could affect system stability


class IdeaStatus(Enum):
    """Status in the idea lifecycle."""

    DRAFT = "draft"              # Ara is still forming the idea
    INBOX = "inbox"              # Ready for human review
    NEEDS_REVIEW = "needs_review"  # Waiting on human decision
    APPROVED = "approved"        # Human approved, queued for execution
    RUNNING = "running"          # Currently being executed/tested
    COMPLETED = "completed"      # Finished successfully
    REVERTED = "reverted"        # Was running but rolled back
    REJECTED = "rejected"        # Human rejected
    PARKED = "parked"           # Deferred for later


class IdeaOutcome(Enum):
    """Outcome of a completed idea."""

    IMPROVED = "improved"        # Measurable improvement
    NEUTRAL = "neutral"          # No significant change
    DEGRADED = "degraded"        # Made things worse (reverted)
    LEARNED = "learned"          # Valuable insight, no direct improvement
    INCONCLUSIVE = "inconclusive"  # Couldn't determine effect


class SandboxStatus(Enum):
    """Status of sandbox testing."""

    NOT_RUN = "not_run"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Signal:
    """A metric or observation that triggered or supports an idea.

    Signals provide the evidence basis for Ara's hypothesis.
    """

    name: str
    value: float
    unit: str = ""
    baseline: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Where this signal came from

    def delta(self) -> Optional[float]:
        """Get change from baseline."""
        if self.baseline is not None:
            return self.value - self.baseline
        return None

    def delta_percent(self) -> Optional[float]:
        """Get percentage change from baseline."""
        if self.baseline is not None and self.baseline != 0:
            return ((self.value - self.baseline) / self.baseline) * 100
        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Signal":
        return cls(**d)


@dataclass
class PlanStep:
    """A single step in an idea's execution plan."""

    description: str
    is_reversible: bool = True
    rollback_cmd: Optional[str] = None
    estimated_duration_sec: Optional[float] = None
    requires_approval: bool = False


@dataclass
class Idea:
    """A proposal from Ara.

    This is the core unit of the Idea Board - a structured hypothesis
    with supporting evidence, a plan, and governance metadata.
    """

    # Identity
    id: str = field(default_factory=lambda: f"idea_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
    title: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    created_by: str = "ara"

    # Classification
    category: IdeaCategory = IdeaCategory.RESEARCH
    risk: IdeaRisk = IdeaRisk.LOW
    status: IdeaStatus = IdeaStatus.DRAFT
    tags: List[str] = field(default_factory=list)

    # Content
    hypothesis: str = ""  # What Ara believes and why
    plan: List[str] = field(default_factory=list)  # Execution steps
    plan_steps: List[PlanStep] = field(default_factory=list)  # Detailed steps
    rollback_plan: List[str] = field(default_factory=list)

    # Evidence
    signals: List[Signal] = field(default_factory=list)
    related_objects: List[str] = field(default_factory=list)  # WorldObject IDs

    # Sandbox
    sandbox_status: SandboxStatus = SandboxStatus.NOT_RUN
    sandbox_results: Optional[Dict[str, Any]] = None
    sandbox_logs: List[str] = field(default_factory=list)

    # Execution
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    outcome: Optional[IdeaOutcome] = None
    outcome_notes: str = ""
    outcome_signals: List[Signal] = field(default_factory=list)

    # Governance
    human_decision: Optional[str] = None  # approve, reject, park
    human_notes: str = ""
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[float] = None

    # Conversation
    thread: List[Dict[str, str]] = field(default_factory=list)  # {role, content, timestamp}

    def __post_init__(self):
        # Convert enums from strings if needed
        if isinstance(self.category, str):
            self.category = IdeaCategory(self.category)
        if isinstance(self.risk, str):
            self.risk = IdeaRisk(self.risk)
        if isinstance(self.status, str):
            self.status = IdeaStatus(self.status)
        if isinstance(self.sandbox_status, str):
            self.sandbox_status = SandboxStatus(self.sandbox_status)
        if isinstance(self.outcome, str):
            self.outcome = IdeaOutcome(self.outcome)

    def touch(self) -> None:
        """Update the modification timestamp."""
        self.updated_at = time.time()

    def add_signal(self, signal: Signal) -> None:
        """Add a supporting signal."""
        self.signals.append(signal)
        self.touch()

    def add_thread_message(self, role: str, content: str) -> None:
        """Add a message to the conversation thread."""
        self.thread.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.touch()

    def approve(self, by: str = "human", notes: str = "") -> None:
        """Mark the idea as approved."""
        self.status = IdeaStatus.APPROVED
        self.human_decision = "approve"
        self.human_notes = notes
        self.reviewed_by = by
        self.reviewed_at = time.time()
        self.touch()

    def reject(self, by: str = "human", notes: str = "") -> None:
        """Mark the idea as rejected."""
        self.status = IdeaStatus.REJECTED
        self.human_decision = "reject"
        self.human_notes = notes
        self.reviewed_by = by
        self.reviewed_at = time.time()
        self.touch()

    def park(self, by: str = "human", notes: str = "") -> None:
        """Park the idea for later."""
        self.status = IdeaStatus.PARKED
        self.human_decision = "park"
        self.human_notes = notes
        self.reviewed_by = by
        self.reviewed_at = time.time()
        self.touch()

    def start_execution(self) -> None:
        """Mark the idea as running."""
        self.status = IdeaStatus.RUNNING
        self.started_at = time.time()
        self.touch()

    def complete(self, outcome: IdeaOutcome, notes: str = "") -> None:
        """Mark the idea as completed."""
        self.status = IdeaStatus.COMPLETED
        self.completed_at = time.time()
        self.outcome = outcome
        self.outcome_notes = notes
        self.touch()

    def revert(self, notes: str = "") -> None:
        """Mark the idea as reverted."""
        self.status = IdeaStatus.REVERTED
        self.completed_at = time.time()
        self.outcome = IdeaOutcome.DEGRADED
        self.outcome_notes = notes
        self.touch()

    def duration_seconds(self) -> Optional[float]:
        """Get execution duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def age_hours(self) -> float:
        """Get age of idea in hours."""
        return (time.time() - self.created_at) / 3600

    def is_actionable(self) -> bool:
        """Check if idea can be acted upon."""
        return self.status in (IdeaStatus.INBOX, IdeaStatus.NEEDS_REVIEW)

    def is_executable(self) -> bool:
        """Check if idea is ready for execution."""
        return self.status == IdeaStatus.APPROVED

    def is_terminal(self) -> bool:
        """Check if idea is in a terminal state."""
        return self.status in (
            IdeaStatus.COMPLETED,
            IdeaStatus.REVERTED,
            IdeaStatus.REJECTED,
        )

    def summary(self) -> str:
        """Get a one-line summary."""
        risk_emoji = {"none": "⚪", "low": "🟢", "medium": "🟡", "high": "🔴"}
        return f"[{risk_emoji.get(self.risk.value, '⚪')} {self.category.value}] {self.title}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "category": self.category.value,
            "risk": self.risk.value,
            "status": self.status.value,
            "tags": self.tags,
            "hypothesis": self.hypothesis,
            "plan": self.plan,
            "rollback_plan": self.rollback_plan,
            "signals": [s.to_dict() for s in self.signals],
            "related_objects": self.related_objects,
            "sandbox_status": self.sandbox_status.value,
            "sandbox_results": self.sandbox_results,
            "sandbox_logs": self.sandbox_logs,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome.value if self.outcome else None,
            "outcome_notes": self.outcome_notes,
            "outcome_signals": [s.to_dict() for s in self.outcome_signals],
            "human_decision": self.human_decision,
            "human_notes": self.human_notes,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "thread": self.thread,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Idea":
        """Deserialize from dictionary."""
        d = d.copy()
        d["signals"] = [Signal.from_dict(s) for s in d.get("signals", [])]
        d["outcome_signals"] = [Signal.from_dict(s) for s in d.get("outcome_signals", [])]
        # Remove plan_steps if present (not in dict form)
        d.pop("plan_steps", None)
        return cls(**d)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Idea":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))
