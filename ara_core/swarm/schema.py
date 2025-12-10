"""
Swarm Schema - Job and Agent Data Types
=======================================

Logging schema for layered intelligence tracking.
"""

from enum import IntEnum, Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import uuid


class AgentLayer(IntEnum):
    """Intelligence layers - higher = smarter + more restricted."""
    L0_REFLEX = 0      # Tiny context, no tools, safe steps
    L1_SPECIALIST = 1  # Domain tools, bounded reasoning
    L2_PLANNER = 2     # Decomposes goals, routes tasks
    L3_GOVERNOR = 3    # Monitors, enforces safety, approvals


class RiskLevel(str, Enum):
    """Job risk classification."""
    LOW = "low"        # No side effects, can auto-execute
    MEDIUM = "medium"  # Limited side effects, L2+ approval
    HIGH = "high"      # Significant side effects, L3/human approval


class JobOutcome(str, Enum):
    """Job completion status."""
    SUCCESS = "success"
    FAIL = "fail"
    NEEDS_FIX = "needs_fix"
    ESCALATED = "escalated"
    ABORTED = "aborted"


@dataclass
class AgentRun:
    """Record of a single agent execution within a job."""
    agent_id: str
    layer: AgentLayer
    cost: float          # tokens, GPU-sec, or unified cost units
    latency_ms: float
    outcome: JobOutcome = JobOutcome.SUCCESS
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "layer": self.layer.value,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "outcome": self.outcome.value,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentRun":
        return cls(
            agent_id=d["agent_id"],
            layer=AgentLayer(d["layer"]),
            cost=d["cost"],
            latency_ms=d["latency_ms"],
            outcome=JobOutcome(d.get("outcome", "success")),
            error=d.get("error"),
        )


@dataclass
class JobFix:
    """Record of a correction from higher layer."""
    from_layer: AgentLayer
    to_layer: AgentLayer
    reason: str
    cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_layer": self.from_layer.value,
            "to_layer": self.to_layer.value,
            "reason": self.reason,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobFix":
        return cls(
            from_layer=AgentLayer(d["from_layer"]),
            to_layer=AgentLayer(d["to_layer"]),
            reason=d["reason"],
            cost=d.get("cost", 0.0),
        )


@dataclass
class JobRecord:
    """Complete record of a job execution."""
    job_id: str
    job_type: str
    risk: RiskLevel
    pattern_id: str
    agents: List[AgentRun] = field(default_factory=list)
    fixes: List[JobFix] = field(default_factory=list)
    outcome: JobOutcome = JobOutcome.SUCCESS
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp_start is None:
            self.timestamp_start = datetime.utcnow()

    @classmethod
    def create(cls, job_type: str, risk: RiskLevel, pattern_id: str) -> "JobRecord":
        """Create a new job record."""
        return cls(
            job_id=f"J_{uuid.uuid4().hex[:12]}",
            job_type=job_type,
            risk=risk,
            pattern_id=pattern_id,
        )

    def add_agent_run(self, run: AgentRun):
        """Add an agent execution record."""
        self.agents.append(run)

    def add_fix(self, fix: JobFix):
        """Record a correction."""
        self.fixes.append(fix)

    def finalize(self, outcome: JobOutcome):
        """Mark job as complete."""
        self.outcome = outcome
        self.timestamp_end = datetime.utcnow()

    @property
    def total_cost(self) -> float:
        """Total cost across all agents and fixes."""
        agent_cost = sum(a.cost for a in self.agents)
        fix_cost = sum(f.cost for f in self.fixes)
        return agent_cost + fix_cost

    @property
    def total_latency_ms(self) -> float:
        """Total latency (sequential sum)."""
        return sum(a.latency_ms for a in self.agents)

    @property
    def duration_ms(self) -> Optional[float]:
        """Wall clock duration."""
        if self.timestamp_start and self.timestamp_end:
            delta = self.timestamp_end - self.timestamp_start
            return delta.total_seconds() * 1000
        return None

    @property
    def max_layer(self) -> AgentLayer:
        """Highest layer used in this job."""
        if not self.agents:
            return AgentLayer.L0_REFLEX
        return max(a.layer for a in self.agents)

    @property
    def correction_count(self) -> int:
        """Number of corrections needed."""
        return len(self.fixes)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON logging."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "risk": self.risk.value,
            "pattern_id": self.pattern_id,
            "agents": [a.to_dict() for a in self.agents],
            "fixes": [f.to_dict() for f in self.fixes],
            "outcome": self.outcome.value,
            "timestamp_start": self.timestamp_start.isoformat() if self.timestamp_start else None,
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobRecord":
        """Deserialize from dict."""
        return cls(
            job_id=d["job_id"],
            job_type=d["job_type"],
            risk=RiskLevel(d["risk"]),
            pattern_id=d["pattern_id"],
            agents=[AgentRun.from_dict(a) for a in d.get("agents", [])],
            fixes=[JobFix.from_dict(f) for f in d.get("fixes", [])],
            outcome=JobOutcome(d.get("outcome", "success")),
            timestamp_start=datetime.fromisoformat(d["timestamp_start"]) if d.get("timestamp_start") else None,
            timestamp_end=datetime.fromisoformat(d["timestamp_end"]) if d.get("timestamp_end") else None,
            metadata=d.get("metadata", {}),
        )

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return json.dumps(self.to_dict())


# =============================================================================
# LAYER CAPABILITIES (what each layer CAN and CANNOT do)
# =============================================================================

LAYER_CAPABILITIES = {
    AgentLayer.L0_REFLEX: {
        "can_use_tools": False,
        "can_write_files": False,
        "can_network": False,
        "can_spend_money": False,
        "can_write_prod_db": False,
        "max_context_tokens": 1000,
        "max_output_tokens": 500,
        "temperature_range": (0.7, 1.0),  # Higher = more diverse
    },
    AgentLayer.L1_SPECIALIST: {
        "can_use_tools": True,
        "can_write_files": True,  # Local only
        "can_network": False,
        "can_spend_money": False,
        "can_write_prod_db": False,
        "max_context_tokens": 4000,
        "max_output_tokens": 2000,
        "temperature_range": (0.3, 0.7),
    },
    AgentLayer.L2_PLANNER: {
        "can_use_tools": True,
        "can_write_files": True,
        "can_network": True,  # Read only
        "can_spend_money": False,
        "can_write_prod_db": False,  # Can propose, not execute
        "max_context_tokens": 16000,
        "max_output_tokens": 4000,
        "temperature_range": (0.1, 0.4),
    },
    AgentLayer.L3_GOVERNOR: {
        "can_use_tools": True,
        "can_write_files": True,
        "can_network": True,
        "can_spend_money": True,  # With budget limits
        "can_write_prod_db": True,  # Final approval
        "max_context_tokens": 32000,
        "max_output_tokens": 8000,
        "temperature_range": (0.0, 0.2),  # Precise
    },
}


def get_layer_capabilities(layer: AgentLayer) -> Dict[str, Any]:
    """Get capabilities for a layer."""
    return LAYER_CAPABILITIES.get(layer, LAYER_CAPABILITIES[AgentLayer.L0_REFLEX])
