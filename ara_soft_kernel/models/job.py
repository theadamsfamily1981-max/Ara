"""
Job Models
==========

Jobs represent discrete units of work for the kernel to execute.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class JobType(str, Enum):
    """Types of jobs."""
    SPAWN_AGENT = "spawn_agent"
    STOP_AGENT = "stop_agent"
    MIGRATE_AGENT = "migrate_agent"
    RESTART_AGENT = "restart_agent"
    CREATE_WORKSPACE = "create_workspace"
    UPDATE_WORKSPACE = "update_workspace"
    DESTROY_WORKSPACE = "destroy_workspace"
    UPDATE_CONFIG = "update_config"
    EXECUTE_COMMAND = "execute_command"


class JobState(str, Enum):
    """Job execution states."""
    PENDING = "pending"
    APPROVED = "approved"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job execution priority."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class GovernanceCheck:
    """Result of a governance policy check."""
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approval_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "requires_approval": self.requires_approval,
            "approved_by": self.approved_by,
            "approval_timestamp": self.approval_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GovernanceCheck:
        return cls(
            passed=data.get("passed", []),
            failed=data.get("failed", []),
            requires_approval=data.get("requires_approval", False),
            approved_by=data.get("approved_by"),
            approval_timestamp=data.get("approval_timestamp"),
        )

    def is_allowed(self) -> bool:
        """Check if job is allowed to proceed."""
        if self.failed:
            return False
        if self.requires_approval and not self.approved_by:
            return False
        return True


@dataclass
class JobExecution:
    """Execution state of a job."""
    state: JobState = JobState.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "result": self.result,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JobExecution:
        state = data.get("state", "pending")
        if isinstance(state, str):
            state = JobState(state)
        return cls(
            state=state,
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            result=data.get("result"),
            attempts=data.get("attempts", 0),
        )


@dataclass
class Job:
    """A discrete unit of work for the kernel."""
    job_id: str
    type: JobType
    payload: Dict[str, Any]

    priority: JobPriority = JobPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    created_by: str = "orchestrator"

    governance: GovernanceCheck = field(default_factory=GovernanceCheck)
    execution: JobExecution = field(default_factory=JobExecution)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Job IDs
    blocks: List[str] = field(default_factory=list)      # Job IDs that depend on this

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "type": self.type.value,
            "payload": self.payload,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "governance": self.governance.to_dict(),
            "execution": self.execution.to_dict(),
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "description": self.description,
            "tags": self.tags,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Job:
        job_type = data.get("type", "execute_command")
        if isinstance(job_type, str):
            job_type = JobType(job_type)
        priority = data.get("priority", "normal")
        if isinstance(priority, str):
            priority = JobPriority(priority)
        return cls(
            job_id=data["job_id"],
            type=job_type,
            payload=data.get("payload", {}),
            priority=priority,
            created_at=data.get("created_at", time.time()),
            created_by=data.get("created_by", "unknown"),
            governance=GovernanceCheck.from_dict(data.get("governance", {})),
            execution=JobExecution.from_dict(data.get("execution", {})),
            depends_on=data.get("depends_on", []),
            blocks=data.get("blocks", []),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Job:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def create(
        cls,
        job_type: JobType,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        created_by: str = "orchestrator",
        description: str = "",
    ) -> Job:
        """Create a new job with generated ID."""
        return cls(
            job_id=f"job-{int(time.time())}-{uuid.uuid4().hex[:8]}",
            type=job_type,
            payload=payload,
            priority=priority,
            created_by=created_by,
            description=description,
        )

    def is_ready(self) -> bool:
        """Check if job is ready to execute."""
        return (
            self.execution.state == JobState.PENDING
            and self.governance.is_allowed()
            and len(self.depends_on) == 0  # No unmet dependencies
        )

    def is_complete(self) -> bool:
        """Check if job is in a terminal state."""
        return self.execution.state in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.REJECTED,
            JobState.CANCELLED,
        )

    def mark_running(self) -> None:
        """Mark job as running."""
        self.execution.state = JobState.RUNNING
        self.execution.started_at = time.time()
        self.execution.attempts += 1

    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark job as completed."""
        self.execution.state = JobState.COMPLETED
        self.execution.completed_at = time.time()
        self.execution.result = result

    def mark_failed(self, error: str) -> None:
        """Mark job as failed."""
        self.execution.state = JobState.FAILED
        self.execution.completed_at = time.time()
        self.execution.error = error

    def mark_rejected(self, reason: str) -> None:
        """Mark job as rejected by governance."""
        self.execution.state = JobState.REJECTED
        self.execution.completed_at = time.time()
        self.execution.error = reason

    def age_seconds(self) -> float:
        """Get job age in seconds."""
        return time.time() - self.created_at

    def duration_seconds(self) -> Optional[float]:
        """Get job execution duration in seconds."""
        if self.execution.started_at is None:
            return None
        end = self.execution.completed_at or time.time()
        return end - self.execution.started_at


# Job creation helpers

def spawn_agent_job(
    agent_spec: Dict[str, Any],
    target_device: str,
    priority: JobPriority = JobPriority.NORMAL,
) -> Job:
    """Create a spawn agent job."""
    return Job.create(
        job_type=JobType.SPAWN_AGENT,
        payload={
            "agent_spec": agent_spec,
            "target_device": target_device,
        },
        priority=priority,
        description=f"Spawn agent {agent_spec.get('name', 'unknown')} on {target_device}",
    )


def stop_agent_job(
    instance_id: str,
    reason: str = "requested",
    priority: JobPriority = JobPriority.NORMAL,
) -> Job:
    """Create a stop agent job."""
    return Job.create(
        job_type=JobType.STOP_AGENT,
        payload={
            "instance_id": instance_id,
            "reason": reason,
        },
        priority=priority,
        description=f"Stop agent {instance_id}",
    )


def migrate_agent_job(
    instance_id: str,
    from_device: str,
    to_device: str,
    priority: JobPriority = JobPriority.NORMAL,
) -> Job:
    """Create a migrate agent job."""
    return Job.create(
        job_type=JobType.MIGRATE_AGENT,
        payload={
            "instance_id": instance_id,
            "from_device": from_device,
            "to_device": to_device,
        },
        priority=priority,
        description=f"Migrate agent {instance_id} from {from_device} to {to_device}",
    )


def create_workspace_job(
    workspace_spec: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
) -> Job:
    """Create a workspace creation job."""
    return Job.create(
        job_type=JobType.CREATE_WORKSPACE,
        payload={
            "workspace_spec": workspace_spec,
        },
        priority=priority,
        description=f"Create workspace {workspace_spec.get('name', 'unknown')}",
    )
