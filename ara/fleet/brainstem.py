"""
The Brainstem - Always-On Orchestrator
=======================================

Tiny, ultra-stable node whose whole job is:
- Run Ara's daemon / scheduler / SMS gateway / health monitors
- Talk to all other nodes (GPU, FPGA, printer, router, etc.)
- Keep state, even if big rigs go down

Role: BRAINSTEM / DISPATCHER
Policy: Never runs heavy training or wild experiments.
        Orchestrates jobs on others, holds global view of the fleet.

Hardware: Old low-power x86 box or NUC, or decent ARM SBC with SSD.
          On a UPS, minimal OS, very few moving parts.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from collections import deque
import time
import hashlib


class JobState(Enum):
    """State of a scheduled job."""
    PENDING = auto()        # Waiting to be dispatched
    DISPATCHED = auto()     # Sent to worker
    RUNNING = auto()        # Confirmed running
    COMPLETED = auto()      # Finished successfully
    FAILED = auto()         # Failed
    CANCELLED = auto()      # Cancelled by user/system


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3            # Safety-related


@dataclass
class ScheduledJob:
    """A job scheduled for execution."""
    job_id: str
    name: str
    target_node: str            # Node ID to run on
    command: str                # What to execute
    priority: JobPriority = JobPriority.NORMAL

    # Timing
    created_at: float = field(default_factory=time.time)
    scheduled_for: Optional[float] = None  # Run after this time
    timeout_seconds: float = 3600.0        # 1 hour default

    # State
    state: JobState = JobState.PENDING
    dispatched_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Results
    exit_code: Optional[int] = None
    output: str = ""
    error: str = ""

    # Requirements
    requires_gpu: bool = False
    requires_network: bool = True
    max_retries: int = 3
    retry_count: int = 0

    def is_ready(self) -> bool:
        """Check if job is ready to run."""
        if self.state != JobState.PENDING:
            return False
        if self.scheduled_for and time.time() < self.scheduled_for:
            return False
        return True

    def mark_dispatched(self):
        """Mark job as dispatched."""
        self.state = JobState.DISPATCHED
        self.dispatched_at = time.time()

    def mark_running(self):
        """Mark job as running."""
        self.state = JobState.RUNNING
        self.started_at = time.time()

    def mark_completed(self, exit_code: int, output: str = ""):
        """Mark job as completed."""
        self.state = JobState.COMPLETED
        self.completed_at = time.time()
        self.exit_code = exit_code
        self.output = output

    def mark_failed(self, error: str):
        """Mark job as failed."""
        self.state = JobState.FAILED
        self.completed_at = time.time()
        self.error = error


@dataclass
class NodeStatus:
    """Status of a node as seen by Brainstem."""
    node_id: str
    is_online: bool = False
    last_heartbeat: float = 0.0
    current_job: Optional[str] = None  # Job ID
    load_percent: float = 0.0
    available_memory_gb: float = 0.0
    jobs_completed: int = 0
    jobs_failed: int = 0


@dataclass
class Brainstem:
    """
    The Brainstem - Always-On Orchestrator.

    Holds global view of the fleet, schedules jobs,
    monitors health, and stays alive when big rigs go down.
    """
    brainstem_id: str = "brainstem-01"

    # Job queue
    _jobs: Dict[str, ScheduledJob] = field(default_factory=dict)
    _job_queue: deque = field(default_factory=deque)  # Job IDs in order

    # Node status
    _nodes: Dict[str, NodeStatus] = field(default_factory=dict)

    # Event log
    _events: deque = field(default_factory=lambda: deque(maxlen=5000))

    # Dispatch callbacks
    _dispatch_fn: Optional[Callable[[str, ScheduledJob], bool]] = None

    # Configuration
    max_jobs_per_node: int = 2
    heartbeat_timeout_seconds: float = 30.0

    def register_node(self, node_id: str):
        """Register a node for job dispatch."""
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeStatus(node_id=node_id)

    def report_heartbeat(self, node_id: str, load: float = 0.0, memory_gb: float = 0.0):
        """Report a heartbeat from a node."""
        if node_id not in self._nodes:
            self.register_node(node_id)

        node = self._nodes[node_id]
        node.is_online = True
        node.last_heartbeat = time.time()
        node.load_percent = load
        node.available_memory_gb = memory_gb

    def report_job_started(self, job_id: str, node_id: str):
        """Report that a job has started on a node."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.mark_running()

            if node_id in self._nodes:
                self._nodes[node_id].current_job = job_id

            self._log_event("job_started", job_id=job_id, node_id=node_id)

    def report_job_completed(self, job_id: str, exit_code: int, output: str = ""):
        """Report that a job has completed."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.mark_completed(exit_code, output)

            # Clear from node
            for node in self._nodes.values():
                if node.current_job == job_id:
                    node.current_job = None
                    if exit_code == 0:
                        node.jobs_completed += 1
                    else:
                        node.jobs_failed += 1

            self._log_event("job_completed", job_id=job_id, exit_code=exit_code)

    def report_job_failed(self, job_id: str, error: str):
        """Report that a job has failed."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.mark_failed(error)

            # Clear from node
            for node in self._nodes.values():
                if node.current_job == job_id:
                    node.current_job = None
                    node.jobs_failed += 1

            # Retry logic
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.state = JobState.PENDING
                self._job_queue.append(job_id)
                self._log_event("job_retry", job_id=job_id, retry=job.retry_count)
            else:
                self._log_event("job_failed", job_id=job_id, error=error)

    def submit_job(self, job: ScheduledJob) -> str:
        """Submit a job for scheduling."""
        self._jobs[job.job_id] = job
        self._job_queue.append(job.job_id)
        self._log_event("job_submitted", job_id=job.job_id, target=job.target_node)
        return job.job_id

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.state in (JobState.PENDING, JobState.DISPATCHED):
                job.state = JobState.CANCELLED
                self._log_event("job_cancelled", job_id=job_id)
                return True
        return False

    def tick(self) -> List[str]:
        """
        Main scheduling tick.

        Returns list of job IDs that were dispatched.
        """
        dispatched = []

        # Update node online status
        now = time.time()
        for node in self._nodes.values():
            if now - node.last_heartbeat > self.heartbeat_timeout_seconds:
                if node.is_online:
                    node.is_online = False
                    self._log_event("node_offline", node_id=node.node_id)

        # Process job queue
        pending_jobs = []
        while self._job_queue:
            job_id = self._job_queue.popleft()
            if job_id not in self._jobs:
                continue

            job = self._jobs[job_id]
            if not job.is_ready():
                pending_jobs.append(job_id)
                continue

            # Try to dispatch
            if self._dispatch_job(job):
                dispatched.append(job_id)
            else:
                pending_jobs.append(job_id)

        # Re-add pending jobs
        for job_id in pending_jobs:
            self._job_queue.append(job_id)

        return dispatched

    def _dispatch_job(self, job: ScheduledJob) -> bool:
        """Attempt to dispatch a job to its target node."""
        target = job.target_node

        # Check if target is available
        if target not in self._nodes:
            return False

        node = self._nodes[target]
        if not node.is_online:
            return False

        if node.current_job is not None:
            return False  # Node is busy

        # Dispatch
        if self._dispatch_fn:
            success = self._dispatch_fn(target, job)
            if success:
                job.mark_dispatched()
                node.current_job = job.job_id
                self._log_event("job_dispatched", job_id=job.job_id, node_id=target)
                return True

        return False

    def _log_event(self, event_type: str, **kwargs):
        """Log an event."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            **kwargs,
        }
        self._events.append(event)

    def wire_dispatch(self, dispatch_fn: Callable[[str, ScheduledJob], bool]):
        """Wire up job dispatch callback."""
        self._dispatch_fn = dispatch_fn

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_node_status(self, node_id: str) -> Optional[NodeStatus]:
        """Get status of a specific node."""
        return self._nodes.get(node_id)

    def get_online_nodes(self) -> List[NodeStatus]:
        """Get all online nodes."""
        return [n for n in self._nodes.values() if n.is_online]

    def get_pending_jobs(self) -> List[ScheduledJob]:
        """Get all pending jobs."""
        return [j for j in self._jobs.values() if j.state == JobState.PENDING]

    def get_running_jobs(self) -> List[ScheduledJob]:
        """Get all running jobs."""
        return [j for j in self._jobs.values() if j.state == JobState.RUNNING]

    def get_recent_events(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent events."""
        return list(self._events)[-count:]

    def get_status(self) -> Dict[str, Any]:
        """Get Brainstem status summary."""
        online = len(self.get_online_nodes())
        total = len(self._nodes)

        pending = len(self.get_pending_jobs())
        running = len(self.get_running_jobs())
        completed = sum(1 for j in self._jobs.values() if j.state == JobState.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.state == JobState.FAILED)

        return {
            "brainstem_id": self.brainstem_id,
            "nodes_online": online,
            "nodes_total": total,
            "jobs_pending": pending,
            "jobs_running": running,
            "jobs_completed": completed,
            "jobs_failed": failed,
            "events_logged": len(self._events),
        }


def create_brainstem(brainstem_id: str = "brainstem-01") -> Brainstem:
    """Create a Brainstem with default configuration."""
    return Brainstem(brainstem_id=brainstem_id)
