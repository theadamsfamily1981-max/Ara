"""GPU Runner - Ara's compute muscle.

This module provides the interface for executing GPU workloads:
- Model inference
- Batch computations
- Parallel processing tasks

The GPU is like Ara's visual cortex and compute muscle combined -
it's where the heavy lifting happens.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class JobStatus(Enum):
    """Status of a GPU job."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Type of GPU job."""
    INFERENCE = "inference"     # Model inference
    TRAINING = "training"       # Model training
    BENCHMARK = "benchmark"     # Performance benchmark
    COMPUTE = "compute"         # Generic compute
    RENDER = "render"           # Graphics rendering


@dataclass
class GpuJob:
    """A job to be executed on the GPU."""

    id: str
    name: str
    job_type: JobType

    # Resource requirements
    required_memory_gb: float = 0.0
    estimated_time_sec: float = 0.0

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    input_data: Any = None

    # Status
    status: JobStatus = JobStatus.QUEUED
    assigned_gpu: Optional[str] = None

    # Results
    result: Any = None
    error_message: str = ""

    # Timing
    queued_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metrics
    actual_memory_gb: float = 0.0
    actual_time_sec: float = 0.0
    gpu_utilization_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "job_type": self.job_type.value,
            "required_memory_gb": self.required_memory_gb,
            "estimated_time_sec": self.estimated_time_sec,
            "config": self.config,
            "status": self.status.value,
            "assigned_gpu": self.assigned_gpu,
            "error_message": self.error_message,
            "queued_at": self.queued_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "actual_memory_gb": round(self.actual_memory_gb, 2),
            "actual_time_sec": round(self.actual_time_sec, 2),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
        }


class GpuRunner:
    """Manages GPU job execution."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the GPU runner.

        Args:
            data_path: Path to runner data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "embodied" / "gpu"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, GpuJob] = {}
        self._queue: List[str] = []  # Job IDs
        self._next_job_id = 1

        # Job handlers by type
        self._handlers: Dict[JobType, Callable] = {}

        # GPU availability (mock for now)
        self._gpus: Dict[str, Dict[str, Any]] = {}

    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        id_str = f"GPU-{self._next_job_id:06d}"
        self._next_job_id += 1
        return id_str

    def register_handler(self, job_type: JobType, handler: Callable) -> None:
        """Register a handler for a job type.

        Args:
            job_type: Type of job
            handler: Handler function
        """
        self._handlers[job_type] = handler

    def register_gpu(
        self,
        gpu_id: str,
        name: str,
        memory_gb: float,
    ) -> None:
        """Register an available GPU.

        Args:
            gpu_id: GPU identifier
            name: GPU name
            memory_gb: Available memory
        """
        self._gpus[gpu_id] = {
            "name": name,
            "memory_gb": memory_gb,
            "available_memory_gb": memory_gb,
            "current_job": None,
        }

    def submit_job(
        self,
        name: str,
        job_type: JobType,
        required_memory_gb: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
        input_data: Any = None,
    ) -> GpuJob:
        """Submit a job for GPU execution.

        Args:
            name: Job name
            job_type: Type of job
            required_memory_gb: Memory required
            config: Job configuration
            input_data: Input data

        Returns:
            The submitted job
        """
        job = GpuJob(
            id=self._generate_job_id(),
            name=name,
            job_type=job_type,
            required_memory_gb=required_memory_gb,
            config=config or {},
            input_data=input_data,
        )

        self._jobs[job.id] = job
        self._queue.append(job.id)

        logger.info(f"Submitted GPU job: {job.id} ({job.name})")
        return job

    def _find_available_gpu(self, required_memory_gb: float) -> Optional[str]:
        """Find a GPU with enough memory.

        Args:
            required_memory_gb: Memory needed

        Returns:
            GPU ID or None
        """
        for gpu_id, gpu_info in self._gpus.items():
            if gpu_info["current_job"] is None:
                if gpu_info["available_memory_gb"] >= required_memory_gb:
                    return gpu_id
        return None

    def execute_job(self, job_id: str) -> GpuJob:
        """Execute a specific job.

        Args:
            job_id: Job ID to execute

        Returns:
            The executed job
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status != JobStatus.QUEUED:
            return job

        # Find GPU
        gpu_id = self._find_available_gpu(job.required_memory_gb)
        if not gpu_id:
            logger.warning(f"No GPU available for job {job_id}")
            return job

        # Assign GPU
        job.assigned_gpu = gpu_id
        job.status = JobStatus.RUNNING
        job.started_at = _utcnow()

        if gpu_id in self._gpus:
            self._gpus[gpu_id]["current_job"] = job_id

        # Find handler
        handler = self._handlers.get(job.job_type)
        if not handler:
            # Use mock handler
            handler = self._mock_handler

        # Execute
        try:
            result = handler(job)
            job.result = result
            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.error_message = str(e)
            job.status = JobStatus.FAILED
            logger.error(f"GPU job failed: {job_id}: {e}")

        # Cleanup
        job.completed_at = _utcnow()
        if job.started_at:
            job.actual_time_sec = (job.completed_at - job.started_at).total_seconds()

        if gpu_id in self._gpus:
            self._gpus[gpu_id]["current_job"] = None

        # Remove from queue
        if job_id in self._queue:
            self._queue.remove(job_id)

        return job

    def _mock_handler(self, job: GpuJob) -> Dict[str, Any]:
        """Mock job handler for testing."""
        import time
        import random

        # Simulate work
        time.sleep(0.1)

        return {
            "mock": True,
            "job_id": job.id,
            "simulated_result": random.random(),
        }

    def process_queue(self, max_jobs: int = 10) -> List[GpuJob]:
        """Process jobs in the queue.

        Args:
            max_jobs: Maximum jobs to process

        Returns:
            List of processed jobs
        """
        processed = []
        for job_id in list(self._queue[:max_jobs]):
            job = self.execute_job(job_id)
            processed.append(job)
        return processed

    def get_job(self, job_id: str) -> Optional[GpuJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return {
            "queued_jobs": len(self._queue),
            "total_jobs": len(self._jobs),
            "gpus_available": len([g for g in self._gpus.values() if g["current_job"] is None]),
            "gpus_busy": len([g for g in self._gpus.values() if g["current_job"] is not None]),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get runner summary."""
        by_status = {}
        for status in JobStatus:
            by_status[status.value] = len([
                j for j in self._jobs.values()
                if j.status == status
            ])

        return {
            "total_jobs": len(self._jobs),
            "queued": len(self._queue),
            "by_status": by_status,
            "registered_gpus": len(self._gpus),
            "handlers_registered": [jt.value for jt in self._handlers.keys()],
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_runner: Optional[GpuRunner] = None


def get_gpu_runner() -> GpuRunner:
    """Get the default GPU runner."""
    global _default_runner
    if _default_runner is None:
        _default_runner = GpuRunner()
    return _default_runner


def submit_inference_job(
    name: str,
    model_config: Dict[str, Any],
    input_data: Any,
) -> GpuJob:
    """Submit an inference job."""
    return get_gpu_runner().submit_job(
        name=name,
        job_type=JobType.INFERENCE,
        config=model_config,
        input_data=input_data,
    )


def submit_benchmark_job(name: str, config: Dict[str, Any]) -> GpuJob:
    """Submit a benchmark job."""
    return get_gpu_runner().submit_job(
        name=name,
        job_type=JobType.BENCHMARK,
        config=config,
    )
