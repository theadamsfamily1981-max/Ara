"""Experiment Scheduler - Ara plans and runs research cycles.

This is the engine room of the Institute:
- Consumes next_actions from research_graph
- Plans experiments based on hardware availability
- Schedules and tracks execution
- Collects results and pushes back to the graph

Think of it as Ara's "lab manager" for experiments.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PROPOSED = "proposed"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentPriority(Enum):
    """Priority levels for experiments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResourceRequirement:
    """Resources required for an experiment."""

    # Hardware
    needs_gpu: bool = False
    needs_fpga: bool = False
    gpu_memory_gb: float = 0.0
    estimated_duration_min: float = 30.0

    # Software
    needs_teachers: List[str] = field(default_factory=list)
    needs_tools: List[str] = field(default_factory=list)

    # Constraints
    max_temp_c: float = 85.0
    max_power_w: float = 300.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "needs_gpu": self.needs_gpu,
            "needs_fpga": self.needs_fpga,
            "gpu_memory_gb": self.gpu_memory_gb,
            "estimated_duration_min": self.estimated_duration_min,
            "needs_teachers": self.needs_teachers,
            "needs_tools": self.needs_tools,
            "max_temp_c": self.max_temp_c,
            "max_power_w": self.max_power_w,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceRequirement":
        return cls(
            needs_gpu=data.get("needs_gpu", False),
            needs_fpga=data.get("needs_fpga", False),
            gpu_memory_gb=data.get("gpu_memory_gb", 0.0),
            estimated_duration_min=data.get("estimated_duration_min", 30.0),
            needs_teachers=data.get("needs_teachers", []),
            needs_tools=data.get("needs_tools", []),
            max_temp_c=data.get("max_temp_c", 85.0),
            max_power_w=data.get("max_power_w", 300.0),
        )


@dataclass
class ExperimentResult:
    """Results from an experiment run."""

    experiment_id: str
    success: bool

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_sec: float = 0.0

    # Logs
    log_path: Optional[str] = None
    error_message: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "success": self.success,
            "metrics": self.metrics,
            "outputs": self.outputs,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_sec": round(self.duration_sec, 2),
            "log_path": self.log_path,
            "error_message": self.error_message,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        return cls(
            experiment_id=data["experiment_id"],
            success=data["success"],
            metrics=data.get("metrics", {}),
            outputs=data.get("outputs", {}),
            duration_sec=data.get("duration_sec", 0.0),
            log_path=data.get("log_path"),
            error_message=data.get("error_message", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class Experiment:
    """A scheduled experiment."""

    id: str
    name: str
    description: str

    # Links
    hypothesis_id: Optional[str] = None
    thread_id: Optional[str] = None

    # Requirements
    requirements: ResourceRequirement = field(default_factory=ResourceRequirement)

    # Scheduling
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    priority: ExperimentPriority = ExperimentPriority.MEDIUM
    scheduled_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    # Execution
    runner_type: str = "generic"  # "benchmark", "viz", "teacher_consult", "generic"
    runner_config: Dict[str, Any] = field(default_factory=dict)

    # Results
    result: Optional[ExperimentResult] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "hypothesis_id": self.hypothesis_id,
            "thread_id": self.thread_id,
            "requirements": self.requirements.to_dict(),
            "status": self.status.value,
            "priority": self.priority.value,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "runner_type": self.runner_type,
            "runner_config": self.runner_config,
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        exp = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            hypothesis_id=data.get("hypothesis_id"),
            thread_id=data.get("thread_id"),
            status=ExperimentStatus(data.get("status", "proposed")),
            priority=ExperimentPriority(data.get("priority", "medium")),
            runner_type=data.get("runner_type", "generic"),
            runner_config=data.get("runner_config", {}),
            tags=data.get("tags", []),
        )

        if data.get("requirements"):
            exp.requirements = ResourceRequirement.from_dict(data["requirements"])

        if data.get("result"):
            exp.result = ExperimentResult.from_dict(data["result"])

        return exp


class ExperimentScheduler:
    """Schedules and manages experiments."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the scheduler.

        Args:
            data_path: Path to experiments data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "institute" / "experiments"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._experiments: Dict[str, Experiment] = {}
        self._loaded = False
        self._next_id = 1
        self._runners: Dict[str, Callable] = {}

    def _load(self, force: bool = False) -> None:
        """Load experiments from disk."""
        if self._loaded and not force:
            return

        self._experiments.clear()

        exp_file = self.data_path / "experiments.json"
        if exp_file.exists():
            try:
                with open(exp_file) as f:
                    data = json.load(f)
                for exp_data in data.get("experiments", []):
                    exp = Experiment.from_dict(exp_data)
                    self._experiments[exp.id] = exp
                self._next_id = data.get("next_id", 1)
            except Exception as e:
                logger.warning(f"Failed to load experiments: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save experiments to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "experiments": [e.to_dict() for e in self._experiments.values()],
            "next_id": self._next_id,
        }
        with open(self.data_path / "experiments.json", "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self) -> str:
        """Generate a unique experiment ID."""
        id_str = f"EXP-{self._next_id:04d}"
        self._next_id += 1
        return id_str

    def register_runner(self, runner_type: str, runner_func: Callable) -> None:
        """Register an experiment runner.

        Args:
            runner_type: Type of runner (benchmark, viz, etc.)
            runner_func: Function to execute experiments
        """
        self._runners[runner_type] = runner_func

    def create_experiment(
        self,
        name: str,
        description: str,
        hypothesis_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        runner_type: str = "generic",
        runner_config: Optional[Dict[str, Any]] = None,
        requirements: Optional[ResourceRequirement] = None,
        priority: ExperimentPriority = ExperimentPriority.MEDIUM,
        tags: Optional[List[str]] = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name
            description: What it tests
            hypothesis_id: Related hypothesis
            thread_id: Related thread
            runner_type: Type of runner to use
            runner_config: Runner configuration
            requirements: Resource requirements
            priority: Experiment priority
            tags: Categorization tags

        Returns:
            The new experiment
        """
        self._load()

        exp = Experiment(
            id=self._generate_id(),
            name=name,
            description=description,
            hypothesis_id=hypothesis_id,
            thread_id=thread_id,
            runner_type=runner_type,
            runner_config=runner_config or {},
            requirements=requirements or ResourceRequirement(),
            priority=priority,
            tags=tags or [],
        )

        self._experiments[exp.id] = exp
        self._save()
        logger.info(f"Created experiment: {exp.id} ({exp.name})")

        return exp

    def schedule_experiment(
        self,
        exp_id: str,
        scheduled_at: Optional[datetime] = None,
        deadline: Optional[datetime] = None,
    ) -> bool:
        """Schedule an experiment for execution.

        Args:
            exp_id: Experiment ID
            scheduled_at: When to run (default: now)
            deadline: When it must complete by

        Returns:
            True if scheduled
        """
        self._load()

        exp = self._experiments.get(exp_id)
        if not exp:
            return False

        exp.scheduled_at = scheduled_at or datetime.utcnow()
        exp.deadline = deadline
        exp.status = ExperimentStatus.SCHEDULED
        exp.updated_at = datetime.utcnow()

        self._save()
        logger.info(f"Scheduled experiment: {exp_id}")
        return True

    def run_experiment(self, exp_id: str) -> Optional[ExperimentResult]:
        """Run an experiment.

        Args:
            exp_id: Experiment ID

        Returns:
            Experiment result or None
        """
        self._load()

        exp = self._experiments.get(exp_id)
        if not exp:
            return None

        if exp.status not in [ExperimentStatus.PROPOSED, ExperimentStatus.SCHEDULED]:
            logger.warning(f"Experiment {exp_id} not in runnable state: {exp.status}")
            return None

        # Get runner
        runner = self._runners.get(exp.runner_type)
        if not runner:
            logger.warning(f"No runner registered for type: {exp.runner_type}")
            # Use default mock runner
            runner = self._mock_runner

        # Run
        exp.status = ExperimentStatus.RUNNING
        exp.updated_at = datetime.utcnow()
        self._save()

        start_time = datetime.utcnow()
        try:
            result = runner(exp)
            result.started_at = start_time
            result.completed_at = datetime.utcnow()
            result.duration_sec = (result.completed_at - start_time).total_seconds()

            exp.result = result
            exp.status = ExperimentStatus.COMPLETED if result.success else ExperimentStatus.FAILED

        except Exception as e:
            result = ExperimentResult(
                experiment_id=exp_id,
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
            )
            result.duration_sec = (result.completed_at - start_time).total_seconds()
            exp.result = result
            exp.status = ExperimentStatus.FAILED
            logger.error(f"Experiment {exp_id} failed: {e}")

        exp.updated_at = datetime.utcnow()
        self._save()

        return exp.result

    def _mock_runner(self, exp: Experiment) -> ExperimentResult:
        """Mock runner for testing."""
        import random
        import time

        # Simulate work
        time.sleep(0.1)

        return ExperimentResult(
            experiment_id=exp.id,
            success=random.random() > 0.2,
            metrics={
                "latency_ms": random.uniform(10, 100),
                "throughput": random.uniform(100, 1000),
            },
            notes="Mock experiment run",
        )

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        self._load()
        return self._experiments.get(exp_id)

    def get_pending_experiments(self) -> List[Experiment]:
        """Get experiments pending execution."""
        self._load()
        return [
            e for e in self._experiments.values()
            if e.status in [ExperimentStatus.PROPOSED, ExperimentStatus.SCHEDULED]
        ]

    def get_scheduled_experiments(self) -> List[Experiment]:
        """Get scheduled experiments, sorted by time."""
        self._load()
        scheduled = [
            e for e in self._experiments.values()
            if e.status == ExperimentStatus.SCHEDULED and e.scheduled_at
        ]
        return sorted(scheduled, key=lambda e: e.scheduled_at)

    def get_experiments_for_hypothesis(self, hyp_id: str) -> List[Experiment]:
        """Get experiments for a hypothesis."""
        self._load()
        return [e for e in self._experiments.values() if e.hypothesis_id == hyp_id]

    def get_completed_experiments(self, days: int = 7) -> List[Experiment]:
        """Get recently completed experiments."""
        self._load()
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [
            e for e in self._experiments.values()
            if e.status == ExperimentStatus.COMPLETED
            and e.result
            and e.result.completed_at
            and e.result.completed_at > cutoff
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get scheduler summary."""
        self._load()

        by_status = {}
        for status in ExperimentStatus:
            by_status[status.value] = len([
                e for e in self._experiments.values()
                if e.status == status
            ])

        recent = self.get_completed_experiments(days=7)
        success_rate = None
        if recent:
            successes = len([e for e in recent if e.result and e.result.success])
            success_rate = successes / len(recent)

        return {
            "total_experiments": len(self._experiments),
            "by_status": by_status,
            "pending": len(self.get_pending_experiments()),
            "scheduled": len(self.get_scheduled_experiments()),
            "recent_success_rate": success_rate,
            "runners_registered": list(self._runners.keys()),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_scheduler: Optional[ExperimentScheduler] = None


def get_experiment_scheduler() -> ExperimentScheduler:
    """Get the default experiment scheduler."""
    global _default_scheduler
    if _default_scheduler is None:
        _default_scheduler = ExperimentScheduler()
    return _default_scheduler


def create_experiment(
    name: str,
    description: str,
    hypothesis_id: Optional[str] = None,
) -> Experiment:
    """Create a new experiment."""
    return get_experiment_scheduler().create_experiment(
        name=name,
        description=description,
        hypothesis_id=hypothesis_id,
    )


def run_experiment(exp_id: str) -> Optional[ExperimentResult]:
    """Run an experiment."""
    return get_experiment_scheduler().run_experiment(exp_id)
