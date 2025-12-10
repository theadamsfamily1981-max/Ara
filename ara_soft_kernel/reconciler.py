"""
Reconciler
==========

The heart of the soft kernel: continuously reconcile
desired state with actual state.
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from ara_soft_kernel.models.supply import SupplyProfile
from ara_soft_kernel.models.demand import DemandProfile
from ara_soft_kernel.models.agent import AgentSpec, AgentInstance
from ara_soft_kernel.models.workspace import WorkspaceSpec
from ara_soft_kernel.models.job import (
    Job, JobType, JobPriority,
    spawn_agent_job, stop_agent_job, migrate_agent_job, create_workspace_job,
)
from ara_soft_kernel.observer import Observer
from ara_soft_kernel.orchestrator import Orchestrator
from ara_soft_kernel.governor import Governor, PolicyCheckResult

logger = logging.getLogger(__name__)


@dataclass
class ReconcileResult:
    """Result of a reconciliation cycle."""
    timestamp: float
    jobs: List[Job] = field(default_factory=list)
    workspaces: List[WorkspaceSpec] = field(default_factory=list)

    agents_to_start: int = 0
    agents_to_stop: int = 0
    agents_to_migrate: int = 0
    governance_rejections: int = 0

    cycle_time_ms: float = 0.0


class Reconciler:
    """
    Reconciles desired state with actual state.

    Called periodically or on significant changes.
    """

    def __init__(
        self,
        observer: Observer,
        orchestrator: Orchestrator,
        governor: Governor,
    ):
        self._lock = threading.RLock()

        self.observer = observer
        self.orchestrator = orchestrator
        self.governor = governor

        # Configuration
        self._reconcile_interval_sec = 5.0

        # State
        self._last_reconcile = 0.0
        self._cycle_count = 0

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start automatic reconciliation."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Reconciler already running")
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AraReconciler",
        )
        self._thread.start()
        logger.info(f"Reconciler started (interval: {self._reconcile_interval_sec}s)")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop automatic reconciliation."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info(f"Reconciler stopped ({self._cycle_count} cycles)")

    def is_running(self) -> bool:
        """Check if reconciler is running."""
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        """Background reconciliation loop."""
        logger.debug("Reconciler loop started")

        while not self._stop.is_set():
            try:
                self.reconcile()
            except Exception as e:
                logger.exception(f"Reconcile error: {e}")

            self._stop.wait(timeout=self._reconcile_interval_sec)

    def reconcile(self, dry_run: bool = False) -> ReconcileResult:
        """
        Perform one reconciliation cycle.

        If dry_run=True, returns what would change without executing.
        """
        start_time = time.time()

        supply = self.observer.get_supply()
        demand = self.observer.get_demand()
        running_agents = self.orchestrator.get_running_agents()

        result = ReconcileResult(timestamp=start_time)

        # ─────────────────────────────────────────────────────────────
        # 1. COMPUTE FEASIBLE CAPABILITIES
        # ─────────────────────────────────────────────────────────────
        feasible = self._compute_feasible(supply)

        # ─────────────────────────────────────────────────────────────
        # 2. EXPAND GOALS INTO DESIRED STATE
        # ─────────────────────────────────────────────────────────────
        active_goals = demand.get_active_goals()
        desired_agents, desired_workspaces = self.orchestrator.plan_from_goals(
            goals=active_goals,
            supply=supply,
            demand=demand,
        )

        # ─────────────────────────────────────────────────────────────
        # 3. DIFF: WHAT TO START / STOP / MIGRATE
        # ─────────────────────────────────────────────────────────────
        to_start, to_stop, to_migrate = self.orchestrator.diff_agents(desired_agents)

        result.agents_to_start = len(to_start)
        result.agents_to_stop = len(to_stop)
        result.agents_to_migrate = len(to_migrate)

        # ─────────────────────────────────────────────────────────────
        # 4. APPLY GOVERNANCE GATES
        # ─────────────────────────────────────────────────────────────
        approved_starts: List[Tuple[AgentSpec, str]] = []  # (spec, device_id)

        for spec in to_start:
            device_id = self.orchestrator.select_device(spec, supply)
            if not device_id:
                logger.warning(f"No suitable device for {spec.name}")
                continue

            check = self.governor.check_spawn(spec, supply, demand, device_id)
            if check.allowed:
                approved_starts.append((spec, device_id))
            else:
                result.governance_rejections += 1
                logger.info(f"Governance rejected spawn of {spec.name}: {check.reason}")

        # ─────────────────────────────────────────────────────────────
        # 5. ENFORCE RESOURCE BUDGETS
        # ─────────────────────────────────────────────────────────────
        approved_starts = self._clamp_to_budgets(approved_starts, supply, demand)

        # Handle resource triage if constrained
        if supply.is_resource_constrained():
            approved_starts, to_stop = self._apply_resource_triage(
                approved_starts, to_stop, supply, active_goals
            )

        # ─────────────────────────────────────────────────────────────
        # 6. EMIT JOBS
        # ─────────────────────────────────────────────────────────────
        jobs: List[Job] = []

        for spec, device_id in approved_starts:
            job = spawn_agent_job(
                agent_spec=spec.to_dict(),
                target_device=device_id,
                priority=self._spec_to_priority(spec),
            )
            jobs.append(job)

        for instance in to_stop:
            job = stop_agent_job(
                instance_id=instance.instance_id,
                reason="no_longer_needed",
            )
            jobs.append(job)

        for instance, new_device in to_migrate:
            job = migrate_agent_job(
                instance_id=instance.instance_id,
                from_device=instance.device_id,
                to_device=new_device,
            )
            jobs.append(job)

        result.jobs = jobs
        result.workspaces = desired_workspaces

        # ─────────────────────────────────────────────────────────────
        # 7. EXECUTE (unless dry run)
        # ─────────────────────────────────────────────────────────────
        if not dry_run:
            self._execute_jobs(jobs)

        result.cycle_time_ms = (time.time() - start_time) * 1000

        with self._lock:
            self._last_reconcile = start_time
            self._cycle_count += 1

        if jobs:
            logger.info(
                f"Reconcile: {len(approved_starts)} starts, {len(to_stop)} stops, "
                f"{len(to_migrate)} migrations ({result.cycle_time_ms:.1f}ms)"
            )

        return result

    def _compute_feasible(self, supply: SupplyProfile) -> Dict[str, Any]:
        """Compute what's feasible given current supply."""
        return {
            "has_gpu": len(supply.get_gpu_rich_devices()) > 0,
            "total_gpu_mem_gb": supply.total_gpu_memory_free(),
            "total_mem_gb": supply.total_memory_free(),
            "device_count": len(supply.get_online_devices()),
            "hive_available": supply.hive.available,
        }

    def _clamp_to_budgets(
        self,
        approved: List[Tuple[AgentSpec, str]],
        supply: SupplyProfile,
        demand: DemandProfile,
    ) -> List[Tuple[AgentSpec, str]]:
        """Clamp approved spawns to resource budgets."""
        # Simple implementation: just limit by GPU memory
        remaining_gpu = supply.total_gpu_memory_free()
        clamped = []

        for spec, device_id in approved:
            if spec.resources.gpu_mem_gb <= remaining_gpu:
                clamped.append((spec, device_id))
                remaining_gpu -= spec.resources.gpu_mem_gb
            else:
                logger.info(f"Clamped {spec.name}: insufficient GPU memory")

        return clamped

    def _apply_resource_triage(
        self,
        approved: List[Tuple[AgentSpec, str]],
        to_stop: List[AgentInstance],
        supply: SupplyProfile,
        goals: List,
    ) -> Tuple[List[Tuple[AgentSpec, str]], List[AgentInstance]]:
        """Apply resource triage when constrained."""
        # In constrained mode, only keep high-priority agents
        filtered_approved = [
            (s, d) for s, d in approved
            if s.resources.priority in (AgentPriority.CRITICAL, AgentPriority.HIGH)
        ]

        # Stop low-priority running agents
        extra_stops = [
            a for a in self.orchestrator.get_running_agents()
            if a.spec.resources.priority in (AgentPriority.LOW, AgentPriority.IDLE)
            and a not in to_stop
        ]

        return filtered_approved, to_stop + extra_stops

    def _spec_to_priority(self, spec: AgentSpec) -> JobPriority:
        """Convert agent priority to job priority."""
        mapping = {
            AgentPriority.CRITICAL: JobPriority.CRITICAL,
            AgentPriority.HIGH: JobPriority.HIGH,
            AgentPriority.NORMAL: JobPriority.NORMAL,
            AgentPriority.LOW: JobPriority.LOW,
            AgentPriority.IDLE: JobPriority.LOW,
        }
        return mapping.get(spec.resources.priority, JobPriority.NORMAL)

    def _execute_jobs(self, jobs: List[Job]) -> None:
        """Execute jobs immediately (synchronous for now)."""
        for job in jobs:
            try:
                if job.type == JobType.SPAWN_AGENT:
                    spec_data = job.payload.get("agent_spec", {})
                    spec = AgentSpec.from_dict(spec_data)
                    device_id = job.payload.get("target_device", "local")
                    self.orchestrator.spawn_agent(spec, device_id)
                    job.mark_completed()

                elif job.type == JobType.STOP_AGENT:
                    instance_id = job.payload.get("instance_id")
                    reason = job.payload.get("reason", "requested")
                    self.orchestrator.stop_agent(instance_id, reason)
                    job.mark_completed()

                elif job.type == JobType.MIGRATE_AGENT:
                    instance_id = job.payload.get("instance_id")
                    to_device = job.payload.get("to_device")
                    self.orchestrator.migrate_agent(instance_id, to_device)
                    job.mark_completed()

            except Exception as e:
                job.mark_failed(str(e))
                logger.error(f"Job {job.job_id} failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get reconciler statistics."""
        with self._lock:
            return {
                "running": self.is_running(),
                "cycle_count": self._cycle_count,
                "last_reconcile": self._last_reconcile,
                "last_reconcile_age_sec": time.time() - self._last_reconcile,
                "interval_sec": self._reconcile_interval_sec,
            }
