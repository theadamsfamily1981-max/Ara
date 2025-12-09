"""
Hive Agent
===========

Base class for Ara worker agents.

Workers are NOT Ara's personality - they're her "hands and legs".
They:
- Follow the covenant, not hold it
- Execute tasks, not make high-level decisions
- Report back to the Queen, not to the outside world

Each agent:
1. Senses pheromones periodically
2. Adjusts behavior based on signals
3. Executes one unit of work
4. Reports results via reward/alarm pheromones
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from .store import PheromoneStore
from .pheromones import Pheromone, PheromoneKind, alarm_pheromone, reward_pheromone

logger = logging.getLogger(__name__)


@dataclass
class WorkResult:
    """Result of a single work unit."""
    success: bool
    work_type: str
    details: Dict[str, Any]
    score: float = 0.0  # 0.0–1.0, for reward calculation
    error: Optional[str] = None


class HiveAgent(ABC):
    """
    Base class for hive worker agents.

    Subclass this to create specific worker types:
    - KDPScanner
    - MerchDesigner
    - TrendScraper
    - ContentDrafter
    - etc.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        store: PheromoneStore,
        tools: Optional[Dict] = None,
    ):
        self.id = agent_id
        self.role = role
        self.store = store
        self.tools = tools or {}

        # State
        self._current_focus: Optional[str] = None
        self._in_safe_mode: bool = False
        self._last_tick: Optional[datetime] = None
        self._tick_count: int = 0

    # =========================================================================
    # Main Loop
    # =========================================================================

    def tick(self) -> Optional[WorkResult]:
        """
        Execute one tick of the agent loop.

        1. Sense pheromones
        2. Check for alarms → enter safe mode if needed
        3. Adjust focus based on priorities
        4. Do one unit of work
        5. Report result

        Returns:
            WorkResult if work was done, None otherwise
        """
        self._tick_count += 1
        self._last_tick = datetime.utcnow()

        # 1. Sense
        signals = self.sense()

        # 2. Check alarms
        alarms = [s for s in signals if s.kind == PheromoneKind.ALARM]
        if alarms and self._should_respond_to_alarms(alarms):
            self._enter_safe_mode(alarms)
            return None

        # Exit safe mode if no relevant alarms
        if self._in_safe_mode and not alarms:
            self._exit_safe_mode()

        if self._in_safe_mode:
            return None

        # 3. Adjust focus
        self._adjust_focus(signals)

        # 4. Do work
        result = self._do_work_unit()

        # 5. Report
        if result:
            self._report_result(result)

        return result

    def sense(self) -> List[Pheromone]:
        """
        Sense relevant pheromones from the store.

        Filters to only pheromones this agent should care about.
        """
        all_pheromones = self.store.snapshot()
        return [p for p in all_pheromones if self._is_relevant(p)]

    def _is_relevant(self, p: Pheromone) -> bool:
        """Check if a pheromone is relevant to this agent."""
        # Global pheromones affect everyone
        if p.kind == PheromoneKind.GLOBAL:
            return True

        # Priority pheromones may target specific roles
        if p.kind == PheromoneKind.PRIORITY:
            targets = p.meta.get("target_roles", [])
            if not targets or self.role in targets:
                return True
            return False

        # Alarm pheromones may affect specific roles
        if p.kind == PheromoneKind.ALARM:
            affected = p.meta.get("affected_roles", [])
            if not affected or self.role in affected:
                return True
            return False

        # Role pheromones for this specific agent
        if p.kind == PheromoneKind.ROLE:
            return p.meta.get("agent_id") == self.id

        # Reward pheromones from same role (for learning)
        if p.kind == PheromoneKind.REWARD:
            return p.key == self.role or self.role in p.meta.get("roles", [])

        return False

    def _should_respond_to_alarms(self, alarms: List[Pheromone]) -> bool:
        """Check if any alarm requires this agent to pause."""
        for alarm in alarms:
            affected = alarm.meta.get("affected_roles", [])
            if not affected or self.role in affected:
                return True
        return False

    def _enter_safe_mode(self, alarms: List[Pheromone]):
        """Enter safe mode due to alarms."""
        self._in_safe_mode = True
        alarm_keys = [a.key for a in alarms]
        logger.info(f"Agent {self.id} entering safe mode due to: {alarm_keys}")

    def _exit_safe_mode(self):
        """Exit safe mode."""
        self._in_safe_mode = False
        logger.info(f"Agent {self.id} exiting safe mode")

    def _adjust_focus(self, signals: List[Pheromone]):
        """Adjust focus based on priority pheromones."""
        priorities = [s for s in signals if s.kind == PheromoneKind.PRIORITY]
        if not priorities:
            return

        # Pick strongest priority
        now = datetime.utcnow()
        priorities.sort(key=lambda p: p.decayed_strength(now), reverse=True)
        strongest = priorities[0]

        if strongest.key != self._current_focus:
            self._current_focus = strongest.key
            logger.debug(f"Agent {self.id} focusing on: {strongest.key}")

    def _report_result(self, result: WorkResult):
        """Report work result via pheromones."""
        if result.success and result.score > 0:
            # Emit reward
            self.store.emit(
                kind=PheromoneKind.REWARD,
                key=self.role,
                strength=result.score,
                ttl=3600,
                emitter=f"agent:{self.id}",
                meta={
                    "work_type": result.work_type,
                    "details": result.details,
                },
            )
        elif not result.success and result.error:
            # Emit alarm for serious errors
            if self._is_serious_error(result.error):
                self.store.emit(
                    kind=PheromoneKind.ALARM,
                    key=f"{self.role}_error",
                    strength=1.0,
                    ttl=300,
                    emitter=f"agent:{self.id}",
                    meta={
                        "error": result.error,
                        "work_type": result.work_type,
                    },
                )

    def _is_serious_error(self, error: str) -> bool:
        """Check if an error is serious enough to raise an alarm."""
        serious_patterns = [
            "rate limit",
            "auth",
            "unauthorized",
            "forbidden",
            "quota",
            "suspended",
        ]
        error_lower = error.lower()
        return any(p in error_lower for p in serious_patterns)

    # =========================================================================
    # Abstract Methods (Implement in Subclasses)
    # =========================================================================

    @abstractmethod
    def _do_work_unit(self) -> Optional[WorkResult]:
        """
        Do one unit of work.

        Implement this in subclasses to define what the agent actually does.

        Returns:
            WorkResult if work was done, None if nothing to do
        """
        pass

    # =========================================================================
    # Status
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "id": self.id,
            "role": self.role,
            "in_safe_mode": self._in_safe_mode,
            "current_focus": self._current_focus,
            "tick_count": self._tick_count,
            "last_tick": self._last_tick.isoformat() if self._last_tick else None,
        }


# =============================================================================
# Example Agent Implementations
# =============================================================================

class GenericWorker(HiveAgent):
    """
    A generic worker that executes arbitrary jobs from a queue.

    This is useful for testing or simple tasks.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        store: PheromoneStore,
        job_queue: Optional[List[Dict]] = None,
        tools: Optional[Dict] = None,
    ):
        super().__init__(agent_id, role, store, tools)
        self.job_queue = job_queue or []

    def add_job(self, job: Dict):
        """Add a job to the queue."""
        self.job_queue.append(job)

    def _do_work_unit(self) -> Optional[WorkResult]:
        """Execute next job in queue."""
        if not self.job_queue:
            return None

        job = self.job_queue.pop(0)
        job_type = job.get("type", "unknown")

        try:
            # Placeholder: actual execution would depend on job type
            logger.info(f"Agent {self.id} executing job: {job_type}")

            return WorkResult(
                success=True,
                work_type=job_type,
                details={"job": job},
                score=0.5,
            )
        except Exception as e:
            return WorkResult(
                success=False,
                work_type=job_type,
                details={"job": job},
                error=str(e),
            )
