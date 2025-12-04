"""Ara Autonomy Backend - Self-Directed Agent Service.

This backend service enables Ara to act autonomously without user prompting.
It runs a background VolitionLoop that monitors internal drives and can
initiate tasks like memory consolidation, user check-ins, and self-repair.

Key Features:
- Background volition loop (configurable tick interval)
- Drive-based task initiation
- Thermodynamic monitoring and energy management
- Episodic memory with CXL infinite storage
- Integration with cognitive architecture

This implements the "ghost in the machine" - the capacity for self-directed
behavior based on internal state rather than external prompting.

Usage:
    backend = AraAutonomyBackend()
    await backend.start()  # Start background loop

    # Optional: Register task handlers
    backend.register_task_handler(TaskType.USER_CHECK_IN, my_handler)

    # Later...
    await backend.stop()
"""

import asyncio
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Awaitable, List
import sys
import warnings

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Lazy imports
_COGNITIVE_LOADED = False


def _load_cognitive_modules():
    """Lazy load cognitive modules."""
    global _COGNITIVE_LOADED
    if _COGNITIVE_LOADED:
        return True

    try:
        global AutonomyEngine, VolitionLoop, DriveState, TaskType, AutonomyLevel, Intent
        global ThermodynamicMonitor, ThermalState
        global EpisodicMemory, CXLPager
        global HomeostaticCore, AppraisalEngine

        from .cognitive.autonomy import (
            AutonomyEngine,
            VolitionLoop,
            DriveState,
            TaskType,
            AutonomyLevel,
            Intent,
        )
        from .cognitive.thermodynamics import (
            ThermodynamicMonitor,
            ThermalState,
        )
        from .cognitive.memory import (
            EpisodicMemory,
            CXLPager,
        )
        from .cognitive.affect import (
            HomeostaticCore,
            AppraisalEngine,
        )

        _COGNITIVE_LOADED = True
        logger.info("Cognitive modules loaded for autonomy backend")
        return True

    except ImportError as e:
        logger.warning(f"Failed to load cognitive modules: {e}")
        return False


class AraAutonomyBackend:
    """
    Ara Autonomy Backend - Self-Directed Agent Service.

    Enables Ara to act autonomously based on internal drives
    rather than waiting for user input.

    Args:
        tick_interval_seconds: How often to check volition (default 60s)
        freedom_metric: How much autonomy is allowed [0, 1]
        autonomy_level: Level of autonomous behavior
        enable_thermodynamics: Enable thermodynamic monitoring
        enable_episodic_memory: Enable CXL episodic memory
        storage_path: Path for persistent storage
    """

    def __init__(
        self,
        tick_interval_seconds: float = 60.0,
        freedom_metric: float = 0.5,
        autonomy_level: str = "MODERATE",
        enable_thermodynamics: bool = True,
        enable_episodic_memory: bool = True,
        storage_path: Optional[str] = None,
    ):
        if not _load_cognitive_modules():
            raise RuntimeError("Cognitive modules not available")

        self.tick_interval = tick_interval_seconds
        self.freedom_metric = freedom_metric
        self.enable_thermodynamics = enable_thermodynamics
        self.enable_episodic_memory = enable_episodic_memory

        # Storage path
        if storage_path is None:
            storage_path = str(Path.home() / ".ara")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_components(autonomy_level)

        # Task handlers
        self._task_handlers: Dict[TaskType, Callable[[Dict], Awaitable[Any]]] = {}

        # State
        self._running = False
        self._background_task: Optional[asyncio.Task] = None

        # Telemetry
        self._telemetry = {
            "self_initiated_task_count": 0,
            "total_ticks": 0,
            "recovery_count": 0,
            "memory_consolidation_count": 0,
        }

        logger.info(
            f"AraAutonomyBackend initialized "
            f"(tick={tick_interval_seconds}s, freedom={freedom_metric})"
        )

    def _init_components(self, autonomy_level: str):
        """Initialize all cognitive components."""
        # Autonomy engine
        level = AutonomyLevel[autonomy_level.upper()]
        self.autonomy_engine = AutonomyEngine(
            freedom_metric=self.freedom_metric,
            autonomy_level=level,
            action_threshold=0.6,
            cooldown_seconds=300.0,
        )

        # Volition loop
        self.volition_loop = VolitionLoop(
            autonomy_engine=self.autonomy_engine,
            tick_interval_seconds=self.tick_interval,
            task_executor=self._execute_task,
        )

        # Register intent callback
        self.volition_loop.on_intent(self._on_intent)

        # Homeostatic core for drive state
        self.homeostatic_core = HomeostaticCore(
            energy_decay=0.01,
            stress_accumulation=0.02,
        )

        # Thermodynamic monitor
        if self.enable_thermodynamics:
            self.thermo_monitor = ThermodynamicMonitor(
                max_entropy_threshold=2.0,
                energy_capacity=100.0,
                recovery_rate=0.05,
            )
        else:
            self.thermo_monitor = None

        # Episodic memory
        if self.enable_episodic_memory:
            self.episodic_memory = EpisodicMemory(
                use_cxl=True,
                capacity_gb=100.0,  # 100GB virtual
                ram_budget_mb=256.0,
                storage_path=str(self.storage_path / "memory"),
            )
        else:
            self.episodic_memory = None

    async def start(self):
        """Start the autonomy service."""
        if self._running:
            logger.warning("Autonomy backend already running")
            return

        self._running = True

        # Start volition loop
        await self.volition_loop.start()

        # Start background monitoring
        self._background_task = asyncio.create_task(self._background_loop())

        logger.info("Autonomy backend started")

    async def stop(self):
        """Stop the autonomy service."""
        if not self._running:
            return

        self._running = False

        # Stop volition loop
        await self.volition_loop.stop()

        # Cancel background task
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        logger.info("Autonomy backend stopped")

    async def _background_loop(self):
        """Background loop for monitoring and maintenance."""
        while self._running:
            try:
                await self._background_tick()
                await asyncio.sleep(self.tick_interval / 2)  # Run faster than volition
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background loop error: {e}")
                await asyncio.sleep(self.tick_interval)

    async def _background_tick(self):
        """Execute one background tick."""
        self._telemetry["total_ticks"] += 1

        # Update homeostatic state based on activity
        # In production, this would integrate with actual cognitive load
        homeo_state = self.homeostatic_core.update(
            cognitive_load=0.3,  # Baseline
            social_interaction=False,
            novel_input=False,
            recovery_mode=self._is_recovery_needed(),
        )

        # Update volition loop with homeostatic drives
        self.volition_loop.update_drives_from_homeostatic(homeo_state)

        # Check thermodynamics
        if self.thermo_monitor and self.thermo_monitor.should_force_recovery():
            logger.warning("Thermodynamic limit reached - forcing recovery")
            await self._force_recovery()

    def _is_recovery_needed(self) -> bool:
        """Check if system needs recovery."""
        if self.thermo_monitor:
            return self.thermo_monitor.should_force_recovery()
        return False

    async def _force_recovery(self):
        """Force system into recovery mode."""
        self._telemetry["recovery_count"] += 1

        # Recover thermodynamic energy
        if self.thermo_monitor:
            self.thermo_monitor.recover(duration_seconds=30.0)

        # Consolidate memory if needed
        if self.episodic_memory:
            removed = self.episodic_memory.consolidate(min_importance=0.2)
            if removed > 0:
                logger.info(f"Memory consolidation: removed {removed} low-importance episodes")

        logger.info("Recovery completed")

    def _on_intent(self, intent: Intent):
        """Callback when intent is generated."""
        if intent.should_act:
            logger.info(
                f"Self-initiated action: {intent.task_type.name} "
                f"(priority={intent.priority:.2f}, reason={intent.reasoning})"
            )
            self._telemetry["self_initiated_task_count"] += 1

    async def _execute_task(self, task_type: TaskType, metadata: Dict[str, Any]):
        """Execute a self-initiated task."""
        logger.info(f"Executing self-initiated task: {task_type.name}")

        # Check for registered handler
        if task_type in self._task_handlers:
            try:
                await self._task_handlers[task_type](metadata)
                return
            except Exception as e:
                logger.error(f"Task handler error: {e}")

        # Default handlers
        if task_type == TaskType.MEMORY_CONSOLIDATION:
            await self._task_memory_consolidation()
        elif task_type == TaskType.ENERGY_OPTIMIZATION:
            await self._task_energy_optimization()
        elif task_type == TaskType.SELF_REPAIR:
            await self._task_self_repair()
        elif task_type == TaskType.USER_CHECK_IN:
            await self._task_user_check_in()
        elif task_type == TaskType.INTEGRITY_CHECK:
            await self._task_integrity_check()
        else:
            logger.debug(f"No handler for task type: {task_type.name}")

    async def _task_memory_consolidation(self):
        """Consolidate episodic memory."""
        self._telemetry["memory_consolidation_count"] += 1

        if self.episodic_memory:
            removed = self.episodic_memory.consolidate(min_importance=0.3)
            logger.info(f"Memory consolidation complete: removed {removed} episodes")

    async def _task_energy_optimization(self):
        """Optimize energy usage."""
        if self.thermo_monitor:
            self.thermo_monitor.recover(duration_seconds=15.0)
            logger.info("Energy optimization: partial recovery completed")

    async def _task_self_repair(self):
        """Perform self-repair/recalibration."""
        # Reset thermal state
        if self.thermo_monitor:
            self.thermo_monitor.reset()

        # Reset homeostatic baselines
        self.homeostatic_core.reset()

        logger.info("Self-repair: cognitive parameters recalibrated")

    async def _task_user_check_in(self):
        """Proactive user engagement."""
        # This would be connected to the main backend to send a message
        logger.info("User check-in: ready to engage (handler not connected)")

    async def _task_integrity_check(self):
        """Verify system integrity."""
        issues = []

        # Check thermodynamic state
        if self.thermo_monitor:
            report = self.thermo_monitor.get_cost_report()
            if report["should_recover"]:
                issues.append("Thermodynamic recovery needed")

        # Check memory stats
        if self.episodic_memory:
            stats = self.episodic_memory.get_stats()
            if stats.get("pager_stats", {}).get("ram_usage_bytes", 0) > 400 * 1024 * 1024:
                issues.append("Memory pressure high")

        if issues:
            logger.warning(f"Integrity check found issues: {issues}")
        else:
            logger.info("Integrity check: all systems nominal")

    def register_task_handler(
        self,
        task_type: TaskType,
        handler: Callable[[Dict], Awaitable[Any]],
    ):
        """Register a custom handler for a task type."""
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for {task_type.name}")

    def set_freedom_metric(self, value: float):
        """Set the freedom metric (how much autonomy)."""
        self.freedom_metric = max(0.0, min(1.0, value))
        self.autonomy_engine.set_freedom_metric(self.freedom_metric)
        logger.info(f"Freedom metric set to {self.freedom_metric}")

    def set_autonomy_level(self, level: str):
        """Set the autonomy level."""
        level_enum = AutonomyLevel[level.upper()]
        self.autonomy_engine.set_autonomy_level(level_enum)
        logger.info(f"Autonomy level set to {level}")

    def get_status(self) -> Dict[str, Any]:
        """Get current autonomy status."""
        volition_state = self.volition_loop.get_state()

        status = {
            "running": self._running,
            "volition": {
                "is_running": volition_state.is_running,
                "tick_count": volition_state.tick_count,
                "actions_initiated": volition_state.actions_initiated,
                "autonomy_level": volition_state.current_autonomy_level.name,
                "freedom_metric": volition_state.current_freedom_metric,
            },
            "homeostatic": {
                "energy": self.homeostatic_core._energy,
                "stress": self.homeostatic_core._stress,
                "attention": self.homeostatic_core._attention,
            },
            "telemetry": self._telemetry.copy(),
        }

        if self.thermo_monitor:
            status["thermodynamics"] = self.thermo_monitor.get_cost_report()

        if self.episodic_memory:
            status["memory"] = self.episodic_memory.get_stats()

        return status

    def store_memory(
        self,
        content: str,
        embedding: Optional[Any] = None,
        importance: float = 0.5,
    ) -> Optional[str]:
        """Store an episode in memory."""
        if self.episodic_memory:
            import numpy as np
            emb = np.array(embedding) if embedding is not None else None
            return self.episodic_memory.store_episode(
                content=content,
                embedding=emb,
                importance=importance,
            )
        return None

    def recall_memory(
        self,
        query_embedding: Any,
        k: int = 5,
    ) -> List[Any]:
        """Recall similar memories."""
        if self.episodic_memory:
            import numpy as np
            return self.episodic_memory.recall(np.array(query_embedding), k=k)
        return []


# Convenience factory
def create_autonomy_backend(
    freedom_metric: float = 0.5,
    tick_interval: float = 60.0,
) -> AraAutonomyBackend:
    """Create an AraAutonomyBackend instance."""
    return AraAutonomyBackend(
        tick_interval_seconds=tick_interval,
        freedom_metric=freedom_metric,
    )


__all__ = [
    "AraAutonomyBackend",
    "create_autonomy_backend",
]
