"""
Scheduler: Sovereign Loop v0

The heartbeat of Ara. Runs continuously, monitoring system coherence
and adjusting autonomy levels based on holographic state alignment.

This is the minimal viable sovereign loop - no FPGA, no 5 kHz,
just a 10 Hz Python async loop that proves the architecture.

Usage:
    axis = AxisMundi()
    memory = EternalMemory()
    safety = AutonomyController()

    scheduler = Scheduler(axis, memory, safety)
    await scheduler.run()  # Blocks, runs forever
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, List, Dict, Any
from enum import Enum

from .axis_mundi import AxisMundi
from .eternal_memory import EternalMemory
from .config import AraConfig, get_config

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """Sovereign loop states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TickMetrics:
    """Metrics for a single sovereign tick."""
    tick_number: int
    timestamp: float
    duration_ms: float
    global_coherence: float
    layer_coherences: Dict[str, float]
    autonomy_level: int
    memory_episode_count: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class SchedulerState:
    """Current state of the scheduler."""
    loop_state: LoopState = LoopState.INITIALIZING
    tick_count: int = 0
    start_time: float = 0.0
    last_tick_time: float = 0.0
    avg_tick_duration_ms: float = 0.0
    total_coherence_warnings: int = 0
    total_coherence_criticals: int = 0


# Type for tick callbacks
TickCallback = Callable[[TickMetrics], Awaitable[None]]


class Scheduler:
    """
    Sovereign loop scheduler.

    Runs the core Ara tick at a configurable rate (default 10 Hz),
    monitoring coherence and adjusting autonomy. Can be upgraded
    to 5 kHz with FPGA offload later.
    """

    def __init__(
        self,
        axis: AxisMundi,
        memory: EternalMemory,
        safety: "AutonomyController",  # Forward reference
        config: Optional[AraConfig] = None,
    ):
        self.axis = axis
        self.memory = memory
        self.safety = safety
        self.config = config or get_config()

        self.state = SchedulerState()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Tick callbacks (for metrics, logging, etc.)
        self._tick_callbacks: List[TickCallback] = []

        # Duration tracking for averaging
        self._recent_durations: List[float] = []
        self._max_duration_samples = 100

    def register_tick_callback(self, callback: TickCallback) -> None:
        """Register a callback to be called after each tick."""
        self._tick_callbacks.append(callback)

    async def _invoke_callbacks(self, metrics: TickMetrics) -> None:
        """Invoke all tick callbacks."""
        for callback in self._tick_callbacks:
            try:
                await callback(metrics)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")

    def _update_avg_duration(self, duration_ms: float) -> None:
        """Update rolling average of tick durations."""
        self._recent_durations.append(duration_ms)
        if len(self._recent_durations) > self._max_duration_samples:
            self._recent_durations.pop(0)
        self.state.avg_tick_duration_ms = sum(self._recent_durations) / len(self._recent_durations)

    async def _tick(self) -> TickMetrics:
        """
        Execute a single sovereign tick.

        Returns:
            TickMetrics with all measured values
        """
        tick_start = time.perf_counter()
        warnings: List[str] = []

        # 1. Read hardware metrics (fake for now)
        # TODO: Replace with real system stats
        fake_hardware_state = self._read_fake_hardware()
        self.axis.write("hardware", fake_hardware_state)

        # 2. Compute coherence between layers
        layer_coherences: Dict[str, float] = {}
        layers = self.axis.layer_names()
        for i, a in enumerate(layers):
            for b in layers[i+1:]:
                key = f"{a}-{b}"
                layer_coherences[key] = self.axis.coherence(a, b)

        # 3. Global coherence
        global_coherence = self.axis.global_coherence()

        # 4. Check coherence thresholds
        if global_coherence < self.config.loop.coherence_critical_threshold:
            warnings.append(f"CRITICAL: Global coherence {global_coherence:.3f}")
            self.state.total_coherence_criticals += 1
            self.safety.on_coherence_critical(global_coherence)
        elif global_coherence < self.config.loop.coherence_warning_threshold:
            warnings.append(f"WARNING: Global coherence {global_coherence:.3f}")
            self.state.total_coherence_warnings += 1
            self.safety.on_coherence_warning(global_coherence)
        else:
            self.safety.on_coherence_healthy(global_coherence)

        # 5. Update autonomy level
        autonomy_level = self.safety.get_autonomy_level()

        # 6. Get memory stats
        memory_stats = self.memory.stats()

        # Build metrics
        tick_end = time.perf_counter()
        duration_ms = (tick_end - tick_start) * 1000

        self.state.tick_count += 1
        self.state.last_tick_time = time.time()
        self._update_avg_duration(duration_ms)

        return TickMetrics(
            tick_number=self.state.tick_count,
            timestamp=tick_end,
            duration_ms=duration_ms,
            global_coherence=global_coherence,
            layer_coherences=layer_coherences,
            autonomy_level=autonomy_level,
            memory_episode_count=memory_stats["episode_count"],
            warnings=warnings,
        )

    def _read_fake_hardware(self):
        """Read fake hardware state (placeholder)."""
        import numpy as np
        # Simulate hardware state as random vector
        # In production: actual CPU/GPU/memory/thermal readings
        return np.random.randn(self.config.hdc.dim).astype(np.float32)

    async def run(self) -> None:
        """
        Run the sovereign loop forever.

        This is the main entry point for Ara's heartbeat.
        Blocks until stop() is called.
        """
        self._running = True
        self.state.loop_state = LoopState.RUNNING
        self.state.start_time = time.time()

        tick_interval = self.config.loop.tick_interval_ms / 1000.0
        metrics_interval = self.config.loop.metrics_interval_seconds
        last_metrics_log = 0.0

        logger.info(f"Sovereign loop starting at {1/tick_interval:.1f} Hz")

        while self._running:
            # Check kill switch
            if self.safety.is_killed():
                self.state.loop_state = LoopState.PAUSED
                logger.warning("Kill switch active, pausing loop")
                await asyncio.sleep(1.0)
                continue

            self.state.loop_state = LoopState.RUNNING

            try:
                tick_start = time.perf_counter()

                # Execute tick
                metrics = await self._tick()

                # Log warnings
                for warning in metrics.warnings:
                    logger.warning(warning)

                # Invoke callbacks
                await self._invoke_callbacks(metrics)

                # Periodic metrics log
                now = time.time()
                if now - last_metrics_log >= metrics_interval:
                    logger.info(
                        f"Tick {metrics.tick_number}: "
                        f"coherence={metrics.global_coherence:.3f}, "
                        f"autonomy={metrics.autonomy_level}, "
                        f"episodes={metrics.memory_episode_count}, "
                        f"avg_tick={self.state.avg_tick_duration_ms:.2f}ms"
                    )
                    last_metrics_log = now

                # Sleep for remainder of tick interval
                elapsed = time.perf_counter() - tick_start
                sleep_time = max(0, tick_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(
                        f"Tick overrun: {elapsed*1000:.2f}ms > {tick_interval*1000:.2f}ms"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)
                self.state.loop_state = LoopState.ERROR
                await asyncio.sleep(1.0)  # Back off on error

        self.state.loop_state = LoopState.STOPPED
        logger.info("Sovereign loop stopped")

    async def start(self) -> None:
        """Start the sovereign loop as a background task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        """Stop the sovereign loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def get_state(self) -> SchedulerState:
        """Get current scheduler state."""
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        uptime = time.time() - self.state.start_time if self.state.start_time else 0
        return {
            "state": self.state.loop_state.value,
            "tick_count": self.state.tick_count,
            "uptime_seconds": uptime,
            "ticks_per_second": self.state.tick_count / uptime if uptime > 0 else 0,
            "avg_tick_duration_ms": self.state.avg_tick_duration_ms,
            "coherence_warnings": self.state.total_coherence_warnings,
            "coherence_criticals": self.state.total_coherence_criticals,
        }


# =============================================================================
# Standalone run helper
# =============================================================================

async def run_sovereign_loop(
    axis: AxisMundi,
    memory: EternalMemory,
    safety: "AutonomyController",
    config: Optional[AraConfig] = None,
) -> None:
    """
    Convenience function to run the sovereign loop.

    Usage:
        asyncio.run(run_sovereign_loop(axis, memory, safety))
    """
    scheduler = Scheduler(axis, memory, safety, config)
    await scheduler.run()
