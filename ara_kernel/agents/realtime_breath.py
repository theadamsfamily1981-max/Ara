"""
Realtime Breath Agent
======================

Tiny wrapper that feeds "heartbeat" events into the kernel:
- Status updates
- Telemetry checks
- Drift control
- Background maintenance

This is the "always-on" agent that keeps Ara responsive.

Thread Safety:
- Uses persistent event loop per thread (not created per-heartbeat)
- Uses threading.Event for stop signal with proper wait semantics
"""

from __future__ import annotations

import asyncio
import threading
import time
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ara_kernel.core.runtime import AraAgentRuntime

logger = logging.getLogger(__name__)


class RealtimeBreathAgent:
    """
    Background agent that periodically sends heartbeat events.

    Use this for:
    - Periodic status checks
    - Memory compaction triggers
    - Drift detection
    - Idle-time background tasks

    Thread Safety:
    Uses a persistent event loop for the thread lifetime.
    """

    def __init__(
        self,
        kernel: AraAgentRuntime,
        interval_sec: float = 30.0,
        domain: str = "realtime",
    ) -> None:
        self.kernel = kernel
        self.interval_sec = interval_sec
        self.domain = domain

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._heartbeat_count = 0
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        """Start the heartbeat loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("RealtimeBreathAgent already running")
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="RealtimeBreathAgent"
        )
        self._thread.start()
        logger.info(f"RealtimeBreathAgent started (interval={self.interval_sec}s)")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the heartbeat loop."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("RealtimeBreathAgent did not stop cleanly")
        logger.info(f"RealtimeBreathAgent stopped ({self._heartbeat_count} heartbeats)")

    def is_running(self) -> bool:
        """Check if the agent is running."""
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        """Main heartbeat loop with persistent event loop."""
        # Create persistent event loop for this thread
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        try:
            while not self._stop.wait(timeout=self.interval_sec):
                self._heartbeat()
        finally:
            # Clean up the event loop
            try:
                self._event_loop.run_until_complete(self._event_loop.shutdown_asyncgens())
            except Exception:
                pass
            self._event_loop.close()
            self._event_loop = None

    def _heartbeat(self) -> None:
        """Execute a single heartbeat using the persistent event loop."""
        self._heartbeat_count += 1
        event: Dict[str, Any] = {
            "type": "heartbeat",
            "domain": self.domain,
            "text": f"Periodic status check #{self._heartbeat_count}",
            "heartbeat_id": self._heartbeat_count,
            "timestamp": time.time(),
        }

        try:
            if self._event_loop is None:
                raise RuntimeError("Event loop not initialized")

            result = self._event_loop.run_until_complete(
                self.kernel.process_input(
                    user_input=event["text"],
                    mode="private",
                    metadata=event,
                )
            )
            logger.debug(f"Heartbeat #{self._heartbeat_count} completed: {result.get('text', '')[:100]}")

        except Exception as e:
            logger.exception(f"Heartbeat #{self._heartbeat_count} failed: {e}")


class DriftMonitor:
    """
    Monitors for "drift" - when Ara's behavior deviates from expected patterns.

    Emits alarms when:
    - Too many failures in a row
    - Unusual patterns in pheromones
    - Memory growth anomalies
    """

    def __init__(
        self,
        kernel: AraAgentRuntime,
        failure_threshold: int = 5,
    ) -> None:
        self.kernel = kernel
        self.failure_threshold = failure_threshold
        self._consecutive_failures = 0

    def check(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Check result for drift indicators.

        Returns drift reason if detected, None otherwise.
        """
        # Check for failures
        if result.get("error"):
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                return f"consecutive_failures:{self._consecutive_failures}"
        else:
            self._consecutive_failures = 0

        # Check pheromone alarms
        if hasattr(self.kernel, "pheromone_store") and self.kernel.pheromone_store:
            pheromones = self.kernel.pheromone_store
            if hasattr(pheromones, "has_alarm") and pheromones.has_alarm():
                return "alarm_active"

        return None

    def reset(self) -> None:
        """Reset drift counters."""
        self._consecutive_failures = 0
