"""
Hive Maintenance Jobs
======================

Background tasks for maintaining hive health:
- Intensity evaporation
- Stale node detection
- Congestion cooling
- Task cleanup

Thread Safety:
All maintenance jobs use threading.Event for stop signals, ensuring
reliable shutdown semantics. The WaggleBoard they access is now
thread-safe (uses RLock + WAL mode).
"""

from __future__ import annotations

import time
import threading
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .waggle_board import WaggleBoard

logger = logging.getLogger(__name__)


# Default timeouts for graceful shutdown
SHUTDOWN_TIMEOUT_SEC = 5.0


class EvaporationJob:
    """
    Periodically decays site intensities.

    This implements pheromone evaporation - without new activity,
    sites gradually become less attractive.

    Thread Safety:
    Uses threading.Event for stop signal, ensuring reliable shutdown.
    """

    def __init__(
        self,
        board: WaggleBoard,
        interval_sec: float = 30.0,
        decay_factor: float = 0.95,
        min_intensity: float = 0.01,
    ) -> None:
        self.board = board
        self.interval_sec = interval_sec
        self.decay_factor = decay_factor
        self.min_intensity = min_intensity

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start evaporation in background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="EvaporationJob")
        self._thread.start()
        logger.info(
            f"EvaporationJob started (interval={self.interval_sec}s, "
            f"decay={self.decay_factor})"
        )

    def stop(self) -> None:
        """Stop the evaporation job."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=SHUTDOWN_TIMEOUT_SEC)
            if self._thread.is_alive():
                logger.warning("EvaporationJob did not stop cleanly")
        logger.info("EvaporationJob stopped")

    def run_once(self) -> int:
        """Run one evaporation pass."""
        affected = self.board.evaporate_intensity(
            decay_factor=self.decay_factor,
            min_intensity=self.min_intensity,
        )
        logger.debug(f"Evaporated intensity on {affected} sites")
        return affected

    def _loop(self) -> None:
        """Main evaporation loop."""
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:
                logger.exception(f"Evaporation error: {e}")

            # Use Event.wait() for interruptible sleep
            self._stop_event.wait(timeout=self.interval_sec)


class StaleNodeDetector:
    """
    Marks nodes with stale heartbeats as offline.

    Thread Safety:
    Uses threading.Event for stop signal, ensuring reliable shutdown.
    """

    def __init__(
        self,
        board: WaggleBoard,
        interval_sec: float = 30.0,
        timeout_sec: float = 60.0,
    ) -> None:
        self.board = board
        self.interval_sec = interval_sec
        self.timeout_sec = timeout_sec

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start detection in background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="StaleNodeDetector")
        self._thread.start()
        logger.info(f"StaleNodeDetector started (timeout={self.timeout_sec}s)")

    def stop(self) -> None:
        """Stop the detector."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=SHUTDOWN_TIMEOUT_SEC)
            if self._thread.is_alive():
                logger.warning("StaleNodeDetector did not stop cleanly")
        logger.info("StaleNodeDetector stopped")

    def run_once(self) -> int:
        """Run one detection pass."""
        affected = self.board.mark_stale_nodes_offline(timeout_sec=self.timeout_sec)
        if affected > 0:
            logger.warning(f"Marked {affected} stale nodes as offline")
        return affected

    def _loop(self) -> None:
        """Main detection loop."""
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:
                logger.exception(f"Stale node detection error: {e}")

            # Use Event.wait() for interruptible sleep
            self._stop_event.wait(timeout=self.interval_sec)


class CongestionCooler:
    """
    Monitors node load and applies cooling pheromones to overloaded nodes.

    Thread Safety:
    Uses threading.Event for stop signal, ensuring reliable shutdown.
    """

    def __init__(
        self,
        board: WaggleBoard,
        interval_sec: float = 10.0,
        cpu_threshold: float = 0.8,
        cooling_factor: float = 0.5,
    ) -> None:
        self.board = board
        self.interval_sec = interval_sec
        self.cpu_threshold = cpu_threshold
        self.cooling_factor = cooling_factor

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start cooling in background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CongestionCooler")
        self._thread.start()
        logger.info(f"CongestionCooler started (threshold={self.cpu_threshold})")

    def stop(self) -> None:
        """Stop the cooler."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=SHUTDOWN_TIMEOUT_SEC)
            if self._thread.is_alive():
                logger.warning("CongestionCooler did not stop cleanly")
        logger.info("CongestionCooler stopped")

    def run_once(self) -> int:
        """Run one cooling pass."""
        cooled = 0
        nodes = self.board.list_nodes(status="online")

        for node in nodes:
            if node.cpu_load > self.cpu_threshold:
                affected = self.board.cool_node_sites(
                    node_id=node.id,
                    factor=self.cooling_factor,
                )
                if affected > 0:
                    logger.info(
                        f"Cooled {affected} sites on overloaded node {node.id} "
                        f"(cpu={node.cpu_load:.1%})"
                    )
                    cooled += affected

        return cooled

    def _loop(self) -> None:
        """Main cooling loop."""
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:
                logger.exception(f"Congestion cooling error: {e}")

            # Use Event.wait() for interruptible sleep
            self._stop_event.wait(timeout=self.interval_sec)


class HiveMaintainer:
    """
    Runs all maintenance jobs together.
    """

    def __init__(
        self,
        board: WaggleBoard,
        evaporation_interval: float = 30.0,
        stale_check_interval: float = 30.0,
        cooling_interval: float = 10.0,
    ) -> None:
        self.evaporation = EvaporationJob(board, interval_sec=evaporation_interval)
        self.stale_detector = StaleNodeDetector(board, interval_sec=stale_check_interval)
        self.cooler = CongestionCooler(board, interval_sec=cooling_interval)

    def start(self) -> None:
        """Start all maintenance jobs."""
        self.evaporation.start()
        self.stale_detector.start()
        self.cooler.start()
        logger.info("HiveMaintainer started all jobs")

    def stop(self) -> None:
        """Stop all maintenance jobs."""
        self.evaporation.stop()
        self.stale_detector.stop()
        self.cooler.stop()
        logger.info("HiveMaintainer stopped all jobs")
