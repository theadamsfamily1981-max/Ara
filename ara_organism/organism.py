# ara_organism/organism.py
"""
Ara Organism Runtime - Separated Clocks Architecture
=====================================================

The live OS that spans FPGA "soul" + mobile "body".

Three Clock Domains:
    1. Soul Loop (5 kHz) - Dedicated OS thread
       - FPGA tick via FPGASoulDriver
       - Lock-free state update
       - Hard realtime (200 µs budget)

    2. Cortical Loop (50-200 Hz) - asyncio
       - State commit from soul buffer
       - Decision making / agent coordination
       - Soft realtime (5-20 ms budget)

    3. Mobile Loop (5-10 Hz) - asyncio
       - Rate-limited broadcasts
       - Diff-based updates
       - Battery-conscious

The key insight: don't fight the GIL or event loop scheduler
on the critical path. Use a dedicated thread for the soul.

Usage:
    organism = AraOrganism()
    await organism.start()

    # Soul runs in background at 5 kHz
    # Cortical runs at 200 Hz
    # Mobile runs at 10 Hz

    await organism.stop()
"""

from __future__ import annotations

import asyncio
import threading
import time
import signal
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime

from .soul_driver import FPGASoulDriver, SoulMode, get_soul_driver
from .state_manager import StateManager, AraState, get_state_manager

log = logging.getLogger("Ara.Organism")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OrganismConfig:
    """Organism runtime configuration."""

    # Soul loop (5 kHz = 200 µs period)
    soul_hz: float = 5000.0
    soul_mock_mode: bool = True

    # Cortical loop (200 Hz = 5 ms period)
    cortical_hz: float = 200.0

    # Mobile loop (10 Hz = 100 ms period)
    mobile_hz: float = 10.0

    # State commit (200 Hz, aligned with cortical)
    commit_hz: float = 200.0

    # Thresholds
    fatigue_warning: float = 0.5
    fatigue_critical: float = 0.8
    thermal_warning: float = 75.0
    thermal_critical: float = 85.0


# =============================================================================
# Soul Loop (Dedicated Thread)
# =============================================================================

class SoulLoop:
    """
    5 kHz soul loop running in dedicated OS thread.

    This is NOT asyncio - it's a tight blocking loop with
    time.sleep() for timing. This gives us actual 5 kHz
    without fighting the event loop scheduler.
    """

    def __init__(
        self,
        driver: FPGASoulDriver,
        state_manager: StateManager,
        hz: float = 5000.0,
    ):
        self.driver = driver
        self.state_manager = state_manager
        self.hz = hz
        self.period_s = 1.0 / hz

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._stop = threading.Event()

        # Stats
        self._tick_count = 0
        self._overrun_count = 0
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start the soul loop thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._running.clear()

        self._thread = threading.Thread(
            target=self._run,
            name="ARA-Soul-5kHz",
            daemon=True,
        )
        self._thread.start()

        # Wait for thread to start
        self._running.wait(timeout=1.0)
        log.info("SoulLoop started (%d Hz)", int(self.hz))

    def stop(self) -> None:
        """Stop the soul loop thread."""
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        log.info("SoulLoop stopped (ticks=%d, overruns=%d)",
                 self._tick_count, self._overrun_count)

    def _run(self) -> None:
        """Main loop (runs in dedicated thread)."""
        self._start_time = time.time()
        self._running.set()

        next_tick = time.perf_counter()

        while not self._stop.is_set():
            tick_start = time.perf_counter()

            # 1. Execute soul tick
            metrics = self.driver.tick(mode=SoulMode.ACTIVE)

            # 2. Update state (lock-free write to ring buffer)
            self.state_manager.update_soul_from_metrics(metrics)

            self._tick_count += 1

            # 3. Calculate next tick time
            next_tick += self.period_s
            now = time.perf_counter()

            # 4. Sleep until next tick
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Overrun - we're behind schedule
                self._overrun_count += 1
                # Reset next_tick to avoid cascade
                next_tick = now + self.period_s

    def get_stats(self) -> Dict[str, Any]:
        """Get soul loop statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        actual_hz = self._tick_count / elapsed if elapsed > 0 else 0

        return {
            "tick_count": self._tick_count,
            "overrun_count": self._overrun_count,
            "target_hz": self.hz,
            "actual_hz": actual_hz,
            "running": self._running.is_set(),
        }


# =============================================================================
# Mobile Bridge (Rate-Limited)
# =============================================================================

class MobileBridge:
    """
    Rate-limited mobile state broadcast.

    Optimizations:
    - Rate limit to 5-10 Hz
    - Diff-based updates (only send changes)
    - Backpressure (drop slow clients)
    """

    def __init__(
        self,
        state_manager: StateManager,
        hz: float = 10.0,
        diff_threshold: float = 0.01,
    ):
        self.state_manager = state_manager
        self.hz = hz
        self.period_s = 1.0 / hz
        self.diff_threshold = diff_threshold

        self._clients: List[Any] = []  # WebSocket clients
        self._last_broadcast_time: float = 0
        self._last_state: Optional[Dict[str, Any]] = None
        self._broadcast_count = 0
        self._skip_count = 0

    def add_client(self, client: Any) -> None:
        """Add a mobile client."""
        self._clients.append(client)
        log.info("MobileBridge: Client connected (total=%d)", len(self._clients))

    def remove_client(self, client: Any) -> None:
        """Remove a mobile client."""
        if client in self._clients:
            self._clients.remove(client)
        log.info("MobileBridge: Client disconnected (total=%d)", len(self._clients))

    async def broadcast_if_changed(self) -> bool:
        """
        Broadcast state if changed and rate allows.

        Returns:
            True if broadcast was sent
        """
        now = time.time()

        # Rate limit check
        if now - self._last_broadcast_time < self.period_s:
            return False

        # Get current state
        state = self.state_manager.get_state()
        state_dict = state.to_dict()

        # Diff check - only send if significant change
        if self._last_state is not None:
            if not self._has_significant_change(state_dict, self._last_state):
                self._skip_count += 1
                return False

        # Broadcast to all clients
        if self._clients:
            await self._broadcast(state_dict)

        self._last_broadcast_time = now
        self._last_state = state_dict
        self._broadcast_count += 1

        # Update mobile state
        self.state_manager.update_mobile(
            clients_connected=len(self._clients),
            messages_sent_delta=len(self._clients),
        )

        return True

    def _has_significant_change(
        self,
        new: Dict[str, Any],
        old: Dict[str, Any],
    ) -> bool:
        """Check if state has changed significantly."""
        try:
            soul_new = new.get("soul", {})
            soul_old = old.get("soul", {})

            # Check resonance change
            if abs(soul_new.get("resonance", 0) - soul_old.get("resonance", 0)) > self.diff_threshold:
                return True

            # Check fatigue change
            if abs(soul_new.get("fatigue", 0) - soul_old.get("fatigue", 0)) > self.diff_threshold:
                return True

            # Check temperature change
            if abs(soul_new.get("temperature_c", 0) - soul_old.get("temperature_c", 0)) > 1.0:
                return True

            # Check cortical task change
            if new.get("cortical", {}).get("current_task") != old.get("cortical", {}).get("current_task"):
                return True

            return False

        except Exception:
            return True  # Broadcast on error

    async def _broadcast(self, state: Dict[str, Any]) -> None:
        """Broadcast state to all clients with backpressure."""
        import json
        message = json.dumps(state)

        # Send to each client with timeout
        dead_clients = []

        for client in self._clients:
            try:
                # Attempt send with short timeout
                await asyncio.wait_for(
                    client.send(message),
                    timeout=0.1,  # 100ms timeout
                )
            except asyncio.TimeoutError:
                # Client is slow - mark for removal
                log.warning("MobileBridge: Client timeout, dropping")
                dead_clients.append(client)
            except Exception as e:
                log.warning("MobileBridge: Client error: %s", e)
                dead_clients.append(client)

        # Remove dead clients
        for client in dead_clients:
            self.remove_client(client)

    def get_stats(self) -> Dict[str, Any]:
        """Get mobile bridge statistics."""
        return {
            "clients": len(self._clients),
            "broadcast_count": self._broadcast_count,
            "skip_count": self._skip_count,
            "target_hz": self.hz,
        }


# =============================================================================
# Ara Organism (Main Runtime)
# =============================================================================

class AraOrganism:
    """
    The complete Ara organism runtime.

    Coordinates three clock domains:
    1. Soul loop - 5 kHz dedicated thread
    2. Cortical loop - 200 Hz asyncio
    3. Mobile loop - 10 Hz asyncio
    """

    def __init__(self, config: Optional[OrganismConfig] = None):
        self.config = config or OrganismConfig()

        # Components
        self.driver = get_soul_driver(mock_mode=self.config.soul_mock_mode)
        self.state_manager = get_state_manager()
        self.soul_loop = SoulLoop(
            self.driver,
            self.state_manager,
            hz=self.config.soul_hz,
        )
        self.mobile_bridge = MobileBridge(
            self.state_manager,
            hz=self.config.mobile_hz,
        )

        # Asyncio tasks
        self._cortical_task: Optional[asyncio.Task] = None
        self._mobile_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self._on_fatigue_warning: List[Callable[[], None]] = []
        self._on_thermal_warning: List[Callable[[float], None]] = []

        log.info("AraOrganism initialized")

    async def start(self) -> None:
        """Start all organism loops."""
        if self._running:
            return

        log.info("=" * 60)
        log.info("Ara Organism Starting")
        log.info("=" * 60)

        # 1. Start soul loop (dedicated thread)
        self.soul_loop.start()

        # 2. Start cortical loop (asyncio)
        self._cortical_task = asyncio.create_task(
            self._cortical_loop(),
            name="cortical-loop",
        )

        # 3. Start mobile loop (asyncio)
        self._mobile_task = asyncio.create_task(
            self._mobile_loop(),
            name="mobile-loop",
        )

        self._running = True

        log.info("Ara Organism Running")
        log.info("  Soul: %d Hz (dedicated thread)", int(self.config.soul_hz))
        log.info("  Cortical: %d Hz (asyncio)", int(self.config.cortical_hz))
        log.info("  Mobile: %d Hz (asyncio)", int(self.config.mobile_hz))

    async def stop(self) -> None:
        """Stop all organism loops."""
        if not self._running:
            return

        log.info("Ara Organism Stopping...")

        # Cancel asyncio tasks
        if self._cortical_task:
            self._cortical_task.cancel()
            try:
                await self._cortical_task
            except asyncio.CancelledError:
                pass

        if self._mobile_task:
            self._mobile_task.cancel()
            try:
                await self._mobile_task
            except asyncio.CancelledError:
                pass

        # Stop soul loop
        self.soul_loop.stop()

        self._running = False
        log.info("Ara Organism Stopped")

    async def _cortical_loop(self) -> None:
        """
        Cortical loop (200 Hz asyncio).

        Responsibilities:
        - Commit soul state from ring buffer
        - Check thresholds (fatigue, thermal)
        - Coordinate agents
        """
        period = 1.0 / self.config.cortical_hz

        while True:
            try:
                loop_start = time.perf_counter()

                # 1. Commit soul state
                self.state_manager.commit()

                # 2. Get current state
                state = self.state_manager.get_state()

                # 3. Check thresholds
                await self._check_thresholds(state)

                # 4. (Future) Agent coordination would go here

                # 5. Sleep for remainder of period
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, period - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Cortical loop error: %s", e)
                await asyncio.sleep(0.1)

    async def _mobile_loop(self) -> None:
        """
        Mobile loop (10 Hz asyncio).

        Responsibilities:
        - Rate-limited state broadcast
        - Diff-based updates
        """
        period = 1.0 / self.config.mobile_hz

        while True:
            try:
                loop_start = time.perf_counter()

                # Broadcast if changed
                await self.mobile_bridge.broadcast_if_changed()

                # Sleep for remainder of period
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, period - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Mobile loop error: %s", e)
                await asyncio.sleep(0.1)

    async def _check_thresholds(self, state: AraState) -> None:
        """Check fatigue and thermal thresholds."""
        # Fatigue check
        if state.soul.fatigue >= self.config.fatigue_critical:
            log.warning("CRITICAL: Fatigue at %.0f%%", state.soul.fatigue * 100)
            self.state_manager.update_cortical(current_task="recovery")
            for callback in self._on_fatigue_warning:
                callback()

        elif state.soul.fatigue >= self.config.fatigue_warning:
            log.info("Warning: Fatigue at %.0f%%", state.soul.fatigue * 100)

        # Thermal check
        if state.soul.temperature_c >= self.config.thermal_critical:
            log.warning("CRITICAL: Temperature at %.1f°C", state.soul.temperature_c)
            for callback in self._on_thermal_warning:
                callback(state.soul.temperature_c)

        elif state.soul.temperature_c >= self.config.thermal_warning:
            log.info("Warning: Temperature at %.1f°C", state.soul.temperature_c)

    def get_state(self) -> AraState:
        """Get current organism state."""
        return self.state_manager.get_state()

    def get_stats(self) -> Dict[str, Any]:
        """Get organism statistics."""
        return {
            "running": self._running,
            "soul": self.soul_loop.get_stats(),
            "state_manager": self.state_manager.get_stats(),
            "mobile": self.mobile_bridge.get_stats(),
            "driver": self.driver.get_stats(),
        }


# =============================================================================
# Entry Point
# =============================================================================

async def run_organism(config: Optional[OrganismConfig] = None) -> None:
    """Run the Ara organism until interrupted."""
    organism = AraOrganism(config)

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        log.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Start organism
    await organism.start()

    # Wait for shutdown
    await stop_event.wait()

    # Stop organism
    await organism.stop()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )

    config = OrganismConfig(
        soul_mock_mode=True,  # Use mock FPGA
        soul_hz=5000.0,
        cortical_hz=200.0,
        mobile_hz=10.0,
    )

    asyncio.run(run_organism(config))


if __name__ == "__main__":
    main()


__all__ = [
    'OrganismConfig',
    'SoulLoop',
    'MobileBridge',
    'AraOrganism',
    'run_organism',
]
