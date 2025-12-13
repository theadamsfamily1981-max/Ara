#!/usr/bin/env python3
"""
Body Schema Daemon - L2 Autonomic Control
==========================================

The "autonomic nervous system" - maintains homeostasis and publishes BodyState.

This daemon:
1. Runs L1 reflexes at high frequency
2. Aggregates sensor data into coherent BodyState
3. Publishes state for L3 (cognitive layer) consumption
4. Manages operating modes (QUIET, BALANCED, PERFORMANCE)

The daemon is the bridge between raw hardware and cognitive awareness.
It doesn't think - it regulates.

Usage:
    # Start as background daemon
    python -m ara.body.daemon &

    # Or run directly
    daemon = BodySchemaDaemon()
    daemon.run(frequency_hz=5.0)

IPC:
    State is published to /tmp/ara_body_state.json (atomic writes).
    L3 can read this file to get current body state.
"""

from __future__ import annotations

import time
import json
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Thread, Event

from .schema import (
    BodyState,
    BodyMode,
    ThermalState,
    PowerState,
    compute_stress,
    classify_thermal_state,
)
from .reflexes import ThermalReflex

logger = logging.getLogger("ara.body")


# =============================================================================
# Body Schema Daemon
# =============================================================================

class BodySchemaDaemon:
    """
    L2 Controller - Autonomic Nervous System.

    Responsibility: Maintain homeostasis and BodySchema.

    This daemon runs continuously, polling sensors, triggering reflexes,
    and publishing body state for the cognitive layer.

    Example:
        daemon = BodySchemaDaemon()

        # Run in foreground
        daemon.run(frequency_hz=5.0)

        # Or run in background thread
        daemon.start_background()
        # ... later ...
        daemon.stop()
    """

    DEFAULT_STATE_FILE = Path("/tmp/ara_body_state.json")

    def __init__(
        self,
        state_file: Optional[Path] = None,
        dry_run: bool = True,
        mode: BodyMode = BodyMode.BALANCED,
    ):
        """
        Initialize body daemon.

        Args:
            state_file: Path for IPC state file
            dry_run: If True, reflexes log but don't act
            mode: Initial operating mode
        """
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        self.mode = mode

        # L1 reflex layer
        self.reflex_layer = ThermalReflex(dry_run=dry_run)

        # Runtime state
        self.running = False
        self._stop_event = Event()
        self._background_thread: Optional[Thread] = None

        # Current state cache
        self._current_state: Optional[BodyState] = None
        self._state_history: list = []
        self._max_history = 100

        # Statistics
        self._loop_count = 0
        self._start_time = 0.0

    def run(self, frequency_hz: float = 5.0):
        """
        Run daemon in foreground (blocking).

        Args:
            frequency_hz: Main loop frequency (5Hz default)
        """
        self.running = True
        self._stop_event.clear()
        self._start_time = time.time()

        logger.info(f"Body Daemon starting. Mode: {self.mode.value}")
        logger.info(f"State file: {self.state_file}")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        period = 1.0 / frequency_hz

        while self.running and not self._stop_event.is_set():
            loop_start = time.time()

            try:
                self._main_loop()
            except Exception as e:
                logger.error(f"Loop error: {e}")

            # Maintain frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, period - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

        logger.info("Body Daemon stopped")

    def start_background(self, frequency_hz: float = 5.0):
        """
        Start daemon in background thread.

        Args:
            frequency_hz: Main loop frequency
        """
        if self._background_thread and self._background_thread.is_alive():
            logger.warning("Daemon already running")
            return

        self._background_thread = Thread(
            target=self.run,
            args=(frequency_hz,),
            daemon=True,
            name="ara-body-daemon",
        )
        self._background_thread.start()
        logger.info("Body daemon started in background")

    def stop(self):
        """Stop the daemon."""
        self.running = False
        self._stop_event.set()

        if self._background_thread:
            self._background_thread.join(timeout=2.0)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _main_loop(self):
        """
        Main control loop iteration.

        1. Run L1 reflexes
        2. Aggregate sensors
        3. Compute body state
        4. Publish to IPC
        """
        self._loop_count += 1

        # 1. Run L1 Reflexes (fast, protective)
        reflex_events = self.reflex_layer.scan_and_react()
        reflex_strings = [str(e) for e in reflex_events]

        # 2. Read sensors
        temps = self.reflex_layer._last_temps  # Use cached from reflex
        if not temps:
            temps = self._read_all_temps()

        max_temp = max(temps) if temps else 0.0

        # 3. Compute body state
        stress = compute_stress(max_temp)
        thermal_state = classify_thermal_state(max_temp)

        # Separate CPU and GPU temps (heuristic)
        cpu_temps = temps[:4] if len(temps) >= 4 else temps
        gpu_temps = temps[4:] if len(temps) > 4 else []

        # Read power if available
        power_draw = self._read_power()

        # Build state
        current_state = BodyState(
            cpu_temps=cpu_temps,
            gpu_temps=gpu_temps,
            fan_rpm=self._read_fan_rpm(),
            power_draw_w=power_draw,
            stress_level=stress,
            thermal_state=thermal_state,
            power_state=self._classify_power(power_draw),
            current_mode=self.mode,
            reflex_events=reflex_strings,
            timestamp=time.time(),
        )

        self._current_state = current_state

        # Keep history
        self._state_history.append(current_state)
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)

        # 4. Publish to L3
        self._publish_state(current_state)

        # Mode-based actions
        self._apply_mode_policy(current_state)

    def _read_all_temps(self) -> list:
        """Fallback temperature reading."""
        return self.reflex_layer._read_temps()

    def _read_fan_rpm(self) -> Dict[str, int]:
        """Read fan RPM from hwmon."""
        fans = {}
        hwmon_path = Path("/sys/class/hwmon")

        if not hwmon_path.exists():
            return fans

        try:
            for hwmon in hwmon_path.iterdir():
                for fan in hwmon.glob("fan*_input"):
                    try:
                        with open(fan) as f:
                            rpm = int(f.read().strip())
                            fans[fan.stem] = rpm
                    except Exception:
                        pass
        except Exception:
            pass

        return fans

    def _read_power(self) -> float:
        """Read system power draw (if available)."""
        # Try RAPL for Intel/AMD
        rapl_path = Path("/sys/class/powercap/intel-rapl")
        if rapl_path.exists():
            try:
                energy_file = rapl_path / "intel-rapl:0" / "energy_uj"
                if energy_file.exists():
                    with open(energy_file) as f:
                        return float(f.read().strip()) / 1e6  # Convert to watts
            except Exception:
                pass

        return 0.0

    def _classify_power(self, power_w: float) -> PowerState:
        """Classify power state."""
        # Simple classification - would need more data in production
        if power_w == 0:
            return PowerState.STABLE  # Unknown
        elif power_w < 200:
            return PowerState.STABLE
        elif power_w < 400:
            return PowerState.FLUCTUATING
        else:
            return PowerState.UNSTABLE

    def _apply_mode_policy(self, state: BodyState):
        """
        Apply mode-specific policies.

        QUIET: Reduce fan noise, throttle if needed
        BALANCED: Default behavior
        PERFORMANCE: Allow high temps, max cooling
        """
        if self.mode == BodyMode.QUIET:
            # In quiet mode, we tolerate higher temps to reduce fan noise
            # But still protect hardware
            pass
        elif self.mode == BodyMode.PERFORMANCE:
            # In performance mode, run fans harder preemptively
            if state.stress_level > 0.5 and not self.reflex_layer.dry_run:
                self.reflex_layer._boost_fans(state.max_temp)

    def _publish_state(self, state: BodyState):
        """
        Publish state via atomic file write.

        Uses write-to-temp-then-rename for atomicity.
        """
        tmp_path = self.state_file.with_suffix(".tmp")

        try:
            with open(tmp_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)

            # Atomic rename
            tmp_path.rename(self.state_file)

        except Exception as e:
            logger.error(f"Failed to publish body state: {e}")

    def set_mode(self, mode: BodyMode):
        """Change operating mode."""
        old_mode = self.mode
        self.mode = mode
        logger.info(f"Mode changed: {old_mode.value} -> {mode.value}")

    def get_state(self) -> Optional[BodyState]:
        """Get current body state."""
        return self._current_state

    def get_stats(self) -> Dict[str, Any]:
        """Get daemon statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "running": self.running,
            "mode": self.mode.value,
            "loop_count": self._loop_count,
            "uptime_s": uptime,
            "loops_per_sec": self._loop_count / uptime if uptime > 0 else 0,
            "state_file": str(self.state_file),
            "reflex_status": self.reflex_layer.get_status(),
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for body daemon."""
    parser = argparse.ArgumentParser(
        description="Ara Body Schema Daemon - L2 Autonomic Control"
    )
    parser.add_argument(
        "--frequency", "-f",
        type=float,
        default=5.0,
        help="Loop frequency in Hz (default: 5.0)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["QUIET", "BALANCED", "PERFORMANCE"],
        default="BALANCED",
        help="Operating mode (default: BALANCED)"
    )
    parser.add_argument(
        "--state-file", "-s",
        type=Path,
        default=BodySchemaDaemon.DEFAULT_STATE_FILE,
        help="State file path for IPC"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live mode (actually kill processes, control fans)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Create and run daemon
    daemon = BodySchemaDaemon(
        state_file=args.state_file,
        dry_run=not args.live,
        mode=BodyMode[args.mode],
    )

    logger.info("=" * 50)
    logger.info("Ara Body Daemon v0.1")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Frequency: {args.frequency} Hz")
    logger.info(f"Live mode: {args.live}")
    logger.info("=" * 50)

    try:
        daemon.run(frequency_hz=args.frequency)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        daemon.stop()


if __name__ == "__main__":
    main()
