#!/usr/bin/env python3
"""
Thermal Reflex - L1 Hardware Protection
========================================

The "fast, rude" loop that protects Ara's physical substrate.

This is the spinal cord - it doesn't ask permission, it acts.
In production, this wraps eBPF code for kernel-level speed.
For v0.1, it's a high-speed Python poller.

Reflex Authority:
    - Can kill ANY process (except itself and vital system processes)
    - Can max fans without warning
    - Can throttle GPUs directly
    - Can trigger emergency shutdown

The reflex layer has ONE job: PREVENT PHYSICAL DAMAGE.

Usage:
    reflex = ThermalReflex()

    while True:
        events = reflex.scan_and_react()
        for event in events:
            log(f"L1 ACTION: {event}")
        time.sleep(0.1)  # 10Hz minimum
"""

from __future__ import annotations

import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..schema import (
    ReflexEvent,
    TEMP_CRITICAL,
    TEMP_WARNING,
    TEMP_EMERGENCY,
)

logger = logging.getLogger("ara.reflex")


# =============================================================================
# Protected Processes (Never Kill These)
# =============================================================================

PROTECTED_PROCESSES = {
    "systemd",
    "init",
    "kernel",
    "kthread",
    "sshd",
    "login",
    "getty",
    "dbus",
    "udev",
    "journald",
    "ara.body",      # Don't kill yourself!
    "body.daemon",
}


# =============================================================================
# Thermal Reflex
# =============================================================================

class ThermalReflex:
    """
    L1 Hardware Loop - Thermal Protection.

    Responsibility: PREVENT PHYSICAL DAMAGE
    Authority: ABSOLUTE (Can kill processes without permission)
    Speed: >10Hz polling

    This is not cognitive - it's reflexive. Like pulling your hand
    from a hot stove, it acts before conscious awareness.

    Example:
        reflex = ThermalReflex()

        # In the main loop
        events = reflex.scan_and_react()
        if events:
            # L1 took action - log it, but don't question it
            for e in events:
                print(f"REFLEX: {e}")
    """

    # Cooldown between kills (prevent thrashing)
    KILL_COOLDOWN_S = 5.0

    # Maximum kills per minute (prevent runaway)
    MAX_KILLS_PER_MINUTE = 6

    def __init__(
        self,
        dry_run: bool = True,
        log_actions: bool = True,
    ):
        """
        Initialize thermal reflex.

        Args:
            dry_run: If True, log but don't actually kill processes
            log_actions: If True, log all reflex actions
        """
        self.dry_run = dry_run
        self.log_actions = log_actions
        self.active = True

        # Kill tracking
        self._last_kill_time = 0.0
        self._kills_this_minute: List[float] = []

        # Event history
        self._recent_events: List[ReflexEvent] = []

        # Sensor cache
        self._last_temps: List[float] = []
        self._last_read_time = 0.0

    def scan_and_react(self) -> List[ReflexEvent]:
        """
        Poll sensors and trigger reflexes if needed.

        This should be called at >10Hz for responsive protection.

        Returns:
            List of ReflexEvent objects describing actions taken
        """
        if not self.active:
            return []

        events = []
        temps = self._read_temps()
        self._last_temps = temps

        if not temps:
            return events

        max_temp = max(temps)

        # Emergency shutdown check (highest priority)
        if max_temp > TEMP_EMERGENCY:
            event = self._emergency_shutdown(max_temp)
            events.append(event)
            return events  # No further action needed

        # Critical intervention
        if max_temp > TEMP_CRITICAL:
            kill_event = self._emergency_brake(max_temp)
            if kill_event:
                events.append(kill_event)

            # Also max fans
            fan_event = self._max_fans(max_temp)
            if fan_event:
                events.append(fan_event)

        # Warning level - just boost fans
        elif max_temp > TEMP_WARNING:
            fan_event = self._boost_fans(max_temp)
            if fan_event:
                events.append(fan_event)

        # Store events
        self._recent_events.extend(events)
        self._recent_events = self._recent_events[-100:]  # Keep last 100

        return events

    def _read_temps(self) -> List[float]:
        """
        Read temperature sensors.

        Tries multiple methods:
        1. psutil sensors
        2. Linux thermal zones
        3. NVIDIA SMI for GPUs
        """
        temps = []

        # Try psutil first
        try:
            import psutil
            if hasattr(psutil, "sensors_temperatures"):
                sensor_data = psutil.sensors_temperatures()
                for name, entries in sensor_data.items():
                    for entry in entries:
                        if entry.current > 0:
                            temps.append(entry.current)
        except Exception:
            pass

        # Try Linux thermal zones
        if not temps:
            temps.extend(self._read_thermal_zones())

        # Try NVIDIA SMI
        gpu_temps = self._read_nvidia_temps()
        temps.extend(gpu_temps)

        self._last_read_time = time.time()
        return temps

    def _read_thermal_zones(self) -> List[float]:
        """Read from /sys/class/thermal/thermal_zone*/temp."""
        temps = []
        thermal_path = Path("/sys/class/thermal")

        if not thermal_path.exists():
            return temps

        try:
            for zone in thermal_path.glob("thermal_zone*"):
                temp_file = zone / "temp"
                if temp_file.exists():
                    with open(temp_file) as f:
                        # Value is in millidegrees
                        temp_milli = int(f.read().strip())
                        temps.append(temp_milli / 1000.0)
        except Exception:
            pass

        return temps

    def _read_nvidia_temps(self) -> List[float]:
        """Read GPU temperatures via nvidia-smi."""
        temps = []

        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        temps.append(float(line.strip()))
        except Exception:
            pass

        return temps

    def _emergency_shutdown(self, temp: float) -> ReflexEvent:
        """
        Trigger emergency shutdown.

        This is the nuclear option - system is about to cook.
        """
        event = ReflexEvent(
            event_type="EMERGENCY_SHUTDOWN",
            trigger_temp=temp,
            target="SYSTEM",
            success=False,  # Will be True if we actually shut down
        )

        logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED! Temp: {temp:.1f}C")

        if not self.dry_run:
            try:
                import subprocess
                # Try graceful shutdown first
                subprocess.run(["shutdown", "-h", "now"], timeout=5.0)
                event.success = True
            except Exception as e:
                logger.error(f"Shutdown failed: {e}")
                # Try harder
                try:
                    subprocess.run(["poweroff", "-f"], timeout=5.0)
                    event.success = True
                except Exception:
                    pass

        return event

    def _emergency_brake(self, temp: float) -> Optional[ReflexEvent]:
        """
        Kill the heaviest process to reduce thermal load.

        Respects cooldown to prevent thrashing.
        """
        # Check cooldown
        now = time.time()
        if now - self._last_kill_time < self.KILL_COOLDOWN_S:
            return ReflexEvent(
                event_type="KILL_COOLDOWN",
                trigger_temp=temp,
                target=None,
                success=False,
            )

        # Check rate limit
        self._kills_this_minute = [
            t for t in self._kills_this_minute
            if now - t < 60.0
        ]
        if len(self._kills_this_minute) >= self.MAX_KILLS_PER_MINUTE:
            return ReflexEvent(
                event_type="KILL_RATELIMIT",
                trigger_temp=temp,
                target=None,
                success=False,
            )

        # Find victim
        victim = self._find_kill_target()
        if victim is None:
            return ReflexEvent(
                event_type="KILL_NO_TARGET",
                trigger_temp=temp,
                target=None,
                success=False,
            )

        # Kill it
        event = ReflexEvent(
            event_type="KILL_PROC",
            trigger_temp=temp,
            target=victim["name"],
            success=False,
        )

        logger.critical(
            f"THERMAL REFLEX: Killing {victim['name']} (PID {victim['pid']}) "
            f"@ {temp:.1f}C (CPU: {victim['cpu']:.1f}%)"
        )

        if not self.dry_run:
            try:
                import psutil
                proc = psutil.Process(victim["pid"])
                proc.kill()
                event.success = True
            except Exception as e:
                logger.error(f"Kill failed: {e}")

        self._last_kill_time = now
        self._kills_this_minute.append(now)

        return event

    def _find_kill_target(self) -> Optional[Dict[str, Any]]:
        """
        Find the heaviest CPU process that's safe to kill.

        Returns dict with pid, name, cpu or None if no valid target.
        """
        try:
            import psutil

            # Get processes sorted by CPU usage
            procs = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
                try:
                    info = proc.info
                    name = info.get("name", "").lower()

                    # Skip protected processes
                    if any(p in name for p in PROTECTED_PROCESSES):
                        continue

                    # Skip our own process
                    if info["pid"] == psutil.Process().pid:
                        continue

                    procs.append({
                        "pid": info["pid"],
                        "name": info["name"],
                        "cpu": info["cpu_percent"] or 0.0,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage
            procs.sort(key=lambda x: x["cpu"], reverse=True)

            # Return heaviest if it's using significant CPU
            if procs and procs[0]["cpu"] > 5.0:
                return procs[0]

            return None

        except Exception:
            return None

    def _max_fans(self, temp: float) -> Optional[ReflexEvent]:
        """Set all fans to maximum speed."""
        event = ReflexEvent(
            event_type="MAX_FANS",
            trigger_temp=temp,
            target="ALL_FANS",
            success=False,
        )

        if self.log_actions:
            logger.warning(f"REFLEX: MAX_FANS triggered @ {temp:.1f}C")

        if not self.dry_run:
            # Try PWM fan control
            success = self._set_pwm_fans(255)  # Max PWM
            event.success = success

        return event

    def _boost_fans(self, temp: float) -> Optional[ReflexEvent]:
        """Boost fan speed (not maximum)."""
        event = ReflexEvent(
            event_type="BOOST_FANS",
            trigger_temp=temp,
            target="ALL_FANS",
            success=False,
        )

        if not self.dry_run:
            # Calculate boost level based on temp
            # 75C = 50%, 85C = 100%
            level = int(128 + (temp - 75) * 12.7)
            level = max(128, min(255, level))
            success = self._set_pwm_fans(level)
            event.success = success

        return event

    def _set_pwm_fans(self, level: int) -> bool:
        """
        Set PWM fan speed.

        Args:
            level: 0-255 PWM duty cycle

        Returns:
            True if at least one fan was set
        """
        success = False
        hwmon_path = Path("/sys/class/hwmon")

        if not hwmon_path.exists():
            return False

        try:
            for hwmon in hwmon_path.iterdir():
                for pwm in hwmon.glob("pwm*"):
                    if pwm.is_file():
                        try:
                            # Enable manual control
                            enable_file = Path(str(pwm) + "_enable")
                            if enable_file.exists():
                                with open(enable_file, "w") as f:
                                    f.write("1")  # Manual mode

                            # Set speed
                            with open(pwm, "w") as f:
                                f.write(str(level))
                            success = True
                        except PermissionError:
                            pass
        except Exception:
            pass

        return success

    def get_recent_events(self, n: int = 10) -> List[ReflexEvent]:
        """Get N most recent reflex events."""
        return self._recent_events[-n:]

    def get_status(self) -> Dict[str, Any]:
        """Get reflex layer status."""
        return {
            "active": self.active,
            "dry_run": self.dry_run,
            "last_temps": self._last_temps,
            "max_temp": max(self._last_temps) if self._last_temps else 0.0,
            "last_kill_time": self._last_kill_time,
            "kills_this_minute": len(self._kills_this_minute),
            "recent_events": len(self._recent_events),
        }

    def enable(self):
        """Enable reflex layer."""
        self.active = True
        logger.info("Thermal reflex ENABLED")

    def disable(self):
        """Disable reflex layer (dangerous!)."""
        self.active = False
        logger.warning("Thermal reflex DISABLED - hardware at risk!")


# =============================================================================
# Tests
# =============================================================================

def test_thermal_reflex():
    """Test thermal reflex (dry run)."""
    print("Testing Thermal Reflex (dry run)")
    print("-" * 40)

    reflex = ThermalReflex(dry_run=True)

    # Run a few cycles
    for i in range(5):
        events = reflex.scan_and_react()
        status = reflex.get_status()

        print(f"  Cycle {i+1}: max_temp={status['max_temp']:.1f}C, "
              f"events={len(events)}")

        for event in events:
            print(f"    {event}")

        time.sleep(0.2)

    print("Thermal reflex test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_thermal_reflex()
