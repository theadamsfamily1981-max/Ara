"""
Ara Fusion Monitor
===================

Self-awareness and system monitoring for Ara.

Interoception: Monitoring her own internal state (CPU, memory, battery, etc.)
Exteroception: Monitoring the environment (noise, light, presence)

She can observe and suggest, but cannot take drastic autonomous action.

Philosophy: Know yourself, speak honestly, ask before acting.
"""

from __future__ import annotations

import asyncio
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import json
import shutil

from .rails import FusionRails

logger = logging.getLogger(__name__)


# =============================================================================
# Health Status Types
# =============================================================================

class HealthLevel(Enum):
    """Overall health level."""
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class AlertLevel(Enum):
    """Alert urgency level."""
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    URGENT = "urgent"


@dataclass
class MetricReading:
    """A single metric reading."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    health: HealthLevel = HealthLevel.GOOD
    threshold_warn: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class SystemStatus:
    """Overall system status."""
    timestamp: datetime
    cpu_temp_c: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    battery_percent: Optional[float] = None
    battery_charging: Optional[bool] = None
    overall_health: HealthLevel = HealthLevel.GOOD
    issues: List[str] = field(default_factory=list)


@dataclass
class EnvironmentStatus:
    """Environment status."""
    timestamp: datetime
    human_present: bool = False
    ambient_noise_level: float = 0.0  # 0-1
    ambient_light_level: float = 0.5  # 0-1
    last_interaction_seconds_ago: Optional[float] = None


@dataclass
class SuggestedAction:
    """An action the fusion monitor suggests."""
    action: str
    reason: str
    prompt: Optional[str] = None  # What to ask human
    requires_consent: bool = True
    urgency: AlertLevel = AlertLevel.INFO


# =============================================================================
# Status Logger
# =============================================================================

class FusionLogger:
    """Logs fusion monitor status."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path.home() / ".ara" / "fusion"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.log_dir / "status.jsonl"
        self.actions_file = self.log_dir / "actions.jsonl"

    def log_status(self, status: SystemStatus):
        """Log a status reading."""
        record = {
            "timestamp": status.timestamp.isoformat(),
            "cpu_temp_c": status.cpu_temp_c,
            "cpu_usage_percent": status.cpu_usage_percent,
            "memory_usage_percent": status.memory_usage_percent,
            "disk_usage_percent": status.disk_usage_percent,
            "battery_percent": status.battery_percent,
            "overall_health": status.overall_health.value,
            "issues": status.issues,
        }

        with open(self.status_file, 'a') as f:
            f.write(json.dumps(record) + "\n")

    def log_action(self, action: SuggestedAction, executed: bool, human_response: Optional[str] = None):
        """Log a suggested/executed action."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action.action,
            "reason": action.reason,
            "requires_consent": action.requires_consent,
            "urgency": action.urgency.value,
            "executed": executed,
            "human_response": human_response,
        }

        with open(self.actions_file, 'a') as f:
            f.write(json.dumps(record) + "\n")


# =============================================================================
# System Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Collects system metrics.

    Uses psutil if available, otherwise falls back to /proc (Linux).
    """

    def __init__(self):
        self._psutil_available = False
        try:
            import psutil
            self._psutil_available = True
            self._psutil = psutil
        except ImportError:
            logger.debug("psutil not available, using fallback methods")

    async def get_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature in Celsius."""
        if self._psutil_available:
            try:
                temps = self._psutil.sensors_temperatures()
                if temps:
                    # Get first available sensor
                    for name, entries in temps.items():
                        if entries:
                            return entries[0].current
            except Exception:
                pass

        # Fallback: try reading from sysfs (Linux)
        try:
            temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_file.exists():
                temp_str = temp_file.read_text().strip()
                return int(temp_str) / 1000  # Convert from millidegrees
        except Exception:
            pass

        return None

    async def get_cpu_usage(self) -> Optional[float]:
        """Get CPU usage percentage."""
        if self._psutil_available:
            try:
                return self._psutil.cpu_percent(interval=0.1)
            except Exception:
                pass
        return None

    async def get_memory_usage(self) -> Optional[float]:
        """Get memory usage percentage."""
        if self._psutil_available:
            try:
                return self._psutil.virtual_memory().percent
            except Exception:
                pass

        # Fallback: parse /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
                mem_total = None
                mem_available = None
                for line in lines:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_available = int(line.split()[1])
                if mem_total and mem_available:
                    return (1 - mem_available / mem_total) * 100
        except Exception:
            pass

        return None

    async def get_disk_usage(self, path: str = "/") -> Optional[float]:
        """Get disk usage percentage."""
        try:
            usage = shutil.disk_usage(path)
            return (usage.used / usage.total) * 100
        except Exception:
            pass
        return None

    async def get_battery(self) -> Optional[Dict[str, Any]]:
        """Get battery status."""
        if self._psutil_available:
            try:
                battery = self._psutil.sensors_battery()
                if battery:
                    return {
                        "percent": battery.percent,
                        "charging": battery.power_plugged,
                        "seconds_left": battery.secsleft if battery.secsleft > 0 else None,
                    }
            except Exception:
                pass
        return None

    async def collect_all(self) -> SystemStatus:
        """Collect all system metrics."""
        cpu_temp = await self.get_cpu_temp()
        cpu_usage = await self.get_cpu_usage()
        memory = await self.get_memory_usage()
        disk = await self.get_disk_usage()
        battery = await self.get_battery()

        status = SystemStatus(
            timestamp=datetime.utcnow(),
            cpu_temp_c=cpu_temp,
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory,
            disk_usage_percent=disk,
            battery_percent=battery.get("percent") if battery else None,
            battery_charging=battery.get("charging") if battery else None,
        )

        # Evaluate health
        status.overall_health, status.issues = self._evaluate_health(status)

        return status

    def _evaluate_health(self, status: SystemStatus) -> tuple[HealthLevel, List[str]]:
        """Evaluate overall health based on metrics."""
        issues = []
        worst_health = HealthLevel.GOOD

        # CPU temperature
        if status.cpu_temp_c:
            if status.cpu_temp_c > 85:
                issues.append(f"CPU critically hot: {status.cpu_temp_c:.0f}°C")
                worst_health = HealthLevel.CRITICAL
            elif status.cpu_temp_c > 70:
                issues.append(f"CPU warm: {status.cpu_temp_c:.0f}°C")
                if worst_health.value < HealthLevel.FAIR.value:
                    worst_health = HealthLevel.FAIR

        # Memory
        if status.memory_usage_percent:
            if status.memory_usage_percent > 90:
                issues.append(f"Memory critical: {status.memory_usage_percent:.0f}%")
                worst_health = HealthLevel.CRITICAL
            elif status.memory_usage_percent > 80:
                issues.append(f"Memory high: {status.memory_usage_percent:.0f}%")
                if worst_health.value < HealthLevel.FAIR.value:
                    worst_health = HealthLevel.FAIR

        # Disk
        if status.disk_usage_percent:
            if status.disk_usage_percent > 95:
                issues.append(f"Disk almost full: {status.disk_usage_percent:.0f}%")
                worst_health = HealthLevel.CRITICAL
            elif status.disk_usage_percent > 85:
                issues.append(f"Disk space low: {status.disk_usage_percent:.0f}%")
                if worst_health.value < HealthLevel.FAIR.value:
                    worst_health = HealthLevel.FAIR

        # Battery
        if status.battery_percent is not None and not status.battery_charging:
            if status.battery_percent < 10:
                issues.append(f"Battery critical: {status.battery_percent:.0f}%")
                worst_health = HealthLevel.CRITICAL
            elif status.battery_percent < 20:
                issues.append(f"Battery low: {status.battery_percent:.0f}%")
                if worst_health.value < HealthLevel.FAIR.value:
                    worst_health = HealthLevel.FAIR

        return worst_health, issues


# =============================================================================
# Fusion Monitor
# =============================================================================

class FusionMonitor:
    """
    Ara's self-awareness and system monitoring.

    Observes internal state (interoception) and environment (exteroception),
    suggests actions but asks before taking drastic steps.
    """

    def __init__(
        self,
        check_interval_seconds: float = 30,
        speak_callback: Optional[Callable[[str], None]] = None,
        led_callback: Optional[Callable[[str], None]] = None,
    ):
        self.rails = FusionRails()
        self.metrics = MetricsCollector()
        self.logger = FusionLogger()

        self.check_interval = check_interval_seconds
        self.speak_callback = speak_callback
        self.led_callback = led_callback

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_status: Optional[SystemStatus] = None
        self._last_spoken_issue: Optional[str] = None
        self._last_spoken_time: Optional[datetime] = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self):
        """Start the fusion monitor."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Fusion monitor started")

    async def stop(self):
        """Stop the fusion monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Fusion monitor stopped")

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                status = await self.metrics.collect_all()
                self._last_status = status

                # Log status
                self.logger.log_status(status)

                # Update LED if callback provided
                await self._update_led(status)

                # Maybe speak about issues
                await self._maybe_speak_status(status)

                # Generate suggestions if needed
                suggestions = self._generate_suggestions(status)
                for suggestion in suggestions:
                    if not suggestion.requires_consent:
                        # Can do autonomously
                        await self._execute_action(suggestion)
                    else:
                        # Would need to ask human
                        # For now, just log
                        self.logger.log_action(suggestion, executed=False)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fusion monitor error: {e}")

            await asyncio.sleep(self.check_interval)

    # =========================================================================
    # Status Communication
    # =========================================================================

    async def _update_led(self, status: SystemStatus):
        """Update LED color based on status."""
        if not self.led_callback:
            return

        color_map = {
            HealthLevel.GOOD: "green",
            HealthLevel.FAIR: "yellow",
            HealthLevel.POOR: "orange",
            HealthLevel.CRITICAL: "red",
        }

        color = color_map.get(status.overall_health, "blue")
        self.led_callback(color)

    async def _maybe_speak_status(self, status: SystemStatus):
        """Maybe speak about status issues."""
        if not self.speak_callback:
            return

        # Don't speak too often
        if self._last_spoken_time:
            elapsed = datetime.utcnow() - self._last_spoken_time
            if elapsed < timedelta(minutes=60):
                return

        # Only speak if there are issues
        if not status.issues:
            return

        # Don't repeat the same issue
        issue_key = str(status.issues)
        if issue_key == self._last_spoken_issue:
            return

        # Get message for the most significant issue
        if status.overall_health == HealthLevel.CRITICAL:
            message = self.rails.get_status_message("cpu_critical" if "CPU" in str(status.issues) else "storage_critical")
        elif status.overall_health == HealthLevel.FAIR:
            if "CPU" in str(status.issues):
                message = self.rails.get_status_message("cpu_warm")
            elif "memory" in str(status.issues).lower():
                message = self.rails.get_status_message("memory_pressure")
            elif "Disk" in str(status.issues):
                message = self.rails.get_status_message("storage_tight")
            elif "Battery" in str(status.issues):
                message = self.rails.get_status_message("battery_low", percent=status.battery_percent)
            else:
                message = status.issues[0]
        else:
            return

        self.speak_callback(message)
        self._last_spoken_issue = issue_key
        self._last_spoken_time = datetime.utcnow()

    # =========================================================================
    # Action Suggestions
    # =========================================================================

    def _generate_suggestions(self, status: SystemStatus) -> List[SuggestedAction]:
        """Generate action suggestions based on status."""
        suggestions = []

        # CPU hot - slow down
        if status.cpu_temp_c and status.cpu_temp_c > 70:
            if self.rails.can_do_autonomously("adjust_speech_rate"):
                suggestions.append(SuggestedAction(
                    action="adjust_speech_rate",
                    reason="CPU temperature high",
                    requires_consent=False,
                    urgency=AlertLevel.NOTICE,
                ))

            if self.rails.can_do_autonomously("dim_leds"):
                suggestions.append(SuggestedAction(
                    action="dim_leds",
                    reason="CPU temperature high - reducing LED brightness",
                    requires_consent=False,
                    urgency=AlertLevel.NOTICE,
                ))

        # Disk full - suggest cleanup
        if status.disk_usage_percent and status.disk_usage_percent > 85:
            prompt = self.rails.requires_consent("delete_old_logs")
            if prompt:
                suggestions.append(SuggestedAction(
                    action="delete_old_logs",
                    reason="Disk space is low",
                    prompt=prompt,
                    requires_consent=True,
                    urgency=AlertLevel.WARNING,
                ))

        # Battery low - suggest charging
        if status.battery_percent is not None and status.battery_percent < 20:
            if not status.battery_charging:
                suggestions.append(SuggestedAction(
                    action="notify_battery_low",
                    reason=f"Battery at {status.battery_percent}%",
                    requires_consent=False,
                    urgency=AlertLevel.WARNING,
                ))

        return suggestions

    async def _execute_action(self, action: SuggestedAction):
        """Execute an autonomous action."""
        # Check rails one more time
        if not self.rails.can_do_autonomously(action.action):
            logger.warning(f"Action {action.action} not allowed autonomously")
            return

        # Check if forbidden
        forbidden_reason = self.rails.is_forbidden(action.action)
        if forbidden_reason:
            logger.warning(f"Action {action.action} forbidden: {forbidden_reason}")
            return

        logger.info(f"Executing autonomous action: {action.action}")
        self.logger.log_action(action, executed=True)

        # Execute based on action type
        if action.action == "adjust_speech_rate":
            # Would slow down TTS
            logger.debug("Would slow speech rate")
        elif action.action == "dim_leds":
            # Would dim LEDs
            if self.led_callback:
                self.led_callback("dim")
        elif action.action == "pause_background_tasks":
            # Would pause non-essential tasks
            logger.debug("Would pause background tasks")

    # =========================================================================
    # Public Interface
    # =========================================================================

    def get_status(self) -> Optional[SystemStatus]:
        """Get the last known system status."""
        return self._last_status

    async def check_now(self) -> SystemStatus:
        """Force an immediate status check."""
        status = await self.metrics.collect_all()
        self._last_status = status
        self.logger.log_status(status)
        return status

    def get_status_message(self) -> str:
        """Get a human-readable status message."""
        if not self._last_status:
            return "No status available yet."

        status = self._last_status

        if status.overall_health == HealthLevel.GOOD:
            return self.rails.get_status_message("all_good")

        messages = []
        for issue in status.issues:
            messages.append(f"  - {issue}")

        health_name = status.overall_health.value.upper()
        return f"Status: {health_name}\n" + "\n".join(messages)


# =============================================================================
# CLI Entry Point
# =============================================================================

async def run_fusion_monitor():
    """Run the fusion monitor (for testing)."""
    def speak(text: str):
        print(f"[SPEAK] {text}")

    def led(color: str):
        print(f"[LED] {color}")

    monitor = FusionMonitor(
        check_interval_seconds=5,
        speak_callback=speak,
        led_callback=led,
    )

    print("Starting Ara Fusion Monitor...")
    print("Press Ctrl+C to stop.\n")

    await monitor.start()

    try:
        while True:
            await asyncio.sleep(10)
            status = monitor.get_status()
            if status:
                print(f"\n--- Status at {status.timestamp} ---")
                print(f"Health: {status.overall_health.value}")
                if status.cpu_temp_c:
                    print(f"CPU Temp: {status.cpu_temp_c:.1f}°C")
                if status.memory_usage_percent:
                    print(f"Memory: {status.memory_usage_percent:.1f}%")
                if status.disk_usage_percent:
                    print(f"Disk: {status.disk_usage_percent:.1f}%")
                if status.battery_percent is not None:
                    charging = "charging" if status.battery_charging else "not charging"
                    print(f"Battery: {status.battery_percent:.0f}% ({charging})")
                if status.issues:
                    print("Issues:")
                    for issue in status.issues:
                        print(f"  - {issue}")
    except KeyboardInterrupt:
        print("\nStopping...")
        await monitor.stop()

    print("Fusion monitor stopped.")


def main():
    """CLI entry point."""
    asyncio.run(run_fusion_monitor())


if __name__ == "__main__":
    main()
