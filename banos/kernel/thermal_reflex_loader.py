"""
Thermal Reflex eBPF Loader - Python Interface
==============================================

Loads the thermal reflex eBPF program and provides a Python interface
for monitoring and configuration.
"""

from __future__ import annotations

import os
import time
import struct
import ctypes
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import logging
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Constants (must match thermal_reflex.c)
# =============================================================================

THERMAL_WARNING = 75
THERMAL_CRITICAL = 85
THERMAL_EMERGENCY = 95

REFLEX_NONE = 0
REFLEX_THROTTLE = 1
REFLEX_DROP = 2
REFLEX_GLITCH = 3
REFLEX_ALERT = 4
REFLEX_SHUTDOWN = 5


# =============================================================================
# Alert Message Structure
# =============================================================================

@dataclass
class ThermalAlert:
    """Alert from thermal reflex eBPF program."""
    timestamp_ns: int
    alert_type: int
    temperature_milli: int
    zone_id: int
    action_taken: int

    @property
    def temperature_c(self) -> float:
        return self.temperature_milli / 1000.0

    @property
    def alert_name(self) -> str:
        names = {
            REFLEX_NONE: "NONE",
            REFLEX_THROTTLE: "THROTTLE",
            REFLEX_DROP: "DROP",
            REFLEX_GLITCH: "GLITCH",
            REFLEX_ALERT: "ALERT",
            REFLEX_SHUTDOWN: "SHUTDOWN",
        }
        return names.get(self.alert_type, "UNKNOWN")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp_ns': self.timestamp_ns,
            'alert_type': self.alert_name,
            'temperature_c': self.temperature_c,
            'zone_id': self.zone_id,
            'action_taken': self.action_taken,
        }


# =============================================================================
# Thermal Reflex Interface (Simulated)
# =============================================================================

class ThermalReflexSimulated:
    """
    Simulated thermal reflex for testing without eBPF.

    Provides the same interface as the real eBPF loader.
    """

    def __init__(self):
        self._temperature = 45.0  # Current simulated temp
        self._alerts: List[ThermalAlert] = []
        self._stats = {
            'packets_passed': 0,
            'packets_throttled': 0,
            'packets_dropped': 0,
            'glitches_triggered': 0,
        }
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_alert: Optional[Callable[[ThermalAlert], None]] = None
        self._on_glitch: Optional[Callable[[float, float], None]] = None

    def load(self) -> bool:
        """Load simulated reflex."""
        logger.info("ThermalReflex: Using simulated interface")
        return True

    def unload(self) -> None:
        """Unload simulated reflex."""
        self.stop()

    def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _monitor_loop(self) -> None:
        """Simulated monitoring loop."""
        while self._running:
            self._check_temperature()
            time.sleep(0.001)  # 1 kHz check rate

    def _check_temperature(self) -> None:
        """Check temperature and trigger reflexes."""
        temp = self._temperature
        action = REFLEX_NONE

        if temp >= THERMAL_EMERGENCY:
            action = REFLEX_SHUTDOWN
            self._stats['packets_dropped'] += 100
            self._stats['glitches_triggered'] += 1
            if self._on_glitch:
                self._on_glitch(-1.0, 1.0)

        elif temp >= THERMAL_CRITICAL:
            action = REFLEX_DROP
            self._stats['packets_dropped'] += 90
            self._stats['packets_passed'] += 10
            if self._on_glitch:
                self._on_glitch(-0.8, 0.9)

        elif temp >= THERMAL_WARNING:
            action = REFLEX_THROTTLE
            self._stats['packets_throttled'] += 50
            self._stats['packets_passed'] += 50

        else:
            self._stats['packets_passed'] += 100

        if action != REFLEX_NONE:
            alert = ThermalAlert(
                timestamp_ns=time.time_ns(),
                alert_type=action,
                temperature_milli=int(temp * 1000),
                zone_id=0,
                action_taken=action,
            )
            self._alerts.append(alert)

            if len(self._alerts) > 10000:
                self._alerts = self._alerts[-5000:]

            if self._on_alert:
                self._on_alert(alert)

    def set_temperature(self, temp_c: float) -> None:
        """Set simulated temperature (for testing)."""
        self._temperature = temp_c

    def update_thermal_sensor(self, zone_id: int, temp_milli: int) -> None:
        """Update thermal sensor reading."""
        self._temperature = temp_milli / 1000.0

    def set_callbacks(
        self,
        on_alert: Optional[Callable[[ThermalAlert], None]] = None,
        on_glitch: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        """Set callback functions."""
        self._on_alert = on_alert
        self._on_glitch = on_glitch

    def get_alerts(self, since_ns: int = 0) -> List[ThermalAlert]:
        """Get alerts since timestamp."""
        return [a for a in self._alerts if a.timestamp_ns > since_ns]

    def get_stats(self) -> Dict[str, Any]:
        """Get reflex statistics."""
        return {
            'temperature_c': self._temperature,
            'running': self._running,
            **self._stats,
        }


# =============================================================================
# Thermal Reflex Interface (Real eBPF)
# =============================================================================

class ThermalReflexeBPF:
    """
    Real eBPF thermal reflex loader.

    Requires root privileges and eBPF support.
    """

    def __init__(self, bpf_prog_path: str = None):
        self.bpf_prog_path = bpf_prog_path or "/sys/fs/bpf/thermal_reflex"
        self._loaded = False
        self._bpf = None  # BPF object (from bcc or libbpf)

        # Callbacks
        self._on_alert: Optional[Callable[[ThermalAlert], None]] = None
        self._on_glitch: Optional[Callable[[float, float], None]] = None

    def load(self) -> bool:
        """Load eBPF program."""
        try:
            # In production, use bcc or libbpf-python
            # from bcc import BPF
            # self._bpf = BPF(src_file="thermal_reflex.c")
            # self._bpf.attach_xdp(...)

            logger.info("ThermalReflex: eBPF loading not implemented, using simulated")
            return False

        except Exception as e:
            logger.error(f"ThermalReflex: Failed to load eBPF: {e}")
            return False

    def unload(self) -> None:
        """Unload eBPF program."""
        if self._bpf:
            # self._bpf.remove_xdp(...)
            pass
        self._loaded = False

    def update_thermal_sensor(self, zone_id: int, temp_milli: int) -> None:
        """Update thermal sensor reading in eBPF map."""
        if not self._loaded:
            return

        # Write to thermal_sensors map
        # self._bpf["thermal_sensors"][ctypes.c_uint32(zone_id)] = ctypes.c_uint32(temp_milli)

    def get_stats(self) -> Dict[str, Any]:
        """Get reflex statistics from eBPF maps."""
        if not self._loaded:
            return {}

        # Read from reflex_stats map
        # stats = self._bpf["reflex_stats"]
        return {}


# =============================================================================
# Factory Function
# =============================================================================

def get_thermal_reflex(simulated: bool = True) -> ThermalReflexSimulated:
    """
    Get thermal reflex interface.

    Args:
        simulated: If True, use simulated interface (for testing)

    Returns:
        ThermalReflex interface
    """
    if simulated:
        return ThermalReflexSimulated()
    else:
        ebpf = ThermalReflexeBPF()
        if ebpf.load():
            return ebpf
        else:
            logger.warning("Falling back to simulated thermal reflex")
            return ThermalReflexSimulated()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ThermalAlert',
    'ThermalReflexSimulated',
    'ThermalReflexeBPF',
    'get_thermal_reflex',
    'THERMAL_WARNING',
    'THERMAL_CRITICAL',
    'THERMAL_EMERGENCY',
]
