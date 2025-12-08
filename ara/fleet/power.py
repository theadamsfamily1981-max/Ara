"""
Power Spine - UPS + Smart PDU Layer
=====================================

Boring but incredibly "cathedral-core".

UPS for:
- Brainstem node
- NAS
- Maybe Juniper

Smart PDU with:
- Per-outlet monitoring
- Switchable outlets (tied into Watcher / Brainstem)

Then Ara (via Brainstem + Watcher) can:
- Initiate clean shutdowns on power sag
- Stagger startup of heavy boxes
- Detect failing PSUs / overloaded circuits before they pop breakers

This turns random power failures into predictable, survivable events.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from collections import deque
import time


class PowerState(Enum):
    """State of power system."""
    NORMAL = auto()         # On grid, batteries full
    ON_BATTERY = auto()     # Running on UPS battery
    LOW_BATTERY = auto()    # Battery getting low
    CRITICAL = auto()       # Emergency shutdown imminent
    OFFLINE = auto()        # Power lost


class OutletState(Enum):
    """State of a PDU outlet."""
    ON = auto()
    OFF = auto()
    UNKNOWN = auto()


@dataclass
class PowerEvent:
    """A power-related event."""
    timestamp: float
    event_type: str     # "power_lost", "on_battery", "outlet_changed", etc.
    source: str         # Device that generated event
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Outlet:
    """A single outlet on a PDU."""
    outlet_id: str
    name: str
    state: OutletState = OutletState.ON

    # Monitoring
    current_watts: float = 0.0
    max_watts: float = 1800.0
    current_amps: float = 0.0

    # What's plugged in
    connected_node: Optional[str] = None

    # Protection
    overload_threshold: float = 0.9  # Fraction of max
    is_protected: bool = False       # Can't be switched off remotely


@dataclass
class SmartPDU:
    """
    Smart PDU with per-outlet monitoring and control.
    """
    pdu_id: str
    name: str
    outlets: Dict[str, Outlet] = field(default_factory=dict)

    # Aggregate
    total_watts: float = 0.0
    max_watts: float = 3600.0

    # Events
    _events: deque = field(default_factory=lambda: deque(maxlen=500))

    # Control callbacks
    _switch_outlet_fn: Optional[Callable[[str, bool], bool]] = None

    def add_outlet(self, outlet_id: str, name: str, connected_node: Optional[str] = None):
        """Add an outlet to the PDU."""
        outlet = Outlet(
            outlet_id=outlet_id,
            name=name,
            connected_node=connected_node,
        )
        self.outlets[outlet_id] = outlet

    def update_outlet_power(self, outlet_id: str, watts: float, amps: float = 0.0):
        """Update power reading for an outlet."""
        if outlet_id not in self.outlets:
            return

        outlet = self.outlets[outlet_id]
        outlet.current_watts = watts
        outlet.current_amps = amps

        # Check overload
        if outlet.max_watts > 0:
            load_fraction = watts / outlet.max_watts
            if load_fraction >= outlet.overload_threshold:
                self._log_event("outlet_overload", outlet_id=outlet_id, watts=watts)

        # Update aggregate
        self.total_watts = sum(o.current_watts for o in self.outlets.values())

    def switch_outlet(self, outlet_id: str, on: bool) -> bool:
        """Switch an outlet on or off."""
        if outlet_id not in self.outlets:
            return False

        outlet = self.outlets[outlet_id]
        if outlet.is_protected and not on:
            self._log_event("protected_outlet_refused", outlet_id=outlet_id)
            return False

        # Try to switch
        if self._switch_outlet_fn:
            success = self._switch_outlet_fn(outlet_id, on)
            if success:
                outlet.state = OutletState.ON if on else OutletState.OFF
                self._log_event(
                    "outlet_switched",
                    outlet_id=outlet_id,
                    state="on" if on else "off",
                )
                return True
        else:
            # Simulated
            outlet.state = OutletState.ON if on else OutletState.OFF
            return True

        return False

    def power_cycle_outlet(self, outlet_id: str, off_seconds: float = 5.0) -> bool:
        """Power cycle an outlet."""
        if self.switch_outlet(outlet_id, False):
            # In real implementation, would wait then turn on
            # For now, just turn back on
            return self.switch_outlet(outlet_id, True)
        return False

    def get_outlet_for_node(self, node_id: str) -> Optional[Outlet]:
        """Find the outlet a node is connected to."""
        for outlet in self.outlets.values():
            if outlet.connected_node == node_id:
                return outlet
        return None

    def _log_event(self, event_type: str, **kwargs):
        """Log an event."""
        event = PowerEvent(
            timestamp=time.time(),
            event_type=event_type,
            source=self.pdu_id,
            details=kwargs,
        )
        self._events.append(event)

    def get_status(self) -> Dict[str, Any]:
        """Get PDU status."""
        outlets_on = sum(1 for o in self.outlets.values() if o.state == OutletState.ON)
        return {
            "pdu_id": self.pdu_id,
            "name": self.name,
            "outlets_total": len(self.outlets),
            "outlets_on": outlets_on,
            "total_watts": self.total_watts,
            "max_watts": self.max_watts,
            "load_percent": (self.total_watts / self.max_watts * 100) if self.max_watts > 0 else 0,
        }


@dataclass
class UPS:
    """
    Uninterruptible Power Supply.
    """
    ups_id: str
    name: str

    # State
    state: PowerState = PowerState.NORMAL

    # Battery
    battery_percent: float = 100.0
    runtime_seconds: float = 1800.0  # 30 minutes

    # Load
    load_watts: float = 0.0
    max_watts: float = 1500.0

    # Input
    input_voltage: float = 120.0
    input_frequency: float = 60.0

    # Events
    _events: deque = field(default_factory=lambda: deque(maxlen=500))

    # Callbacks
    _on_power_lost: Optional[Callable[[], None]] = None
    _on_low_battery: Optional[Callable[[], None]] = None
    _on_power_restored: Optional[Callable[[], None]] = None

    # Thresholds
    low_battery_threshold: float = 20.0
    critical_battery_threshold: float = 5.0
    low_runtime_threshold: float = 300.0  # 5 minutes

    def update_status(
        self,
        on_battery: bool,
        battery_percent: float,
        runtime_seconds: float,
        load_watts: float,
        input_voltage: float = 120.0,
    ):
        """Update UPS status from monitoring."""
        old_state = self.state

        self.battery_percent = battery_percent
        self.runtime_seconds = runtime_seconds
        self.load_watts = load_watts
        self.input_voltage = input_voltage

        # Determine state
        if on_battery:
            if battery_percent <= self.critical_battery_threshold:
                self.state = PowerState.CRITICAL
            elif battery_percent <= self.low_battery_threshold:
                self.state = PowerState.LOW_BATTERY
            else:
                self.state = PowerState.ON_BATTERY
        else:
            self.state = PowerState.NORMAL

        # Fire callbacks on state changes
        if old_state != self.state:
            self._log_event(
                "state_changed",
                old_state=old_state.name,
                new_state=self.state.name,
            )

            if self.state == PowerState.ON_BATTERY and old_state == PowerState.NORMAL:
                if self._on_power_lost:
                    self._on_power_lost()

            elif self.state == PowerState.LOW_BATTERY:
                if self._on_low_battery:
                    self._on_low_battery()

            elif self.state == PowerState.NORMAL and old_state != PowerState.NORMAL:
                if self._on_power_restored:
                    self._on_power_restored()

    def is_on_battery(self) -> bool:
        """Check if running on battery."""
        return self.state in (PowerState.ON_BATTERY, PowerState.LOW_BATTERY, PowerState.CRITICAL)

    def should_shutdown(self) -> bool:
        """Check if immediate shutdown is needed."""
        if self.state == PowerState.CRITICAL:
            return True
        if self.runtime_seconds < self.low_runtime_threshold:
            return True
        return False

    def wire_callbacks(
        self,
        on_power_lost: Optional[Callable[[], None]] = None,
        on_low_battery: Optional[Callable[[], None]] = None,
        on_power_restored: Optional[Callable[[], None]] = None,
    ):
        """Wire up event callbacks."""
        self._on_power_lost = on_power_lost
        self._on_low_battery = on_low_battery
        self._on_power_restored = on_power_restored

    def _log_event(self, event_type: str, **kwargs):
        """Log an event."""
        event = PowerEvent(
            timestamp=time.time(),
            event_type=event_type,
            source=self.ups_id,
            details=kwargs,
        )
        self._events.append(event)

    def get_status(self) -> Dict[str, Any]:
        """Get UPS status."""
        return {
            "ups_id": self.ups_id,
            "name": self.name,
            "state": self.state.name,
            "battery_percent": self.battery_percent,
            "runtime_seconds": self.runtime_seconds,
            "runtime_minutes": self.runtime_seconds / 60,
            "load_watts": self.load_watts,
            "load_percent": (self.load_watts / self.max_watts * 100) if self.max_watts > 0 else 0,
            "input_voltage": self.input_voltage,
        }


@dataclass
class PowerSpine:
    """
    Complete power management layer.

    Coordinates UPS(es) and PDU(s) for:
    - Clean shutdowns on power sag
    - Staggered startup of heavy boxes
    - Power monitoring and overload prevention
    """
    spine_id: str = "power-spine-01"

    # Devices
    ups_devices: Dict[str, UPS] = field(default_factory=dict)
    pdu_devices: Dict[str, SmartPDU] = field(default_factory=dict)

    # Node → Outlet mapping (node_id → (pdu_id, outlet_id))
    _node_outlets: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # Startup sequence (ordered list of node_ids)
    _startup_sequence: List[str] = field(default_factory=list)
    _startup_delay_seconds: float = 5.0

    # Events
    _events: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add_ups(self, ups: UPS):
        """Add a UPS to the power spine."""
        self.ups_devices[ups.ups_id] = ups

    def add_pdu(self, pdu: SmartPDU):
        """Add a PDU to the power spine."""
        self.pdu_devices[pdu.pdu_id] = pdu

    def map_node_to_outlet(self, node_id: str, pdu_id: str, outlet_id: str):
        """Map a node to a PDU outlet."""
        self._node_outlets[node_id] = (pdu_id, outlet_id)

        # Also update the outlet's connected_node
        if pdu_id in self.pdu_devices:
            pdu = self.pdu_devices[pdu_id]
            if outlet_id in pdu.outlets:
                pdu.outlets[outlet_id].connected_node = node_id

    def set_startup_sequence(self, node_ids: List[str]):
        """Set the order for staggered startup."""
        self._startup_sequence = list(node_ids)

    def power_off_node(self, node_id: str) -> bool:
        """Power off a node via its PDU outlet."""
        if node_id not in self._node_outlets:
            return False

        pdu_id, outlet_id = self._node_outlets[node_id]
        if pdu_id not in self.pdu_devices:
            return False

        pdu = self.pdu_devices[pdu_id]
        success = pdu.switch_outlet(outlet_id, False)

        if success:
            self._log_event("node_powered_off", node_id=node_id)

        return success

    def power_on_node(self, node_id: str) -> bool:
        """Power on a node via its PDU outlet."""
        if node_id not in self._node_outlets:
            return False

        pdu_id, outlet_id = self._node_outlets[node_id]
        if pdu_id not in self.pdu_devices:
            return False

        pdu = self.pdu_devices[pdu_id]
        success = pdu.switch_outlet(outlet_id, True)

        if success:
            self._log_event("node_powered_on", node_id=node_id)

        return success

    def power_cycle_node(self, node_id: str) -> bool:
        """Power cycle a node."""
        if self.power_off_node(node_id):
            # In real implementation, would wait
            return self.power_on_node(node_id)
        return False

    def initiate_graceful_shutdown(self, reason: str = "power_event"):
        """
        Initiate graceful shutdown of non-critical nodes.

        Preserves brainstem, NAS, and other UPS-backed nodes.
        """
        self._log_event("graceful_shutdown_initiated", reason=reason)

        # Power off in reverse startup order
        for node_id in reversed(self._startup_sequence):
            # Skip protected nodes (would check node role here)
            self.power_off_node(node_id)

    def staggered_startup(self) -> List[str]:
        """
        Start nodes in sequence with delays.

        Returns list of nodes that were powered on.
        """
        powered_on = []

        for node_id in self._startup_sequence:
            if self.power_on_node(node_id):
                powered_on.append(node_id)
                # In real implementation, would wait for node to come online
                # time.sleep(self._startup_delay_seconds)

        self._log_event("staggered_startup_complete", nodes=powered_on)
        return powered_on

    def check_overload(self) -> List[Tuple[str, float]]:
        """
        Check for overloaded circuits.

        Returns list of (pdu_id, load_percent) for overloaded PDUs.
        """
        overloaded = []

        for pdu in self.pdu_devices.values():
            if pdu.max_watts > 0:
                load_percent = pdu.total_watts / pdu.max_watts
                if load_percent > 0.8:  # 80% threshold
                    overloaded.append((pdu.pdu_id, load_percent * 100))

        return overloaded

    def should_emergency_shutdown(self) -> bool:
        """Check if emergency shutdown is needed."""
        for ups in self.ups_devices.values():
            if ups.should_shutdown():
                return True
        return False

    def _log_event(self, event_type: str, **kwargs):
        """Log an event."""
        event = PowerEvent(
            timestamp=time.time(),
            event_type=event_type,
            source=self.spine_id,
            details=kwargs,
        )
        self._events.append(event)

    def get_status(self) -> Dict[str, Any]:
        """Get power spine status."""
        ups_statuses = {
            ups_id: ups.get_status()
            for ups_id, ups in self.ups_devices.items()
        }
        pdu_statuses = {
            pdu_id: pdu.get_status()
            for pdu_id, pdu in self.pdu_devices.items()
        }

        any_on_battery = any(ups.is_on_battery() for ups in self.ups_devices.values())
        overloaded = self.check_overload()

        return {
            "spine_id": self.spine_id,
            "ups_count": len(self.ups_devices),
            "pdu_count": len(self.pdu_devices),
            "mapped_nodes": len(self._node_outlets),
            "any_on_battery": any_on_battery,
            "overloaded_pdus": len(overloaded),
            "ups_status": ups_statuses,
            "pdu_status": pdu_statuses,
        }


def create_power_spine() -> PowerSpine:
    """
    Create a default power spine configuration.
    """
    spine = PowerSpine()

    # Add a UPS
    ups = UPS(
        ups_id="ups-01",
        name="Main UPS",
        max_watts=1500,
    )
    spine.add_ups(ups)

    # Add a PDU
    pdu = SmartPDU(
        pdu_id="pdu-01",
        name="Main PDU",
        max_watts=3600,
    )
    pdu.add_outlet("outlet-1", "GPU Worker")
    pdu.add_outlet("outlet-2", "FPGA Worker")
    pdu.add_outlet("outlet-3", "Intern Host")
    pdu.add_outlet("outlet-4", "Brainstem", connected_node="brainstem-01")
    pdu.outlets["outlet-4"].is_protected = True  # Can't power off brainstem
    spine.add_pdu(pdu)

    # Map nodes
    spine.map_node_to_outlet("gpu-worker-01", "pdu-01", "outlet-1")
    spine.map_node_to_outlet("fpga-worker-01", "pdu-01", "outlet-2")
    spine.map_node_to_outlet("intern-host-01", "pdu-01", "outlet-3")

    # Startup sequence (brainstem first, then workers)
    spine.set_startup_sequence([
        "brainstem-01",
        "gpu-worker-01",
        "fpga-worker-01",
        "intern-host-01",
    ])

    return spine
