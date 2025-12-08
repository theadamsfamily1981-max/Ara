"""
The Watcher - Out-of-Band Safety Controller
============================================

Tiny microcontroller or SBC that sits outside the main rigs and can:
- Watch power, temps, fans, and pings
- Cut power or reboot boxes via relays / smart PDUs
- Stay alive even when Linux / GPU / FPGA are hard-locked

Role: MEDIC
Authority: Can turn things off, never change configs or code.

The Watcher is "below" everything:
- It can kill power to a specific box
- It can force reboots
- It keeps logging health even when the OS is dead
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from collections import deque
import time
import hashlib


class WatcherState(Enum):
    """State of the Watcher."""
    HEALTHY = auto()        # All monitored nodes responding
    ALERT = auto()          # Some nodes degraded
    EMERGENCY = auto()      # Critical thresholds exceeded
    LOCKDOWN = auto()       # Watcher has taken action


class PowerAction(Enum):
    """Actions the Watcher can take."""
    NONE = auto()
    POWER_CYCLE = auto()    # Turn off, wait, turn on
    POWER_OFF = auto()      # Kill power immediately
    POWER_ON = auto()       # Turn on
    FORCE_REBOOT = auto()   # ATX reset button press
    GRACEFUL_SHUTDOWN = auto()  # Request OS shutdown first


@dataclass
class WatcherEvent:
    """An event recorded by the Watcher."""
    timestamp: float
    node_id: str
    event_type: str         # "ping_fail", "temp_high", "power_action", etc.
    value: Optional[float] = None
    action_taken: Optional[PowerAction] = None
    details: str = ""


@dataclass
class NodeHealth:
    """Health state for a monitored node."""
    node_id: str
    last_ping: float = 0.0
    last_temp: float = 0.0
    last_power_watts: float = 0.0
    ping_failures: int = 0
    temp_warnings: int = 0
    is_powered: bool = True

    def update_ping(self, success: bool):
        """Update ping status."""
        if success:
            self.last_ping = time.time()
            self.ping_failures = 0
        else:
            self.ping_failures += 1

    def update_temp(self, temp_c: float, warning_threshold: float = 80.0):
        """Update temperature reading."""
        self.last_temp = temp_c
        if temp_c > warning_threshold:
            self.temp_warnings += 1
        else:
            self.temp_warnings = max(0, self.temp_warnings - 1)


@dataclass
class WatcherPolicy:
    """Policy for Watcher actions."""
    # Ping failure thresholds
    ping_timeout_seconds: float = 5.0
    ping_failures_before_alert: int = 3
    ping_failures_before_action: int = 10

    # Temperature thresholds
    temp_warning_c: float = 80.0
    temp_critical_c: float = 95.0
    temp_warnings_before_action: int = 5

    # Power thresholds
    power_max_watts: float = 1500.0

    # Cooldowns
    action_cooldown_seconds: float = 300.0  # 5 min between actions per node

    # Auto-recovery
    auto_power_cycle: bool = True
    auto_power_off_on_critical_temp: bool = True


@dataclass
class Watcher:
    """
    The Watcher - Out-of-Band Safety Controller.

    Monitors nodes and can take emergency actions
    even when the main systems are unresponsive.
    """
    watcher_id: str = "watcher-01"
    policy: WatcherPolicy = field(default_factory=WatcherPolicy)

    # Monitored nodes
    _nodes: Dict[str, NodeHealth] = field(default_factory=dict)

    # Event log
    _events: deque = field(default_factory=lambda: deque(maxlen=1000))

    # State
    _state: WatcherState = WatcherState.HEALTHY
    _last_action_time: Dict[str, float] = field(default_factory=dict)

    # Power control callbacks (to be wired to actual hardware)
    _power_off_fn: Optional[Callable[[str], bool]] = None
    _power_on_fn: Optional[Callable[[str], bool]] = None
    _power_cycle_fn: Optional[Callable[[str], bool]] = None

    def register_node(self, node_id: str):
        """Register a node for monitoring."""
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeHealth(node_id=node_id)

    def unregister_node(self, node_id: str):
        """Stop monitoring a node."""
        if node_id in self._nodes:
            del self._nodes[node_id]

    def report_ping(self, node_id: str, success: bool):
        """Report a ping result for a node."""
        if node_id not in self._nodes:
            self.register_node(node_id)

        node = self._nodes[node_id]
        node.update_ping(success)

        if not success:
            self._log_event(WatcherEvent(
                timestamp=time.time(),
                node_id=node_id,
                event_type="ping_fail",
                value=float(node.ping_failures),
            ))

            # Check if action needed
            if node.ping_failures >= self.policy.ping_failures_before_action:
                self._consider_action(node_id, "ping_failures")

    def report_temp(self, node_id: str, temp_c: float):
        """Report a temperature reading for a node."""
        if node_id not in self._nodes:
            self.register_node(node_id)

        node = self._nodes[node_id]
        node.update_temp(temp_c, self.policy.temp_warning_c)

        # Log if concerning
        if temp_c >= self.policy.temp_warning_c:
            self._log_event(WatcherEvent(
                timestamp=time.time(),
                node_id=node_id,
                event_type="temp_warning",
                value=temp_c,
            ))

        # Critical temperature = immediate action
        if temp_c >= self.policy.temp_critical_c:
            self._log_event(WatcherEvent(
                timestamp=time.time(),
                node_id=node_id,
                event_type="temp_critical",
                value=temp_c,
            ))
            if self.policy.auto_power_off_on_critical_temp:
                self._execute_action(node_id, PowerAction.POWER_OFF, "critical_temp")

        # Accumulated warnings
        elif node.temp_warnings >= self.policy.temp_warnings_before_action:
            self._consider_action(node_id, "temp_warnings")

    def report_power(self, node_id: str, watts: float):
        """Report power consumption for a node."""
        if node_id not in self._nodes:
            self.register_node(node_id)

        node = self._nodes[node_id]
        node.last_power_watts = watts

        if watts > self.policy.power_max_watts:
            self._log_event(WatcherEvent(
                timestamp=time.time(),
                node_id=node_id,
                event_type="power_high",
                value=watts,
            ))

    def _consider_action(self, node_id: str, reason: str):
        """Consider whether to take action on a node."""
        # Check cooldown
        last_action = self._last_action_time.get(node_id, 0)
        if time.time() - last_action < self.policy.action_cooldown_seconds:
            return  # Still in cooldown

        # Decide action
        if reason == "ping_failures" and self.policy.auto_power_cycle:
            self._execute_action(node_id, PowerAction.POWER_CYCLE, reason)
        elif reason == "temp_warnings":
            # Throttle first, then power off if continues
            node = self._nodes[node_id]
            if node.temp_warnings >= self.policy.temp_warnings_before_action * 2:
                self._execute_action(node_id, PowerAction.POWER_OFF, reason)

    def _execute_action(self, node_id: str, action: PowerAction, reason: str):
        """Execute a power action on a node."""
        success = False

        if action == PowerAction.POWER_OFF:
            if self._power_off_fn:
                success = self._power_off_fn(node_id)
                if success and node_id in self._nodes:
                    self._nodes[node_id].is_powered = False

        elif action == PowerAction.POWER_ON:
            if self._power_on_fn:
                success = self._power_on_fn(node_id)
                if success and node_id in self._nodes:
                    self._nodes[node_id].is_powered = True

        elif action == PowerAction.POWER_CYCLE:
            if self._power_cycle_fn:
                success = self._power_cycle_fn(node_id)

        # Log the action
        self._log_event(WatcherEvent(
            timestamp=time.time(),
            node_id=node_id,
            event_type="power_action",
            action_taken=action,
            details=f"reason={reason}, success={success}",
        ))

        self._last_action_time[node_id] = time.time()
        self._update_state()

    def _log_event(self, event: WatcherEvent):
        """Log an event."""
        self._events.append(event)

    def _update_state(self):
        """Update overall Watcher state."""
        # Count issues
        critical_nodes = 0
        alert_nodes = 0

        for node in self._nodes.values():
            if node.ping_failures >= self.policy.ping_failures_before_action:
                critical_nodes += 1
            elif node.ping_failures >= self.policy.ping_failures_before_alert:
                alert_nodes += 1

            if node.temp_warnings >= self.policy.temp_warnings_before_action:
                critical_nodes += 1
            elif node.temp_warnings > 0:
                alert_nodes += 1

        if critical_nodes > 0:
            self._state = WatcherState.EMERGENCY
        elif alert_nodes > 0:
            self._state = WatcherState.ALERT
        else:
            self._state = WatcherState.HEALTHY

    def get_state(self) -> WatcherState:
        """Get current Watcher state."""
        return self._state

    def get_node_health(self, node_id: str) -> Optional[NodeHealth]:
        """Get health info for a specific node."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> Dict[str, NodeHealth]:
        """Get health info for all nodes."""
        return dict(self._nodes)

    def get_recent_events(self, count: int = 20) -> List[WatcherEvent]:
        """Get recent events."""
        return list(self._events)[-count:]

    def get_status(self) -> Dict[str, Any]:
        """Get Watcher status summary."""
        healthy = sum(1 for n in self._nodes.values() if n.ping_failures == 0)
        powered = sum(1 for n in self._nodes.values() if n.is_powered)

        return {
            "watcher_id": self.watcher_id,
            "state": self._state.name,
            "nodes_monitored": len(self._nodes),
            "nodes_healthy": healthy,
            "nodes_powered": powered,
            "events_logged": len(self._events),
        }

    def wire_power_control(
        self,
        power_off: Callable[[str], bool],
        power_on: Callable[[str], bool],
        power_cycle: Callable[[str], bool],
    ):
        """Wire up power control callbacks."""
        self._power_off_fn = power_off
        self._power_on_fn = power_on
        self._power_cycle_fn = power_cycle

    def manual_power_off(self, node_id: str) -> bool:
        """Manually request power off for a node."""
        self._execute_action(node_id, PowerAction.POWER_OFF, "manual")
        return True

    def manual_power_on(self, node_id: str) -> bool:
        """Manually request power on for a node."""
        self._execute_action(node_id, PowerAction.POWER_ON, "manual")
        return True

    def manual_power_cycle(self, node_id: str) -> bool:
        """Manually request power cycle for a node."""
        self._execute_action(node_id, PowerAction.POWER_CYCLE, "manual")
        return True


def create_watcher(watcher_id: str = "watcher-01") -> Watcher:
    """Create a Watcher with default policy."""
    return Watcher(watcher_id=watcher_id)
