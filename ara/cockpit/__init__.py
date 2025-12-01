"""
Ara Dynamic Cockpit Integration

Provides real-time monitoring and control of Ara's autonomous systems via D-Bus.
Connects to L3 Metacontrol service for PAD visualization and mode switching.

Features:
1. Real-time PAD Signal Display - Valence/Arousal/Dominance visualization
2. Workspace Mode Control - Switch between focus/creative/relaxed modes
3. Stability Monitoring - Topology gap and EPR-CV alerts
4. Control Modulation View - Temperature/Memory/Attention multipliers
5. Event History - Log of signal changes and mode transitions

Usage:
    from ara.cockpit import CockpitClient, PADDisplay, run_cockpit

    # Start cockpit with terminal UI
    run_cockpit()

    # Or use programmatic interface
    client = CockpitClient()
    client.connect()
    client.on_pad_changed(my_callback)
    client.set_workspace_mode("focus")
"""

import sys
import json
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from enum import Enum

# Add project root
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("ara.cockpit")

# D-Bus configuration
DBUS_BUS_NAME = "org.ara.metacontrol"
DBUS_OBJECT_PATH = "/org/ara/metacontrol/L3"
DBUS_INTERFACE = "org.ara.metacontrol.L3Interface"

# Try D-Bus import
try:
    import dbus
    from dbus.mainloop.glib import DBusGMainLoop
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False
    dbus = None


class WorkspaceMode(str, Enum):
    """Workspace modes for L3 Metacontrol."""
    FOCUS = "focus"       # Deep work, low temperature
    CREATIVE = "creative" # Exploration, high temperature
    RELAXED = "relaxed"   # Balanced, medium temperature
    URGENT = "urgent"     # High priority, aggressive
    REVIEW = "review"     # Analysis mode


@dataclass
class PADState:
    """Pleasure-Arousal-Dominance state."""
    valence: float = 0.0       # Pleasure: -1 (negative) to +1 (positive)
    arousal: float = 0.5       # Arousal: 0 (calm) to 1 (excited)
    dominance: float = 0.5     # Dominance: 0 (submissive) to 1 (dominant)

    # Derived signals
    topology_gap: float = 0.0
    temperature_mult: float = 1.0
    memory_mult: float = 1.0
    attention_gain: float = 1.0

    # Status flags
    needs_suppression: bool = False
    stability_warning: bool = False

    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PADState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CockpitEvent:
    """Event in cockpit history."""
    event_type: str
    data: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class CockpitClient:
    """
    D-Bus client for Ara Cockpit integration.

    Connects to L3 Metacontrol service and provides:
    - Real-time PAD signal updates
    - Workspace mode control
    - Event history
    """

    def __init__(
        self,
        bus_name: str = DBUS_BUS_NAME,
        object_path: str = DBUS_OBJECT_PATH,
        interface_name: str = DBUS_INTERFACE,
    ):
        self.bus_name = bus_name
        self.object_path = object_path
        self.interface_name = interface_name

        self._bus: Optional[Any] = None
        self._proxy: Optional[Any] = None
        self._interface: Optional[Any] = None
        self._connected = False

        # Current state
        self.pad_state = PADState()
        self.workspace_mode = WorkspaceMode.FOCUS
        self.event_history: List[CockpitEvent] = []

        # Callbacks
        self._pad_callbacks: List[Callable[[PADState], None]] = []
        self._mode_callbacks: List[Callable[[WorkspaceMode], None]] = []
        self._modulation_callbacks: List[Callable[[Dict], None]] = []

        logger.info("CockpitClient initialized")

    def connect(self) -> bool:
        """
        Connect to the D-Bus service.

        Returns True if connection successful.
        """
        if not DBUS_AVAILABLE:
            logger.warning("D-Bus not available, using mock mode")
            self._connected = False
            return False

        try:
            DBusGMainLoop(set_as_default=True)
            self._bus = dbus.SessionBus()
            self._proxy = self._bus.get_object(self.bus_name, self.object_path)
            self._interface = dbus.Interface(self._proxy, self.interface_name)

            # Connect signal handlers
            self._bus.add_signal_receiver(
                self._on_pad_changed,
                signal_name="PADChanged",
                dbus_interface=self.interface_name,
            )
            self._bus.add_signal_receiver(
                self._on_mode_changed,
                signal_name="ModeChanged",
                dbus_interface=self.interface_name,
            )
            self._bus.add_signal_receiver(
                self._on_modulation_changed,
                signal_name="ModulationChanged",
                dbus_interface=self.interface_name,
            )

            self._connected = True
            logger.info(f"Connected to D-Bus: {self.bus_name}")

            # Get initial status
            self._fetch_status()

            return True

        except Exception as e:
            logger.error(f"D-Bus connection failed: {e}")
            self._connected = False
            return False

    def _on_pad_changed(self, data_json: str):
        """Handle PADChanged signal."""
        try:
            data = json.loads(data_json)
            self.pad_state = PADState.from_dict(data)
            self.pad_state.timestamp = datetime.utcnow().isoformat()

            self.event_history.append(CockpitEvent(
                event_type="PADChanged",
                data=data,
            ))

            # Invoke callbacks
            for cb in self._pad_callbacks:
                try:
                    cb(self.pad_state)
                except Exception as e:
                    logger.error(f"PAD callback error: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid PAD JSON: {e}")

    def _on_mode_changed(self, mode_str: str):
        """Handle ModeChanged signal."""
        try:
            self.workspace_mode = WorkspaceMode(mode_str)

            self.event_history.append(CockpitEvent(
                event_type="ModeChanged",
                data={"mode": mode_str},
            ))

            for cb in self._mode_callbacks:
                try:
                    cb(self.workspace_mode)
                except Exception as e:
                    logger.error(f"Mode callback error: {e}")

        except ValueError:
            logger.warning(f"Unknown workspace mode: {mode_str}")

    def _on_modulation_changed(self, data_json: str):
        """Handle ModulationChanged signal."""
        try:
            data = json.loads(data_json)

            self.event_history.append(CockpitEvent(
                event_type="ModulationChanged",
                data=data,
            ))

            for cb in self._modulation_callbacks:
                try:
                    cb(data)
                except Exception as e:
                    logger.error(f"Modulation callback error: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid modulation JSON: {e}")

    def _fetch_status(self):
        """Fetch current status from service."""
        if not self._connected or not self._interface:
            return

        try:
            status_json = self._interface.GetStatus()
            status = json.loads(status_json)

            if "mode" in status:
                try:
                    self.workspace_mode = WorkspaceMode(status["mode"])
                except ValueError:
                    pass

            logger.debug(f"Fetched status: {status}")

        except Exception as e:
            logger.warning(f"Failed to fetch status: {e}")

    def on_pad_changed(self, callback: Callable[[PADState], None]):
        """Register callback for PAD changes."""
        self._pad_callbacks.append(callback)

    def on_mode_changed(self, callback: Callable[[WorkspaceMode], None]):
        """Register callback for mode changes."""
        self._mode_callbacks.append(callback)

    def on_modulation_changed(self, callback: Callable[[Dict], None]):
        """Register callback for modulation changes."""
        self._modulation_callbacks.append(callback)

    def set_workspace_mode(self, mode: WorkspaceMode | str) -> bool:
        """
        Set workspace mode via D-Bus.

        Args:
            mode: WorkspaceMode enum or string

        Returns:
            True if successful
        """
        if isinstance(mode, WorkspaceMode):
            mode_str = mode.value
        else:
            mode_str = str(mode)

        if not self._connected or not self._interface:
            logger.warning("Not connected, setting mode locally")
            self.workspace_mode = WorkspaceMode(mode_str)
            return True

        try:
            result_json = self._interface.SetWorkspaceMode(mode_str)
            result = json.loads(result_json)

            if result.get("status") == "ok":
                self.workspace_mode = WorkspaceMode(mode_str)
                return True
            else:
                logger.error(f"SetWorkspaceMode failed: {result}")
                return False

        except Exception as e:
            logger.error(f"SetWorkspaceMode error: {e}")
            return False

    def get_pad_gating(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        goal_valence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute PAD gating via D-Bus.

        Returns control outputs (temperature, memory, attention multipliers).
        """
        if not self._connected or not self._interface:
            # Local fallback computation
            temp = 0.8 + 0.5 * arousal
            mem = 0.95 + 0.25 * (valence + 1) / 2
            attn = 0.8 + 0.4 * dominance
            return {
                "temperature_mult": temp,
                "memory_mult": mem,
                "attention_gain": attn,
                "local_fallback": True,
            }

        try:
            result_json = self._interface.ComputePADGating(
                valence, arousal, dominance, goal_valence
            )
            return json.loads(result_json)

        except Exception as e:
            logger.error(f"ComputePADGating error: {e}")
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get full service status."""
        status = {
            "connected": self._connected,
            "workspace_mode": self.workspace_mode.value,
            "pad_state": self.pad_state.to_dict(),
            "event_count": len(self.event_history),
            "dbus_available": DBUS_AVAILABLE,
        }

        if self._connected and self._interface:
            try:
                remote_status = json.loads(self._interface.GetStatus())
                status["remote"] = remote_status
            except Exception as e:
                status["remote_error"] = str(e)

        return status

    def disconnect(self):
        """Disconnect from D-Bus."""
        self._connected = False
        self._bus = None
        self._proxy = None
        self._interface = None
        logger.info("Disconnected from D-Bus")


class PADDisplay:
    """
    Terminal-based PAD state display.

    Shows real-time visualization of:
    - Valence bar (negative/positive)
    - Arousal bar (calm/excited)
    - Dominance bar (submissive/dominant)
    - Control multipliers
    - Status flags
    """

    def __init__(self, client: CockpitClient):
        self.client = client
        self._running = False

    def format_bar(self, value: float, width: int = 30, min_val: float = -1, max_val: float = 1) -> str:
        """Format a value as an ASCII bar."""
        normalized = (value - min_val) / (max_val - min_val)
        filled = int(normalized * width)
        filled = max(0, min(width, filled))

        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def format_pad_state(self, state: PADState) -> str:
        """Format PAD state for display."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║                    ARA COCKPIT - PAD STATE                   ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ Mode: {self.client.workspace_mode.value:15s}  │  Time: {state.timestamp[-12:]:<12s} ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ Valence   {self.format_bar(state.valence, 30, -1, 1)} {state.valence:+.3f} ║",
            f"║ Arousal   {self.format_bar(state.arousal, 30, 0, 1)} {state.arousal:+.3f} ║",
            f"║ Dominance {self.format_bar(state.dominance, 30, 0, 1)} {state.dominance:+.3f} ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ CONTROL MODULATION                                           ║",
            f"║   Temperature: {state.temperature_mult:.3f}   Memory: {state.memory_mult:.3f}   Attention: {state.attention_gain:.3f} ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ Topology Gap: {state.topology_gap:.4f}  │  Suppress: {'YES' if state.needs_suppression else 'NO ':3s}  │  Warn: {'YES' if state.stability_warning else 'NO ':3s} ║",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def display(self):
        """Display current state."""
        print("\033[2J\033[H")  # Clear screen
        print(self.format_pad_state(self.client.pad_state))

    def run(self, refresh_rate: float = 0.5):
        """Run continuous display loop."""
        self._running = True

        def on_update(state):
            if self._running:
                self.display()

        self.client.on_pad_changed(on_update)
        self.display()

        try:
            while self._running:
                time.sleep(refresh_rate)
        except KeyboardInterrupt:
            self._running = False

    def stop(self):
        """Stop display loop."""
        self._running = False


def run_cockpit(
    connect_dbus: bool = True,
    demo_mode: bool = False,
):
    """
    Run the cockpit terminal interface.

    Args:
        connect_dbus: Whether to connect to D-Bus service
        demo_mode: If True, generate demo PAD updates
    """
    print("Ara Cockpit Starting...")

    client = CockpitClient()

    if connect_dbus:
        if client.connect():
            print("Connected to D-Bus service")
        else:
            print("D-Bus connection failed, running in demo mode")
            demo_mode = True

    display = PADDisplay(client)

    # Demo mode: generate fake updates
    if demo_mode:
        import random

        def demo_updater():
            while display._running:
                client.pad_state = PADState(
                    valence=random.uniform(-0.8, 0.8),
                    arousal=random.uniform(0.2, 0.9),
                    dominance=random.uniform(0.3, 0.7),
                    topology_gap=random.uniform(0, 0.5),
                    temperature_mult=random.uniform(0.8, 1.3),
                    memory_mult=random.uniform(0.7, 1.2),
                    attention_gain=random.uniform(0.8, 1.2),
                    needs_suppression=random.random() > 0.8,
                    stability_warning=random.random() > 0.9,
                    timestamp=datetime.utcnow().isoformat(),
                )
                display.display()
                time.sleep(0.5)

        thread = threading.Thread(target=demo_updater, daemon=True)
        thread.start()

    print("Press Ctrl+C to exit")
    display.run()


# Exports
__all__ = [
    "CockpitClient",
    "PADDisplay",
    "PADState",
    "WorkspaceMode",
    "CockpitEvent",
    "run_cockpit",
    "DBUS_AVAILABLE",
]
