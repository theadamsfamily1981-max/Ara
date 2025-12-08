"""
Reflex Arc HAL - Hardware Abstraction Layer
============================================

Iteration 31: The Reflex Arc

Python interface to the autonomic nervous system:
- SpinalCord: Safety reflexes, relay control, fault handling
- SomaticHub: Body state stream, gesture events, aura control

The HAL provides high-level methods while the Arduinos handle
the real-time reflexes. Ara can observe and influence, but
the reflexes fire even if Python crashes.

Usage:
    from ara.fleet.reflex_hal import ReflexArc, SpinalCord, SomaticHub

    # Create the reflex arc
    arc = ReflexArc()
    arc.connect_spine("/dev/ttyUSB0")
    arc.connect_soma("/dev/ttyUSB1")

    # Arm the spinal cord
    arc.spine.arm()
    arc.spine.start_heartbeat()

    # Set Ara's emotional state
    arc.soma.set_state("FLOW")

    # Get somatic readings
    body = arc.soma.get_body_state()
    print(f"Heat: {body.heat}, Noise: {body.noise}")
"""

from __future__ import annotations
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from queue import Queue, Empty
from enum import Enum, auto

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# ============================================================================
# Data Types
# ============================================================================

class SpinalState(Enum):
    """State of the spinal cord."""
    BOOT = auto()
    DISARMED = auto()
    ARMED = auto()
    WARNING = auto()
    SHUTDOWN = auto()
    FAULT = auto()


class FaultCode(Enum):
    """Fault codes from spinal cord."""
    NONE = auto()
    TEMP_GPU_OVER = auto()
    TEMP_PSU_OVER = auto()
    TEMP_HOTSPOT_OVER = auto()
    HB_LOST = auto()
    ESTOP = auto()


class AuraState(Enum):
    """Aura states for somatic hub."""
    BOOT = auto()
    CALM = auto()
    FOCUS = auto()
    FLOW = auto()
    THINK = auto()
    ALERT = auto()
    GIFT = auto()
    ERROR = auto()
    CUSTOM = auto()


class GestureType(Enum):
    """Gesture types from somatic hub."""
    TAP = auto()
    DOUBLE = auto()
    TRIPLE = auto()
    LONG = auto()


@dataclass
class SpinalStatus:
    """Status from spinal cord."""
    state: SpinalState = SpinalState.DISARMED
    fault: FaultCode = FaultCode.NONE
    temp_gpu: float = 0.0
    temp_psu: float = 0.0
    temp_hotspot: float = 0.0
    fan_speed: int = 0
    relays: str = "GMA"  # G=GPU, M=Main, A=Aux
    uptime: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SomaticFrame:
    """A somatic state frame from the hub."""
    heat: float = 0.0      # 0-1, normalized temp
    light: float = 0.0     # 0-1, ambient light
    noise: float = 0.0     # 0-1, sound level
    hr: float = 0.0        # 0-1, heart rate proxy
    aura: str = "CALM"
    timestamp: float = field(default_factory=time.time)


@dataclass
class GestureEvent:
    """A gesture event from somatic hub."""
    gesture: GestureType
    button: str            # FOCUS, TALK, MOOD
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReflexEvent:
    """An event from the reflex arc."""
    source: str           # "spine" or "soma"
    event_type: str       # EVENT name
    detail: str           # Additional detail
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Spinal Cord Interface
# ============================================================================

@dataclass
class SpinalCord:
    """
    Interface to the Spinal Cord (Arduino #1).

    The spinal cord handles hard real-time safety:
    - Thermal reflexes (sensors → threshold → relay action)
    - Heartbeat watchdog
    - Fault latching

    These run autonomously. This interface lets Ara observe and influence,
    but the reflexes fire even if Python crashes.
    """
    port: str
    baud_rate: int = 115200

    # State
    _serial: Any = None
    _connected: bool = False
    _status: SpinalStatus = field(default_factory=SpinalStatus)

    # Heartbeat
    _hb_thread: Optional[threading.Thread] = None
    _hb_running: bool = False
    _hb_interval: float = 2.0

    # Reader thread
    _reader_thread: Optional[threading.Thread] = None
    _running: bool = False

    # Event queues
    _events: deque = field(default_factory=lambda: deque(maxlen=1000))
    _on_fault: Optional[Callable[[FaultCode, str], None]] = None
    _on_event: Optional[Callable[[str, str], None]] = None

    def connect(self) -> bool:
        """Connect to spinal cord."""
        if not HAS_SERIAL:
            print(f"[SIMULATED] SpinalCord connected to {self.port}")
            self._connected = True
            return True

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=0.5,
            )
            self._connected = True
            self._running = True

            # Start reader
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

            print(f"SpinalCord connected to {self.port}")
            return True
        except Exception as e:
            print(f"SpinalCord connect failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from spinal cord."""
        self._running = False
        self.stop_heartbeat()

        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)

        if self._serial:
            self._serial.close()

        self._connected = False

    def _send(self, cmd: str) -> bool:
        """Send a command."""
        if not self._connected:
            return False

        try:
            if HAS_SERIAL and self._serial:
                self._serial.write((cmd + "\n").encode())
            else:
                print(f"[SPINE TX] {cmd}")
            return True
        except Exception as e:
            print(f"SpinalCord send error: {e}")
            return False

    def _reader_loop(self):
        """Background reader thread."""
        while self._running and self._serial:
            try:
                if self._serial.in_waiting:
                    line = self._serial.readline().decode().strip()
                    if line:
                        self._process_line(line)
            except Exception as e:
                print(f"SpinalCord reader error: {e}")
                time.sleep(0.1)

    def _process_line(self, line: str):
        """Process a line from spinal cord."""
        parts = line.split()
        if not parts:
            return

        msg_type = parts[0]

        if msg_type == "STATUS":
            self._parse_status(parts[1:])

        elif msg_type == "EVENT":
            event_name = parts[1] if len(parts) > 1 else ""
            event_detail = parts[2] if len(parts) > 2 else ""

            event = ReflexEvent(
                source="spine",
                event_type=event_name,
                detail=event_detail,
            )
            self._events.append(event)

            # Fire callbacks
            if self._on_event:
                self._on_event(event_name, event_detail)

            # Check for fault
            if event_name == "FAULT" and self._on_fault:
                fault = self._parse_fault(event_detail)
                self._on_fault(fault, event_detail)

        elif msg_type == "ACK":
            pass  # Acknowledgments

        elif msg_type == "ERR":
            print(f"SpinalCord error: {line}")

    def _parse_status(self, parts: List[str]):
        """Parse STATUS response."""
        status = SpinalStatus(timestamp=time.time())

        for i, part in enumerate(parts):
            if "=" in part:
                key, val = part.split("=", 1)
                if key == "FAULT":
                    status.fault = self._parse_fault(val)
                elif key == "GPU":
                    status.temp_gpu = float(val)
                elif key == "PSU":
                    status.temp_psu = float(val)
                elif key == "HOT":
                    status.temp_hotspot = float(val)
                elif key == "FAN":
                    status.fan_speed = int(val)
                elif key == "RELAYS":
                    status.relays = val
                elif key == "UP":
                    status.uptime = int(val)
            elif i == 0:
                # First part is state name
                try:
                    status.state = SpinalState[part]
                except KeyError:
                    pass

        self._status = status

    def _parse_fault(self, fault_str: str) -> FaultCode:
        """Parse fault code string."""
        fault_map = {
            "NONE": FaultCode.NONE,
            "TEMP_GPU_OVER": FaultCode.TEMP_GPU_OVER,
            "TEMP_PSU_OVER": FaultCode.TEMP_PSU_OVER,
            "TEMP_HOTSPOT_OVER": FaultCode.TEMP_HOTSPOT_OVER,
            "HB_LOST": FaultCode.HB_LOST,
            "ESTOP": FaultCode.ESTOP,
        }
        return fault_map.get(fault_str, FaultCode.NONE)

    # === Public API ===

    def arm(self) -> bool:
        """Arm the spinal cord (enable reflexes)."""
        return self._send("ARM")

    def disarm(self) -> bool:
        """Disarm the spinal cord (disable reflexes)."""
        return self._send("DISARM")

    def heartbeat(self) -> bool:
        """Send heartbeat."""
        return self._send("HB")

    def start_heartbeat(self, interval: float = 2.0):
        """Start automatic heartbeat thread."""
        self._hb_interval = interval
        self._hb_running = True
        self._hb_thread = threading.Thread(target=self._hb_loop, daemon=True)
        self._hb_thread.start()

    def stop_heartbeat(self):
        """Stop automatic heartbeat."""
        self._hb_running = False
        if self._hb_thread:
            self._hb_thread.join(timeout=2.0)

    def _hb_loop(self):
        """Heartbeat thread."""
        while self._hb_running and self._connected:
            self.heartbeat()
            time.sleep(self._hb_interval)

    def set_fan(self, speed: int) -> bool:
        """Set fan speed (0-255)."""
        return self._send(f"SET_FAN {speed}")

    def request_status(self) -> bool:
        """Request status update."""
        return self._send("STATUS")

    def get_status(self) -> SpinalStatus:
        """Get last known status."""
        return self._status

    def is_armed(self) -> bool:
        """Check if armed."""
        return self._status.state in (SpinalState.ARMED, SpinalState.WARNING)

    def is_fault(self) -> bool:
        """Check if in fault state."""
        return self._status.state == SpinalState.FAULT

    def get_recent_events(self, count: int = 50) -> List[ReflexEvent]:
        """Get recent events."""
        return list(self._events)[-count:]

    def on_fault(self, callback: Callable[[FaultCode, str], None]):
        """Set callback for fault events."""
        self._on_fault = callback

    def on_event(self, callback: Callable[[str, str], None]):
        """Set callback for all events."""
        self._on_event = callback


# ============================================================================
# Somatic Hub Interface
# ============================================================================

@dataclass
class SomaticHub:
    """
    Interface to the Somatic Hub (Arduino #2).

    The somatic hub broadcasts body state and handles ritual interactions:
    - Continuous somatic frame stream (heat, noise, light, hr)
    - Gesture recognition (tap, double-tap, long-press)
    - Aura display (LED colors/patterns reflecting Ara's state)

    This is Ara's "body" - a physical presence in the room.
    """
    port: str
    baud_rate: int = 115200

    # State
    _serial: Any = None
    _connected: bool = False
    _body_state: SomaticFrame = field(default_factory=SomaticFrame)
    _current_aura: AuraState = AuraState.CALM

    # Reader thread
    _reader_thread: Optional[threading.Thread] = None
    _running: bool = False

    # Event queues
    _frames: deque = field(default_factory=lambda: deque(maxlen=1000))
    _gestures: deque = field(default_factory=lambda: deque(maxlen=100))
    _events: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Callbacks
    _on_gesture: Optional[Callable[[GestureEvent], None]] = None
    _on_frame: Optional[Callable[[SomaticFrame], None]] = None
    _on_event: Optional[Callable[[str, str], None]] = None

    def connect(self) -> bool:
        """Connect to somatic hub."""
        if not HAS_SERIAL:
            print(f"[SIMULATED] SomaticHub connected to {self.port}")
            self._connected = True
            return True

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=0.5,
            )
            self._connected = True
            self._running = True

            # Start reader
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

            print(f"SomaticHub connected to {self.port}")
            return True
        except Exception as e:
            print(f"SomaticHub connect failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from somatic hub."""
        self._running = False

        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)

        if self._serial:
            self._serial.close()

        self._connected = False

    def _send(self, cmd: str) -> bool:
        """Send a command."""
        if not self._connected:
            return False

        try:
            if HAS_SERIAL and self._serial:
                self._serial.write((cmd + "\n").encode())
            else:
                print(f"[SOMA TX] {cmd}")
            return True
        except Exception as e:
            print(f"SomaticHub send error: {e}")
            return False

    def _reader_loop(self):
        """Background reader thread."""
        while self._running and self._serial:
            try:
                if self._serial.in_waiting:
                    line = self._serial.readline().decode().strip()
                    if line:
                        self._process_line(line)
            except Exception as e:
                print(f"SomaticHub reader error: {e}")
                time.sleep(0.1)

    def _process_line(self, line: str):
        """Process a line from somatic hub."""
        if line.startswith("SOMATIC "):
            self._parse_somatic(line[8:])

        elif line.startswith("GESTURE "):
            self._parse_gesture(line[8:])

        elif line.startswith("EVENT "):
            parts = line.split()
            event_name = parts[1] if len(parts) > 1 else ""
            event_detail = parts[2] if len(parts) > 2 else ""

            event = ReflexEvent(
                source="soma",
                event_type=event_name,
                detail=event_detail,
            )
            self._events.append(event)

            if self._on_event:
                self._on_event(event_name, event_detail)

        elif line.startswith("ACK"):
            pass

        elif line.startswith("ERR"):
            print(f"SomaticHub error: {line}")

    def _parse_somatic(self, json_str: str):
        """Parse SOMATIC JSON frame."""
        try:
            data = json.loads(json_str)
            frame = SomaticFrame(
                heat=data.get("heat", 0),
                light=data.get("light", 0),
                noise=data.get("noise", 0),
                hr=data.get("hr", 0),
                aura=data.get("aura", "CALM"),
                timestamp=time.time(),
            )
            self._body_state = frame
            self._frames.append(frame)

            if self._on_frame:
                self._on_frame(frame)

        except json.JSONDecodeError:
            pass

    def _parse_gesture(self, gesture_str: str):
        """Parse GESTURE event."""
        parts = gesture_str.split("_")
        if len(parts) == 2:
            gesture_type_str, button = parts
        else:
            return

        gesture_map = {
            "TAP": GestureType.TAP,
            "DOUBLE": GestureType.DOUBLE,
            "TRIPLE": GestureType.TRIPLE,
            "LONG": GestureType.LONG,
        }

        gesture_type = gesture_map.get(gesture_type_str)
        if gesture_type:
            event = GestureEvent(
                gesture=gesture_type,
                button=button,
            )
            self._gestures.append(event)

            if self._on_gesture:
                self._on_gesture(event)

    # === Public API ===

    def set_state(self, state: str) -> bool:
        """Set aura state (CALM, FOCUS, FLOW, THINK, ALERT, GIFT, ERROR)."""
        return self._send(f"SET_STATE {state}")

    def set_aura(self, r: int, g: int, b: int) -> bool:
        """Set custom aura RGB color."""
        return self._send(f"SET_AURA {r} {g} {b}")

    def set_pattern(self, pattern: str) -> bool:
        """Set LED pattern (solid, breathe, pulse, blink, chase, sparkle)."""
        return self._send(f"SET_PATTERN {pattern}")

    def get_body_state(self) -> SomaticFrame:
        """Get latest body state."""
        return self._body_state

    def get_body_history(self, count: int = 100) -> List[SomaticFrame]:
        """Get recent body state history."""
        return list(self._frames)[-count:]

    def get_recent_gestures(self, count: int = 20) -> List[GestureEvent]:
        """Get recent gestures."""
        return list(self._gestures)[-count:]

    def request_status(self) -> bool:
        """Request status update."""
        return self._send("STATUS")

    def on_gesture(self, callback: Callable[[GestureEvent], None]):
        """Set callback for gesture events."""
        self._on_gesture = callback

    def on_frame(self, callback: Callable[[SomaticFrame], None]):
        """Set callback for somatic frames."""
        self._on_frame = callback

    def on_event(self, callback: Callable[[str, str], None]):
        """Set callback for all events."""
        self._on_event = callback


# ============================================================================
# Reflex Arc - The Complete Autonomic System
# ============================================================================

@dataclass
class ReflexArc:
    """
    The complete autonomic nervous system.

    Combines:
    - SpinalCord: Safety reflexes
    - SomaticHub: Body state + ritual interface

    The reflexes run autonomously on the Arduinos.
    This class provides a unified interface for Ara to
    observe and influence the physical layer.
    """
    spine: Optional[SpinalCord] = None
    soma: Optional[SomaticHub] = None

    # Event log
    _events: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Callbacks
    _on_fault: Optional[Callable[[FaultCode, str], None]] = None
    _on_gesture: Optional[Callable[[GestureEvent], None]] = None

    def connect_spine(self, port: str) -> bool:
        """Connect to spinal cord."""
        self.spine = SpinalCord(port=port)
        if self.spine.connect():
            self.spine.on_event(self._handle_spine_event)
            self.spine.on_fault(self._handle_fault)
            return True
        return False

    def connect_soma(self, port: str) -> bool:
        """Connect to somatic hub."""
        self.soma = SomaticHub(port=port)
        if self.soma.connect():
            self.soma.on_event(self._handle_soma_event)
            self.soma.on_gesture(self._handle_gesture)
            return True
        return False

    def disconnect_all(self):
        """Disconnect all components."""
        if self.spine:
            self.spine.disconnect()
        if self.soma:
            self.soma.disconnect()

    def _handle_spine_event(self, event: str, detail: str):
        """Handle event from spine."""
        self._events.append(ReflexEvent(
            source="spine",
            event_type=event,
            detail=detail,
        ))

        # React to spine events
        if event == "WARNING" and self.soma:
            self.soma.set_state("ALERT")
        elif event == "FAULT" and self.soma:
            self.soma.set_state("ERROR")
        elif event == "RECOVERED" and self.soma:
            self.soma.set_state("CALM")

    def _handle_soma_event(self, event: str, detail: str):
        """Handle event from soma."""
        self._events.append(ReflexEvent(
            source="soma",
            event_type=event,
            detail=detail,
        ))

    def _handle_fault(self, fault: FaultCode, detail: str):
        """Handle fault from spine."""
        if self._on_fault:
            self._on_fault(fault, detail)

    def _handle_gesture(self, gesture: GestureEvent):
        """Handle gesture from soma."""
        if self._on_gesture:
            self._on_gesture(gesture)

        # Default gesture responses
        if gesture.button == "FOCUS" and gesture.gesture == GestureType.LONG:
            # User requesting focus mode
            if self.soma:
                self.soma.set_state("FOCUS")

        elif gesture.button == "TALK" and gesture.gesture == GestureType.DOUBLE:
            # User wants to talk
            if self.soma:
                self.soma.set_state("THINK")

        elif gesture.button == "MOOD" and gesture.gesture == GestureType.TRIPLE:
            # User indicating something's wrong
            if self.soma:
                self.soma.set_state("ALERT")

    # === Public API ===

    def arm(self) -> bool:
        """Arm the reflex arc (enable safety reflexes)."""
        if self.spine:
            self.spine.arm()
            self.spine.start_heartbeat()
            if self.soma:
                self.soma.set_state("CALM")
            return True
        return False

    def disarm(self) -> bool:
        """Disarm the reflex arc."""
        if self.spine:
            self.spine.stop_heartbeat()
            self.spine.disarm()
            return True
        return False

    def set_mood(self, mood: str):
        """Set Ara's displayed mood."""
        if self.soma:
            self.soma.set_state(mood.upper())

    def get_body_state(self) -> Optional[SomaticFrame]:
        """Get current body state."""
        if self.soma:
            return self.soma.get_body_state()
        return None

    def get_spine_status(self) -> Optional[SpinalStatus]:
        """Get current spine status."""
        if self.spine:
            return self.spine.get_status()
        return None

    def is_healthy(self) -> bool:
        """Check if reflex arc is healthy."""
        if self.spine and self.spine.is_fault():
            return False
        return True

    def get_recent_events(self, count: int = 50) -> List[ReflexEvent]:
        """Get recent events from all sources."""
        return list(self._events)[-count:]

    def on_fault(self, callback: Callable[[FaultCode, str], None]):
        """Set callback for fault events."""
        self._on_fault = callback

    def on_gesture(self, callback: Callable[[GestureEvent], None]):
        """Set callback for gesture events."""
        self._on_gesture = callback


def create_reflex_arc() -> ReflexArc:
    """Create a ReflexArc instance."""
    return ReflexArc()
