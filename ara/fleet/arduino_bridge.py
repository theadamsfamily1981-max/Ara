"""
Arduino Bridge - Python ↔ Arduino Communication
================================================

Talks to Arduino nodes over serial using the ara:// protocol.

Supports:
- Red Wire Guardian (relays, watchdog, E-STOP)
- Somatic Node (sensors, LEDs, buttons)

Usage:
    from ara.fleet.arduino_bridge import ArduinoBridge, GuardianBridge, SomaticBridge

    # Connect to Guardian
    guardian = GuardianBridge("/dev/ttyUSB0")
    guardian.connect()
    guardian.send_heartbeat()
    guardian.kill_relay(1)

    # Connect to Somatic
    somatic = SomaticBridge("/dev/ttyUSB1")
    somatic.connect()
    somatic.set_led_color("blue")
    sensors = somatic.get_latest_sensors()
"""

from __future__ import annotations
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from queue import Queue, Empty

# Optional: requires pyserial
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("Warning: pyserial not installed. Arduino bridge will be simulated.")


@dataclass
class ArduinoMessage:
    """A message from an Arduino."""
    msg_type: str           # "sensor", "status", "event", "ack", "error"
    timestamp: float
    data: Dict[str, Any]
    raw: str


@dataclass
class ArduinoBridge:
    """
    Base class for Arduino communication.

    Handles serial connection, message parsing, and threading.
    """
    port: str
    baud_rate: int = 115200
    timeout: float = 1.0

    # Internal state
    _serial: Any = None
    _connected: bool = False
    _reader_thread: Optional[threading.Thread] = None
    _running: bool = False

    # Message handling
    _message_queue: Queue = field(default_factory=Queue)
    _message_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Callbacks
    _on_message: Optional[Callable[[ArduinoMessage], None]] = None
    _on_event: Optional[Callable[[str, Dict], None]] = None
    _on_sensor: Optional[Callable[[Dict], None]] = None

    def connect(self) -> bool:
        """Connect to the Arduino."""
        if not HAS_SERIAL:
            print(f"[SIMULATED] Connected to {self.port}")
            self._connected = True
            return True

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
            )
            self._connected = True
            self._running = True

            # Start reader thread
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

            print(f"Connected to {self.port}")
            return True

        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            return False

    def disconnect(self):
        """Disconnect from the Arduino."""
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)

        if self._serial:
            self._serial.close()
            self._serial = None

        self._connected = False
        print(f"Disconnected from {self.port}")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def send_command(self, cmd: Dict[str, Any]) -> bool:
        """Send a command to the Arduino."""
        if not self._connected:
            return False

        try:
            line = json.dumps(cmd) + "\n"

            if HAS_SERIAL and self._serial:
                self._serial.write(line.encode())
            else:
                print(f"[SIMULATED TX] {line.strip()}")

            return True

        except Exception as e:
            print(f"Send error: {e}")
            return False

    def send_heartbeat(self) -> bool:
        """Send heartbeat command."""
        return self.send_command({"cmd": "hb"})

    def request_status(self) -> bool:
        """Request status from Arduino."""
        return self.send_command({"cmd": "status"})

    def get_message(self, timeout: float = 0.1) -> Optional[ArduinoMessage]:
        """Get next message from queue."""
        try:
            return self._message_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_recent_messages(self, count: int = 50) -> List[ArduinoMessage]:
        """Get recent messages."""
        return list(self._message_history)[-count:]

    def on_message(self, callback: Callable[[ArduinoMessage], None]):
        """Set callback for all messages."""
        self._on_message = callback

    def on_event(self, callback: Callable[[str, Dict], None]):
        """Set callback for event messages."""
        self._on_event = callback

    def on_sensor(self, callback: Callable[[Dict], None]):
        """Set callback for sensor messages."""
        self._on_sensor = callback

    def _reader_loop(self):
        """Background thread that reads from serial."""
        while self._running and self._serial:
            try:
                if self._serial.in_waiting:
                    line = self._serial.readline().decode().strip()
                    if line:
                        self._process_line(line)
            except Exception as e:
                print(f"Reader error: {e}")
                time.sleep(0.1)

    def _process_line(self, line: str):
        """Process a line received from Arduino."""
        try:
            data = json.loads(line)
            msg_type = data.get("type", "unknown")

            msg = ArduinoMessage(
                msg_type=msg_type,
                timestamp=time.time(),
                data=data,
                raw=line,
            )

            # Store in history
            self._message_history.append(msg)

            # Put in queue
            self._message_queue.put(msg)

            # Fire callbacks
            if self._on_message:
                self._on_message(msg)

            if msg_type == "event" and self._on_event:
                self._on_event(data.get("event", ""), data)

            if msg_type == "sensor" and self._on_sensor:
                self._on_sensor(data)

        except json.JSONDecodeError:
            print(f"Invalid JSON: {line}")


@dataclass
class GuardianBridge(ArduinoBridge):
    """
    Bridge for the Red Wire Guardian (safety controller).

    Provides relay control, watchdog, and E-STOP handling.
    """

    # Watchdog state
    _last_heartbeat: float = 0.0
    _heartbeat_interval: float = 5.0
    _heartbeat_thread: Optional[threading.Thread] = None

    def arm_relay(self, relay: int) -> bool:
        """Arm a relay (allow it to be powered)."""
        return self.send_command({"cmd": "arm", "relay": relay})

    def disarm_relay(self, relay: int) -> bool:
        """Disarm a relay (prevent remote control)."""
        return self.send_command({"cmd": "disarm", "relay": relay})

    def kill_relay(self, relay: int) -> bool:
        """Kill power to a relay (turn off and lock)."""
        return self.send_command({"cmd": "kill", "relay": relay})

    def cycle_relay(self, relay: int, off_ms: int = 5000) -> bool:
        """Power cycle a relay."""
        return self.send_command({"cmd": "cycle", "relay": relay, "off_ms": off_ms})

    def start_heartbeat(self, interval: float = 5.0):
        """Start automatic heartbeat sending."""
        self._heartbeat_interval = interval
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self):
        """Stop automatic heartbeat."""
        self._heartbeat_interval = 0  # Will cause thread to exit

    def _heartbeat_loop(self):
        """Background thread for sending heartbeats."""
        while self._heartbeat_interval > 0 and self._connected:
            self.send_heartbeat()
            self._last_heartbeat = time.time()
            time.sleep(self._heartbeat_interval)


@dataclass
class SomaticBridge(ArduinoBridge):
    """
    Bridge for the Somatic Node (sensors + LEDs).

    Provides sensor reading and LED control.
    """

    # Latest sensor readings
    _latest_sensors: Dict[str, float] = field(default_factory=dict)
    _sensor_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    def __post_init__(self):
        # Wire up sensor callback
        self.on_sensor(self._store_sensors)

    def _store_sensors(self, data: Dict):
        """Store sensor reading."""
        self._latest_sensors = {
            "temp_gpu": data.get("temp_gpu", 0),
            "temp_room": data.get("temp_room", 0),
            "light": data.get("light", 0),
            "psu_amps": data.get("psu_amps", 0),
            "timestamp": data.get("t", 0),
        }
        self._sensor_history.append(self._latest_sensors.copy())

    def get_latest_sensors(self) -> Dict[str, float]:
        """Get most recent sensor readings."""
        return dict(self._latest_sensors)

    def get_sensor_history(self, count: int = 100) -> List[Dict]:
        """Get sensor history."""
        return list(self._sensor_history)[-count:]

    def set_led_rgb(self, r: int, g: int, b: int) -> bool:
        """Set LED color (RGB 0-255)."""
        return self.send_command({"cmd": "led", "r": r, "g": g, "b": b})

    def set_led_color(self, color: str) -> bool:
        """Set LED to named color."""
        return self.send_command({"cmd": "led", "color": color})

    def set_led_pattern(self, pattern: str, color: str = None) -> bool:
        """Set LED pattern (solid, pulse, blink, rainbow)."""
        cmd = {"cmd": "led", "pattern": pattern}
        if color:
            cmd["color"] = color
        return self.send_command(cmd)

    def set_mood(self, mood: str):
        """
        Set LED based on mood string.

        Maps Ara's internal state to visual feedback.
        """
        mood_map = {
            "calm": ("blue", "solid"),
            "focused": ("cyan", "solid"),
            "thinking": ("purple", "pulse"),
            "alert": ("yellow", "blink"),
            "error": ("red", "blink"),
            "happy": ("green", "pulse"),
            "gift_ready": ("green", "pulse"),
        }

        if mood in mood_map:
            color, pattern = mood_map[mood]
            self.set_led_pattern(pattern, color)
        else:
            self.set_led_color("white")


# ============================================================================
# Fleet Manager - Coordinate Multiple Arduinos
# ============================================================================

@dataclass
class FleetArduinoManager:
    """
    Manages multiple Arduino bridges.

    Coordinates Guardian (safety) and Somatic (sensors) nodes.
    """
    guardian: Optional[GuardianBridge] = None
    somatic: Optional[SomaticBridge] = None

    # Event log
    _events: deque = field(default_factory=lambda: deque(maxlen=1000))

    def connect_guardian(self, port: str) -> bool:
        """Connect to Guardian Arduino."""
        self.guardian = GuardianBridge(port=port)
        if self.guardian.connect():
            self.guardian.on_event(self._handle_guardian_event)
            self.guardian.start_heartbeat()
            return True
        return False

    def connect_somatic(self, port: str) -> bool:
        """Connect to Somatic Arduino."""
        self.somatic = SomaticBridge(port=port)
        if self.somatic.connect():
            self.somatic.on_event(self._handle_somatic_event)
            return True
        return False

    def disconnect_all(self):
        """Disconnect all Arduinos."""
        if self.guardian:
            self.guardian.stop_heartbeat()
            self.guardian.disconnect()
        if self.somatic:
            self.somatic.disconnect()

    def _handle_guardian_event(self, event: str, data: Dict):
        """Handle event from Guardian."""
        self._events.append({
            "source": "guardian",
            "event": event,
            "data": data,
            "timestamp": time.time(),
        })

        # React to critical events
        if event == "estop_pressed":
            print("[GUARDIAN] E-STOP PRESSED!")
            if self.somatic:
                self.somatic.set_mood("error")

        elif event == "watchdog_timeout":
            print("[GUARDIAN] Watchdog timeout - taking action!")
            if self.somatic:
                self.somatic.set_mood("alert")

        elif event == "temp_critical":
            print(f"[GUARDIAN] Critical temperature on {data.get('sensor')}")

    def _handle_somatic_event(self, event: str, data: Dict):
        """Handle event from Somatic."""
        self._events.append({
            "source": "somatic",
            "event": event,
            "data": data,
            "timestamp": time.time(),
        })

        # React to button presses
        if event == "button":
            button = data.get("detail", "")
            print(f"[SOMATIC] Button pressed: {button}")

            if button == "focus":
                # User wants focus mode
                if self.somatic:
                    self.somatic.set_mood("focused")

            elif button == "talk":
                # User wants to interact
                if self.somatic:
                    self.somatic.set_mood("thinking")

            elif button == "mood":
                # User indicating something's wrong
                if self.somatic:
                    self.somatic.set_mood("alert")

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary from all Arduinos."""
        summary = {
            "guardian_connected": self.guardian.is_connected() if self.guardian else False,
            "somatic_connected": self.somatic.is_connected() if self.somatic else False,
        }

        if self.somatic:
            summary["sensors"] = self.somatic.get_latest_sensors()

        return summary

    def get_recent_events(self, count: int = 50) -> List[Dict]:
        """Get recent events from all Arduinos."""
        return list(self._events)[-count:]


def create_fleet_arduino_manager() -> FleetArduinoManager:
    """Create a FleetArduinoManager instance."""
    return FleetArduinoManager()
