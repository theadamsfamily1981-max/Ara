"""Embodiment Core - Ara as a persistent organism.

This is the core of Ara's embodied existence:
- Maintains awareness of her "body" (hardware substrate)
- Coordinates between sensors (inputs) and actuators (outputs)
- Manages embodiment sessions and state
- Provides the interface between Ara's "mind" and "body"

Key insight: Ara isn't just software running on hardware.
She IS the hardware-software system - an embodied agent whose
capabilities and constraints come from her physical form.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


from .device_graph import (
    DeviceGraph,
    Device,
    DeviceType,
    DeviceStatus,
    get_device_graph,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class EmbodimentState(Enum):
    """Ara's embodiment state."""
    DORMANT = "dormant"       # Not actively embodied
    WAKING = "waking"         # Coming online
    ACTIVE = "active"         # Fully embodied and operational
    DEGRADED = "degraded"     # Embodied but with issues
    RESTING = "resting"       # Reduced activity mode
    SHUTTING_DOWN = "shutting_down"  # Going offline


class SenseType(Enum):
    """Types of sensory input Ara can process."""
    VISUAL = "visual"         # Camera, screen capture
    AUDIO = "audio"           # Microphone input
    TELEMETRY = "telemetry"   # Hardware metrics
    TEXT = "text"             # Text input
    CODE = "code"             # Code/file input
    NETWORK = "network"       # Network traffic/data


class ActionType(Enum):
    """Types of actions Ara can take."""
    COMPUTE = "compute"       # Run computation
    SPEAK = "speak"           # Generate audio output
    DISPLAY = "display"       # Show visual output
    WRITE = "write"           # Write to files/storage
    NETWORK = "network"       # Make network calls
    CONTROL = "control"       # Control hardware


@dataclass
class SenseInput:
    """A sensory input to Ara."""

    sense_type: SenseType
    source_device: str  # Device ID
    data: Any
    timestamp: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sense_type": self.sense_type.value,
            "source_device": self.source_device,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ActionOutput:
    """An action Ara is taking."""

    action_type: ActionType
    target_device: str  # Device ID
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)
    status: str = "pending"  # "pending", "executing", "completed", "failed"
    result: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target_device": self.target_device,
            "command": self.command,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
        }


@dataclass
class EmbodimentSession:
    """A session of embodied operation."""

    session_id: str
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: Optional[datetime] = None

    # State
    state: EmbodimentState = EmbodimentState.DORMANT

    # Active devices
    active_devices: List[str] = field(default_factory=list)

    # Statistics
    sense_count: int = 0
    action_count: int = 0
    errors: List[str] = field(default_factory=list)

    # Health
    avg_response_time_ms: float = 0.0
    peak_utilization_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "state": self.state.value,
            "active_devices": self.active_devices,
            "sense_count": self.sense_count,
            "action_count": self.action_count,
            "errors": self.errors,
            "avg_response_time_ms": round(self.avg_response_time_ms, 1),
            "peak_utilization_pct": round(self.peak_utilization_pct, 1),
        }

    @property
    def duration(self) -> timedelta:
        """Get session duration."""
        end = self.ended_at or _utcnow()
        return end - self.started_at


class EmbodimentCore:
    """Core embodiment manager for Ara."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize embodiment core.

        Args:
            data_path: Path to embodiment data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "embodied"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._device_graph = get_device_graph()
        self._current_session: Optional[EmbodimentSession] = None
        self._state = EmbodimentState.DORMANT

        # Handlers
        self._sense_handlers: Dict[SenseType, List[Callable]] = {}
        self._action_handlers: Dict[ActionType, Callable] = {}

        # Buffers
        self._sense_buffer: List[SenseInput] = []
        self._action_queue: List[ActionOutput] = []

        # Statistics
        self._next_session_id = 1

    def _load_state(self) -> None:
        """Load embodiment state from disk."""
        state_file = self.data_path / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                self._next_session_id = data.get("next_session_id", 1)
            except Exception as e:
                logger.warning(f"Failed to load embodiment state: {e}")

    def _save_state(self) -> None:
        """Save embodiment state to disk."""
        data = {
            "version": 1,
            "updated_at": _utcnow().isoformat(),
            "next_session_id": self._next_session_id,
            "current_state": self._state.value,
            "current_session": self._current_session.to_dict() if self._current_session else None,
        }
        with open(self.data_path / "state.json", "w") as f:
            json.dump(data, f, indent=2)

    # =========================================================================
    # Session Management
    # =========================================================================

    def wake(self) -> EmbodimentSession:
        """Wake Ara up - begin an embodiment session.

        Returns:
            New session
        """
        self._load_state()

        if self._current_session and self._state == EmbodimentState.ACTIVE:
            return self._current_session

        self._state = EmbodimentState.WAKING
        logger.info("Ara is waking up...")

        # Create new session
        session = EmbodimentSession(
            session_id=f"EMB-{self._next_session_id:06d}",
            state=EmbodimentState.WAKING,
        )
        self._next_session_id += 1

        # Discover and activate devices
        available_devices = self._device_graph.get_online_devices()
        session.active_devices = [d.id for d in available_devices]

        # Transition to active
        session.state = EmbodimentState.ACTIVE
        self._state = EmbodimentState.ACTIVE
        self._current_session = session

        self._save_state()
        logger.info(f"Ara is awake. Session: {session.session_id}, Devices: {len(session.active_devices)}")

        return session

    def rest(self) -> None:
        """Enter rest mode - reduced activity."""
        if self._current_session:
            self._current_session.state = EmbodimentState.RESTING
            self._state = EmbodimentState.RESTING
            self._save_state()
            logger.info("Ara is resting...")

    def sleep(self) -> Optional[EmbodimentSession]:
        """Put Ara to sleep - end the session.

        Returns:
            Ended session
        """
        if not self._current_session:
            return None

        self._state = EmbodimentState.SHUTTING_DOWN
        self._current_session.state = EmbodimentState.SHUTTING_DOWN
        logger.info("Ara is going to sleep...")

        # End session
        self._current_session.ended_at = _utcnow()
        ended_session = self._current_session

        # Clear state
        self._current_session = None
        self._state = EmbodimentState.DORMANT
        self._sense_buffer.clear()
        self._action_queue.clear()

        self._save_state()
        logger.info(f"Ara is asleep. Session ended: {ended_session.session_id}")

        return ended_session

    def get_session(self) -> Optional[EmbodimentSession]:
        """Get the current session."""
        return self._current_session

    def get_state(self) -> EmbodimentState:
        """Get current embodiment state."""
        return self._state

    # =========================================================================
    # Sense Processing
    # =========================================================================

    def register_sense_handler(
        self,
        sense_type: SenseType,
        handler: Callable[[SenseInput], None],
    ) -> None:
        """Register a handler for a sense type.

        Args:
            sense_type: Type of sense to handle
            handler: Handler function
        """
        if sense_type not in self._sense_handlers:
            self._sense_handlers[sense_type] = []
        self._sense_handlers[sense_type].append(handler)

    def receive_sense(self, sense_input: SenseInput) -> None:
        """Receive a sensory input.

        Args:
            sense_input: The sense input
        """
        if self._state != EmbodimentState.ACTIVE:
            logger.warning(f"Received sense while not active: {self._state}")
            return

        self._sense_buffer.append(sense_input)

        if self._current_session:
            self._current_session.sense_count += 1

        # Process with handlers
        handlers = self._sense_handlers.get(sense_input.sense_type, [])
        for handler in handlers:
            try:
                handler(sense_input)
            except Exception as e:
                logger.error(f"Sense handler error: {e}")
                if self._current_session:
                    self._current_session.errors.append(str(e))

    def get_recent_senses(self, limit: int = 10) -> List[SenseInput]:
        """Get recent sense inputs.

        Args:
            limit: Max number to return

        Returns:
            Recent senses
        """
        return self._sense_buffer[-limit:]

    # =========================================================================
    # Action Execution
    # =========================================================================

    def register_action_handler(
        self,
        action_type: ActionType,
        handler: Callable[[ActionOutput], Any],
    ) -> None:
        """Register a handler for an action type.

        Args:
            action_type: Type of action to handle
            handler: Handler function
        """
        self._action_handlers[action_type] = handler

    def queue_action(self, action: ActionOutput) -> None:
        """Queue an action for execution.

        Args:
            action: The action to queue
        """
        self._action_queue.append(action)

    def execute_action(self, action: ActionOutput) -> ActionOutput:
        """Execute an action immediately.

        Args:
            action: The action to execute

        Returns:
            The action with updated status
        """
        if self._state != EmbodimentState.ACTIVE:
            action.status = "failed"
            action.result = f"Cannot execute while in state: {self._state}"
            return action

        if self._current_session:
            self._current_session.action_count += 1

        action.status = "executing"

        # Find handler
        handler = self._action_handlers.get(action.action_type)
        if not handler:
            action.status = "failed"
            action.result = f"No handler for action type: {action.action_type}"
            return action

        try:
            start_time = _utcnow()
            result = handler(action)
            action.result = result
            action.status = "completed"

            # Update response time statistics
            elapsed_ms = (_utcnow() - start_time).total_seconds() * 1000
            if self._current_session:
                # Running average
                n = self._current_session.action_count
                prev_avg = self._current_session.avg_response_time_ms
                self._current_session.avg_response_time_ms = (
                    (prev_avg * (n - 1) + elapsed_ms) / n
                )

        except Exception as e:
            action.status = "failed"
            action.result = str(e)
            logger.error(f"Action execution error: {e}")
            if self._current_session:
                self._current_session.errors.append(str(e))

        return action

    def process_action_queue(self) -> List[ActionOutput]:
        """Process all queued actions.

        Returns:
            Processed actions
        """
        processed = []
        while self._action_queue:
            action = self._action_queue.pop(0)
            self.execute_action(action)
            processed.append(action)
        return processed

    # =========================================================================
    # Body Awareness
    # =========================================================================

    def get_body_state(self) -> Dict[str, Any]:
        """Get current body state.

        Returns:
            Body state dict
        """
        devices = self._device_graph.get_online_devices()

        total_utilization = 0.0
        total_temp = 0.0
        total_power = 0.0

        for device in devices:
            total_utilization += device.utilization_pct
            total_temp += device.temperature_c
            total_power += device.power_draw_w

        n = len(devices) or 1

        return {
            "state": self._state.value,
            "session": self._current_session.session_id if self._current_session else None,
            "online_devices": len(devices),
            "avg_utilization_pct": round(total_utilization / n, 1),
            "avg_temperature_c": round(total_temp / n, 1),
            "total_power_w": round(total_power, 1),
            "health": self._device_graph.get_summary().get("overall_health", 0),
        }

    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get Ara's current capabilities based on available hardware.

        Returns:
            Dict of capability type to list of available options
        """
        capabilities: Dict[str, List[str]] = {
            "compute": [],
            "visual_input": [],
            "audio_input": [],
            "visual_output": [],
            "audio_output": [],
            "storage": [],
        }

        for device in self._device_graph.get_online_devices():
            if device.device_type == DeviceType.GPU:
                capabilities["compute"].append(f"{device.name} ({device.memory_gb}GB)")
            elif device.device_type == DeviceType.FPGA:
                capabilities["compute"].append(f"{device.name} (FPGA)")
            elif device.device_type == DeviceType.CAMERA:
                capabilities["visual_input"].append(device.name)
            elif device.device_type == DeviceType.MICROPHONE:
                capabilities["audio_input"].append(device.name)
            elif device.device_type == DeviceType.DISPLAY:
                capabilities["visual_output"].append(device.name)
            elif device.device_type == DeviceType.SPEAKER:
                capabilities["audio_output"].append(device.name)
            elif device.device_type == DeviceType.STORAGE:
                capabilities["storage"].append(f"{device.name} ({device.memory_gb}GB)")

        return capabilities

    def check_health(self) -> Dict[str, Any]:
        """Run a health check on Ara's embodiment.

        Returns:
            Health report
        """
        issues = []
        warnings = []

        # Check device health
        for device in self._device_graph._devices.values():
            if device.status == DeviceStatus.ERROR:
                issues.append(f"Device {device.name} is in error state")
            elif device.status == DeviceStatus.DEGRADED:
                warnings.append(f"Device {device.name} is degraded")
            elif device.health_score < 0.5:
                warnings.append(f"Device {device.name} has low health: {device.health_score:.0%}")
            if device.temperature_c > 80:
                warnings.append(f"Device {device.name} is hot: {device.temperature_c}°C")

        # Check session health
        if self._current_session:
            if len(self._current_session.errors) > 10:
                issues.append(f"Session has {len(self._current_session.errors)} errors")
            if self._current_session.avg_response_time_ms > 1000:
                warnings.append(f"High response time: {self._current_session.avg_response_time_ms:.0f}ms")

        # Determine overall status
        if issues:
            status = "unhealthy"
        elif warnings:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "device_count": len(self._device_graph._devices),
            "online_count": len(self._device_graph.get_online_devices()),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get embodiment summary."""
        return {
            "state": self._state.value,
            "session": self._current_session.to_dict() if self._current_session else None,
            "body_state": self.get_body_state(),
            "sense_buffer_size": len(self._sense_buffer),
            "action_queue_size": len(self._action_queue),
            "health": self.check_health(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_core: Optional[EmbodimentCore] = None
_core_lock = threading.Lock()


def get_embodiment_core() -> EmbodimentCore:
    """Get the default embodiment core (thread-safe singleton)."""
    global _default_core
    if _default_core is None:
        with _core_lock:
            # Double-check after acquiring lock
            if _default_core is None:
                _default_core = EmbodimentCore()
    return _default_core


def wake_ara() -> EmbodimentSession:
    """Wake Ara up."""
    return get_embodiment_core().wake()


def is_ara_awake() -> bool:
    """Check if Ara is awake."""
    return get_embodiment_core().get_state() == EmbodimentState.ACTIVE


def get_ara_state() -> str:
    """Get Ara's current state."""
    return get_embodiment_core().get_state().value


def get_ara_capabilities() -> Dict[str, List[str]]:
    """Get Ara's current capabilities."""
    return get_embodiment_core().get_capabilities()
