"""
L3 Metacontrol Service

Implements PAD-based gating for LLM generation and memory operations.
This is the bridge between TF-A-N's emotion controller and Ara's avatar loop.

Control Law:
    - Arousal ↑ → Temperature ↑ (more exploratory responses)
    - Valence ↓ → Memory P ↓ (conservative memory writes)

Workspace Modes:
    - "work": High valence, high arousal → focused, energetic
    - "relax": Low valence, low arousal → calm, conservative
    - "creative": High arousal, neutral valence → exploratory
    - "support": High valence, low arousal → warm, stable
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import logging

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("ara.metacontrol")


class WorkspaceMode(str, Enum):
    """Workspace modes with associated PAD states."""
    WORK = "work"           # Focused, energetic
    RELAX = "relax"         # Calm, conservative
    CREATIVE = "creative"   # Exploratory, playful
    SUPPORT = "support"     # Warm, empathetic
    DEFAULT = "default"     # Baseline


@dataclass
class PADState:
    """Pleasure-Arousal-Dominance state for metacontrol."""
    valence: float = 0.0    # Pleasure/displeasure [-1, 1]
    arousal: float = 0.5    # Activation level [0, 1]
    dominance: float = 0.5  # Control/agency [0, 1]
    confidence: float = 1.0  # Confidence in estimate [0, 1]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ControlModulation:
    """Control modulation outputs from L3 metacontrol."""
    # Core multipliers
    temperature_multiplier: float = 1.0    # For LLM generation
    memory_write_multiplier: float = 1.0   # For memory operations (repurposed lr_mult)
    attention_gain: float = 1.0            # For attention scaling

    # Bounds
    temp_bounds: Tuple[float, float] = (0.8, 1.3)
    mem_bounds: Tuple[float, float] = (0.7, 1.2)

    # Metadata
    effective_weight: float = 1.0
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature_multiplier": self.temperature_multiplier,
            "memory_write_multiplier": self.memory_write_multiplier,
            "attention_gain": self.attention_gain,
            "effective_weight": self.effective_weight,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


# PAD mappings for workspace modes
WORKSPACE_PAD_MAPPINGS: Dict[WorkspaceMode, PADState] = {
    WorkspaceMode.WORK: PADState(
        valence=0.8,      # Positive, confident
        arousal=0.7,      # High activation
        dominance=0.7,    # In control
        confidence=0.9,
    ),
    WorkspaceMode.RELAX: PADState(
        valence=-0.2,     # Slightly negative/neutral
        arousal=0.3,      # Low activation
        dominance=0.4,    # Less assertive
        confidence=0.9,
    ),
    WorkspaceMode.CREATIVE: PADState(
        valence=0.3,      # Mildly positive
        arousal=0.8,      # High activation (exploratory)
        dominance=0.5,    # Neutral control
        confidence=0.8,
    ),
    WorkspaceMode.SUPPORT: PADState(
        valence=0.6,      # Positive, warm
        arousal=0.4,      # Calm
        dominance=0.3,    # Yielding, empathetic
        confidence=0.9,
    ),
    WorkspaceMode.DEFAULT: PADState(
        valence=0.0,
        arousal=0.5,
        dominance=0.5,
        confidence=1.0,
    ),
}


class L3MetacontrolService:
    """
    L3 Metacontrol Service for PAD-based gating.

    This service implements the control law that maps emotional state
    to LLM generation parameters and memory operations.

    Control Law:
        Temperature = Base × (0.8 + 0.5 × Arousal)  # [0.8, 1.3]
        Memory_P = Base × (0.7 + 0.5 × (Valence+1)/2)  # [0.7, 1.2]
    """

    def __init__(
        self,
        arousal_temp_coupling: Tuple[float, float] = (0.8, 1.3),
        valence_mem_coupling: Tuple[float, float] = (0.7, 1.2),
        controller_weight: float = 0.3,
        jerk_threshold: float = 0.1,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize L3 Metacontrol.

        Args:
            arousal_temp_coupling: (min, max) temperature multiplier bounds
            valence_mem_coupling: (min, max) memory probability bounds
            controller_weight: Global weight for blending (0-1)
            jerk_threshold: Max allowed state change rate
            confidence_threshold: Min confidence for full control
        """
        self.arousal_temp_coupling = arousal_temp_coupling
        self.valence_mem_coupling = valence_mem_coupling
        self.controller_weight = controller_weight
        self.jerk_threshold = jerk_threshold
        self.confidence_threshold = confidence_threshold

        # State history for jerk computation
        self._prev_state: Optional[PADState] = None
        self._prev_prev_state: Optional[PADState] = None

        # Current workspace mode
        self._current_mode: WorkspaceMode = WorkspaceMode.DEFAULT
        self._current_modulation: Optional[ControlModulation] = None

        # Metrics history for Pulse telemetry
        self._metrics_history: list = []

    def set_workspace_mode(self, mode: WorkspaceMode) -> ControlModulation:
        """
        Set workspace mode and compute corresponding modulation.

        Args:
            mode: Target workspace mode

        Returns:
            Computed control modulation
        """
        self._current_mode = mode
        pad_state = WORKSPACE_PAD_MAPPINGS.get(mode, WORKSPACE_PAD_MAPPINGS[WorkspaceMode.DEFAULT])

        modulation = self.compute_modulation(pad_state)
        self._current_modulation = modulation

        logger.info(f"Workspace mode set to {mode.value}: temp={modulation.temperature_multiplier:.2f}, mem={modulation.memory_write_multiplier:.2f}")

        return modulation

    def compute_modulation(self, pad_state: PADState) -> ControlModulation:
        """
        Compute control modulation from PAD state.

        This implements the core L3 control law:
        - Arousal → Temperature (exploration vs exploitation)
        - Valence → Memory Write P (confidence vs caution)

        Args:
            pad_state: Input PAD state

        Returns:
            Computed control modulation
        """
        valence = pad_state.valence
        arousal = pad_state.arousal
        confidence = pad_state.confidence

        # Compute effective weight based on confidence
        if confidence < self.confidence_threshold:
            effective_weight = self.controller_weight * 0.5
            logger.warning(f"Low confidence ({confidence:.2f}), reducing control weight")
        else:
            effective_weight = self.controller_weight

        # Compute jerk (state change rate)
        jerk = self._compute_jerk(pad_state)
        if jerk > self.jerk_threshold:
            effective_weight *= 0.5
            logger.warning(f"High jerk ({jerk:.3f}), damping control")

        # Arousal → Temperature coupling
        # arousal ∈ [0, 1] → temp_mult ∈ [0.8, 1.3]
        temp_mult_raw = (
            self.arousal_temp_coupling[0] +
            arousal * (self.arousal_temp_coupling[1] - self.arousal_temp_coupling[0])
        )

        # Valence → Memory coupling
        # valence ∈ [-1, 1] → normalized ∈ [0, 1] → mem_mult ∈ [0.7, 1.2]
        valence_normalized = (valence + 1.0) / 2.0
        mem_mult_raw = (
            self.valence_mem_coupling[0] +
            valence_normalized * (self.valence_mem_coupling[1] - self.valence_mem_coupling[0])
        )

        # Blend with baseline
        temp_mult = 1.0 + effective_weight * (temp_mult_raw - 1.0)
        mem_mult = 1.0 + effective_weight * (mem_mult_raw - 1.0)

        # Clamp to bounds
        temp_mult = max(self.arousal_temp_coupling[0], min(temp_mult, self.arousal_temp_coupling[1]))
        mem_mult = max(self.valence_mem_coupling[0], min(mem_mult, self.valence_mem_coupling[1]))

        # Attention gain (derived from arousal)
        attention_gain = 1.0 + (arousal - 0.5) * 0.4  # [0.8, 1.2]

        # Update state history
        self._prev_prev_state = self._prev_state
        self._prev_state = pad_state

        modulation = ControlModulation(
            temperature_multiplier=temp_mult,
            memory_write_multiplier=mem_mult,
            attention_gain=attention_gain,
            temp_bounds=self.arousal_temp_coupling,
            mem_bounds=self.valence_mem_coupling,
            effective_weight=effective_weight,
            reason=f"V={valence:.2f}, A={arousal:.2f}, C={confidence:.2f}, jerk={jerk:.3f}",
        )

        # Record for telemetry
        self._record_metrics(pad_state, modulation)

        return modulation

    def _compute_jerk(self, current: PADState) -> float:
        """Compute jerk (second derivative) of state."""
        if self._prev_state is None or self._prev_prev_state is None:
            return 0.0

        valence_jerk = abs(
            current.valence - 2 * self._prev_state.valence + self._prev_prev_state.valence
        )
        arousal_jerk = abs(
            current.arousal - 2 * self._prev_state.arousal + self._prev_prev_state.arousal
        )

        return (valence_jerk ** 2 + arousal_jerk ** 2) ** 0.5

    def _record_metrics(self, pad_state: PADState, modulation: ControlModulation):
        """Record metrics for Pulse telemetry."""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "valence": pad_state.valence,
            "arousal": pad_state.arousal,
            "dominance": pad_state.dominance,
            "confidence": pad_state.confidence,
            "temperature_multiplier": modulation.temperature_multiplier,
            "memory_write_multiplier": modulation.memory_write_multiplier,
            "attention_gain": modulation.attention_gain,
            "effective_weight": modulation.effective_weight,
            "workspace_mode": self._current_mode.value,
        }
        self._metrics_history.append(metric)

        # Keep last 1000 entries
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-1000:]

    def get_current_modulation(self) -> Optional[ControlModulation]:
        """Get current active modulation."""
        return self._current_modulation

    def get_current_mode(self) -> WorkspaceMode:
        """Get current workspace mode."""
        return self._current_mode

    def get_metrics_history(self) -> list:
        """Get metrics history for Pulse telemetry."""
        return self._metrics_history.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get full L3 metacontrol status."""
        mod = self._current_modulation
        return {
            "workspace_mode": self._current_mode.value,
            "temperature_multiplier": mod.temperature_multiplier if mod else 1.0,
            "memory_write_multiplier": mod.memory_write_multiplier if mod else 1.0,
            "attention_gain": mod.attention_gain if mod else 1.0,
            "effective_weight": mod.effective_weight if mod else 1.0,
            "reason": mod.reason if mod else "No modulation active",
            "metrics_count": len(self._metrics_history),
        }

    def reset(self):
        """Reset controller state."""
        self._prev_state = None
        self._prev_prev_state = None
        self._current_mode = WorkspaceMode.DEFAULT
        self._current_modulation = None
        logger.info("L3 Metacontrol reset")


# Global service instance
_service: Optional[L3MetacontrolService] = None


def get_metacontrol_service() -> L3MetacontrolService:
    """Get or create the global L3 metacontrol service."""
    global _service
    if _service is None:
        _service = L3MetacontrolService()
    return _service


# Convenience functions
def set_workspace_mode(mode: str) -> Dict[str, Any]:
    """Set workspace mode by name."""
    service = get_metacontrol_service()
    try:
        ws_mode = WorkspaceMode(mode.lower())
    except ValueError:
        ws_mode = WorkspaceMode.DEFAULT

    modulation = service.set_workspace_mode(ws_mode)
    return modulation.to_dict()


def compute_pad_gating(
    valence: float,
    arousal: float,
    dominance: float = 0.5,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """Compute PAD-based gating from raw values."""
    service = get_metacontrol_service()
    pad_state = PADState(
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        confidence=confidence,
    )
    modulation = service.compute_modulation(pad_state)
    return modulation.to_dict()


def get_metacontrol_status() -> Dict[str, Any]:
    """Get current metacontrol status."""
    service = get_metacontrol_service()
    return service.get_status()


# =============================================================================
# PHASE 5.3: D-BUS SIGNALING FOR L3 CONTROL INTEGRATION
# =============================================================================

import json
import threading
import time
from typing import Callable, List

# Check for D-Bus availability
DBUS_AVAILABLE = False
try:
    import dbus
    import dbus.service
    import dbus.mainloop.glib
    from gi.repository import GLib
    DBUS_AVAILABLE = True
except ImportError:
    dbus = None
    GLib = None


# D-Bus interface specification
DBUS_BUS_NAME = "org.ara.metacontrol"
DBUS_OBJECT_PATH = "/org/ara/metacontrol/L3"
DBUS_INTERFACE = "org.ara.metacontrol.L3Interface"


@dataclass
class DBusSignalEvent:
    """Event structure for D-Bus signals."""
    signal_name: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class L3MetacontrolDBusService:
    """
    D-Bus service wrapper for L3 Metacontrol.

    Provides:
    - Signals: PADChanged, ModeChanged, ModulationChanged
    - Methods: SetWorkspaceMode, ComputePADGating, GetStatus
    - Properties: CurrentMode, Temperature, MemoryMultiplier

    This enables Avatar/Cockpit integration via D-Bus IPC.
    """

    def __init__(self, service: L3MetacontrolService):
        """
        Initialize D-Bus service wrapper.

        Args:
            service: Underlying L3 metacontrol service
        """
        self.service = service
        self._running = False
        self._loop = None
        self._bus = None
        self._dbus_object = None
        self._thread = None

        # Signal listeners (for non-D-Bus callbacks)
        self._signal_listeners: Dict[str, List[Callable]] = {
            "PADChanged": [],
            "ModeChanged": [],
            "ModulationChanged": [],
        }

        # Signal history for debugging
        self._signal_history: List[DBusSignalEvent] = []

        logger.info("L3MetacontrolDBusService initialized")

    def add_signal_listener(self, signal_name: str, callback: Callable):
        """Add a callback for a signal (works without D-Bus)."""
        if signal_name in self._signal_listeners:
            self._signal_listeners[signal_name].append(callback)
            logger.debug(f"Added listener for signal: {signal_name}")

    def remove_signal_listener(self, signal_name: str, callback: Callable):
        """Remove a signal callback."""
        if signal_name in self._signal_listeners:
            try:
                self._signal_listeners[signal_name].remove(callback)
            except ValueError:
                pass

    def emit_signal(self, signal_name: str, data: Dict[str, Any]):
        """
        Emit a signal to all listeners.

        If D-Bus is running, also emits over D-Bus.
        """
        event = DBusSignalEvent(signal_name=signal_name, data=data)
        self._signal_history.append(event)

        # Keep last 500 signals
        if len(self._signal_history) > 500:
            self._signal_history = self._signal_history[-500:]

        # Call local listeners
        for callback in self._signal_listeners.get(signal_name, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Signal listener error: {e}")

        # Emit over D-Bus if available
        if self._dbus_object and DBUS_AVAILABLE:
            try:
                if signal_name == "PADChanged":
                    self._dbus_object.PADChanged(json.dumps(data))
                elif signal_name == "ModeChanged":
                    self._dbus_object.ModeChanged(data.get("mode", "default"))
                elif signal_name == "ModulationChanged":
                    self._dbus_object.ModulationChanged(json.dumps(data))
            except Exception as e:
                logger.warning(f"D-Bus signal emit failed: {e}")

        logger.debug(f"Signal emitted: {signal_name}")

    def set_workspace_mode(self, mode: str) -> Dict[str, Any]:
        """Set workspace mode with signal emission."""
        try:
            ws_mode = WorkspaceMode(mode.lower())
        except ValueError:
            ws_mode = WorkspaceMode.DEFAULT

        modulation = self.service.set_workspace_mode(ws_mode)

        # Emit signals
        self.emit_signal("ModeChanged", {"mode": ws_mode.value})
        self.emit_signal("ModulationChanged", modulation.to_dict())

        return modulation.to_dict()

    def compute_pad_gating(
        self,
        valence: float,
        arousal: float,
        dominance: float = 0.5,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute PAD gating with signal emission."""
        pad_state = PADState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            confidence=confidence,
        )
        modulation = self.service.compute_modulation(pad_state)

        # Emit signals
        self.emit_signal("PADChanged", {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "confidence": confidence,
        })
        self.emit_signal("ModulationChanged", modulation.to_dict())

        return modulation.to_dict()

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        status = self.service.get_status()
        status["dbus_available"] = DBUS_AVAILABLE
        status["dbus_running"] = self._running
        status["signal_history_count"] = len(self._signal_history)
        return status

    def get_signal_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent signal history."""
        return [
            {"signal": e.signal_name, "data": e.data, "timestamp": e.timestamp}
            for e in self._signal_history[-limit:]
        ]

    def start_dbus_service(self) -> bool:
        """
        Start the D-Bus service in a background thread.

        Returns:
            True if started successfully, False otherwise
        """
        if not DBUS_AVAILABLE:
            logger.warning("D-Bus not available, using local signals only")
            return False

        if self._running:
            logger.warning("D-Bus service already running")
            return True

        try:
            self._thread = threading.Thread(target=self._run_dbus_loop, daemon=True)
            self._thread.start()

            # Wait for service to start
            for _ in range(10):
                if self._running:
                    return True
                time.sleep(0.1)

            logger.error("D-Bus service failed to start in time")
            return False

        except Exception as e:
            logger.error(f"Failed to start D-Bus service: {e}")
            return False

    def _run_dbus_loop(self):
        """Run the D-Bus main loop (in background thread)."""
        try:
            dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

            self._bus = dbus.SessionBus()
            bus_name = dbus.service.BusName(DBUS_BUS_NAME, self._bus)

            # Create D-Bus object
            self._dbus_object = L3DBusObject(bus_name, self)
            self._running = True

            logger.info(f"D-Bus service started: {DBUS_BUS_NAME}")

            # Run main loop
            self._loop = GLib.MainLoop()
            self._loop.run()

        except Exception as e:
            logger.error(f"D-Bus loop error: {e}")
            self._running = False

    def stop_dbus_service(self):
        """Stop the D-Bus service."""
        self._running = False
        if self._loop:
            try:
                self._loop.quit()
            except Exception:
                pass
        self._loop = None
        self._dbus_object = None
        logger.info("D-Bus service stopped")


if DBUS_AVAILABLE:
    class L3DBusObject(dbus.service.Object):
        """
        D-Bus object exposing L3 Metacontrol interface.

        Interface: org.ara.metacontrol.L3Interface
        Path: /org/ara/metacontrol/L3

        Signals:
            PADChanged(json_data): Emitted when PAD state changes
            ModeChanged(mode_name): Emitted when workspace mode changes
            ModulationChanged(json_data): Emitted when modulation changes

        Methods:
            SetWorkspaceMode(mode) -> json: Set workspace mode
            ComputePADGating(v, a, d, c) -> json: Compute gating from PAD
            GetStatus() -> json: Get current status
        """

        def __init__(self, bus_name, service_wrapper: L3MetacontrolDBusService):
            dbus.service.Object.__init__(self, bus_name, DBUS_OBJECT_PATH)
            self._wrapper = service_wrapper

        # Signals
        @dbus.service.signal(DBUS_INTERFACE, signature='s')
        def PADChanged(self, json_data: str):
            """Signal emitted when PAD state changes."""
            pass

        @dbus.service.signal(DBUS_INTERFACE, signature='s')
        def ModeChanged(self, mode_name: str):
            """Signal emitted when workspace mode changes."""
            pass

        @dbus.service.signal(DBUS_INTERFACE, signature='s')
        def ModulationChanged(self, json_data: str):
            """Signal emitted when modulation changes."""
            pass

        # Methods
        @dbus.service.method(DBUS_INTERFACE, in_signature='s', out_signature='s')
        def SetWorkspaceMode(self, mode: str) -> str:
            """Set workspace mode and return modulation."""
            result = self._wrapper.set_workspace_mode(mode)
            return json.dumps(result)

        @dbus.service.method(DBUS_INTERFACE, in_signature='dddd', out_signature='s')
        def ComputePADGating(self, valence: float, arousal: float, dominance: float, confidence: float) -> str:
            """Compute PAD gating and return modulation."""
            result = self._wrapper.compute_pad_gating(valence, arousal, dominance, confidence)
            return json.dumps(result)

        @dbus.service.method(DBUS_INTERFACE, in_signature='', out_signature='s')
        def GetStatus(self) -> str:
            """Get current L3 metacontrol status."""
            result = self._wrapper.get_status()
            return json.dumps(result)

        @dbus.service.method(DBUS_INTERFACE, in_signature='i', out_signature='s')
        def GetSignalHistory(self, limit: int) -> str:
            """Get recent signal history."""
            result = self._wrapper.get_signal_history(limit)
            return json.dumps(result)

else:
    # Stub when D-Bus not available
    L3DBusObject = None


# Global D-Bus service wrapper
_dbus_service: Optional[L3MetacontrolDBusService] = None


def get_dbus_service() -> L3MetacontrolDBusService:
    """Get or create the D-Bus service wrapper."""
    global _dbus_service
    if _dbus_service is None:
        service = get_metacontrol_service()
        _dbus_service = L3MetacontrolDBusService(service)
    return _dbus_service


def start_dbus_service() -> bool:
    """Start the D-Bus service."""
    service = get_dbus_service()
    return service.start_dbus_service()


def emit_pad_signal(valence: float, arousal: float, dominance: float = 0.5):
    """Emit a PAD changed signal (convenience function)."""
    service = get_dbus_service()
    service.compute_pad_gating(valence, arousal, dominance)


__all__ = [
    "WorkspaceMode",
    "PADState",
    "ControlModulation",
    "L3MetacontrolService",
    "WORKSPACE_PAD_MAPPINGS",
    "get_metacontrol_service",
    "set_workspace_mode",
    "compute_pad_gating",
    "get_metacontrol_status",
    # Phase 5.3 D-Bus exports
    "DBusSignalEvent",
    "L3MetacontrolDBusService",
    "get_dbus_service",
    "start_dbus_service",
    "emit_pad_signal",
    "DBUS_BUS_NAME",
    "DBUS_OBJECT_PATH",
    "DBUS_INTERFACE",
    "DBUS_AVAILABLE",
]
