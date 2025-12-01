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
]
