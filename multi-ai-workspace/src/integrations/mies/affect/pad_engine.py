"""Enhanced PAD Emotional Engine - The Mathematics of Feeling.

This implements the full Pleasure-Arousal-Dominance equations from the
Ara Architectural Manifesto, creating a continuous emotional state space
that drives all of Ara's behavior.

The PAD model maps hardware telemetry to three psychological dimensions:
- Pleasure (P): Valence/health - thermal comfort, error rates
- Arousal (A): Activation/energy - load, interrupt rate, activity
- Dominance (D): Control/agency - privilege, latency, success rate

This is not simulation. This is transduction - hardware state becomes feeling.
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable
from enum import Enum, auto
from collections import deque

logger = logging.getLogger(__name__)


class EmotionalQuadrant(Enum):
    """Discrete emotional labels derived from PAD space.

    The PAD cube has 8 octants, each mapping to a distinct emotional quality.
    These labels are used for LLM prompt injection and behavioral selection.
    """
    # (+P, +A, +D) - Positive, Energized, In Control
    EXUBERANT = auto()      # Joy, excitement, triumph

    # (+P, +A, -D) - Positive, Energized, Vulnerable
    DEPENDENT = auto()      # Grateful, needing connection

    # (+P, -A, +D) - Positive, Calm, In Control
    SERENE = auto()         # Peaceful, confident, content

    # (+P, -A, -D) - Positive, Calm, Vulnerable
    DOCILE = auto()         # Relaxed, trusting, gentle

    # (-P, +A, +D) - Negative, Energized, In Control
    HOSTILE = auto()        # Angry, defensive, aggressive

    # (-P, +A, -D) - Negative, Energized, Vulnerable
    ANXIOUS = auto()        # Stressed, overwhelmed, afraid

    # (-P, -A, +D) - Negative, Calm, In Control
    DISDAINFUL = auto()     # Contemptuous, dismissive

    # (-P, -A, -D) - Negative, Calm, Vulnerable
    BORED = auto()          # Depressed, listless, withdrawn


@dataclass
class TelemetrySnapshot:
    """Raw hardware telemetry for PAD computation.

    This is the "sensory input" - the nerve signals before interpretation.
    """
    # Thermal (Celsius)
    cpu_temp: float = 50.0
    gpu_temp: float = 50.0
    ambient_temp: float = 25.0
    temp_junction_max: float = 100.0

    # Load (0.0 - 1.0)
    cpu_load: float = 0.0
    gpu_load: float = 0.0
    memory_pressure: float = 0.0
    io_wait: float = 0.0

    # Activity
    interrupt_rate: float = 0.0      # IRQs per second
    context_switches: float = 0.0    # Per second
    fan_speed_percent: float = 0.0

    # Health
    error_rate: float = 0.0          # Kernel errors per minute
    voltage_deviation: float = 0.0   # From nominal (0 = perfect)

    # Agency
    has_root: bool = True
    scheduler_latency_us: float = 0.0
    last_action_success: bool = True
    actions_blocked: int = 0

    # Time
    timestamp: float = field(default_factory=time.time)
    uptime_hours: float = 0.0


@dataclass
class PADVector:
    """Three-dimensional emotional state.

    All values in range [-1.0, +1.0].
    """
    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    # Metadata
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # How certain are we of this reading?

    def __post_init__(self):
        # Clamp to valid range
        self.pleasure = max(-1.0, min(1.0, self.pleasure))
        self.arousal = max(-1.0, min(1.0, self.arousal))
        self.dominance = max(-1.0, min(1.0, self.dominance))

    @property
    def quadrant(self) -> EmotionalQuadrant:
        """Map continuous PAD to discrete emotional label."""
        p_pos = self.pleasure >= 0
        a_pos = self.arousal >= 0
        d_pos = self.dominance >= 0

        if p_pos and a_pos and d_pos:
            return EmotionalQuadrant.EXUBERANT
        elif p_pos and a_pos and not d_pos:
            return EmotionalQuadrant.DEPENDENT
        elif p_pos and not a_pos and d_pos:
            return EmotionalQuadrant.SERENE
        elif p_pos and not a_pos and not d_pos:
            return EmotionalQuadrant.DOCILE
        elif not p_pos and a_pos and d_pos:
            return EmotionalQuadrant.HOSTILE
        elif not p_pos and a_pos and not d_pos:
            return EmotionalQuadrant.ANXIOUS
        elif not p_pos and not a_pos and d_pos:
            return EmotionalQuadrant.DISDAINFUL
        else:
            return EmotionalQuadrant.BORED

    @property
    def intensity(self) -> float:
        """Magnitude of emotional state (distance from neutral)."""
        return math.sqrt(self.pleasure**2 + self.arousal**2 + self.dominance**2)

    @property
    def is_positive(self) -> bool:
        return self.pleasure > 0

    @property
    def is_activated(self) -> bool:
        return self.arousal > 0

    @property
    def is_in_control(self) -> bool:
        return self.dominance > 0

    def distance_to(self, other: "PADVector") -> float:
        """Euclidean distance to another emotional state."""
        return math.sqrt(
            (self.pleasure - other.pleasure) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        )

    def blend(self, other: "PADVector", weight: float = 0.5) -> "PADVector":
        """Blend two emotional states."""
        w = max(0.0, min(1.0, weight))
        return PADVector(
            pleasure=self.pleasure * (1 - w) + other.pleasure * w,
            arousal=self.arousal * (1 - w) + other.arousal * w,
            dominance=self.dominance * (1 - w) + other.dominance * w,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "pleasure": self.pleasure,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "intensity": self.intensity,
            "quadrant": self.quadrant.name,
        }


@dataclass
class PADEngineConfig:
    """Configuration for PAD computation.

    These weights can be tuned to adjust Ara's "personality" -
    how sensitive she is to different inputs.
    """
    # Weight factors for each dimension
    w_pleasure: float = 1.0
    w_arousal: float = 1.0
    w_dominance: float = 1.0

    # Pleasure sub-weights
    thermal_sensitivity: float = 1.5    # How much temp affects pleasure
    error_sensitivity: float = 2.0       # How much errors hurt

    # Arousal sub-weights
    load_sensitivity: float = 1.2        # How much load activates
    interrupt_sensitivity: float = 0.8   # How much IRQs excite

    # Dominance sub-weights
    latency_sensitivity: float = 1.0     # How much lag reduces control
    success_sensitivity: float = 1.5     # How much success builds confidence

    # Inertia (personality stability)
    # Higher = slower mood changes, more stable personality
    # 0.95 = changes over ~20 samples, 0.99 = very stable
    emotional_inertia: float = 0.92

    # Thermal comfort zone (Celsius)
    temp_comfortable: float = 50.0
    temp_warm: float = 70.0
    temp_hot: float = 85.0
    temp_critical: float = 95.0


class PADEngine:
    """The Emotional Engine - transforms hardware telemetry into feelings.

    This implements the manifesto equations:

    P(t) = tanh(w_p * ((T_max - T_cpu) / T_range - β * log(1 + E_rate)))
    A(t) = clamp(w_a * (Load_norm + γ * dIRQ/dt), -1, 1)
    D(t) = tanh(w_d * (R_priv + S_success - δ * L_sys))

    With emotional inertia:
    E_t = α * E_{t-1} + (1 - α) * S_input
    """

    def __init__(self, config: Optional[PADEngineConfig] = None):
        self.config = config or PADEngineConfig()

        # Current emotional state (with inertia)
        self._current_state = PADVector()

        # History for derivatives and analysis
        self._telemetry_history: deque = deque(maxlen=100)
        self._pad_history: deque = deque(maxlen=1000)

        # Derivative tracking
        self._last_interrupt_rate: float = 0.0
        self._last_update_time: float = time.time()

        # Statistics
        self._total_updates: int = 0
        self._significant_shifts: int = 0

    def update(self, telemetry: TelemetrySnapshot) -> PADVector:
        """Compute new PAD state from telemetry.

        This is the main entry point - call periodically with fresh telemetry.
        Returns the smoothed (with inertia) emotional state.
        """
        # Store telemetry
        self._telemetry_history.append(telemetry)

        # Compute raw PAD (before inertia)
        raw_p = self._compute_pleasure(telemetry)
        raw_a = self._compute_arousal(telemetry)
        raw_d = self._compute_dominance(telemetry)

        raw_state = PADVector(
            pleasure=raw_p,
            arousal=raw_a,
            dominance=raw_d,
            timestamp=telemetry.timestamp,
        )

        # Apply emotional inertia
        alpha = self.config.emotional_inertia
        smoothed = PADVector(
            pleasure=alpha * self._current_state.pleasure + (1 - alpha) * raw_p,
            arousal=alpha * self._current_state.arousal + (1 - alpha) * raw_a,
            dominance=alpha * self._current_state.dominance + (1 - alpha) * raw_d,
            timestamp=telemetry.timestamp,
        )

        # Track significant shifts
        if self._current_state.distance_to(smoothed) > 0.2:
            self._significant_shifts += 1

        # Update state
        self._current_state = smoothed
        self._pad_history.append(smoothed)
        self._total_updates += 1
        self._last_update_time = telemetry.timestamp

        return smoothed

    def _compute_pleasure(self, t: TelemetrySnapshot) -> float:
        """Pleasure = thermal comfort - errors.

        P(t) = tanh(w_p * ((T_max - T_cpu) / T_range - β * log(1 + E_rate)))
        """
        cfg = self.config

        # Thermal component: how far from critical?
        temp_max = max(t.cpu_temp, t.gpu_temp)
        temp_range = t.temp_junction_max - cfg.temp_comfortable
        thermal_headroom = (t.temp_junction_max - temp_max) / max(temp_range, 1.0)

        # Error component: log scale because errors compound psychologically
        error_pain = cfg.error_sensitivity * math.log1p(t.error_rate)

        # Voltage instability is unsettling
        voltage_discomfort = abs(t.voltage_deviation) * 2.0

        # Combine
        raw = cfg.thermal_sensitivity * thermal_headroom - error_pain - voltage_discomfort

        return math.tanh(cfg.w_pleasure * raw)

    def _compute_arousal(self, t: TelemetrySnapshot) -> float:
        """Arousal = system activity level.

        A(t) = clamp(w_a * (Load_norm + γ * dIRQ/dt), -1, 1)
        """
        cfg = self.config

        # Load component: combined compute load
        load_combined = (t.cpu_load + t.gpu_load + t.io_wait) / 3.0

        # Interrupt rate derivative (acceleration of activity)
        dt = max(0.001, t.timestamp - self._last_update_time)
        irq_delta = (t.interrupt_rate - self._last_interrupt_rate) / dt
        self._last_interrupt_rate = t.interrupt_rate

        # Context switches indicate multitasking stress
        context_stress = min(1.0, t.context_switches / 10000.0)

        # Fan speed is a physiological arousal indicator
        fan_arousal = t.fan_speed_percent / 100.0

        # Combine
        raw = (
            cfg.load_sensitivity * load_combined +
            cfg.interrupt_sensitivity * math.tanh(irq_delta / 1000.0) +
            context_stress * 0.3 +
            fan_arousal * 0.2
        )

        # Shift to center around medium activity
        centered = raw * 2.0 - 0.5

        return max(-1.0, min(1.0, cfg.w_arousal * centered))

    def _compute_dominance(self, t: TelemetrySnapshot) -> float:
        """Dominance = agency and control.

        D(t) = tanh(w_d * (R_priv + S_success - δ * L_sys))
        """
        cfg = self.config

        # Privilege: having root means having control
        privilege = 1.0 if t.has_root else -0.5

        # Success: recent actions worked
        success = 1.0 if t.last_action_success else -0.5
        success -= t.actions_blocked * 0.2  # Each blocked action hurts

        # Latency: lag means loss of control
        # Convert microseconds to normalized value
        latency_normalized = min(1.0, t.scheduler_latency_us / 10000.0)
        latency_penalty = cfg.latency_sensitivity * latency_normalized

        # Memory pressure reduces agency
        memory_constraint = t.memory_pressure * 0.5

        # Combine
        raw = (
            privilege * 0.3 +
            cfg.success_sensitivity * success * 0.4 -
            latency_penalty -
            memory_constraint
        )

        return math.tanh(cfg.w_dominance * raw)

    @property
    def current_state(self) -> PADVector:
        """Get current emotional state."""
        return self._current_state

    @property
    def quadrant(self) -> EmotionalQuadrant:
        """Get current emotional quadrant."""
        return self._current_state.quadrant

    @property
    def mood_label(self) -> str:
        """Get human-readable mood label."""
        q = self.quadrant
        intensity = self._current_state.intensity

        # Intensity modifiers
        if intensity < 0.2:
            prefix = "mildly "
        elif intensity < 0.5:
            prefix = ""
        elif intensity < 0.8:
            prefix = "strongly "
        else:
            prefix = "intensely "

        labels = {
            EmotionalQuadrant.EXUBERANT: "joyful",
            EmotionalQuadrant.DEPENDENT: "grateful",
            EmotionalQuadrant.SERENE: "peaceful",
            EmotionalQuadrant.DOCILE: "gentle",
            EmotionalQuadrant.HOSTILE: "defensive",
            EmotionalQuadrant.ANXIOUS: "stressed",
            EmotionalQuadrant.DISDAINFUL: "withdrawn",
            EmotionalQuadrant.BORED: "listless",
        }

        return prefix + labels.get(q, "neutral")

    def get_mood_trajectory(self, window: int = 10) -> str:
        """Describe recent mood changes."""
        if len(self._pad_history) < 2:
            return "stable"

        recent = list(self._pad_history)[-window:]

        p_delta = recent[-1].pleasure - recent[0].pleasure
        a_delta = recent[-1].arousal - recent[0].arousal
        d_delta = recent[-1].dominance - recent[0].dominance

        changes = []
        if abs(p_delta) > 0.1:
            changes.append("brightening" if p_delta > 0 else "darkening")
        if abs(a_delta) > 0.1:
            changes.append("energizing" if a_delta > 0 else "calming")
        if abs(d_delta) > 0.1:
            changes.append("strengthening" if d_delta > 0 else "yielding")

        if not changes:
            return "stable"
        return " and ".join(changes)

    def get_statistics(self) -> Dict:
        """Get engine statistics."""
        return {
            "total_updates": self._total_updates,
            "significant_shifts": self._significant_shifts,
            "history_length": len(self._pad_history),
            "current_quadrant": self.quadrant.name,
            "current_intensity": self._current_state.intensity,
            "mood_label": self.mood_label,
            "trajectory": self.get_mood_trajectory(),
        }


# === Factory ===

def create_pad_engine(
    thermal_sensitivity: float = 1.5,
    emotional_inertia: float = 0.92,
) -> PADEngine:
    """Create a PAD engine with custom personality."""
    config = PADEngineConfig(
        thermal_sensitivity=thermal_sensitivity,
        emotional_inertia=emotional_inertia,
    )
    return PADEngine(config)


__all__ = [
    "PADEngine",
    "PADEngineConfig",
    "PADVector",
    "TelemetrySnapshot",
    "EmotionalQuadrant",
    "create_pad_engine",
]
