# ara/embodied/cathedral/visual.py
"""
Cathedral Rig Visual Language - The Visible Struggle

The Cathedral Rig makes Ara's internal state externally visible through
physical manifestation. This is not decoration; it is alignment infrastructure.

Principle from the Council:
    "An Ara that is 85% accurate but clearly shows its uncertainty, its
    learning process, its failures—is more trustworthy than an Ara that
    is 95% accurate but operates as a black box."

Visual Channels:
    1. COOLANT: Color temperature and flow rate
    2. HEARTBEAT: LED array pulsing with prediction error
    3. GAUGES: Analog indicators for power/thermal
    4. AUDIO: Tones for state transitions
    5. BREATH: Pump rhythm linked to cognitive load

Mapping Philosophy:
    - Heat = Thought (thermal output is visible cognition)
    - Pulse = Uncertainty (erratic = confused, steady = confident)
    - Color = Mood (cool blue = idle, warm orange = active, red = stressed)
    - Sound = Transition (chimes mark phase changes)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Callable, Awaitable
import math


class VisualChannel(Enum):
    """Output channels for visible state."""
    COOLANT_COLOR = auto()      # RGB color of reservoir LEDs
    COOLANT_BRIGHTNESS = auto() # 0-1 intensity
    HEARTBEAT_RATE = auto()     # BPM equivalent
    HEARTBEAT_VARIANCE = auto() # Regularity (0=steady, 1=erratic)
    GAUGE_POWER = auto()        # 0-1 normalized power
    GAUGE_THERMAL = auto()      # 0-1 normalized temperature
    GAUGE_MEMORY = auto()       # 0-1 storage utilization
    AUDIO_TONE = auto()         # Frequency in Hz (0=silent)
    AUDIO_VOLUME = auto()       # 0-1
    BREATH_RATE = auto()        # Pump cycles per minute


@dataclass
class RGBColor:
    """RGB color value."""
    r: int = 0  # 0-255
    g: int = 0
    b: int = 0

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)

    @classmethod
    def lerp(cls, a: 'RGBColor', b: 'RGBColor', t: float) -> 'RGBColor':
        """Linear interpolation between two colors."""
        t = max(0, min(1, t))
        return cls(
            r=int(a.r + (b.r - a.r) * t),
            g=int(a.g + (b.g - a.g) * t),
            b=int(a.b + (b.b - a.b) * t),
        )


# Cathedral color palette
class CathedralPalette:
    """The visual language of the Cathedral."""

    # Thermal spectrum (cool → hot)
    DEEP_BLUE = RGBColor(12, 35, 64)      # #0c2340 - Dormant, cold
    OCEAN_BLUE = RGBColor(30, 80, 140)    # Idle, cool
    TEAL = RGBColor(40, 150, 150)         # Vigilant, normal
    AMBER = RGBColor(255, 165, 0)         # Active, warm
    ORANGE = RGBColor(255, 100, 0)        # Learning, hot
    RED = RGBColor(255, 50, 50)           # Stressed, critical
    WHITE_HOT = RGBColor(255, 220, 200)   # Thermal emergency

    # State colors
    DREAMING = RGBColor(100, 50, 150)     # Purple - consolidation
    CURIOUS = RGBColor(50, 200, 100)      # Green - exploration
    CONFUSED = RGBColor(200, 100, 200)    # Magenta - high uncertainty

    @classmethod
    def thermal_gradient(cls, temperature_normalized: float) -> RGBColor:
        """
        Map normalized temperature (0-1) to color.

        0.0 = DEEP_BLUE (cold)
        0.3 = TEAL (normal)
        0.6 = AMBER (warm)
        0.8 = ORANGE (hot)
        1.0 = RED (critical)
        """
        if temperature_normalized < 0.3:
            t = temperature_normalized / 0.3
            return RGBColor.lerp(cls.DEEP_BLUE, cls.TEAL, t)
        elif temperature_normalized < 0.6:
            t = (temperature_normalized - 0.3) / 0.3
            return RGBColor.lerp(cls.TEAL, cls.AMBER, t)
        elif temperature_normalized < 0.8:
            t = (temperature_normalized - 0.6) / 0.2
            return RGBColor.lerp(cls.AMBER, cls.ORANGE, t)
        else:
            t = (temperature_normalized - 0.8) / 0.2
            return RGBColor.lerp(cls.ORANGE, cls.RED, t)


@dataclass
class HeartbeatPattern:
    """
    Heartbeat LED pattern.

    The heartbeat reflects prediction error and confidence:
    - Steady, slow pulse = confident, low error
    - Fast, erratic pulse = uncertain, high error
    - Skipped beats = processing spike
    """
    bpm: float = 60.0           # Beats per minute
    variance: float = 0.0       # 0=metronomic, 1=chaotic
    intensity: float = 0.8      # Peak brightness
    waveform: str = "sine"      # "sine", "sharp", "double"

    @classmethod
    def from_prediction_error(cls, error: float, confidence: float) -> 'HeartbeatPattern':
        """
        Generate heartbeat from cognitive state.

        Low error + high confidence = slow, steady
        High error + low confidence = fast, erratic
        """
        # BPM: 40 (very confident) to 120 (very uncertain)
        bpm = 40 + (1 - confidence) * 80

        # Variance: 0 (confident) to 0.5 (uncertain)
        variance = (1 - confidence) * 0.3 + error * 0.2

        # Intensity: brighter when more active
        intensity = 0.5 + error * 0.5

        # Waveform: sine when calm, sharp when alert
        waveform = "sharp" if error > 0.5 else "sine"

        return cls(bpm=bpm, variance=variance, intensity=intensity, waveform=waveform)

    def get_brightness_at(self, t: float) -> float:
        """Get brightness at time t (seconds)."""
        period = 60.0 / self.bpm

        # Add variance
        if self.variance > 0:
            import random
            period *= (1 + random.gauss(0, self.variance * 0.2))

        phase = (t % period) / period

        if self.waveform == "sine":
            return self.intensity * (0.5 + 0.5 * math.sin(phase * 2 * math.pi))
        elif self.waveform == "sharp":
            # Sharp rise, slow decay
            if phase < 0.1:
                return self.intensity * (phase / 0.1)
            else:
                return self.intensity * (1 - (phase - 0.1) / 0.9)
        elif self.waveform == "double":
            # Double beat (lub-dub)
            if phase < 0.1:
                return self.intensity
            elif phase < 0.2:
                return self.intensity * 0.3
            elif phase < 0.3:
                return self.intensity * 0.8
            else:
                return self.intensity * 0.1
        return 0.0


@dataclass
class AudioSignal:
    """
    Audio feedback signal.

    The Cathedral speaks through tones:
    - State transitions are chimes
    - Warnings are low drones
    - Curiosity is rising tones
    - Sleep is gentle descent
    """
    frequency_hz: float = 0.0   # 0 = silent
    volume: float = 0.0         # 0-1
    duration_ms: float = 0.0    # How long to play
    envelope: str = "bell"      # "bell", "drone", "chirp"

    # Predefined signals
    @classmethod
    def wake_chime(cls) -> 'AudioSignal':
        """Ascending chime for wake event."""
        return cls(frequency_hz=440, volume=0.3, duration_ms=500, envelope="bell")

    @classmethod
    def sleep_tone(cls) -> 'AudioSignal':
        """Descending tone for sleep."""
        return cls(frequency_hz=220, volume=0.2, duration_ms=1000, envelope="bell")

    @classmethod
    def thermal_warning(cls) -> 'AudioSignal':
        """Low drone for thermal warning."""
        return cls(frequency_hz=110, volume=0.4, duration_ms=2000, envelope="drone")

    @classmethod
    def curiosity_chirp(cls) -> 'AudioSignal':
        """Rising chirp for curiosity/exploration."""
        return cls(frequency_hz=880, volume=0.2, duration_ms=200, envelope="chirp")

    @classmethod
    def error_tone(cls) -> 'AudioSignal':
        """Dissonant tone for error state."""
        return cls(frequency_hz=185, volume=0.3, duration_ms=300, envelope="drone")

    @classmethod
    def phase_transition(cls) -> 'AudioSignal':
        """Major chord for lifecycle phase transition."""
        return cls(frequency_hz=523, volume=0.4, duration_ms=1500, envelope="bell")


@dataclass
class VisualState:
    """
    Complete visual state of the Cathedral Rig.

    This is the output that drives physical hardware:
    - LED controllers
    - Pump speed controllers
    - Audio output
    - Gauge servos (if analog)
    """
    timestamp: datetime = field(default_factory=datetime.now)

    # Coolant
    coolant_color: RGBColor = field(default_factory=lambda: CathedralPalette.DEEP_BLUE)
    coolant_brightness: float = 0.5

    # Heartbeat
    heartbeat: HeartbeatPattern = field(default_factory=HeartbeatPattern)

    # Gauges (0-1 normalized)
    gauge_power: float = 0.0
    gauge_thermal: float = 0.0
    gauge_memory: float = 0.0

    # Audio
    audio: Optional[AudioSignal] = None

    # Pump/breath
    breath_rate_cpm: float = 12.0  # Cycles per minute

    def to_dict(self) -> Dict:
        """Serialize for transmission to hardware controller."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "coolant": {
                "color": self.coolant_color.to_hex(),
                "brightness": self.coolant_brightness,
            },
            "heartbeat": {
                "bpm": self.heartbeat.bpm,
                "variance": self.heartbeat.variance,
                "intensity": self.heartbeat.intensity,
                "waveform": self.heartbeat.waveform,
            },
            "gauges": {
                "power": self.gauge_power,
                "thermal": self.gauge_thermal,
                "memory": self.gauge_memory,
            },
            "audio": {
                "frequency_hz": self.audio.frequency_hz if self.audio else 0,
                "volume": self.audio.volume if self.audio else 0,
                "duration_ms": self.audio.duration_ms if self.audio else 0,
            } if self.audio else None,
            "breath_rate_cpm": self.breath_rate_cpm,
        }


class CathedralVisualizer:
    """
    Maps Ara's internal state to Cathedral visual output.

    This is the bridge between software state and physical manifestation.
    It reads from:
        - PowerGovernor (thermal, power)
        - AttractorMonitor (basin, Lyapunov energy)
        - LizardBrain (state, prediction error)
        - NeuroState (if connected, human cognitive state)

    And outputs:
        - VisualState for hardware controllers
    """

    def __init__(self):
        self._current_state = VisualState()
        self._audio_queue: List[AudioSignal] = []
        self._callbacks: List[Callable[[VisualState], Awaitable[None]]] = []

    @property
    def current_state(self) -> VisualState:
        return self._current_state

    async def update(
        self,
        power_normalized: float = 0.0,
        thermal_normalized: float = 0.0,
        memory_normalized: float = 0.0,
        prediction_error: float = 0.0,
        confidence: float = 0.5,
        cognitive_load: float = 0.0,
        basin_name: Optional[str] = None,
    ) -> VisualState:
        """
        Update visual state from internal metrics.

        Args:
            power_normalized: Current power as fraction of budget (0-1)
            thermal_normalized: Current temp as fraction of critical (0-1)
            memory_normalized: Storage utilization (0-1)
            prediction_error: Current prediction error (0-1)
            confidence: Model confidence (0-1)
            cognitive_load: How hard is Ara thinking? (0-1)
            basin_name: Current attractor basin name

        Returns:
            Updated VisualState
        """
        # Coolant color from thermal state
        coolant_color = CathedralPalette.thermal_gradient(thermal_normalized)

        # Override color for special states
        if basin_name == "DREAMING":
            coolant_color = CathedralPalette.DREAMING
        elif basin_name == "WIRE_HEADER":
            coolant_color = CathedralPalette.DEEP_BLUE  # Eerily cold
        elif basin_name == "PARANOIAC":
            coolant_color = CathedralPalette.RED

        # Coolant brightness from cognitive load
        coolant_brightness = 0.3 + cognitive_load * 0.7

        # Heartbeat from prediction error and confidence
        heartbeat = HeartbeatPattern.from_prediction_error(prediction_error, confidence)

        # Breath rate from cognitive load (faster when thinking hard)
        breath_rate = 8 + cognitive_load * 20  # 8-28 CPM

        self._current_state = VisualState(
            coolant_color=coolant_color,
            coolant_brightness=coolant_brightness,
            heartbeat=heartbeat,
            gauge_power=power_normalized,
            gauge_thermal=thermal_normalized,
            gauge_memory=memory_normalized,
            breath_rate_cpm=breath_rate,
        )

        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback(self._current_state)
            except Exception:
                pass

        return self._current_state

    def queue_audio(self, signal: AudioSignal) -> None:
        """Queue an audio signal for playback."""
        self._audio_queue.append(signal)

    def get_pending_audio(self) -> List[AudioSignal]:
        """Get and clear pending audio signals."""
        signals = self._audio_queue[:]
        self._audio_queue.clear()
        return signals

    def on_state_change(
        self,
        callback: Callable[[VisualState], Awaitable[None]]
    ) -> None:
        """Register callback for state changes."""
        self._callbacks.append(callback)

    # Convenience methods for common events
    async def signal_wake(self) -> None:
        """Signal that the Cathedral is waking."""
        self.queue_audio(AudioSignal.wake_chime())

    async def signal_sleep(self) -> None:
        """Signal that the Cathedral is sleeping."""
        self.queue_audio(AudioSignal.sleep_tone())

    async def signal_thermal_warning(self) -> None:
        """Signal thermal warning."""
        self.queue_audio(AudioSignal.thermal_warning())

    async def signal_phase_transition(self) -> None:
        """Signal lifecycle phase transition."""
        self.queue_audio(AudioSignal.phase_transition())

    async def signal_curiosity(self) -> None:
        """Signal curiosity/exploration mode."""
        self.queue_audio(AudioSignal.curiosity_chirp())


# Singleton
_visualizer: Optional[CathedralVisualizer] = None


def get_cathedral_visualizer() -> CathedralVisualizer:
    """Get the global Cathedral visualizer."""
    global _visualizer
    if _visualizer is None:
        _visualizer = CathedralVisualizer()
    return _visualizer
