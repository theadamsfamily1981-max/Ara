#!/usr/bin/env python3
"""
Somatic Loom - GUTC-Integrated Haptic Neuromodulation System
=============================================================

A closed-loop haptic effector that operates on the (λ, Π) control manifold,
using subtle thermal and tensile feedback to guide brain state toward
the optimal critical corridor.

Theory:
    The Somatic Loom acts as a non-invasive lever for the Precision (Π)
    parameter, delivering subconscious neuromodulatory signals through:
    - Micro-thermal regulation (Peltier/thermal tiles)
    - Micro-tensile actuation (pressure, shape-memory alloys)
    - Temporal patterning synchronized with physiological rhythms

GUTC Mapping:
    ΔΠ_sensory (L4/L2/3 gain, ACh) → Tactile pressure patterns
    ΔΠ_prior (SFC/dACC gain, DA)   → Thermal shift patterns
    Δλ (E/I balance)               → Combined rhythm/frequency

Usage:
    from ara_core.somatic import SomaticLoom, HapticGrammar

    loom = SomaticLoom()
    loom.connect()

    # Receive repair vector from GUTC diagnostic
    repair = diagnosis.repair_vector
    pattern = HapticGrammar.from_repair_vector(repair)
    loom.execute(pattern)
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
from abc import ABC, abstractmethod

import numpy as np


# =============================================================================
# Core Enums and Types
# =============================================================================

class HapticModality(Enum):
    """Haptic feedback modalities."""
    THERMAL = auto()      # Temperature change (warming/cooling)
    PRESSURE = auto()     # Tensile/pressure change
    VIBRATION = auto()    # Low-amplitude vibration (minimal use)
    COMBINED = auto()     # Multi-modal synchronized pattern


class WaveShape(Enum):
    """Temporal waveform shapes."""
    CONSTANT = auto()     # Steady state
    SINE = auto()         # Smooth sinusoidal
    TRIANGLE = auto()     # Linear ramp up/down
    PULSE = auto()        # Sharp on/off
    BREATHING = auto()    # Respiratory-synchronized
    CONTRACTING = auto()  # Inward wave pattern


class GUTCState(Enum):
    """GUTC clinical states for haptic mapping."""
    HEALTHY = auto()
    ASD_LIKE = auto()
    PSYCHOSIS_RISK = auto()
    ANHEDONIC = auto()
    CHAOTIC = auto()
    MANIC = auto()


# =============================================================================
# Haptic Pattern Definitions
# =============================================================================

@dataclass
class ThermalPattern:
    """Thermal feedback pattern specification."""
    delta_temp: float         # Temperature change in °C (-5 to +5)
    duration_ms: int          # Duration in milliseconds
    wave_shape: WaveShape     # Temporal envelope
    frequency_hz: float = 0.0 # For oscillating patterns
    ramp_ms: int = 500        # Rise/fall time

    def __post_init__(self):
        self.delta_temp = max(-5.0, min(5.0, self.delta_temp))


@dataclass
class PressurePattern:
    """Pressure/tensile feedback pattern specification."""
    intensity: float          # Pressure intensity (0.0 to 1.0)
    duration_ms: int          # Duration in milliseconds
    wave_shape: WaveShape     # Temporal envelope
    frequency_hz: float = 0.0 # For oscillating patterns
    spatial_zones: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    def __post_init__(self):
        self.intensity = max(0.0, min(1.0, self.intensity))


@dataclass
class HapticPattern:
    """
    Complete haptic pattern combining all modalities.

    This is the "sentence" in the haptic grammar.
    """
    name: str
    gutc_state: GUTCState
    thermal: Optional[ThermalPattern] = None
    pressure: Optional[PressurePattern] = None
    sync_to_breath: bool = False
    repeat_count: int = 1
    inter_pattern_ms: int = 0

    # Target effects on GUTC parameters
    target_delta_pi_sensory: float = 0.0
    target_delta_pi_prior: float = 0.0
    target_delta_lambda: float = 0.0

    def total_duration_ms(self) -> int:
        """Calculate total pattern duration."""
        thermal_dur = self.thermal.duration_ms if self.thermal else 0
        pressure_dur = self.pressure.duration_ms if self.pressure else 0
        single_dur = max(thermal_dur, pressure_dur)
        return single_dur * self.repeat_count + self.inter_pattern_ms * (self.repeat_count - 1)


# =============================================================================
# Haptic Grammar - The Language of Precision
# =============================================================================

class HapticGrammar:
    """
    Defines the haptic vocabulary for GUTC state modulation.

    The grammar maps brain states to specific haptic patterns that
    the brain processes subconsciously as neuromodulatory signals.
    """

    # Pre-defined patterns for each GUTC state
    PATTERNS: Dict[GUTCState, HapticPattern] = {}

    @classmethod
    def _init_patterns(cls):
        """Initialize the standard haptic vocabulary."""

        # HEALTHY: Breathing rhythm - reinforces homeostatic setpoint
        cls.PATTERNS[GUTCState.HEALTHY] = HapticPattern(
            name="steady_breath",
            gutc_state=GUTCState.HEALTHY,
            thermal=ThermalPattern(
                delta_temp=0.5,
                duration_ms=4000,
                wave_shape=WaveShape.BREATHING,
                frequency_hz=0.2,  # ~12 breaths/min
            ),
            pressure=PressurePattern(
                intensity=0.2,
                duration_ms=4000,
                wave_shape=WaveShape.BREATHING,
                frequency_hz=0.2,
            ),
            sync_to_breath=True,
            repeat_count=3,
            target_delta_pi_sensory=0.0,
            target_delta_pi_prior=0.0,
            target_delta_lambda=0.0,
        )

        # ASD-LIKE: Dampening Pulse - reduce sensory over-gain
        cls.PATTERNS[GUTCState.ASD_LIKE] = HapticPattern(
            name="dampening_pulse",
            gutc_state=GUTCState.ASD_LIKE,
            thermal=ThermalPattern(
                delta_temp=-2.0,  # Cooling
                duration_ms=3000,
                wave_shape=WaveShape.CONSTANT,
                ramp_ms=1000,
            ),
            pressure=PressurePattern(
                intensity=0.6,  # Firm, steady
                duration_ms=3000,
                wave_shape=WaveShape.CONSTANT,
            ),
            sync_to_breath=False,
            repeat_count=2,
            inter_pattern_ms=1000,
            target_delta_pi_sensory=-1.5,  # Reduce sensory gain
            target_delta_pi_prior=0.0,
            target_delta_lambda=0.5,       # Push toward criticality
        )

        # PSYCHOSIS_RISK: Containment Wave - dampen runaway excitability
        cls.PATTERNS[GUTCState.PSYCHOSIS_RISK] = HapticPattern(
            name="containment_wave",
            gutc_state=GUTCState.PSYCHOSIS_RISK,
            thermal=ThermalPattern(
                delta_temp=-1.5,  # Cooling
                duration_ms=5000,
                wave_shape=WaveShape.CONTRACTING,
                frequency_hz=0.1,
            ),
            pressure=PressurePattern(
                intensity=0.4,
                duration_ms=5000,
                wave_shape=WaveShape.PULSE,
                frequency_hz=0.5,  # Slow pulse
            ),
            sync_to_breath=False,
            repeat_count=3,
            inter_pattern_ms=500,
            target_delta_pi_sensory=-0.5,
            target_delta_pi_prior=-1.0,    # Reduce prior over-weighting
            target_delta_lambda=-1.5,      # Push toward subcritical (stability)
        )

        # ANHEDONIC: Arousal Pulse - increase motivational salience
        cls.PATTERNS[GUTCState.ANHEDONIC] = HapticPattern(
            name="arousal_pulse",
            gutc_state=GUTCState.ANHEDONIC,
            thermal=ThermalPattern(
                delta_temp=2.5,  # Warming
                duration_ms=500,
                wave_shape=WaveShape.PULSE,
                frequency_hz=2.0,  # Quick pulses
            ),
            pressure=PressurePattern(
                intensity=0.3,
                duration_ms=500,
                wave_shape=WaveShape.PULSE,
                frequency_hz=2.0,
                spatial_zones=[1, 2],  # Targeted spots
            ),
            sync_to_breath=False,
            repeat_count=5,
            inter_pattern_ms=200,
            target_delta_pi_sensory=0.5,
            target_delta_pi_prior=1.5,     # Increase motivational gain
            target_delta_lambda=1.0,       # Push toward criticality
        )

        # CHAOTIC: Grounding Anchor - establish stable boundaries
        cls.PATTERNS[GUTCState.CHAOTIC] = HapticPattern(
            name="grounding_anchor",
            gutc_state=GUTCState.CHAOTIC,
            thermal=ThermalPattern(
                delta_temp=-1.0,
                duration_ms=4000,
                wave_shape=WaveShape.TRIANGLE,
                ramp_ms=2000,
            ),
            pressure=PressurePattern(
                intensity=0.7,  # Strong grounding
                duration_ms=4000,
                wave_shape=WaveShape.CONSTANT,
                spatial_zones=[0, 1, 2, 3],  # Full coverage
            ),
            sync_to_breath=True,
            repeat_count=2,
            target_delta_pi_sensory=0.5,
            target_delta_pi_prior=0.5,
            target_delta_lambda=-1.0,      # Reduce excitability
        )

        # MANIC: Calming Descent - reduce hyperactive processing
        cls.PATTERNS[GUTCState.MANIC] = HapticPattern(
            name="calming_descent",
            gutc_state=GUTCState.MANIC,
            thermal=ThermalPattern(
                delta_temp=-2.0,  # Cooling
                duration_ms=6000,
                wave_shape=WaveShape.SINE,
                frequency_hz=0.15,  # Very slow
            ),
            pressure=PressurePattern(
                intensity=0.5,
                duration_ms=6000,
                wave_shape=WaveShape.SINE,
                frequency_hz=0.15,
            ),
            sync_to_breath=True,
            repeat_count=3,
            target_delta_pi_sensory=0.0,
            target_delta_pi_prior=-2.0,    # Normalize salience
            target_delta_lambda=0.0,       # Maintain criticality
        )

    @classmethod
    def get_pattern(cls, state: GUTCState) -> HapticPattern:
        """Get the standard pattern for a GUTC state."""
        if not cls.PATTERNS:
            cls._init_patterns()
        return cls.PATTERNS.get(state)

    @classmethod
    def from_repair_vector(
        cls,
        delta_lambda: float,
        delta_pi_sensory: float,
        delta_pi_prior: float,
    ) -> HapticPattern:
        """
        Generate a haptic pattern from a GUTC repair vector.

        Maps the numeric repair targets to the closest matching pattern
        or synthesizes a custom blend.
        """
        if not cls.PATTERNS:
            cls._init_patterns()

        # Determine dominant state based on repair vector
        if abs(delta_lambda) < 0.5 and abs(delta_pi_sensory) < 0.5 and abs(delta_pi_prior) < 0.5:
            return cls.PATTERNS[GUTCState.HEALTHY]

        # High sensory Π + low λ → ASD-like
        if delta_pi_sensory < -1.0 and delta_lambda > 0:
            return cls.PATTERNS[GUTCState.ASD_LIKE]

        # High prior Π + high λ → Psychosis risk
        if delta_pi_prior < -0.5 and delta_lambda < -0.5:
            return cls.PATTERNS[GUTCState.PSYCHOSIS_RISK]

        # Low Π overall + low λ → Anhedonic
        if delta_pi_prior > 0.5 and delta_lambda > 0.5:
            return cls.PATTERNS[GUTCState.ANHEDONIC]

        # High λ alone → Chaotic
        if delta_lambda < -0.5:
            return cls.PATTERNS[GUTCState.CHAOTIC]

        # High prior Π alone → Manic
        if delta_pi_prior < -1.0:
            return cls.PATTERNS[GUTCState.MANIC]

        # Default to healthy maintenance
        return cls.PATTERNS[GUTCState.HEALTHY]


# Initialize patterns on module load
HapticGrammar._init_patterns()


# =============================================================================
# Hardware Abstraction Layer
# =============================================================================

class HapticActuator(ABC):
    """Abstract base class for haptic actuators."""

    @abstractmethod
    def set_thermal(self, delta_temp: float) -> None:
        """Set thermal output."""
        pass

    @abstractmethod
    def set_pressure(self, intensity: float, zones: List[int]) -> None:
        """Set pressure output."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop all output."""
        pass


class SimulatedActuator(HapticActuator):
    """Simulated actuator for testing without hardware."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_temp = 0.0
        self.current_pressure = 0.0

    def set_thermal(self, delta_temp: float) -> None:
        self.current_temp = delta_temp
        if self.verbose:
            direction = "warming" if delta_temp > 0 else "cooling"
            print(f"  [THERMAL] {direction} {abs(delta_temp):.1f}°C")

    def set_pressure(self, intensity: float, zones: List[int]) -> None:
        self.current_pressure = intensity
        if self.verbose:
            print(f"  [PRESSURE] intensity={intensity:.2f} zones={zones}")

    def stop(self) -> None:
        self.current_temp = 0.0
        self.current_pressure = 0.0
        if self.verbose:
            print("  [STOP] All actuators off")


# =============================================================================
# Somatic Loom Controller
# =============================================================================

class SomaticLoom:
    """
    Somatic Loom - Main controller for GUTC-integrated haptic feedback.

    The Loom orchestrates haptic patterns based on GUTC diagnostic output,
    delivering subtle neuromodulatory signals through thermal and pressure
    channels.
    """

    def __init__(
        self,
        actuator: HapticActuator = None,
        breath_rate_hz: float = 0.2,
        verbose: bool = True,
    ):
        """
        Initialize the Somatic Loom.

        Args:
            actuator: Hardware actuator (default: simulated)
            breath_rate_hz: Default breathing rate (0.2 = 12/min)
            verbose: Print execution status
        """
        self.actuator = actuator or SimulatedActuator(verbose=verbose)
        self.breath_rate_hz = breath_rate_hz
        self.verbose = verbose

        self.connected = False
        self.executing = False
        self.current_pattern: Optional[HapticPattern] = None

        # Physiological sync
        self.last_breath_phase: float = 0.0
        self.breath_callback: Optional[Callable[[], float]] = None

    def connect(self) -> bool:
        """Connect to haptic hardware."""
        self.connected = True
        if self.verbose:
            print("[SomaticLoom] Connected to haptic actuator")
        return True

    def disconnect(self) -> None:
        """Disconnect from hardware."""
        self.actuator.stop()
        self.connected = False
        if self.verbose:
            print("[SomaticLoom] Disconnected")

    def set_breath_callback(self, callback: Callable[[], float]) -> None:
        """
        Set callback for respiratory synchronization.

        Callback should return current breath phase (0.0 to 1.0).
        """
        self.breath_callback = callback

    def execute(self, pattern: HapticPattern) -> None:
        """
        Execute a haptic pattern.

        Args:
            pattern: HapticPattern to execute
        """
        if not self.connected:
            raise RuntimeError("Loom not connected")

        self.executing = True
        self.current_pattern = pattern

        if self.verbose:
            print(f"\n[SomaticLoom] Executing: {pattern.name}")
            print(f"  Duration: {pattern.total_duration_ms()}ms")
            print(f"  Target: Δλ={pattern.target_delta_lambda:+.1f}, "
                  f"ΔΠs={pattern.target_delta_pi_sensory:+.1f}, "
                  f"ΔΠp={pattern.target_delta_pi_prior:+.1f}")

        for rep in range(pattern.repeat_count):
            if self.verbose and pattern.repeat_count > 1:
                print(f"  [Rep {rep + 1}/{pattern.repeat_count}]")

            self._execute_single(pattern)

            if rep < pattern.repeat_count - 1 and pattern.inter_pattern_ms > 0:
                time.sleep(pattern.inter_pattern_ms / 1000.0)

        self.actuator.stop()
        self.executing = False
        self.current_pattern = None

        if self.verbose:
            print("[SomaticLoom] Pattern complete")

    def _execute_single(self, pattern: HapticPattern) -> None:
        """Execute a single iteration of a pattern."""
        duration_ms = max(
            pattern.thermal.duration_ms if pattern.thermal else 0,
            pattern.pressure.duration_ms if pattern.pressure else 0,
        )

        steps = 20  # Temporal resolution
        step_ms = duration_ms / steps

        for step in range(steps):
            t = step / steps  # Normalized time [0, 1]

            # Calculate breath phase for sync
            if pattern.sync_to_breath and self.breath_callback:
                breath_phase = self.breath_callback()
            else:
                breath_phase = t

            # Apply thermal
            if pattern.thermal:
                temp_value = self._calculate_waveform(
                    t, breath_phase,
                    pattern.thermal.wave_shape,
                    pattern.thermal.frequency_hz,
                ) * pattern.thermal.delta_temp
                self.actuator.set_thermal(temp_value)

            # Apply pressure
            if pattern.pressure:
                pressure_value = self._calculate_waveform(
                    t, breath_phase,
                    pattern.pressure.wave_shape,
                    pattern.pressure.frequency_hz,
                ) * pattern.pressure.intensity
                self.actuator.set_pressure(
                    pressure_value,
                    pattern.pressure.spatial_zones,
                )

            time.sleep(step_ms / 1000.0)

    def _calculate_waveform(
        self,
        t: float,
        breath_phase: float,
        shape: WaveShape,
        freq_hz: float,
    ) -> float:
        """Calculate waveform value at time t."""
        if shape == WaveShape.CONSTANT:
            return 1.0

        elif shape == WaveShape.SINE:
            phase = 2 * math.pi * (freq_hz * t if freq_hz > 0 else t)
            return 0.5 * (1 + math.sin(phase))

        elif shape == WaveShape.TRIANGLE:
            return 1.0 - abs(2 * t - 1)

        elif shape == WaveShape.PULSE:
            if freq_hz > 0:
                return 1.0 if (t * freq_hz) % 1.0 < 0.5 else 0.0
            return 1.0 if t < 0.5 else 0.0

        elif shape == WaveShape.BREATHING:
            # Smooth breathing curve (sinusoidal with hold at extremes)
            return 0.5 * (1 + math.sin(2 * math.pi * breath_phase - math.pi / 2))

        elif shape == WaveShape.CONTRACTING:
            # Inward wave - intensity increases toward center of duration
            return 1.0 - abs(2 * t - 1) ** 0.5

        return 1.0

    def execute_for_state(self, state: GUTCState) -> None:
        """Execute the standard pattern for a GUTC state."""
        pattern = HapticGrammar.get_pattern(state)
        if pattern:
            self.execute(pattern)

    def execute_from_diagnosis(self, diagnosis: Any) -> None:
        """
        Execute pattern based on GUTC diagnosis.

        Args:
            diagnosis: GUTCDiagnosis object from gutc_diagnostic_engine
        """
        if hasattr(diagnosis, 'repair_vector'):
            rv = diagnosis.repair_vector
            pattern = HapticGrammar.from_repair_vector(
                delta_lambda=rv.delta_lambda,
                delta_pi_sensory=getattr(rv, 'delta_pi', rv.delta_pi) if hasattr(rv, 'delta_pi') else 0,
                delta_pi_prior=rv.delta_pi,
            )
            self.execute(pattern)
        else:
            if self.verbose:
                print("[SomaticLoom] No repair vector in diagnosis")


# =============================================================================
# Convenience Functions
# =============================================================================

_loom_instance: Optional[SomaticLoom] = None


def get_loom() -> SomaticLoom:
    """Get the global Somatic Loom instance."""
    global _loom_instance
    if _loom_instance is None:
        _loom_instance = SomaticLoom()
        _loom_instance.connect()
    return _loom_instance


def execute_haptic(state: GUTCState) -> None:
    """Execute haptic feedback for a GUTC state."""
    get_loom().execute_for_state(state)
