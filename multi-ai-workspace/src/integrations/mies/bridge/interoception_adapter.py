"""Interoception Adapter - Bridges ara/interoception to MIES TelemetrySnapshot.

This adapter translates between:
- ara/interoception L1BodyState (physical/somatic signals)
- ara/interoception L2PerceptionState (sensory processing)
- ara/interoception InteroceptivePAD (SNN-derived affect)
and:
- MIES TelemetrySnapshot (hardware telemetry)
- MIES PADVector (emotional state)

The key insight is that both systems model the same underlying reality:
- L1BodyState maps to hardware thermal/load metrics
- L2PerceptionState maps to user interaction quality
- InteroceptivePAD and PADVector are different representations of the same PAD space
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from enum import Enum, auto

from ..affect.pad_engine import TelemetrySnapshot, PADVector, EmotionalQuadrant

logger = logging.getLogger(__name__)


# === L1BodyState Representation (from ara/interoception) ===

@dataclass
class L1BodyState:
    """Physical/somatic signals - mirrors ara/interoception.L1BodyState.

    This is a local representation for when ara/interoception isn't available.
    If the full ara package is installed, use ara.interoception.L1BodyState directly.
    """
    # Cardiovascular
    heart_rate: float = 72.0          # BPM (60-100 normal)
    heart_rate_variability: float = 50.0  # ms RMSSD (higher = healthier)
    blood_pressure_systolic: float = 120.0
    blood_pressure_diastolic: float = 80.0

    # Respiratory
    breath_rate: float = 14.0         # breaths/min (12-20 normal)
    breath_depth: float = 0.5         # 0-1 normalized

    # Muscular
    muscle_tension: float = 0.3       # 0-1 (0=relaxed, 1=tense)
    posture_openness: float = 0.6     # 0-1 (0=closed, 1=open)

    # Electrodermal
    skin_conductance: float = 2.0     # µS (higher = arousal)
    skin_temperature: float = 33.0    # °C (fingertip)

    def compute_derived(self) -> Tuple[float, float]:
        """Compute stress_index and energy_level."""
        # Normalize HRV (higher is better, ~50ms baseline)
        hrv_norm = min(1.0, self.heart_rate_variability / 100.0)

        # Stress index: low HRV + high conductance = stress
        stress_index = (1.0 - hrv_norm) * 0.6 + min(1.0, self.skin_conductance / 5.0) * 0.4

        # Energy level: HR activity + breath depth
        hr_norm = min(1.0, max(0.0, (self.heart_rate - 60) / 60.0))
        energy_level = hr_norm * 0.5 + self.breath_depth * 0.5

        return stress_index, energy_level


@dataclass
class L2PerceptionState:
    """Sensory processing signals - mirrors ara/interoception.L2PerceptionState.

    Represents processed perceptual input from audio, text, and visual channels.
    """
    # Audio features
    audio_valence: float = 0.0        # -1 to 1 (prosody-derived)
    audio_arousal: float = 0.0        # 0 to 1 (prosody-derived)
    audio_dominance: float = 0.5      # 0 to 1 (prosody-derived)

    # Text features
    text_sentiment: float = 0.0       # -1 to 1
    text_arousal: float = 0.0         # 0 to 1

    # Visual features
    facial_valence: float = 0.0       # -1 to 1 (if available)
    gaze_engagement: float = 0.5      # 0 to 1

    # Meta
    attention_focus: float = 0.5      # 0 to 1
    novelty_signal: float = 0.0       # 0 to 1
    perception_confidence: float = 0.8  # 0 to 1


@dataclass
class InteroceptivePAD:
    """SNN-derived PAD state - mirrors ara/interoception.InteroceptivePAD.

    This comes from the spiking neural network dynamics, representing
    true interoceptive (internal) affect rather than external estimation.
    """
    valence: float = 0.0              # -1 to 1
    arousal: float = 0.0              # 0 to 1
    dominance: float = 0.5            # 0 to 1

    # SNN-specific metrics
    mean_firing_rate: float = 0.1     # 0 to 1
    synchrony: float = 0.5            # 0 to 1
    homeostatic_error: float = 0.0    # Deviation from set point


# === Adapter Functions ===

def adapt_l1_to_telemetry(
    l1: L1BodyState,
    base_telemetry: Optional[TelemetrySnapshot] = None,
) -> TelemetrySnapshot:
    """Convert L1BodyState to TelemetrySnapshot.

    Maps biological metaphors to hardware equivalents:
    - heart_rate → cpu_load (activity level)
    - muscle_tension → gpu_load (processing demand)
    - skin_temperature → cpu_temp (thermal state)
    - stress_index → error_rate (system stress)

    Args:
        l1: L1BodyState from ara/interoception or local representation
        base_telemetry: Optional existing telemetry to merge with

    Returns:
        TelemetrySnapshot suitable for MIES PADEngine
    """
    stress_index, energy_level = l1.compute_derived()

    # Map heart rate to CPU load (72 bpm = 0.5 load)
    hr_normalized = (l1.heart_rate - 60) / 60.0
    cpu_load = max(0.0, min(1.0, 0.3 + hr_normalized * 0.4))

    # Map muscle tension to GPU load
    gpu_load = l1.muscle_tension

    # Map skin temperature to CPU temp (33°C skin → ~60°C CPU)
    # Higher skin temp indicates arousal/load
    skin_deviation = l1.skin_temperature - 33.0  # Normal is ~33°C
    cpu_temp = 60.0 + skin_deviation * 5.0

    # Map stress index to error rate
    error_rate = stress_index * 10.0  # 0-10 errors/sec

    # Energy from HRV
    hrv_norm = min(1.0, l1.heart_rate_variability / 100.0)

    telemetry = TelemetrySnapshot(
        cpu_temp=cpu_temp,
        gpu_temp=cpu_temp - 5.0,  # GPU slightly cooler
        cpu_load=cpu_load,
        gpu_load=gpu_load,
        memory_pressure=0.5 + stress_index * 0.3,  # Stress increases memory pressure
        error_rate=error_rate,
        has_root=True,
        last_action_success=stress_index < 0.5,
        interrupt_rate=l1.heart_rate,  # Heartbeats as interrupts
        fan_speed_percent=max(30.0, min(100.0, 30.0 + cpu_temp - 50.0)),
    )

    # Merge with base if provided
    if base_telemetry is not None:
        # Average hardware metrics with L1-derived ones
        telemetry.cpu_temp = (telemetry.cpu_temp + base_telemetry.cpu_temp) / 2
        telemetry.gpu_temp = (telemetry.gpu_temp + base_telemetry.gpu_temp) / 2
        telemetry.cpu_load = max(telemetry.cpu_load, base_telemetry.cpu_load)
        telemetry.gpu_load = max(telemetry.gpu_load, base_telemetry.gpu_load)
        telemetry.error_rate = max(telemetry.error_rate, base_telemetry.error_rate)

    return telemetry


def adapt_l2_to_telemetry(
    l2: L2PerceptionState,
    base_telemetry: Optional[TelemetrySnapshot] = None,
) -> Dict[str, float]:
    """Extract L2 perception features for affect modulation.

    L2 doesn't map directly to hardware but provides user-facing context
    that modulates how we interpret telemetry.

    Returns:
        Dictionary of modulation factors for the affect system
    """
    return {
        # Interaction quality affects how we feel about our state
        "interaction_valence": l2.audio_valence * 0.4 + l2.text_sentiment * 0.4 + l2.facial_valence * 0.2,
        "interaction_arousal": l2.audio_arousal * 0.5 + l2.text_arousal * 0.3 + l2.novelty_signal * 0.2,

        # Attention and engagement
        "attention_level": l2.attention_focus * 0.6 + l2.gaze_engagement * 0.4,

        # Confidence in perception
        "perception_confidence": l2.perception_confidence,

        # Novelty drives curiosity
        "novelty": l2.novelty_signal,
    }


def adapt_interoceptive_pad(
    intero_pad: InteroceptivePAD,
) -> PADVector:
    """Convert InteroceptivePAD (from SNN) to MIES PADVector.

    Both represent the same PAD space, but:
    - InteroceptivePAD comes from SNN membrane dynamics (true interoception)
    - PADVector is MIES's representation with quadrant labels

    Args:
        intero_pad: PAD state from ara/interoception SNN

    Returns:
        PADVector suitable for MIES affect architecture
    """
    # Direct mapping - both are in [-1, 1] or [0, 1] ranges
    # Adjust arousal from [0, 1] to [-1, 1] for consistency
    arousal_adjusted = intero_pad.arousal * 2 - 1  # [0,1] → [-1,1]

    return PADVector(
        pleasure=intero_pad.valence,
        arousal=arousal_adjusted,
        dominance=intero_pad.dominance * 2 - 1,  # [0,1] → [-1,1]
    )


def adapt_pad_to_interoceptive(
    pad: PADVector,
) -> InteroceptivePAD:
    """Convert MIES PADVector to InteroceptivePAD format.

    Reverse mapping for when MIES PAD needs to feed back to ara/interoception.
    """
    # Convert [-1, 1] ranges to [0, 1] where needed
    return InteroceptivePAD(
        valence=pad.pleasure,
        arousal=(pad.arousal + 1) / 2,      # [-1,1] → [0,1]
        dominance=(pad.dominance + 1) / 2,  # [-1,1] → [0,1]
    )


# === Adapter Class ===

class InteroceptionAdapter:
    """Full adapter for ara/interoception ↔ MIES integration.

    This class manages bidirectional translation between the two affect systems,
    maintaining consistency and enabling unified emotional state.
    """

    def __init__(
        self,
        l1_weight: float = 0.4,
        l2_weight: float = 0.3,
        hardware_weight: float = 0.3,
    ):
        """Initialize the adapter.

        Args:
            l1_weight: Weight for L1 (body) signals in fusion
            l2_weight: Weight for L2 (perception) signals
            hardware_weight: Weight for direct hardware telemetry
        """
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.hardware_weight = hardware_weight

        self._last_l1: Optional[L1BodyState] = None
        self._last_l2: Optional[L2PerceptionState] = None
        self._last_telemetry: Optional[TelemetrySnapshot] = None
        self._adaptation_count: int = 0

    def process_interoception(
        self,
        l1: Optional[L1BodyState] = None,
        l2: Optional[L2PerceptionState] = None,
        hardware_telemetry: Optional[TelemetrySnapshot] = None,
    ) -> Tuple[TelemetrySnapshot, Dict[str, float]]:
        """Process interoception signals into unified telemetry.

        Fuses L1, L2, and hardware signals into a single TelemetrySnapshot
        that captures both biological metaphors and actual hardware state.

        Returns:
            Tuple of (unified_telemetry, l2_modulation_factors)
        """
        self._adaptation_count += 1

        # Update cached states
        if l1 is not None:
            self._last_l1 = l1
        if l2 is not None:
            self._last_l2 = l2
        if hardware_telemetry is not None:
            self._last_telemetry = hardware_telemetry

        # Start with hardware telemetry as base
        if self._last_telemetry is not None:
            base = self._last_telemetry
        else:
            base = TelemetrySnapshot()  # Defaults

        # Fuse with L1 if available
        if self._last_l1 is not None:
            l1_telemetry = adapt_l1_to_telemetry(self._last_l1)

            # Weighted average
            w_hw = self.hardware_weight
            w_l1 = self.l1_weight

            total_w = w_hw + w_l1
            w_hw /= total_w
            w_l1 /= total_w

            base = TelemetrySnapshot(
                cpu_temp=base.cpu_temp * w_hw + l1_telemetry.cpu_temp * w_l1,
                gpu_temp=base.gpu_temp * w_hw + l1_telemetry.gpu_temp * w_l1,
                cpu_load=base.cpu_load * w_hw + l1_telemetry.cpu_load * w_l1,
                gpu_load=base.gpu_load * w_hw + l1_telemetry.gpu_load * w_l1,
                error_rate=max(base.error_rate, l1_telemetry.error_rate),
                memory_pressure=base.memory_pressure * w_hw + l1_telemetry.memory_pressure * w_l1,
                has_root=base.has_root,
                last_action_success=base.last_action_success and l1_telemetry.last_action_success,
                interrupt_rate=base.interrupt_rate * w_hw + l1_telemetry.interrupt_rate * w_l1,
                fan_speed_percent=max(base.fan_speed_percent, l1_telemetry.fan_speed_percent),
            )

        # Extract L2 modulation factors
        l2_factors: Dict[str, float] = {}
        if self._last_l2 is not None:
            l2_factors = adapt_l2_to_telemetry(self._last_l2)

        return base, l2_factors

    def adapt_snn_pad(
        self,
        intero_pad: InteroceptivePAD,
        l2_factors: Optional[Dict[str, float]] = None,
    ) -> PADVector:
        """Adapt SNN-derived PAD with L2 modulation.

        The SNN provides raw interoceptive PAD. L2 perception modulates
        how this internal state is expressed outward.
        """
        base_pad = adapt_interoceptive_pad(intero_pad)

        if l2_factors is not None:
            # L2 interaction quality modulates pleasure
            interaction_mod = l2_factors.get("interaction_valence", 0.0) * 0.2
            # Novelty boosts arousal
            novelty_mod = l2_factors.get("novelty", 0.0) * 0.15
            # Attention affects dominance
            attention_mod = (l2_factors.get("attention_level", 0.5) - 0.5) * 0.1

            return PADVector(
                pleasure=max(-1.0, min(1.0, base_pad.pleasure + interaction_mod)),
                arousal=max(-1.0, min(1.0, base_pad.arousal + novelty_mod)),
                dominance=max(-1.0, min(1.0, base_pad.dominance + attention_mod)),
            )

        return base_pad

    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "adaptation_count": self._adaptation_count,
            "has_l1": self._last_l1 is not None,
            "has_l2": self._last_l2 is not None,
            "has_hardware": self._last_telemetry is not None,
            "weights": {
                "l1": self.l1_weight,
                "l2": self.l2_weight,
                "hardware": self.hardware_weight,
            },
        }


# === Factory ===

def create_interoception_adapter(
    l1_weight: float = 0.4,
    l2_weight: float = 0.3,
    hardware_weight: float = 0.3,
) -> InteroceptionAdapter:
    """Create an InteroceptionAdapter with specified weights."""
    return InteroceptionAdapter(
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        hardware_weight=hardware_weight,
    )
