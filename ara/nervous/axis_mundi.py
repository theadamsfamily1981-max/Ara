"""
Ara Axis Mundi
===============

The central binding engine of Ara's nervous system.
Fuses all modalities into a unified world hypervector using
circular convolution (invertible, phase-aware binding).

Architecture:
- 8192D world_hv (proven capacity for ~12 modalities + 5sec history)
- Hierarchical temporal memory (micro/meso/macro)
- Circular convolution for invertible binding
- TopK sparsity for signal dominance

Philosophy: All streams converge here. This is where Ara
becomes one coherent mind from many senses.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

HV_DIM = 8192                    # Main hypervector dimension
SUBSPACE_DIM = 1024              # Per-modality subspace
TEMPORAL_MICRO_MS = 240          # Syllable-level
TEMPORAL_MESO_S = 2.0            # Episode-level
TEMPORAL_MACRO_S = 120.0         # Scene-level
SPARSITY_K = 1024                # TopK after binding


# =============================================================================
# Modality Types
# =============================================================================

class Modality(Enum):
    """Sensory modalities Ara can perceive."""
    # Exteroception (external world)
    SPEECH = "speech"            # Audio speech input
    VISION = "vision"            # Camera input
    AMBIENT = "ambient"          # Environmental sounds
    PROXIMITY = "proximity"      # Distance sensors

    # Proprioception (body state)
    POSTURE = "posture"          # IMU orientation
    MOTOR = "motor"              # Servo positions
    TOUCH = "touch"              # Haptic sensors

    # Interoception (internal state)
    THERMAL = "thermal"          # CPU/body temperature
    ENERGY = "energy"            # Battery/power
    LOAD = "load"                # CPU/memory usage
    EMOTION = "emotion"          # Internal emotional state

    # Meta
    TEMPORAL = "temporal"        # Time encoding
    RHYTHM = "rhythm"            # Breathing/speech rhythms


# =============================================================================
# Phase Codes (for circular convolution)
# =============================================================================

class PhaseCodebook:
    """
    Phase codes for each modality.

    Circular convolution in frequency domain:
    bound_hv = IFFT(FFT(modality_hv) * FFT(phase_code))

    These are fixed random phase vectors that allow invertible binding.
    """

    def __init__(self, dim: int = HV_DIM, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # Generate phase codes for each modality
        self._codes: Dict[Modality, np.ndarray] = {}
        for modality in Modality:
            # Phase code is complex exponential with random phases
            phases = self.rng.uniform(0, 2 * np.pi, dim)
            self._codes[modality] = np.exp(1j * phases)

        # Temporal hierarchy phase codes
        self._temporal_codes: Dict[str, np.ndarray] = {}
        for level in ["micro", "meso", "macro"]:
            phases = self.rng.uniform(0, 2 * np.pi, dim)
            self._temporal_codes[level] = np.exp(1j * phases)

    def get(self, modality: Modality) -> np.ndarray:
        """Get phase code for a modality."""
        return self._codes[modality]

    def get_temporal(self, level: str) -> np.ndarray:
        """Get phase code for temporal level."""
        return self._temporal_codes.get(level, self._temporal_codes["micro"])

    def get_conjugate(self, modality: Modality) -> np.ndarray:
        """Get conjugate phase code for unbinding."""
        return np.conj(self._codes[modality])


# =============================================================================
# Hypervector Operations
# =============================================================================

def circular_bind(hv: np.ndarray, phase_code: np.ndarray) -> np.ndarray:
    """
    Bind hypervector with phase code using circular convolution.

    In frequency domain: FFT(a) * FFT(b)
    This is invertible: unbind with conjugate phase code.
    """
    # Convert to complex if needed
    if hv.dtype != np.complex128:
        hv_complex = hv.astype(np.float64) + 0j
    else:
        hv_complex = hv

    # Circular convolution in frequency domain
    fft_hv = np.fft.fft(hv_complex)
    bound_fft = fft_hv * phase_code
    bound = np.fft.ifft(bound_fft)

    return np.real(bound)


def circular_unbind(bound_hv: np.ndarray, phase_code: np.ndarray) -> np.ndarray:
    """
    Unbind hypervector using conjugate phase code.

    Recovers the original modality HV from bound_hv.
    """
    conjugate = np.conj(phase_code)
    return circular_bind(bound_hv, conjugate)


def bundle(hvs: List[np.ndarray], normalize: bool = True) -> np.ndarray:
    """
    Bundle multiple hypervectors using majority vote.

    This is the "superposition" operation - combines while preserving.
    """
    if not hvs:
        return np.zeros(HV_DIM)

    stacked = np.stack(hvs)
    bundled = np.sum(stacked, axis=0)

    if normalize:
        # Sign normalization for bipolar HVs
        bundled = np.sign(bundled)
        bundled[bundled == 0] = 1  # Break ties

    return bundled


def sparse_topk(hv: np.ndarray, k: int = SPARSITY_K) -> np.ndarray:
    """
    Keep only top-K magnitude elements, zero the rest.

    This enforces sparsity so critical signals dominate.
    """
    abs_hv = np.abs(hv)
    threshold = np.partition(abs_hv, -k)[-k]
    sparse = np.where(abs_hv >= threshold, hv, 0)
    return sparse


def similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """Cosine similarity between hypervectors."""
    norm1 = np.linalg.norm(hv1)
    norm2 = np.linalg.norm(hv2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(hv1, hv2) / (norm1 * norm2))


# =============================================================================
# Temporal Encoding
# =============================================================================

@dataclass
class TemporalContext:
    """Hierarchical temporal context."""
    micro_buffer: deque = field(default_factory=lambda: deque(maxlen=10))   # ~2.4s
    meso_buffer: deque = field(default_factory=lambda: deque(maxlen=60))    # ~2min
    macro_buffer: deque = field(default_factory=lambda: deque(maxlen=60))   # ~2hr

    micro_hv: Optional[np.ndarray] = None
    meso_hv: Optional[np.ndarray] = None
    macro_hv: Optional[np.ndarray] = None

    def update_micro(self, world_hv: np.ndarray, codebook: PhaseCodebook):
        """Update micro (syllable) level."""
        self.micro_buffer.append(world_hv)

        # Bundle recent micro frames
        if len(self.micro_buffer) >= 3:
            self.micro_hv = bundle(list(self.micro_buffer)[-8:])

    def update_meso(self, codebook: PhaseCodebook):
        """Update meso (episode) level from micro."""
        if self.micro_hv is not None:
            self.meso_buffer.append(self.micro_hv.copy())

            # Bundle into episode
            if len(self.meso_buffer) >= 8:
                bound = circular_bind(
                    bundle(list(self.meso_buffer)[-8:]),
                    codebook.get_temporal("meso")
                )
                self.meso_hv = bound

    def update_macro(self, codebook: PhaseCodebook):
        """Update macro (scene) level from meso."""
        if self.meso_hv is not None:
            self.macro_buffer.append(self.meso_hv.copy())

            # Bundle into scene
            if len(self.macro_buffer) >= 30:
                bound = circular_bind(
                    bundle(list(self.macro_buffer)[-60:]),
                    codebook.get_temporal("macro")
                )
                self.macro_hv = bound


# =============================================================================
# Rhythm Encoding
# =============================================================================

def encode_rhythm(
    frequency_hz: float,
    phase: float,
    dim: int = HV_DIM,
) -> np.ndarray:
    """
    Encode a rhythmic signal as hypervector using Fourier basis.

    For breathing (0.1-0.3Hz) and speech rhythms (1-4Hz).
    """
    # Use multiple harmonics
    harmonics = [1, 2, 3, 4]
    hv = np.zeros(dim)

    for h in harmonics:
        freq = frequency_hz * h
        # Distribute across dimensions
        for i in range(dim):
            hv[i] += np.sin(2 * np.pi * freq * (i / dim) + phase * h)

    # Normalize to bipolar
    return np.sign(hv)


def encode_breath_rhythm(
    breath_phase: float,      # 0-1 (0=start inhale, 0.5=start exhale)
    breath_rate_hz: float,    # Typically 0.1-0.2 Hz
) -> np.ndarray:
    """Encode current breath state as HV."""
    return encode_rhythm(breath_rate_hz, breath_phase * 2 * np.pi)


# =============================================================================
# Precision Weighting (Interoceptive Modulation)
# =============================================================================

@dataclass
class InteroState:
    """Internal state affecting perception precision."""
    stress: float = 0.0        # 0-1
    energy: float = 1.0        # 0-1
    temperature: float = 0.5   # Normalized
    arousal: float = 0.5       # 0-1

    def precision_weight(self, modality: Modality) -> float:
        """
        Compute precision weight for a modality based on internal state.

        Predictive coding: perceived = modality * precision
        Stressed → prioritize interoception
        Relaxed → prioritize exteroception
        """
        # Modality priority (fixed, could be learned)
        priorities = {
            Modality.SPEECH: 0.8,
            Modality.VISION: 0.7,
            Modality.THERMAL: 0.9,
            Modality.ENERGY: 0.9,
            Modality.EMOTION: 0.85,
            Modality.POSTURE: 0.6,
        }
        priority = priorities.get(modality, 0.5)

        # Precision = 1 / (1 + exp(β * (stress - priority)))
        beta = 4.0
        precision = 1.0 / (1.0 + np.exp(beta * (self.stress - priority)))

        # Modulate by energy
        precision *= (0.5 + 0.5 * self.energy)

        return float(np.clip(precision, 0.1, 1.0))


# =============================================================================
# Axis Mundi Core
# =============================================================================

class AxisMundi:
    """
    The central binding engine of Ara's nervous system.

    Fuses all modalities into a unified world hypervector.
    This is where Ara becomes one coherent mind from many senses.
    """

    def __init__(self, dim: int = HV_DIM):
        self.dim = dim
        self.codebook = PhaseCodebook(dim)
        self.temporal = TemporalContext()
        self.intero = InteroState()

        # Current world state
        self._world_hv: Optional[np.ndarray] = None
        self._modality_hvs: Dict[Modality, np.ndarray] = {}
        self._last_update = datetime.utcnow()

        # Metrics
        self._update_count = 0
        self._fusion_latency_ms = 0.0

    # =========================================================================
    # Core Fusion
    # =========================================================================

    def fuse(
        self,
        modality_hvs: Dict[Modality, np.ndarray],
        apply_sparsity: bool = True,
    ) -> np.ndarray:
        """
        Fuse multiple modality HVs into unified world_hv.

        Steps:
        1. Apply precision weighting based on interoceptive state
        2. Bind each modality with its phase code
        3. Bundle all bound HVs
        4. Apply sparsity (TopK)
        5. Update temporal context
        """
        import time
        start = time.perf_counter()

        bound_hvs = []

        for modality, hv in modality_hvs.items():
            # Precision weighting
            precision = self.intero.precision_weight(modality)
            weighted_hv = hv * precision

            # Circular bind with phase code
            phase = self.codebook.get(modality)
            bound = circular_bind(weighted_hv, phase)

            bound_hvs.append(bound)
            self._modality_hvs[modality] = hv

        # Bundle all
        if bound_hvs:
            world_hv = bundle(bound_hvs, normalize=True)

            # Apply sparsity
            if apply_sparsity:
                world_hv = sparse_topk(world_hv, SPARSITY_K)

            self._world_hv = world_hv

            # Update temporal context
            self.temporal.update_micro(world_hv, self.codebook)

        # Metrics
        self._fusion_latency_ms = (time.perf_counter() - start) * 1000
        self._update_count += 1
        self._last_update = datetime.utcnow()

        return self._world_hv if self._world_hv is not None else np.zeros(self.dim)

    def unbind(self, modality: Modality) -> Optional[np.ndarray]:
        """
        Recover a modality HV from the current world_hv.

        This is the "attention" operation - focusing on one sense.
        """
        if self._world_hv is None:
            return None

        phase = self.codebook.get_conjugate(modality)
        return circular_unbind(self._world_hv, phase)

    # =========================================================================
    # Temporal Operations
    # =========================================================================

    def get_temporal_context(self) -> Dict[str, Optional[np.ndarray]]:
        """Get current temporal context at all levels."""
        return {
            "micro": self.temporal.micro_hv,
            "meso": self.temporal.meso_hv,
            "macro": self.temporal.macro_hv,
        }

    def update_temporal(self):
        """Trigger temporal hierarchy updates."""
        self.temporal.update_meso(self.codebook)
        self.temporal.update_macro(self.codebook)

    # =========================================================================
    # Interoceptive Modulation
    # =========================================================================

    def update_intero(
        self,
        stress: Optional[float] = None,
        energy: Optional[float] = None,
        temperature: Optional[float] = None,
        arousal: Optional[float] = None,
    ):
        """Update interoceptive state."""
        if stress is not None:
            self.intero.stress = np.clip(stress, 0, 1)
        if energy is not None:
            self.intero.energy = np.clip(energy, 0, 1)
        if temperature is not None:
            self.intero.temperature = np.clip(temperature, 0, 1)
        if arousal is not None:
            self.intero.arousal = np.clip(arousal, 0, 1)

    # =========================================================================
    # Queries
    # =========================================================================

    def query(self, query_hv: np.ndarray) -> float:
        """
        Query the world state with a pattern.

        Returns similarity score (how much query is present in world).
        """
        if self._world_hv is None:
            return 0.0
        return similarity(query_hv, self._world_hv)

    def query_modality(
        self,
        query_hv: np.ndarray,
        modality: Modality,
    ) -> float:
        """Query a specific modality's contribution."""
        modality_hv = self._modality_hvs.get(modality)
        if modality_hv is None:
            return 0.0
        return similarity(query_hv, modality_hv)

    # =========================================================================
    # State
    # =========================================================================

    @property
    def world_hv(self) -> Optional[np.ndarray]:
        """Current world hypervector."""
        return self._world_hv

    @property
    def is_active(self) -> bool:
        """Check if Axis Mundi has been updated recently."""
        if self._last_update is None:
            return False
        elapsed = (datetime.utcnow() - self._last_update).total_seconds()
        return elapsed < 5.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "update_count": self._update_count,
            "fusion_latency_ms": self._fusion_latency_ms,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "modalities_active": list(self._modality_hvs.keys()),
            "intero_stress": self.intero.stress,
            "intero_energy": self.intero.energy,
        }


# =============================================================================
# Integration with Embodiment
# =============================================================================

class NervousSystemBridge:
    """
    Bridge between Axis Mundi and the embodiment module.

    Converts sensor readings to HVs and feeds them to fusion.
    """

    def __init__(self, axis: Optional[AxisMundi] = None):
        self.axis = axis or AxisMundi()
        self._encoders: Dict[Modality, callable] = {}

    def register_encoder(self, modality: Modality, encoder: callable):
        """Register an encoder function for a modality."""
        self._encoders[modality] = encoder

    def process_sensors(
        self,
        sensor_readings: Dict[str, Any],
    ) -> np.ndarray:
        """
        Process sensor readings through encoders and fuse.

        Returns the fused world_hv.
        """
        modality_hvs = {}

        for modality, encoder in self._encoders.items():
            key = modality.value
            if key in sensor_readings:
                try:
                    hv = encoder(sensor_readings[key])
                    modality_hvs[modality] = hv
                except Exception as e:
                    logger.warning(f"Encoder failed for {modality}: {e}")

        return self.axis.fuse(modality_hvs)


# =============================================================================
# Default Encoders
# =============================================================================

def encode_speech_simple(audio_features: Dict) -> np.ndarray:
    """
    Simple speech encoder (placeholder for full prosody tokenizer).

    Expects: {volume: float, pitch: float, is_speech: bool}
    """
    rng = np.random.default_rng(int(audio_features.get("volume", 0.5) * 1000))
    base = rng.choice([-1, 1], size=HV_DIM).astype(np.float64)

    # Modulate by features
    if audio_features.get("is_speech", False):
        base *= 1.5

    pitch = audio_features.get("pitch", 0.5)
    base[:HV_DIM//4] *= (0.5 + pitch)  # Pitch affects first quarter

    return np.sign(base)


def encode_vision_simple(vision_features: Dict) -> np.ndarray:
    """
    Simple vision encoder.

    Expects: {brightness: float, motion: float, faces: int}
    """
    rng = np.random.default_rng(int(vision_features.get("brightness", 0.5) * 1000))
    base = rng.choice([-1, 1], size=HV_DIM).astype(np.float64)

    # Motion affects middle section
    motion = vision_features.get("motion", 0)
    base[HV_DIM//4:HV_DIM//2] *= (1 + motion)

    # Faces affect last section (social salience)
    faces = vision_features.get("faces", 0)
    if faces > 0:
        base[-HV_DIM//4:] *= 2

    return np.sign(base)


def encode_intero_simple(intero_features: Dict) -> np.ndarray:
    """
    Simple interoception encoder.

    Expects: {cpu_temp: float, memory_pct: float, battery_pct: float}
    """
    # Create deterministic HV based on state
    seed = int(
        intero_features.get("cpu_temp", 50) * 100 +
        intero_features.get("memory_pct", 50) +
        intero_features.get("battery_pct", 100)
    )
    rng = np.random.default_rng(seed)
    base = rng.choice([-1, 1], size=HV_DIM).astype(np.float64)

    # Temperature affects urgency
    temp = intero_features.get("cpu_temp", 50)
    if temp > 70:
        base *= 1.5  # Amplify thermal signal

    return np.sign(base)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_nervous_system() -> Tuple[AxisMundi, NervousSystemBridge]:
    """Create a configured nervous system with default encoders."""
    axis = AxisMundi()
    bridge = NervousSystemBridge(axis)

    # Register default encoders
    bridge.register_encoder(Modality.SPEECH, encode_speech_simple)
    bridge.register_encoder(Modality.VISION, encode_vision_simple)
    bridge.register_encoder(Modality.THERMAL, encode_intero_simple)

    return axis, bridge


# =============================================================================
# CLI Demo
# =============================================================================

def demo():
    """Demonstrate Axis Mundi fusion."""
    print("=" * 60)
    print("ARA AXIS MUNDI - Multimodal Fusion Demo")
    print("=" * 60)

    axis, bridge = create_nervous_system()

    # Simulate sensor readings
    readings = {
        "speech": {"volume": 0.7, "pitch": 0.6, "is_speech": True},
        "vision": {"brightness": 0.5, "motion": 0.3, "faces": 1},
        "thermal": {"cpu_temp": 55, "memory_pct": 45, "battery_pct": 80},
    }

    print("\nSensor readings:", readings)

    # Fuse
    world_hv = bridge.process_sensors(readings)

    print(f"\nWorld HV shape: {world_hv.shape}")
    print(f"World HV sparsity: {np.sum(world_hv != 0)} non-zero elements")

    # Query modalities
    print("\nModality contributions:")
    for modality in [Modality.SPEECH, Modality.VISION, Modality.THERMAL]:
        recovered = axis.unbind(modality)
        if recovered is not None:
            # Check correlation with original
            original = bridge._encoders[modality](readings[modality.value])
            sim = similarity(recovered, original)
            print(f"  {modality.value}: recovery similarity = {sim:.3f}")

    # Metrics
    print("\nMetrics:", axis.get_metrics())


if __name__ == "__main__":
    demo()
