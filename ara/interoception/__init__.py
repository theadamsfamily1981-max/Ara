"""
Multi-Modal Spiking Interoception Core with L1/L2 Closed-Loop

Generates internal affective state (L1/L2 homeostatic signals) directly from
raw sensory data using SNN membrane potential as the TRUE PAD signal.

Key Features:
1. L1 Body Layer - Physical/somatic signals (heart rate, breath, muscle tension)
2. L2 Perception Layer - Sensory processing (prosody, semantic, visual)
3. L1/L2 Closed Loop - L3 metacontrol feeds back to modulate L1/L2 populations
4. True Interoception - LIF membrane potential (v) as real-time affect
5. Trainable Populations - AEPO-driven parameter optimization (tau, v_th)

The core's internal LIF firing rate and membrane state BECOMES the PAD signal,
replacing external EmotionPrediction with true interoceptive control.

Architecture:
    L3 Metacontrol ←→ L2 Perception ←→ L1 Body
           ↓              ↓              ↓
        [PAD]         [Features]     [Somatic]
           ↓              ↓              ↓
    ┌─────────────────────────────────────────┐
    │         Interoception Core              │
    │   (Valence/Arousal/Dominance SNNs)      │
    └─────────────────────────────────────────┘

Usage:
    from ara.interoception import InteroceptionCore, L1BodyState, L2PerceptionState

    core = InteroceptionCore()

    # Process with L1/L2 integration
    pad_state = core.process_with_layers(
        l1_body=L1BodyState(heart_rate=72, breath_rate=16),
        l2_perception=L2PerceptionState(audio_features=prosody),
    )

    # Receive L3 feedback
    core.receive_l3_feedback(temp_mult=0.9, mem_mult=1.1, attention_gain=0.8)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime
from enum import Enum
import logging
import math
import time
import pickle

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("ara.interoception")

# Try numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class SensoryModality(str, Enum):
    """Supported sensory modalities."""
    AUDIO = "audio"       # Voice prosody
    TEXT = "text"         # Semantic content
    VISUAL = "visual"     # Facial/body cues
    INTERNAL = "internal" # Internal state feedback


# ============================================================================
# L1 BODY LAYER - Physical/Somatic Signals
# ============================================================================

@dataclass
class L1BodyState:
    """
    L1 Body Layer - Physical/somatic state.

    Represents internal body signals that drive low-level homeostatic responses.
    These are the "gut feelings" that influence affect.
    """
    # Cardiovascular
    heart_rate: float = 72.0         # BPM
    heart_rate_variability: float = 50.0  # ms RMSSD
    blood_pressure_sys: float = 120.0
    blood_pressure_dia: float = 80.0

    # Respiratory
    breath_rate: float = 16.0        # Breaths per minute
    breath_depth: float = 0.5        # Normalized 0-1

    # Muscular
    muscle_tension: float = 0.3      # Normalized 0-1
    posture_openness: float = 0.5    # Closed (0) to open (1)

    # Electrodermal
    skin_conductance: float = 2.0    # µS
    skin_temperature: float = 33.0   # Celsius

    # Derived metrics
    stress_index: float = 0.5        # Computed from HRV + conductance
    energy_level: float = 0.5        # Computed from HR + breath

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def compute_derived(self):
        """Compute derived metrics from raw signals."""
        # Stress index: low HRV + high conductance = stress
        hrv_norm = max(0, min(1, self.heart_rate_variability / 100))
        self.stress_index = (1 - hrv_norm) * 0.6 + min(1, self.skin_conductance / 10) * 0.4

        # Energy level: moderate HR + deep breath = high energy
        hr_norm = max(0, min(1, (self.heart_rate - 50) / 100))
        self.energy_level = hr_norm * 0.5 + self.breath_depth * 0.5

    def to_current_vector(self) -> Tuple[float, float, float]:
        """
        Convert to input currents for SNN populations.

        Returns (valence_current, arousal_current, dominance_current)
        """
        self.compute_derived()

        # Valence: Low stress + moderate HR = positive
        valence_current = (1 - self.stress_index) * 0.7 + (1 - abs(self.heart_rate - 72) / 50) * 0.3

        # Arousal: High HR + high conductance = high arousal
        hr_arousal = max(0, min(1, (self.heart_rate - 60) / 60))
        arousal_current = hr_arousal * 0.5 + min(1, self.skin_conductance / 5) * 0.5

        # Dominance: Open posture + low tension = high dominance
        dominance_current = self.posture_openness * 0.6 + (1 - self.muscle_tension) * 0.4

        return valence_current, arousal_current, dominance_current

    def to_dict(self) -> Dict[str, Any]:
        return {
            "heart_rate": self.heart_rate,
            "hrv": self.heart_rate_variability,
            "breath_rate": self.breath_rate,
            "muscle_tension": self.muscle_tension,
            "skin_conductance": self.skin_conductance,
            "stress_index": self.stress_index,
            "energy_level": self.energy_level,
        }


# ============================================================================
# L2 PERCEPTION LAYER - Sensory Processing
# ============================================================================

@dataclass
class L2PerceptionState:
    """
    L2 Perception Layer - Processed sensory state.

    Represents higher-level perception of external stimuli
    that influences affect through appraisal.
    """
    # Audio features
    audio_valence: float = 0.0       # Prosody-derived valence
    audio_arousal: float = 0.5       # Prosody-derived arousal
    audio_dominance: float = 0.5
    speech_rate: float = 1.0         # Words per second
    pitch_mean: float = 150.0        # Hz
    pitch_variance: float = 20.0     # Hz

    # Text/semantic features
    text_sentiment: float = 0.0      # -1 to 1
    text_arousal: float = 0.5
    semantic_coherence: float = 0.8  # How coherent is the conversation

    # Visual features (if available)
    facial_valence: float = 0.0
    facial_arousal: float = 0.5
    gaze_engagement: float = 0.5     # How engaged is the user

    # Attention/salience
    attention_focus: float = 0.8     # How focused on the task
    novelty_signal: float = 0.2      # How novel is the current input

    # Uncertainty
    perception_confidence: float = 0.7

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def from_audio_features(cls, features: Dict) -> "L2PerceptionState":
        """Create from raw audio features dict."""
        pitch = features.get("pitch_mean", 150)
        energy = features.get("energy", 0.5)
        rate = features.get("speaking_rate", 1.0)
        pitch_var = features.get("pitch_variance", 20)

        # Derive valence/arousal from prosody
        valence = 0.5 * (pitch_var / 50) + 0.3 * (1 - abs(energy - 0.5))
        arousal = 0.6 * energy + 0.4 * min(1, rate / 1.5)
        dominance = 0.5 * (1 - pitch / 300) + 0.5 * (1 - pitch_var / 50)

        return cls(
            audio_valence=valence,
            audio_arousal=arousal,
            audio_dominance=dominance,
            speech_rate=rate,
            pitch_mean=pitch,
            pitch_variance=pitch_var,
        )

    @classmethod
    def from_text_embedding(cls, embedding: List[float]) -> "L2PerceptionState":
        """Create from text embedding."""
        if not embedding:
            return cls()

        mean_val = sum(embedding) / len(embedding)
        max_abs = max(abs(x) for x in embedding) if embedding else 1

        return cls(
            text_sentiment=mean_val / (max_abs + 1e-8),
            text_arousal=min(1, sum(abs(x) for x in embedding[:10]) / 5),
            semantic_coherence=0.8,
        )

    def to_current_vector(self) -> Tuple[float, float, float]:
        """
        Convert to input currents for SNN populations.

        Returns (valence_current, arousal_current, dominance_current)
        """
        # Weighted combination of modalities
        valence_current = (
            self.audio_valence * 0.4 +
            self.text_sentiment * 0.3 +
            self.facial_valence * 0.3
        )

        arousal_current = (
            self.audio_arousal * 0.5 +
            self.text_arousal * 0.3 +
            self.novelty_signal * 0.2
        )

        dominance_current = (
            self.audio_dominance * 0.4 +
            self.gaze_engagement * 0.3 +
            self.perception_confidence * 0.3
        )

        return valence_current, arousal_current, dominance_current

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_valence": self.audio_valence,
            "audio_arousal": self.audio_arousal,
            "text_sentiment": self.text_sentiment,
            "attention_focus": self.attention_focus,
            "novelty_signal": self.novelty_signal,
            "perception_confidence": self.perception_confidence,
        }


# ============================================================================
# L3 FEEDBACK - Metacontrol Modulation
# ============================================================================

@dataclass
class L3Feedback:
    """
    Feedback from L3 Metacontrol to modulate L1/L2 processing.

    This closes the loop by allowing higher-level control to
    influence lower-level processing.
    """
    # Modulation gains
    temperature_multiplier: float = 1.0  # LLM temperature mod
    memory_multiplier: float = 1.0       # Memory retrieval mod
    attention_gain: float = 1.0          # Attention focus mod

    # Population modulation (directly affects SNN parameters)
    tau_modifier: float = 0.0            # Add to tau (slows/speeds neurons)
    threshold_modifier: float = 0.0       # Add to threshold (harder/easier to spike)

    # Homeostatic goal adjustment
    goal_valence_shift: float = 0.0      # Shift target valence
    goal_arousal_shift: float = 0.0      # Shift target arousal

    # Gating
    suppress_arousal: bool = False       # Emergency arousal suppression
    boost_attention: bool = False        # Emergency attention boost

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temp_mult": self.temperature_multiplier,
            "mem_mult": self.memory_multiplier,
            "attn_gain": self.attention_gain,
            "tau_mod": self.tau_modifier,
            "thresh_mod": self.threshold_modifier,
        }


@dataclass
class SensoryStream:
    """Raw sensory input stream."""
    modality: SensoryModality
    timestamp: float  # Unix timestamp
    features: List[float]  # Feature vector
    confidence: float = 1.0
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "timestamp": self.timestamp,
            "features": self.features[:10] if len(self.features) > 10 else self.features,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
        }


@dataclass
class LIFState:
    """Leaky Integrate-and-Fire neuron state."""
    membrane_potential: float = 0.0  # v(t)
    spike_count: int = 0
    last_spike_time: float = 0.0
    refractory_remaining: float = 0.0

    # Learnable parameters
    tau: float = 10.0      # Membrane time constant (ms)
    v_threshold: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": self.membrane_potential,
            "spikes": self.spike_count,
            "tau": self.tau,
            "threshold": self.v_threshold,
        }


@dataclass
class InteroceptivePAD:
    """
    Internal PAD state derived from SNN membrane dynamics.

    This is the TRUE affect signal - generated from internal neural state,
    not external prediction.
    """
    # Core PAD
    valence: float = 0.0     # Pleasure/displeasure from v_mean
    arousal: float = 0.5     # Activation from firing rate
    dominance: float = 0.5   # Control from population sync

    # Derived from LIF dynamics
    membrane_mean: float = 0.0
    membrane_variance: float = 0.0
    firing_rate: float = 0.0
    population_sync: float = 0.0

    # Homeostatic error
    homeostatic_error: float = 0.0
    goal_valence: float = 0.0
    goal_arousal: float = 0.5

    # Confidence based on signal quality
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "membrane_mean": self.membrane_mean,
            "firing_rate": self.firing_rate,
            "homeostatic_error": self.homeostatic_error,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class SpikeRateEncoder:
    """
    Encode continuous values to spike rates.

    Maps input features to Poisson spike rates for SNN processing.
    """

    def __init__(
        self,
        max_rate: float = 100.0,  # Max spikes per second
        time_step_ms: float = 1.0,
    ):
        self.max_rate = max_rate
        self.time_step = time_step_ms / 1000.0  # Convert to seconds

    def encode(self, value: float, duration_ms: float = 10.0) -> List[int]:
        """
        Encode value to spike train.

        Args:
            value: Input value [0, 1]
            duration_ms: Duration in milliseconds

        Returns:
            List of spike times (1 = spike, 0 = no spike)
        """
        # Compute rate
        rate = self.max_rate * max(0, min(1, value))
        prob_per_step = rate * self.time_step

        # Generate spikes
        num_steps = int(duration_ms / (self.time_step * 1000))
        if NUMPY_AVAILABLE:
            spikes = (np.random.random(num_steps) < prob_per_step).astype(int).tolist()
        else:
            import random
            spikes = [1 if random.random() < prob_per_step else 0 for _ in range(num_steps)]

        return spikes


class InteroceptionPopulation:
    """
    Population of LIF neurons for interoceptive processing.

    Each population processes one aspect of sensory input and
    contributes to the overall PAD state.

    Supports L3 feedback modulation of tau and threshold.
    """

    def __init__(
        self,
        size: int = 64,
        tau_range: Tuple[float, float] = (5.0, 20.0),
        threshold_range: Tuple[float, float] = (0.8, 1.2),
        name: str = "population",
    ):
        self.size = size
        self.name = name
        self.tau_range = tau_range
        self.threshold_range = threshold_range

        # Initialize neurons with varied parameters
        if NUMPY_AVAILABLE:
            self.base_tau = np.random.uniform(tau_range[0], tau_range[1], size)
            self.base_v_th = np.random.uniform(threshold_range[0], threshold_range[1], size)
            self.tau = self.base_tau.copy()
            self.v_th = self.base_v_th.copy()
            self.v = np.zeros(size)
            self.spikes = np.zeros(size)
        else:
            import random
            self.base_tau = [random.uniform(tau_range[0], tau_range[1]) for _ in range(size)]
            self.base_v_th = [random.uniform(threshold_range[0], threshold_range[1]) for _ in range(size)]
            self.tau = self.base_tau.copy()
            self.v_th = self.base_v_th.copy()
            self.v = [0.0] * size
            self.spikes = [0] * size

        self.spike_history = []
        self.membrane_history = []
        self.dt = 1.0  # Time step in ms

        # L3 modulation state
        self.tau_modifier = 0.0
        self.threshold_modifier = 0.0

    def apply_l3_modulation(self, tau_mod: float = 0.0, thresh_mod: float = 0.0):
        """
        Apply L3 feedback modulation to population parameters.

        Args:
            tau_mod: Additive modifier to tau (positive = slower)
            thresh_mod: Additive modifier to threshold (positive = harder to spike)
        """
        self.tau_modifier = tau_mod
        self.threshold_modifier = thresh_mod

        if NUMPY_AVAILABLE:
            self.tau = np.clip(self.base_tau + tau_mod, 1.0, 100.0)
            self.v_th = np.clip(self.base_v_th + thresh_mod, 0.1, 5.0)
        else:
            self.tau = [max(1.0, min(100.0, t + tau_mod)) for t in self.base_tau]
            self.v_th = [max(0.1, min(5.0, v + thresh_mod)) for v in self.base_v_th]

    def step(self, current: float, suppress: bool = False) -> Tuple[float, float]:
        """
        Advance one time step with input current.

        Args:
            current: Input current to all neurons
            suppress: If True, clamp firing (emergency suppression)

        Returns:
            (mean_voltage, firing_rate)
        """
        if NUMPY_AVAILABLE:
            # LIF dynamics: dv/dt = -(v - v_rest)/tau + I
            dv = (-self.v / self.tau + current) * self.dt
            self.v += dv

            # Check for spikes
            if suppress:
                self.spikes = np.zeros(self.size)
            else:
                self.spikes = (self.v >= self.v_th).astype(float)
                spike_mask = self.spikes > 0
                self.v[spike_mask] = 0.0  # Reset

            # Track history
            self.spike_history.append(self.spikes.sum())
            self.membrane_history.append(float(self.v.mean()))

            return float(self.v.mean()), float(self.spikes.mean())
        else:
            # Simple fallback
            for i in range(self.size):
                dv = (-self.v[i] / self.tau[i] + current) * self.dt
                self.v[i] += dv
                if not suppress and self.v[i] >= self.v_th[i]:
                    self.spikes[i] = 1
                    self.v[i] = 0.0
                else:
                    self.spikes[i] = 0

            mean_v = sum(self.v) / self.size
            mean_spikes = sum(self.spikes) / self.size
            self.spike_history.append(sum(self.spikes))
            self.membrane_history.append(mean_v)

            return mean_v, mean_spikes

    def reset(self):
        """Reset population state."""
        if NUMPY_AVAILABLE:
            self.v = np.zeros(self.size)
            self.spikes = np.zeros(self.size)
        else:
            self.v = [0.0] * self.size
            self.spikes = [0] * self.size

    def get_statistics(self) -> Dict[str, float]:
        """Get population statistics."""
        if NUMPY_AVAILABLE:
            recent_spikes = self.spike_history[-100:] if self.spike_history else [0]
            recent_membrane = self.membrane_history[-100:] if self.membrane_history else [0]
            return {
                "v_mean": float(self.v.mean()),
                "v_std": float(self.v.std()),
                "firing_rate": float(np.mean(recent_spikes)),
                "sync": float(np.std(recent_spikes)) if len(recent_spikes) > 10 else 0.0,
                "membrane_trend": float(np.mean(recent_membrane[-10:]) - np.mean(recent_membrane[:10])) if len(recent_membrane) > 10 else 0.0,
                "tau_mean": float(self.tau.mean()),
                "v_th_mean": float(self.v_th.mean()),
            }
        else:
            recent = self.spike_history[-100:] if self.spike_history else [0]
            return {
                "v_mean": sum(self.v) / self.size,
                "v_std": 0.1,
                "firing_rate": sum(recent) / len(recent) if recent else 0.0,
                "sync": 0.1,
                "membrane_trend": 0.0,
                "tau_mean": sum(self.tau) / len(self.tau),
                "v_th_mean": sum(self.v_th) / len(self.v_th),
            }

    def get_trainable_params(self) -> Dict[str, Any]:
        """Get parameters that can be trained via TP-RL."""
        if NUMPY_AVAILABLE:
            return {
                "tau": self.base_tau.tolist(),
                "v_th": self.base_v_th.tolist(),
            }
        return {
            "tau": self.base_tau,
            "v_th": self.base_v_th,
        }

    def set_trainable_params(self, params: Dict[str, Any]):
        """Set trainable parameters from TP-RL optimization."""
        if "tau" in params:
            if NUMPY_AVAILABLE:
                self.base_tau = np.array(params["tau"])
            else:
                self.base_tau = params["tau"]
        if "v_th" in params:
            if NUMPY_AVAILABLE:
                self.base_v_th = np.array(params["v_th"])
            else:
                self.base_v_th = params["v_th"]

        # Re-apply modulation
        self.apply_l3_modulation(self.tau_modifier, self.threshold_modifier)


class InteroceptionCore:
    """
    Multi-Modal Spiking Interoception Core with L1/L2 Closed-Loop.

    Generates PAD state from internal SNN dynamics rather than
    external prediction. This is TRUE interoceptive control.

    Architecture:
    - L1 Body Layer → Valence/Arousal/Dominance currents
    - L2 Perception Layer → Valence/Arousal/Dominance currents
    - L3 Feedback → Modulates tau/v_th parameters
    - Valence population: responds to positive/negative signals
    - Arousal population: responds to intensity/activation
    - Dominance population: responds to control/agency signals
    """

    def __init__(
        self,
        population_size: int = 64,
        homeostatic_gain: float = 0.1,
        temporal_window_ms: float = 100.0,
        l1_weight: float = 0.4,  # Weight for body signals
        l2_weight: float = 0.6,  # Weight for perception signals
    ):
        self.population_size = population_size
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

        # Create specialized populations
        self.valence_pop = InteroceptionPopulation(
            size=population_size,
            tau_range=(10.0, 30.0),  # Slower for valence
            name="valence",
        )
        self.arousal_pop = InteroceptionPopulation(
            size=population_size,
            tau_range=(3.0, 10.0),  # Faster for arousal
            name="arousal",
        )
        self.dominance_pop = InteroceptionPopulation(
            size=population_size,
            tau_range=(15.0, 40.0),  # Very slow for dominance
            name="dominance",
        )

        # Homeostatic parameters
        self.homeostatic_gain = homeostatic_gain
        self.goal_valence = 0.0
        self.goal_arousal = 0.5

        # Temporal alignment
        self.temporal_window = temporal_window_ms
        self.stream_buffer: Dict[SensoryModality, List[SensoryStream]] = {
            m: [] for m in SensoryModality
        }

        # Encoder
        self.encoder = SpikeRateEncoder()

        # History
        self.pad_history: List[InteroceptivePAD] = []

        # L3 Feedback state
        self.current_l3_feedback: Optional[L3Feedback] = None
        self.l3_feedback_history: List[L3Feedback] = []

        # L1/L2 state tracking
        self.current_l1_state: Optional[L1BodyState] = None
        self.current_l2_state: Optional[L2PerceptionState] = None

        logger.info(f"InteroceptionCore initialized with {population_size} neurons per population")

    def _extract_prosody_features(self, audio_features: Dict) -> Tuple[float, float, float]:
        """
        Extract valence/arousal/dominance signals from prosody.

        Args:
            audio_features: Dict with pitch, energy, rate, etc.

        Returns:
            (valence_input, arousal_input, dominance_input) as currents
        """
        # Extract features (normalize to 0-1 range)
        pitch = audio_features.get("pitch_mean", 150) / 300  # Normalize
        energy = audio_features.get("energy", 0.5)
        rate = audio_features.get("speaking_rate", 1.0)
        pitch_var = audio_features.get("pitch_variance", 0.1)

        # Map to PAD dimensions
        # Valence: Higher pitch variance + moderate energy → positive
        valence_input = 0.5 * pitch_var + 0.3 * (1 - abs(energy - 0.5))

        # Arousal: High energy + fast rate → high arousal
        arousal_input = 0.6 * energy + 0.4 * min(1, rate / 1.5)

        # Dominance: Lower pitch + steady energy → higher dominance
        dominance_input = 0.5 * (1 - pitch) + 0.5 * (1 - pitch_var)

        return valence_input, arousal_input, dominance_input

    def _extract_text_features(self, text_embedding: List[float]) -> Tuple[float, float, float]:
        """
        Extract PAD signals from text embedding.

        Uses embedding statistics as proxy for sentiment.
        """
        if not text_embedding:
            return 0.0, 0.5, 0.5

        if NUMPY_AVAILABLE:
            emb = np.array(text_embedding)
            mean_val = float(np.mean(emb))
            std_val = float(np.std(emb))
            max_val = float(np.max(np.abs(emb)))
        else:
            mean_val = sum(text_embedding) / len(text_embedding)
            std_val = 0.1
            max_val = max(abs(x) for x in text_embedding)

        # Simple heuristic mapping
        valence_input = 0.5 + mean_val * 0.5  # Center around 0.5
        arousal_input = min(1, std_val * 2)   # Variance → arousal
        dominance_input = 0.5  # Neutral for text

        return valence_input, arousal_input, dominance_input

    def process_sensory(
        self,
        audio_features: Optional[Dict] = None,
        text_embedding: Optional[List[float]] = None,
        visual_features: Optional[Dict] = None,
        num_steps: int = 10,
    ) -> InteroceptivePAD:
        """
        Process multi-modal sensory input and generate internal PAD state.

        This is the core interoception function - the SNN dynamics
        BECOME the affect signal.

        Args:
            audio_features: Prosody features (pitch, energy, rate)
            text_embedding: Text embedding vector
            visual_features: Visual/facial features (optional)
            num_steps: Simulation steps

        Returns:
            InteroceptivePAD state derived from SNN membrane dynamics
        """
        # Initialize currents
        val_current = 0.0
        aro_current = 0.0
        dom_current = 0.0

        # Process audio modality
        if audio_features:
            v, a, d = self._extract_prosody_features(audio_features)
            val_current += v * 0.6  # Audio weighted higher for arousal
            aro_current += a * 0.7
            dom_current += d * 0.4

        # Process text modality
        if text_embedding:
            v, a, d = self._extract_text_features(text_embedding)
            val_current += v * 0.4  # Text weighted for valence
            aro_current += a * 0.3
            dom_current += d * 0.6

        # Add homeostatic drive (push toward goal)
        homeostatic_valence = self.homeostatic_gain * (self.goal_valence - val_current)
        homeostatic_arousal = self.homeostatic_gain * (self.goal_arousal - aro_current)

        val_current += homeostatic_valence
        aro_current += homeostatic_arousal

        # Run SNN simulation
        for _ in range(num_steps):
            v_mean, v_rate = self.valence_pop.step(val_current)
            a_mean, a_rate = self.arousal_pop.step(aro_current)
            d_mean, d_rate = self.dominance_pop.step(dom_current)

        # Get population statistics
        val_stats = self.valence_pop.get_statistics()
        aro_stats = self.arousal_pop.get_statistics()
        dom_stats = self.dominance_pop.get_statistics()

        # Map SNN state to PAD
        # Valence: from membrane mean (negative = displeasure)
        valence = math.tanh(val_stats["v_mean"] * 2)

        # Arousal: from firing rate (more spikes = higher arousal)
        arousal = min(1.0, aro_stats["firing_rate"] / 0.3)  # Normalize

        # Dominance: from population synchrony (more sync = more control)
        dominance = 0.5 + 0.5 * math.tanh(dom_stats["v_mean"])

        # Compute homeostatic error
        homeostatic_error = math.sqrt(
            (valence - self.goal_valence) ** 2 +
            (arousal - self.goal_arousal) ** 2
        )

        # Confidence based on signal stability
        confidence = max(0.3, 1.0 - val_stats.get("v_std", 0) - aro_stats.get("v_std", 0))

        # Create PAD state
        pad = InteroceptivePAD(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            membrane_mean=(val_stats["v_mean"] + aro_stats["v_mean"] + dom_stats["v_mean"]) / 3,
            membrane_variance=val_stats.get("v_std", 0),
            firing_rate=(val_stats["firing_rate"] + aro_stats["firing_rate"] + dom_stats["firing_rate"]) / 3,
            population_sync=dom_stats.get("sync", 0),
            homeostatic_error=homeostatic_error,
            goal_valence=self.goal_valence,
            goal_arousal=self.goal_arousal,
            confidence=confidence,
        )

        self.pad_history.append(pad)

        logger.debug(f"Interoceptive PAD: V={valence:.2f}, A={arousal:.2f}, D={dominance:.2f}")

        return pad

    def set_homeostatic_goal(self, valence: float, arousal: float):
        """Set homeostatic goal state."""
        self.goal_valence = valence
        self.goal_arousal = arousal
        logger.info(f"Homeostatic goal set: V={valence:.2f}, A={arousal:.2f}")

    def receive_l3_feedback(
        self,
        temp_mult: float = 1.0,
        mem_mult: float = 1.0,
        attention_gain: float = 1.0,
        tau_modifier: float = 0.0,
        threshold_modifier: float = 0.0,
        goal_valence_shift: float = 0.0,
        goal_arousal_shift: float = 0.0,
        suppress_arousal: bool = False,
        boost_attention: bool = False,
    ):
        """
        Receive feedback from L3 Metacontrol to modulate L1/L2 processing.

        This CLOSES THE LOOP - L3 decisions affect future interoceptive processing.

        Args:
            temp_mult: LLM temperature multiplier (higher = more creative)
            mem_mult: Memory retrieval multiplier (higher = more context)
            attention_gain: Attention focus multiplier
            tau_modifier: Additive modifier to neuron time constants
            threshold_modifier: Additive modifier to spike thresholds
            goal_valence_shift: Adjust homeostatic valence target
            goal_arousal_shift: Adjust homeostatic arousal target
            suppress_arousal: Emergency arousal suppression
            boost_attention: Emergency attention boost
        """
        # Create feedback object
        feedback = L3Feedback(
            temperature_multiplier=temp_mult,
            memory_multiplier=mem_mult,
            attention_gain=attention_gain,
            tau_modifier=tau_modifier,
            threshold_modifier=threshold_modifier,
            goal_valence_shift=goal_valence_shift,
            goal_arousal_shift=goal_arousal_shift,
            suppress_arousal=suppress_arousal,
            boost_attention=boost_attention,
        )

        self.current_l3_feedback = feedback
        self.l3_feedback_history.append(feedback)

        # Apply modulation to SNN populations
        self.valence_pop.apply_l3_modulation(tau_modifier, threshold_modifier)
        self.arousal_pop.apply_l3_modulation(tau_modifier * 0.5, threshold_modifier)  # Less effect on arousal
        self.dominance_pop.apply_l3_modulation(tau_modifier, threshold_modifier)

        # Adjust homeostatic goals
        self.goal_valence = max(-1, min(1, self.goal_valence + goal_valence_shift))
        self.goal_arousal = max(0, min(1, self.goal_arousal + goal_arousal_shift))

        logger.debug(
            f"L3 Feedback applied: tau_mod={tau_modifier:.2f}, "
            f"thresh_mod={threshold_modifier:.2f}, suppress={suppress_arousal}"
        )

    def process_with_layers(
        self,
        l1_body: Optional[L1BodyState] = None,
        l2_perception: Optional[L2PerceptionState] = None,
        num_steps: int = 10,
    ) -> InteroceptivePAD:
        """
        Process L1 Body and L2 Perception layers to generate internal PAD.

        This is the full L1/L2 closed-loop processing path.

        Args:
            l1_body: L1 Body state (physical/somatic signals)
            l2_perception: L2 Perception state (sensory processing)
            num_steps: Number of SNN simulation steps

        Returns:
            InteroceptivePAD state derived from combined L1/L2 processing
        """
        # Store states for reference
        self.current_l1_state = l1_body
        self.current_l2_state = l2_perception

        # Initialize currents
        val_current = 0.0
        aro_current = 0.0
        dom_current = 0.0

        # Process L1 Body Layer
        if l1_body:
            l1_v, l1_a, l1_d = l1_body.to_current_vector()
            val_current += l1_v * self.l1_weight
            aro_current += l1_a * self.l1_weight
            dom_current += l1_d * self.l1_weight

        # Process L2 Perception Layer
        if l2_perception:
            l2_v, l2_a, l2_d = l2_perception.to_current_vector()
            val_current += l2_v * self.l2_weight
            aro_current += l2_a * self.l2_weight
            dom_current += l2_d * self.l2_weight

        # Apply homeostatic drive (closed-loop regulation)
        val_current += self.homeostatic_gain * (self.goal_valence - val_current)
        aro_current += self.homeostatic_gain * (self.goal_arousal - aro_current)

        # Check for L3 suppression/boost
        suppress_arousal = False
        if self.current_l3_feedback:
            suppress_arousal = self.current_l3_feedback.suppress_arousal
            if self.current_l3_feedback.boost_attention:
                aro_current *= 1.5  # Boost attention increases arousal current

        # Run SNN simulation with L3 modulation applied
        for _ in range(num_steps):
            v_mean, v_rate = self.valence_pop.step(val_current)
            a_mean, a_rate = self.arousal_pop.step(aro_current, suppress=suppress_arousal)
            d_mean, d_rate = self.dominance_pop.step(dom_current)

        # Get population statistics
        val_stats = self.valence_pop.get_statistics()
        aro_stats = self.arousal_pop.get_statistics()
        dom_stats = self.dominance_pop.get_statistics()

        # Map SNN state to PAD (this is the TRUE interoceptive signal)
        valence = math.tanh(val_stats["v_mean"] * 2)
        arousal = min(1.0, aro_stats["firing_rate"] / 0.3)
        dominance = 0.5 + 0.5 * math.tanh(dom_stats["v_mean"])

        # Compute homeostatic error
        homeostatic_error = math.sqrt(
            (valence - self.goal_valence) ** 2 +
            (arousal - self.goal_arousal) ** 2
        )

        # Confidence based on signal stability and L3 feedback
        confidence = max(0.3, 1.0 - val_stats.get("v_std", 0) - aro_stats.get("v_std", 0))
        if self.current_l3_feedback:
            confidence *= self.current_l3_feedback.attention_gain

        # Create PAD state
        pad = InteroceptivePAD(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            membrane_mean=(val_stats["v_mean"] + aro_stats["v_mean"] + dom_stats["v_mean"]) / 3,
            membrane_variance=val_stats.get("v_std", 0),
            firing_rate=(val_stats["firing_rate"] + aro_stats["firing_rate"] + dom_stats["firing_rate"]) / 3,
            population_sync=dom_stats.get("sync", 0),
            homeostatic_error=homeostatic_error,
            goal_valence=self.goal_valence,
            goal_arousal=self.goal_arousal,
            confidence=confidence,
        )

        self.pad_history.append(pad)

        logger.debug(
            f"L1/L2 PAD: V={valence:.2f}, A={arousal:.2f}, D={dominance:.2f}, "
            f"error={homeostatic_error:.3f}"
        )

        return pad

    def get_internal_state(self) -> Dict[str, Any]:
        """Get current internal state for debugging/monitoring."""
        return {
            "valence_pop": self.valence_pop.get_statistics(),
            "arousal_pop": self.arousal_pop.get_statistics(),
            "dominance_pop": self.dominance_pop.get_statistics(),
            "homeostatic_goal": {
                "valence": self.goal_valence,
                "arousal": self.goal_arousal,
            },
            "l3_feedback": self.current_l3_feedback.to_dict() if self.current_l3_feedback else None,
            "l1_state": self.current_l1_state.to_dict() if self.current_l1_state else None,
            "l2_state": self.current_l2_state.to_dict() if self.current_l2_state else None,
            "history_length": len(self.pad_history),
        }

    def get_recent_pad(self, n: int = 10) -> List[Dict]:
        """Get recent PAD history."""
        return [p.to_dict() for p in self.pad_history[-n:]]

    def get_trainable_params(self) -> Dict[str, Dict]:
        """Get all trainable SNN parameters for TP-RL optimization."""
        return {
            "valence": self.valence_pop.get_trainable_params(),
            "arousal": self.arousal_pop.get_trainable_params(),
            "dominance": self.dominance_pop.get_trainable_params(),
        }

    def set_trainable_params(self, params: Dict[str, Dict]):
        """Set trainable parameters from TP-RL optimization."""
        if "valence" in params:
            self.valence_pop.set_trainable_params(params["valence"])
        if "arousal" in params:
            self.arousal_pop.set_trainable_params(params["arousal"])
        if "dominance" in params:
            self.dominance_pop.set_trainable_params(params["dominance"])

    def reset_populations(self):
        """Reset all SNN populations to initial state."""
        self.valence_pop.reset()
        self.arousal_pop.reset()
        self.dominance_pop.reset()


# Singleton
_interoception_core: Optional[InteroceptionCore] = None


def get_interoception_core() -> InteroceptionCore:
    """Get or create global interoception core."""
    global _interoception_core
    if _interoception_core is None:
        _interoception_core = InteroceptionCore()
    return _interoception_core


def process_sensory_input(
    audio_features: Optional[Dict] = None,
    text_embedding: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to process sensory input.

    Returns PAD state derived from internal SNN dynamics.
    """
    core = get_interoception_core()
    pad = core.process_sensory(
        audio_features=audio_features,
        text_embedding=text_embedding,
    )
    return pad.to_dict()


__all__ = [
    # Enums
    "SensoryModality",
    # L1 Body Layer
    "L1BodyState",
    # L2 Perception Layer
    "L2PerceptionState",
    # L3 Feedback
    "L3Feedback",
    # Data classes
    "SensoryStream",
    "LIFState",
    "InteroceptivePAD",
    # Processing
    "SpikeRateEncoder",
    "InteroceptionPopulation",
    "InteroceptionCore",
    # Convenience functions
    "get_interoception_core",
    "process_sensory_input",
]
