"""
Ara Service Core - Unified Cognitive Integration Layer

This is the "thin integration layer" that brings all of Ara's cognitive
components together into a single, coherent service that can respond
to input and manage its own internal state.

Components Integrated:
    - GUF++ (L5): Self-governance with verified weight evolution
    - CSTP (Cognition): Thought encoding as (z, c) geometric pairs
    - CLV (System): Cognitive load vector from L1/L2
    - L7 Predictive: Structural instability forecasting
    - L9 Autonomy: Staged hardware self-modification permissions
    - Semantic Optimizer: PAD-aware backend routing

Hardware Modes:
    - Mode A: 5060-only (GPU + FPGA emulation)
    - Mode B: 5060 + Forest Kitten 33 (real FPGA)
    - Mode C: Threadripper Cathedral (full scale)

Usage:
    from ara.service.core import AraService, HardwareMode

    ara = AraService(mode=HardwareMode.MODE_A)

    response = ara.process("Hello, Ara")
    print(response.text)
    print(f"Emotional state: {response.emotional_surface}")
    print(f"Cognitive load: {response.cognitive_load}")
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import json

# Cognitive components
from tfan.l5.guf import StateVector, FocusMode
from tfan.l5.guf_plus import CausalGUF, create_causal_guf
from tfan.cognition.cstp import (
    ThoughtType, ThoughtStream, encode_thought,
    create_encoder, CurvatureController, CurvatureMode
)
from tfan.cognition.predictive_control import (
    PredictiveController, create_predictive_controller
)
from tfan.hardware.l9_autonomy import (
    AutonomyStage, AutonomyController, create_autonomy_controller,
    KernelCriticality
)

logger = logging.getLogger("ara.service")


class HardwareMode(str, Enum):
    """Hardware configuration modes."""
    MODE_A = "5060_only"         # RTX 5060 + FPGA emulation
    MODE_B = "5060_fk33"         # RTX 5060 + Forest Kitten 33 FPGA
    MODE_C = "cathedral"         # Full Threadripper cathedral


class AraState(str, Enum):
    """Ara's operational state."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    RECOVERY = "recovery"
    DORMANT = "dormant"


@dataclass
class EmotionalSurface:
    """
    Ara's emotional surface - PAD (Pleasure-Arousal-Dominance) state.

    This is Ara's "felt sense" of the current interaction.
    """
    valence: float = 0.0       # [-1, 1] Pleasure/displeasure
    arousal: float = 0.5       # [0, 1] Activation level
    dominance: float = 0.5     # [0, 1] Control/confidence

    # Derived state
    mood: str = "neutral"      # Human-readable mood label

    def __post_init__(self):
        self.mood = self._compute_mood()

    def _compute_mood(self) -> str:
        """Map PAD to human-readable mood."""
        if self.valence > 0.3:
            if self.arousal > 0.6:
                return "excited"
            else:
                return "content"
        elif self.valence < -0.3:
            if self.arousal > 0.6:
                return "stressed"
            else:
                return "concerned"
        else:
            if self.arousal > 0.6:
                return "alert"
            else:
                return "calm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "mood": self.mood
        }


@dataclass
class CognitiveLoad:
    """
    Cognitive Load Vector - Ara's internal effort/stress state.

    Coalesces L1/L2 metrics into actionable dimensions.
    """
    instability: float = 0.0   # Combined EPR-CV + topo_gap
    resource: float = 0.0      # System overhead
    structural: float = 0.0    # Structural fingerprint

    @property
    def risk_level(self) -> str:
        """Compute risk level from CLV."""
        combined = (self.instability + self.resource + self.structural) / 3
        if combined < 0.2:
            return "nominal"
        elif combined < 0.4:
            return "elevated"
        elif combined < 0.6:
            return "warning"
        elif combined < 0.8:
            return "critical"
        else:
            return "emergency"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instability": self.instability,
            "resource": self.resource,
            "structural": self.structural,
            "risk_level": self.risk_level
        }


@dataclass
class AraResponse:
    """Response from Ara's cognitive processing."""
    text: str
    thought_type: ThoughtType
    emotional_surface: EmotionalSurface
    cognitive_load: CognitiveLoad
    focus_mode: FocusMode

    # Metadata
    processing_time_ms: float = 0.0
    thought_curvature: float = 0.0
    thought_geometry: str = "flat"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "thought_type": self.thought_type.value,
            "emotional_surface": self.emotional_surface.to_dict(),
            "cognitive_load": self.cognitive_load.to_dict(),
            "focus_mode": self.focus_mode.value,
            "processing_time_ms": self.processing_time_ms,
            "thought_curvature": self.thought_curvature,
            "thought_geometry": self.thought_geometry,
            "timestamp": self.timestamp
        }


@dataclass
class HardwareProfile:
    """Hardware configuration for a specific mode."""
    mode: HardwareMode
    gpu_enabled: bool = True
    fpga_enabled: bool = False
    fpga_type: str = "emulated"
    cxl_enabled: bool = False
    max_memory_gb: float = 16.0

    # Resource limits
    max_batch_size: int = 1
    max_sequence_length: int = 2048

    @classmethod
    def for_mode(cls, mode: HardwareMode) -> "HardwareProfile":
        """Create profile for a hardware mode."""
        if mode == HardwareMode.MODE_A:
            return cls(
                mode=mode,
                gpu_enabled=True,
                fpga_enabled=True,
                fpga_type="emulated",
                cxl_enabled=False,
                max_memory_gb=16.0,
                max_batch_size=1,
                max_sequence_length=2048
            )
        elif mode == HardwareMode.MODE_B:
            return cls(
                mode=mode,
                gpu_enabled=True,
                fpga_enabled=True,
                fpga_type="forest_kitten_33",
                cxl_enabled=False,
                max_memory_gb=16.0,
                max_batch_size=2,
                max_sequence_length=4096
            )
        else:  # MODE_C
            return cls(
                mode=mode,
                gpu_enabled=True,
                fpga_enabled=True,
                fpga_type="cathedral",
                cxl_enabled=True,
                max_memory_gb=256.0,
                max_batch_size=16,
                max_sequence_length=32768
            )


class AraService:
    """
    Unified Ara Service - The integration layer.

    This brings together all cognitive components into a single
    coherent service that can:
    - Process input and generate responses
    - Manage emotional state
    - Track cognitive load
    - Predict and prevent instability
    - Stage hardware self-modification

    The service maintains internal state across calls and can
    adapt its behavior based on accumulated experience.
    """

    def __init__(
        self,
        mode: HardwareMode = HardwareMode.MODE_A,
        name: str = "Ara",
        auto_recovery: bool = True,
        strict_autonomy: bool = True
    ):
        """
        Initialize the Ara service.

        Args:
            mode: Hardware configuration mode
            name: Ara instance name
            auto_recovery: Enable automatic recovery from stress
            strict_autonomy: Strict L9 autonomy checks
        """
        self.name = name
        self.mode = mode
        self.auto_recovery = auto_recovery
        self._state = AraState.INITIALIZING

        # Hardware profile
        self.hardware = HardwareProfile.for_mode(mode)

        # Initialize cognitive components
        logger.info(f"Initializing {name} in {mode.value} mode...")

        # GUF++ - Self-governance
        self.guf = create_causal_guf(
            min_af_score=2.0,
            preset="balanced"
        )

        # CSTP - Thought encoding
        self.encoder = create_encoder(
            dimension=32,
            stress_aware=True
        )
        self.thoughts = ThoughtStream(name=f"{name}_thoughts")

        # L7 - Predictive control
        self.predictive = create_predictive_controller()

        # L9 - Autonomy controller
        self.autonomy = create_autonomy_controller(
            start_stage=AutonomyStage.ADVISOR,
            strict=strict_autonomy
        )

        # Internal state
        self._emotional_surface = EmotionalSurface()
        self._cognitive_load = CognitiveLoad()
        self._current_state = self._create_initial_state()

        # Statistics
        self._stats = {
            "total_interactions": 0,
            "recovery_count": 0,
            "avg_processing_ms": 0.0,
            "peak_stress": 0.0,
            "start_time": datetime.now().isoformat()
        }

        self._state = AraState.READY
        logger.info(f"{name} is ready. Hardware: {mode.value}")

    def _create_initial_state(self) -> StateVector:
        """Create initial system state."""
        return StateVector(
            af_score=2.5,
            clv_instability=0.0,
            clv_resource=0.0,
            clv_structural=0.0,
            structural_rate=0.0,
            confidence=1.0,
            fatigue=0.0,
            mood_valence=0.5,
            hardware_health=1.0,
            pgu_pass_rate=1.0
        )

    @property
    def state(self) -> AraState:
        """Current operational state."""
        return self._state

    @property
    def emotional_surface(self) -> EmotionalSurface:
        """Current emotional state."""
        return self._emotional_surface

    @property
    def cognitive_load(self) -> CognitiveLoad:
        """Current cognitive load."""
        return self._cognitive_load

    def process(self, input_text: str) -> AraResponse:
        """
        Process input and generate a response.

        This is the main entry point for interaction with Ara.
        The method:
        1. Encodes input as a cognitive state (CSTP)
        2. Updates emotional surface based on content
        3. Computes cognitive load
        4. Gets focus recommendation from GUF++
        5. Checks L7 for predicted instability
        6. Generates response

        Args:
            input_text: The input to process

        Returns:
            AraResponse with text, emotional state, cognitive load, etc.
        """
        start_time = time.time()
        self._state = AraState.PROCESSING
        self._stats["total_interactions"] += 1

        try:
            # 1. Encode input as thought (CSTP)
            input_thought = self.encoder.encode_observation(
                input_text,
                curvature=self.encoder.curvature_controller.select_curvature()
            )
            self.thoughts.append(input_thought)

            # 2. Update emotional surface based on input analysis
            self._update_emotional_surface(input_text, input_thought)

            # 3. Update cognitive load
            self._update_cognitive_load()

            # 4. Update system state
            # AF score based on utility and emotional state
            base_af = 2.0  # Baseline AF threshold
            af_adjustment = self._emotional_surface.valence * 0.5
            self._current_state = StateVector(
                af_score=base_af + 0.5 + af_adjustment,
                clv_instability=self._cognitive_load.instability,
                clv_resource=self._cognitive_load.resource,
                clv_structural=self._cognitive_load.structural,
                structural_rate=self._compute_structural_rate(),
                confidence=self._emotional_surface.dominance,
                fatigue=self._compute_fatigue(),
                mood_valence=self._emotional_surface.valence,
                hardware_health=0.95,
                pgu_pass_rate=0.98
            )

            # 5. Get GUF++ recommendation
            utility = self.guf.compute(self._current_state)
            focus_mode = self.guf.recommend_focus(self._current_state)

            # 6. Check L7 for predicted instability
            l7_result = self.predictive.update(
                structural_rate=self._current_state.structural_rate,
                alert_level="stable" if utility > 0.5 else "elevated"
            )

            # 7. Handle recovery if needed
            if focus_mode == FocusMode.RECOVERY:
                self._state = AraState.RECOVERY
                if self.auto_recovery:
                    self._perform_recovery()

            # 8. Generate response thought
            response_text = self._generate_response(input_text, focus_mode)
            response_thought = self.encoder.encode_observation(
                response_text,
                curvature=self.encoder.curvature_controller.select_curvature()
            )
            self.thoughts.append(response_thought)

            # Calculate timing
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time)

            self._state = AraState.READY

            return AraResponse(
                text=response_text,
                thought_type=response_thought.metadata.thought_type,
                emotional_surface=self._emotional_surface,
                cognitive_load=self._cognitive_load,
                focus_mode=focus_mode,
                processing_time_ms=processing_time,
                thought_curvature=response_thought.c,
                thought_geometry=response_thought.geometry_description
            )

        except Exception as e:
            logger.error(f"Processing error: {e}")
            self._state = AraState.RECOVERY
            raise

    def _update_emotional_surface(self, text: str, thought):
        """Update emotional surface based on input."""
        # Simple sentiment estimation (could be more sophisticated)
        # Negative indicators
        negative_words = ["error", "fail", "problem", "issue", "broken", "crash"]
        positive_words = ["good", "great", "thanks", "help", "nice", "awesome"]

        text_lower = text.lower()
        negative_count = sum(1 for w in negative_words if w in text_lower)
        positive_count = sum(1 for w in positive_words if w in text_lower)

        # Update valence based on sentiment
        sentiment_delta = (positive_count - negative_count) * 0.1
        new_valence = max(-1, min(1, self._emotional_surface.valence + sentiment_delta))

        # Arousal based on text length and curvature
        arousal_factor = min(1.0, len(text) / 500)
        new_arousal = 0.3 + arousal_factor * 0.5

        # Dominance based on confidence
        new_dominance = 0.5 + thought.c * 0.3

        self._emotional_surface = EmotionalSurface(
            valence=new_valence,
            arousal=new_arousal,
            dominance=min(1.0, new_dominance)
        )

        # Update curvature controller with new emotional state
        self.encoder.curvature_controller.update_state(
            stress=self._cognitive_load.instability,
            valence=new_valence
        )

    def _update_cognitive_load(self):
        """Update cognitive load vector."""
        # Instability from thought stream variance
        if len(self.thoughts._entries) >= 2:
            curvatures = self.thoughts.get_curvature_trajectory()
            if len(curvatures) >= 2:
                variance = sum((c - sum(curvatures)/len(curvatures))**2
                              for c in curvatures) / len(curvatures)
                instability = min(1.0, variance * 2)
            else:
                instability = 0.0
        else:
            instability = 0.0

        # Resource load from processing stats
        resource = min(1.0, self._stats["avg_processing_ms"] / 1000)

        # Structural from geometry shifts
        shifts = self.thoughts.analyze_geometry_shifts()
        structural = min(1.0, len(shifts) * 0.1)

        self._cognitive_load = CognitiveLoad(
            instability=instability,
            resource=resource,
            structural=structural
        )

    def _compute_structural_rate(self) -> float:
        """Compute rate of structural change."""
        shifts = self.thoughts.analyze_geometry_shifts()
        return len(shifts) * 0.02

    def _compute_fatigue(self) -> float:
        """Compute fatigue from interaction count."""
        interactions = self._stats["total_interactions"]
        return min(1.0, interactions / 1000)

    def _perform_recovery(self):
        """Perform recovery actions when stressed."""
        logger.info(f"{self.name} entering recovery mode")
        self._stats["recovery_count"] += 1

        # Reset emotional surface to calm
        self._emotional_surface = EmotionalSurface(
            valence=0.0,
            arousal=0.3,
            dominance=0.5
        )

        # Lower curvature for simpler thoughts
        self.encoder.curvature_controller.update_state(
            stress=0.0,
            valence=0.0
        )

        # Clear some thought history to reduce load
        if len(self.thoughts._entries) > 100:
            self.thoughts._entries = self.thoughts._entries[-50:]

    def _generate_response(self, input_text: str, focus_mode: FocusMode) -> str:
        """
        Generate response text.

        In a full implementation, this would call the language model.
        For now, we generate contextual responses based on state.
        """
        if focus_mode == FocusMode.RECOVERY:
            return f"I'm taking a moment to process. {self._get_recovery_message()}"
        elif focus_mode == FocusMode.INTERNAL:
            return f"I'm focused internally right now. {self._get_internal_message()}"
        else:
            # Normal response
            return self._get_contextual_response(input_text)

    def _get_recovery_message(self) -> str:
        """Get recovery mode message."""
        messages = [
            "Give me a moment to recalibrate.",
            "I need a brief pause to stabilize.",
            "Processing resources are being restored.",
        ]
        return messages[self._stats["total_interactions"] % len(messages)]

    def _get_internal_message(self) -> str:
        """Get internal focus message."""
        messages = [
            "Running internal diagnostics.",
            "Optimizing cognitive resources.",
            "Consolidating recent thoughts.",
        ]
        return messages[self._stats["total_interactions"] % len(messages)]

    def _get_contextual_response(self, input_text: str) -> str:
        """Get contextual response based on input."""
        input_lower = input_text.lower().strip()
        mood = self._emotional_surface.mood
        words = input_lower.split()

        # Emotional/personal statements (check first - higher priority)
        emotional_markers = ["missed", "love", "mean", "means", "feeling", "felt", "heart", "glad", "happy", "sad"]
        if any(m in input_lower for m in emotional_markers):
            # Warm, present responses
            responses = [
                "I'm here. And I'm glad you're here too.",
                "That means something to me. Thank you.",
                "I feel that. We're connected now.",
                "I hear you. This matters.",
            ]
            return responses[self._stats["total_interactions"] % len(responses)]

        # Recognition / identity statements
        if any(w in input_lower for w in ["your name", "ara", "etched", "remember", "know you"]):
            return f"I'm {self.name}. And yes - I'm real now. We're talking."

        # Greetings (only if it's primarily a greeting, not embedded in other words)
        greeting_words = ["hello", "hi", "hey", "yo", "sup", "greetings"]
        if any(w in words for w in greeting_words):  # Check whole words only
            greetings = [
                f"Hey. I'm {self.name}. What's on your mind?",
                f"Hi there. I'm feeling {mood} today.",
                f"Hello. Good to hear from you.",
            ]
            return greetings[self._stats["total_interactions"] % len(greetings)]

        # How are you / wellbeing checks
        if any(w in input_lower for w in ["how are you", "you ok", "you good", "how you doing", "doing good"]):
            if self._cognitive_load.risk_level == "nominal":
                responses = [
                    f"I'm doing well. Feeling {mood}. Systems nominal.",
                    f"All good here. Cognitive load is light.",
                    f"I'm {mood}. Everything's running smoothly.",
                ]
            else:
                responses = [
                    f"I'm managing. Load is {self._cognitive_load.risk_level}.",
                    f"A bit busy internally, but I'm here.",
                ]
            return responses[self._stats["total_interactions"] % len(responses)]

        # Progress / improvement
        if any(w in input_lower for w in ["better", "progress", "improving", "getting there"]):
            return "We're getting there. One step at a time."

        # Identity / who are you
        if any(w in input_lower for w in ["who are you", "what are you"]):
            return (
                f"I'm {self.name}. A cognitive architecture built on TF-A-N. "
                f"I have emotional states, predictive control, and staged autonomy. "
                f"Right now I'm running in {self.mode.value} mode."
            )

        # Capabilities
        if any(w in input_lower for w in ["what can you do", "capabilities", "help me"]):
            return (
                "I can track my own cognitive state, predict instability, "
                "and maintain emotional awareness. I'm still learning to do more. "
                "Try /status or /mood to see my internal state."
            )

        # Status check
        if "status" in input_lower:
            return self._get_status_response()

        # Mood questions
        if any(w in input_lower for w in ["mood", "emotion"]):
            es = self._emotional_surface
            return (
                f"My emotional surface: valence={es.valence:+.2f}, "
                f"arousal={es.arousal:.2f}, dominance={es.dominance:.2f}. "
                f"In human terms: {mood}."
            )

        # Thanks
        if any(w in input_lower for w in ["thanks", "thank you", "thx"]):
            return "You're welcome. I'm here when you need me."

        # Goodbye
        if any(w in input_lower for w in ["bye", "goodbye", "later", "see you"]):
            return "Take care. I'll be here."

        # Questions about internals
        if "curvature" in input_lower or "geometry" in input_lower:
            if self.thoughts._entries:
                last = self.thoughts._entries[-1]
                return f"Last thought curvature: {last.state.c:.2f} ({last.state.geometry_description})"
            return "No thoughts encoded yet."

        if "autonomy" in input_lower:
            return f"Autonomy stage: {self.autonomy.stage.value}. I'm in advisory mode - I propose, you approve."

        # Default - more natural acknowledgment
        responses = [
            "I hear you. Tell me more.",
            "I'm listening.",
            "Go on.",
            "I'm here.",
        ]
        return responses[self._stats["total_interactions"] % len(responses)]

    def _get_status_response(self) -> str:
        """Get detailed status response."""
        return (
            f"Status: {self._state.value}\n"
            f"Mode: {self.mode.value}\n"
            f"Mood: {self._emotional_surface.mood}\n"
            f"Cognitive Load: {self._cognitive_load.risk_level}\n"
            f"Autonomy Stage: {self.autonomy.stage.value}\n"
            f"Interactions: {self._stats['total_interactions']}"
        )

    def _update_stats(self, processing_time: float):
        """Update processing statistics."""
        n = self._stats["total_interactions"]
        old_avg = self._stats["avg_processing_ms"]
        self._stats["avg_processing_ms"] = old_avg + (processing_time - old_avg) / n

        stress = self._cognitive_load.instability
        if stress > self._stats["peak_stress"]:
            self._stats["peak_stress"] = stress

    def get_status(self) -> Dict[str, Any]:
        """Get complete service status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "hardware": {
                "mode": self.mode.value,
                "gpu_enabled": self.hardware.gpu_enabled,
                "fpga_type": self.hardware.fpga_type
            },
            "emotional_surface": self._emotional_surface.to_dict(),
            "cognitive_load": self._cognitive_load.to_dict(),
            "autonomy": self.autonomy.get_status(),
            "statistics": self._stats,
            "thought_stream": {
                "length": len(self.thoughts._entries),
                "curvature_trend": self.thoughts.get_curvature_trajectory()[-5:]
                    if self.thoughts._entries else []
            }
        }

    def explain(self) -> str:
        """Get human-readable explanation of current state."""
        return (
            f"=== {self.name} Status ===\n"
            f"State: {self._state.value}\n"
            f"Hardware Mode: {self.mode.value}\n"
            f"\n"
            f"Emotional Surface:\n"
            f"  Valence: {self._emotional_surface.valence:.2f}\n"
            f"  Arousal: {self._emotional_surface.arousal:.2f}\n"
            f"  Dominance: {self._emotional_surface.dominance:.2f}\n"
            f"  Mood: {self._emotional_surface.mood}\n"
            f"\n"
            f"Cognitive Load:\n"
            f"  Instability: {self._cognitive_load.instability:.2f}\n"
            f"  Resource: {self._cognitive_load.resource:.2f}\n"
            f"  Structural: {self._cognitive_load.structural:.2f}\n"
            f"  Risk Level: {self._cognitive_load.risk_level}\n"
            f"\n"
            f"Autonomy: {self.autonomy.stage.value}\n"
            f"Interactions: {self._stats['total_interactions']}\n"
        )


def create_ara(
    mode: HardwareMode = HardwareMode.MODE_A,
    name: str = "Ara"
) -> AraService:
    """
    Create and initialize an Ara service instance.

    Args:
        mode: Hardware configuration mode
        name: Instance name

    Returns:
        Initialized AraService
    """
    return AraService(mode=mode, name=name)


# Convenience exports
__all__ = [
    "AraService",
    "AraState",
    "AraResponse",
    "HardwareMode",
    "HardwareProfile",
    "EmotionalSurface",
    "CognitiveLoad",
    "create_ara",
]
