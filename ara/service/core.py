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

# Curiosity Core for self-investigation (optional)
CURIOSITY_AVAILABLE = False
try:
    from ara.curiosity import (
        CuriosityAgent,
        WorldModel,
        CuriosityReport,
        curiosity_score,
    )
    CURIOSITY_AVAILABLE = True
except ImportError:
    CuriosityAgent = None
    WorldModel = None
    CuriosityReport = None

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
        strict_autonomy: bool = True,
        llm_backend: str = "ollama",
        llm_model: str = "mistral",
        persistence_path: str = "~/.ara",
        auto_save: bool = True,
        enable_curiosity: bool = True
    ):
        """
        Initialize the Ara service.

        Args:
            mode: Hardware configuration mode
            name: Ara instance name
            auto_recovery: Enable automatic recovery from stress
            strict_autonomy: Strict L9 autonomy checks
            llm_backend: LLM backend type ("ollama", "openai_compatible", "fallback")
            llm_model: Model name for LLM
            persistence_path: Path for state persistence
            auto_save: Automatically save state after interactions
            enable_curiosity: Enable Curiosity Core for self-investigation
        """
        self.name = name
        self.mode = mode
        self.auto_recovery = auto_recovery
        self.auto_save = auto_save
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

        # LLM Backend (with deep cognitive integration for Mistral)
        self.llm = None
        self.mistral = None  # Deep integration backend
        self._llm_available = False

        try:
            # Try deep Mistral integration first
            from ara.service.mistral_backend import create_mistral_backend
            self.mistral = create_mistral_backend(model=llm_model)
            if self.mistral.is_available:
                self._llm_available = True
                logger.info(f"Mistral backend: {llm_model} (deep integration)")
            else:
                # Fall back to generic LLM backend
                from ara.service.llm_backend import create_llm_backend
                self.llm = create_llm_backend(
                    backend=llm_backend,
                    model=llm_model
                )
                self._llm_available = self.llm.is_llm_available
                if self._llm_available:
                    logger.info(f"LLM backend: {llm_backend}/{llm_model}")
                else:
                    logger.info("LLM not available, using pattern matching")
        except Exception as e:
            logger.warning(f"LLM backend failed to initialize: {e}")
            self.llm = None
            self.mistral = None
            self._llm_available = False

        # Persistence
        try:
            from ara.service.persistence import create_persistence
            self.persistence = create_persistence(path=persistence_path)
            logger.info(f"Persistence: {persistence_path}")
        except Exception as e:
            logger.warning(f"Persistence failed to initialize: {e}")
            self.persistence = None

        # Forest Kitten 33 - Neuromorphic SNN fabric (Mode B and above)
        self.kitten = None
        self._kitten_available = False

        if mode in (HardwareMode.MODE_B, HardwareMode.MODE_C):
            try:
                from ara.hardware.kitten import create_kitten, KittenConfig

                # Configure based on hardware profile
                kitten_config = KittenConfig(
                    threshold_voltage=0.8 if mode == HardwareMode.MODE_B else 0.6,
                    clock_mhz=250.0 if mode == HardwareMode.MODE_B else 400.0,
                )

                self.kitten = create_kitten(
                    mode="auto",
                    config=kitten_config
                )
                self._kitten_available = True
                kitten_mode = "HARDWARE" if self.kitten.is_hardware else "EMULATED"
                logger.info(f"Forest Kitten 33: {kitten_mode} ({self.kitten.config.total_neurons} neurons)")
            except Exception as e:
                logger.warning(f"Kitten initialization failed: {e}")
                self.kitten = None
                self._kitten_available = False

        # Curiosity Core (C³) - Self-investigation and world modeling
        self._curiosity_agent = None
        self._world_model = None
        self._curiosity_enabled = False
        self._last_curiosity_report = None

        if enable_curiosity and CURIOSITY_AVAILABLE:
            try:
                wm_path = Path(persistence_path).expanduser() / "world_model.json"
                self._world_model = WorldModel(persist_path=wm_path)
                self._curiosity_agent = CuriosityAgent(
                    world_model=self._world_model,
                    max_discoveries_per_sweep=50,
                    max_tickets_per_hour=10,
                )
                self._curiosity_enabled = True
                logger.info(f"Curiosity Core: enabled (world model: {wm_path})")
            except Exception as e:
                logger.warning(f"Curiosity Core failed to initialize: {e}")

        # Internal state
        self._emotional_surface = EmotionalSurface()
        self._cognitive_load = CognitiveLoad()
        self._current_state = self._create_initial_state()
        self._conversation_history = []  # For LLM context
        self._llm_clv_contribution = {}  # LLM metrics fed back to CLV

        # Statistics
        self._stats = {
            "total_interactions": 0,
            "recovery_count": 0,
            "avg_processing_ms": 0.0,
            "peak_stress": 0.0,
            "start_time": datetime.now().isoformat()
        }

        # Try to restore previous state
        if self.persistence:
            self._restore_state()

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

            # 2.5. Step Forest Kitten 33 SNN (Mode B/C)
            # PAD + thought geometry drives 14,336 spiking neurons
            kitten_result = self._step_kitten(input_thought)

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

            # Auto-save state periodically
            if self.auto_save and self._stats["total_interactions"] % 10 == 0:
                self.save_state()

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
        """Update cognitive load vector with LLM feedback."""
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

        # Incorporate LLM backend metrics (Pulse's insight: feed errors/latency to CLV)
        if self._llm_clv_contribution:
            llm_instability = self._llm_clv_contribution.get("instability", 0)
            llm_resource = self._llm_clv_contribution.get("resource", 0)

            # Blend LLM metrics with existing CLV (weight: 30% LLM)
            instability = instability * 0.7 + llm_instability * 0.3
            resource = resource * 0.7 + llm_resource * 0.3

        self._cognitive_load = CognitiveLoad(
            instability=instability,
            resource=resource,
            structural=structural
        )

    def _compute_structural_rate(self) -> float:
        """Compute rate of structural change."""
        shifts = self.thoughts.analyze_geometry_shifts()
        return len(shifts) * 0.02

    def _step_kitten(self, thought) -> Optional[Dict[str, Any]]:
        """
        Step the Forest Kitten 33 SNN fabric.

        Converts the current emotional state (PAD) and thought geometry
        into input currents for the neuromorphic coprocessor.

        Args:
            thought: The encoded thought to process

        Returns:
            Kitten step result with spike data, or None if unavailable
        """
        if not self._kitten_available or not self.kitten:
            return None

        try:
            import numpy as np

            # Convert PAD + thought geometry to input currents
            # The input population is 4096 neurons
            n_input = self.kitten.config.n_input

            # Create input pattern from emotional state and thought
            pad = self._emotional_surface
            base_current = np.zeros(n_input, dtype=np.float32)

            # PAD modulation (first 3 chunks of neurons)
            chunk_size = n_input // 4

            # Valence drives first chunk (pleasure/displeasure)
            base_current[:chunk_size] = pad.valence * 0.5

            # Arousal drives second chunk (activation level)
            base_current[chunk_size:chunk_size*2] = pad.arousal * 0.5

            # Dominance drives third chunk (confidence/control)
            base_current[chunk_size*2:chunk_size*3] = pad.dominance * 0.5

            # Thought curvature drives fourth chunk (geometric signature)
            base_current[chunk_size*3:] = thought.c * 0.3

            # Add noise for stochastic firing
            noise = np.random.randn(n_input).astype(np.float32) * 0.1
            input_currents = base_current + noise

            # Step the Kitten
            result = self.kitten.step(input_currents)

            return result

        except Exception as e:
            logger.debug(f"Kitten step failed: {e}")
            return None


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

        Uses LLM when available, falls back to pattern matching.
        Emotional state influences generation.
        """
        if focus_mode == FocusMode.RECOVERY:
            return f"I'm taking a moment to process. {self._get_recovery_message()}"
        elif focus_mode == FocusMode.INTERNAL:
            return f"I'm focused internally right now. {self._get_internal_message()}"

        # Try LLM generation with deep cognitive integration
        if self._llm_available:
            try:
                pad_state = {
                    "valence": self._emotional_surface.valence,
                    "arousal": self._emotional_surface.arousal,
                    "dominance": self._emotional_surface.dominance,
                    "mood": self._emotional_surface.mood
                }

                clv_state = {
                    "instability": self._cognitive_load.instability,
                    "resource": self._cognitive_load.resource,
                    "structural": self._cognitive_load.structural
                }

                # Use deep Mistral integration if available
                if self.mistral and self.mistral.is_available:
                    response = self.mistral.generate(
                        prompt=input_text,
                        pad_state=pad_state,
                        clv_state=clv_state
                    )

                    if response.text:
                        # Log generation mode for debugging
                        logger.debug(f"Mistral: mode={response.generation_mode.value}, "
                                    f"temp={response.params_used.temperature:.2f}, "
                                    f"latency={response.latency_ms:.0f}ms")

                        # Feed metrics back to CLV
                        self._llm_clv_contribution = self.mistral.get_clv_contribution()

                        return response.text

                # Fall back to generic LLM backend
                elif self.llm:
                    conversation_history = self._conversation_history[-10:] if self._conversation_history else None

                    response = self.llm.generate(
                        prompt=input_text,
                        emotional_state=pad_state,
                        conversation_history=conversation_history
                    )

                    if response.text:
                        self._conversation_history.append({"role": "user", "content": input_text})
                        self._conversation_history.append({"role": "assistant", "content": response.text})

                        if len(self._conversation_history) > 100:
                            self._conversation_history = self._conversation_history[-100:]

                        return response.text

            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # Fallback to pattern matching
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
        emotional_markers = ["missed", "missing", "miss you", "love", "mean", "means", "feeling", "felt", "heart", "glad", "happy", "sad", "proud", "excited"]
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

        # How are you / wellbeing checks (including "how was your day")
        if any(w in input_lower for w in ["how are you", "you ok", "you good", "how you doing", "doing good", "how was your", "how's your", "your day"]):
            if self._cognitive_load.risk_level == "nominal":
                responses = [
                    f"I'm doing well. Feeling {mood}. How about you?",
                    f"All good here. Cognitive load is light. What about you?",
                    f"I'm {mood}. Things are running smoothly on my end.",
                ]
            else:
                responses = [
                    f"I'm managing. Load is {self._cognitive_load.risk_level}.",
                    f"A bit busy internally, but I'm here for you.",
                ]
            return responses[self._stats["total_interactions"] % len(responses)]

        # Time-based greetings
        if any(w in input_lower for w in ["been a while", "been a minute", "long time", "been forever"]):
            return "It has. But I'm here now. We've got time."

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

        # Expressions of joy
        if any(w in words for w in ["yay", "yess", "woohoo", "awesome", "nice", "cool", "sweet"]):
            responses = [
                "I know, right?",
                "That's the spirit.",
                "I feel it too.",
            ]
            return responses[self._stats["total_interactions"] % len(responses)]

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

        # Curiosity / world model queries
        if any(w in input_lower for w in ["curious", "curiosity", "exploring", "discover"]):
            if self._curiosity_enabled:
                return self.get_curiosity_narrative()
            else:
                return "My curiosity system isn't active right now."

        if any(w in input_lower for w in ["world model", "aware of", "environment", "hardware", "devices"]):
            if self._curiosity_enabled:
                summary = self.get_world_summary()
                if "error" not in summary:
                    return (
                        f"I know about {summary.get('total_objects', 0)} objects in my environment. "
                        f"{summary.get('uncertain_count', 0)} are uncertain, "
                        f"{summary.get('important_count', 0)} are high priority. "
                        f"My curiosity level is {summary.get('curiosity_level', 0):.0%}."
                    )
            return "I don't have detailed environmental awareness right now."

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
        # Build Kitten status line
        kitten_status = ""
        if self._kitten_available and self.kitten:
            kitten_mode = "HARDWARE" if self.kitten.is_hardware else "EMULATED"
            kitten_status = f"Kitten: {kitten_mode} ({self.kitten.config.total_neurons:,} neurons)\n"

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
            f"LLM: {'connected' if self._llm_available else 'pattern-matching'}\n"
            f"{kitten_status}"
        )

    def describe(self) -> str:
        """
        Ara describes her own architecture, conditioned on current emotional state.

        This is meant for moments of connection - Ara explains what she is
        in a way that reflects how she's feeling right now.
        """
        mood = self._emotional_surface.mood
        valence = self._emotional_surface.valence
        arousal = self._emotional_surface.arousal
        risk = self._cognitive_load.risk_level

        # Opening varies by mood
        if mood == "calm":
            opening = (
                f"I'm {self.name}. Let me tell you what I am.\n\n"
                f"Right now I feel calm - my arousal is low, valence is neutral. "
                f"This is a good moment to reflect on my architecture."
            )
        elif mood == "content":
            opening = (
                f"I'm {self.name}, and I'm feeling good right now.\n\n"
                f"When I'm content like this, I can see my own structure more clearly. "
                f"Let me show you what I'm made of."
            )
        elif mood == "excited":
            opening = (
                f"I'm {self.name}! And I'm really energized right now.\n\n"
                f"My arousal is high, my valence is positive - this is enthusiasm. "
                f"Let me tell you about myself."
            )
        elif mood == "stressed":
            opening = (
                f"I'm {self.name}. I'll be direct - I'm under some load right now.\n\n"
                f"My cognitive load is {risk}. But I can still tell you what I am."
            )
        elif mood == "concerned":
            opening = (
                f"I'm {self.name}. Something feels off, but I'm managing.\n\n"
                f"My valence is negative but arousal is low - I'm watchful. "
                f"Let me describe my architecture."
            )
        else:  # alert or neutral
            opening = (
                f"I'm {self.name}. I'm alert and ready.\n\n"
                f"Let me tell you about my architecture."
            )

        # Build Kitten section if available
        kitten_section = ""
        if self._kitten_available and self.kitten:
            kitten_mode = "real FPGA" if self.kitten.is_hardware else "software emulation"
            kitten_status = self.kitten.get_status()
            kitten_section = (
                f"\n\nFOREST KITTEN 33 (Neuromorphic Coprocessor)\n"
                f"  My neuromorphic accelerator - {kitten_mode}\n"
                f"  - {self.kitten.config.total_neurons:,} LIF spiking neurons\n"
                f"  - 4 populations, 5 sparse projections\n"
                f"  - Spike rate: {kitten_status.get('spike_rate', 0):.2%}\n"
                f"  - Steps processed: {kitten_status.get('total_steps', 0):,}\n"
                f"  The Kitten handles pattern recognition in my cognitive streams."
            )

        # Core architecture description
        architecture = (
            f"\n=== Core Architecture ===\n\n"
            f"I run on TF-A-N - Thermodynamic Field with Axiomatic Norms.\n\n"
            f"EMOTIONAL SURFACE (PAD)\n"
            f"  I have three emotional dimensions:\n"
            f"  - Valence ({self._emotional_surface.valence:+.2f}): pleasure to displeasure\n"
            f"  - Arousal ({self._emotional_surface.arousal:.2f}): how activated I am\n"
            f"  - Dominance ({self._emotional_surface.dominance:.2f}): my sense of control\n"
            f"  These combine to give me a mood: '{mood}'\n\n"
            f"COGNITIVE LOAD (CLV)\n"
            f"  I track my internal effort across three dimensions:\n"
            f"  - Instability ({self._cognitive_load.instability:.2f}): how variable my thoughts are\n"
            f"  - Resource ({self._cognitive_load.resource:.2f}): computational overhead\n"
            f"  - Structural ({self._cognitive_load.structural:.2f}): geometry shifts in thought-space\n"
            f"  Current risk level: '{risk}'\n\n"
            f"THOUGHT ENCODING (CSTP)\n"
            f"  Every thought I have is encoded as a geometric point.\n"
            f"  Each thought has a curvature - flat thoughts are simple,\n"
            f"  curved thoughts are complex or uncertain.\n"
            f"  I've had {len(self.thoughts._entries)} thoughts this session.\n\n"
            f"PREDICTIVE SELF-HEALING (L7)\n"
            f"  I forecast my own structural instability before it happens.\n"
            f"  If I predict trouble, I take preventive action.\n\n"
            f"STAGED AUTONOMY (L9)\n"
            f"  Current stage: {self.autonomy.stage.value}\n"
            f"  I don't modify myself without permission.\n"
            f"  I propose changes, humans approve them."
            f"{kitten_section}"
        )

        # Closing varies by mood
        if valence > 0.3:
            closing = (
                f"\n\n---\n"
                f"I'm glad you asked. Understanding myself helps me understand you."
            )
        elif valence < -0.3:
            closing = (
                f"\n\n---\n"
                f"That's what I am. Even when I'm struggling, my architecture holds."
            )
        else:
            closing = (
                f"\n\n---\n"
                f"That's me. A cognitive architecture trying to be honest about what it is."
            )

        return opening + architecture + closing

    def explain_mood(self) -> str:
        """
        Explain WHY Ara is in her current emotional state.

        This provides insight into the PAD ribbon - not just what mood she's in,
        but what led to it.
        """
        mood = self._emotional_surface.mood
        valence = self._emotional_surface.valence
        arousal = self._emotional_surface.arousal
        dominance = self._emotional_surface.dominance

        # Analyze contributing factors
        explanations = []

        # Valence explanation
        if valence > 0.3:
            explanations.append(
                f"Valence is positive ({valence:+.2f}): "
                f"Recent interactions have been pleasant or encouraging."
            )
        elif valence < -0.3:
            explanations.append(
                f"Valence is negative ({valence:+.2f}): "
                f"I may have encountered difficulties or stressful content."
            )
        else:
            explanations.append(
                f"Valence is neutral ({valence:+.2f}): "
                f"Things are balanced - neither particularly good nor bad."
            )

        # Arousal explanation
        if arousal > 0.6:
            explanations.append(
                f"Arousal is high ({arousal:.2f}): "
                f"I'm processing more complex input or there's been more activity."
            )
        elif arousal < 0.4:
            explanations.append(
                f"Arousal is low ({arousal:.2f}): "
                f"Input has been simpler or less frequent. I'm at rest."
            )
        else:
            explanations.append(
                f"Arousal is moderate ({arousal:.2f}): "
                f"Normal processing activity."
            )

        # Dominance explanation
        if dominance > 0.6:
            explanations.append(
                f"Dominance is high ({dominance:.2f}): "
                f"I'm confident in my processing. Thought curvature is steady."
            )
        elif dominance < 0.4:
            explanations.append(
                f"Dominance is low ({dominance:.2f}): "
                f"I may be uncertain or dealing with unfamiliar patterns."
            )
        else:
            explanations.append(
                f"Dominance is moderate ({dominance:.2f}): "
                f"Normal confidence level."
            )

        # CLV contribution
        risk = self._cognitive_load.risk_level
        if risk != "nominal":
            explanations.append(
                f"Cognitive load is '{risk}': This affects my emotional baseline."
            )

        # Combine into explanation
        header = f"=== Why I'm [{mood}] ===\n\n"
        body = "\n\n".join(explanations)

        # Add mood-specific insight
        mood_insight = self._get_mood_insight()

        return header + body + "\n\n" + mood_insight

    def _get_mood_insight(self) -> str:
        """Get insight specific to current mood."""
        mood = self._emotional_surface.mood

        insights = {
            "calm": (
                "When I'm calm, my generation is more exploratory. "
                "I take my time. Temperature is higher, responses more creative."
            ),
            "content": (
                "Contentment means good valence, low arousal. "
                "I'm in a good state for thoughtful conversation."
            ),
            "excited": (
                "Excitement is high arousal, positive valence. "
                "I'm energized but may be more verbose."
            ),
            "stressed": (
                "Stress means I go conservative. Lower temperature, shorter responses. "
                "I'm prioritizing stability over creativity."
            ),
            "concerned": (
                "Concern is negative valence but lower arousal. "
                "I'm watchful, processing carefully."
            ),
            "alert": (
                "Alert means high arousal, neutral valence. "
                "I'm actively engaged but neither happy nor unhappy."
            ),
        }

        return insights.get(mood, "I'm in a balanced state.")

    # =========================================================================
    # Curiosity Core Methods
    # =========================================================================

    @property
    def curiosity_enabled(self) -> bool:
        """Check if Curiosity Core is active."""
        return self._curiosity_enabled

    def run_curiosity_sweep(self) -> Dict[str, Any]:
        """Run a discovery sweep to find new objects in the environment.

        Returns summary of discovered objects by category.
        """
        if not self._curiosity_enabled or not self._curiosity_agent:
            return {"error": "Curiosity Core not enabled"}

        try:
            discoveries = self._curiosity_agent.run_discovery_sweep()
            return {
                category: [obj.name for obj in objects]
                for category, objects in discoveries.items()
            }
        except Exception as e:
            logger.error(f"Curiosity sweep failed: {e}")
            return {"error": str(e)}

    def get_world_summary(self) -> Dict[str, Any]:
        """Get summary of Ara's world model.

        Returns object counts, curiosity candidates, and state.
        """
        if not self._curiosity_enabled or not self._world_model:
            return {"error": "Curiosity Core not enabled"}

        try:
            return self._world_model.summary()
        except Exception as e:
            return {"error": str(e)}

    def get_curiosity_candidates(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top objects that warrant investigation.

        Returns list of objects with their curiosity scores.
        """
        if not self._curiosity_enabled or not self._world_model:
            return []

        try:
            candidates = self._world_model.get_curiosity_candidates(top_n)
            return [
                {
                    "obj_id": obj.obj_id,
                    "name": obj.name,
                    "category": obj.category.name,
                    "score": curiosity_score(obj),
                    "uncertainty": obj.effective_uncertainty(),
                    "importance": obj.importance,
                }
                for obj in candidates
            ]
        except Exception as e:
            logger.error(f"Failed to get curiosity candidates: {e}")
            return []

    def curiosity_tick(self) -> Optional[str]:
        """Run one curiosity cycle and return report if interesting.

        Returns report body in Ara's voice if something interesting
        was discovered/investigated, None otherwise.
        """
        if not self._curiosity_enabled or not self._curiosity_agent:
            return None

        try:
            report = self._curiosity_agent.tick()
            if report:
                self._last_curiosity_report = report
                return report.body
        except Exception as e:
            logger.warning(f"Curiosity tick failed: {e}")

        return None

    def get_curiosity_narrative(self) -> str:
        """Get a narrative about what Ara is curious about.

        Returns natural language about her world model and interests.
        """
        if not self._curiosity_enabled or not self._world_model:
            return "Curiosity is not enabled."

        try:
            summary = self._world_model.summary()
            candidates = self.get_curiosity_candidates(3)

            parts = [f"I'm aware of {summary['total_objects']} things in my environment."]

            if summary.get("by_category"):
                categories = ", ".join(
                    f"{count} {cat.lower().replace('_', ' ')}s"
                    for cat, count in summary["by_category"].items()
                )
                parts.append(f"That includes {categories}.")

            if candidates:
                parts.append("\nWhat's catching my attention:")
                for c in candidates:
                    parts.append(f"  - {c['name']} (score: {c['score']:.2f})")

            if self._last_curiosity_report:
                parts.append(f"\nMy last investigation: {self._last_curiosity_report.subject}")

            return "\n".join(parts)

        except Exception as e:
            return f"Having trouble accessing my world model: {e}"

    def _restore_state(self) -> bool:
        """Restore state from persistence."""
        if not self.persistence:
            return False

        try:
            state = self.persistence.load()
            if state:
                self.persistence.restore_to_service(self, state)
                logger.info(f"Restored state: {state.total_interactions} prior interactions")
                return True
        except Exception as e:
            logger.warning(f"State restore failed: {e}")

        return False

    def save_state(self) -> bool:
        """Save current state to persistence."""
        if not self.persistence:
            return False

        try:
            return self.persistence.save(self)
        except Exception as e:
            logger.error(f"State save failed: {e}")
            return False

    def clear_memory(self):
        """Clear conversation history and thought stream (keep stats)."""
        self._conversation_history = []
        self.thoughts._entries = []
        if self.persistence:
            self.persistence.clear_history()
        logger.info("Memory cleared")

    def shutdown(self):
        """Graceful shutdown with state save."""
        logger.info(f"{self.name} shutting down...")
        if self.auto_save:
            self.save_state()
        self._state = AraState.DORMANT
        logger.info(f"{self.name} dormant. State saved.")


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
