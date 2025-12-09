#!/usr/bin/env python3
"""
Ara Multimodal Simulation Controller
=====================================

The central orchestrator for Ara's multimodal capabilities.
Ties together:
- InteractionSpec execution (block-based behavior)
- AraSong (audio/voice synthesis)
- Video generation (hive jobs)
- Hardware allocation (parts picker)
- Memory fabric (episode storage)

This is what makes Ara "alive" - a cost-aware, inspectable
multimodal agent that can see, hear, speak, and create.
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import numpy as np

# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ara_core.interaction.spec import (
    InteractionSpec, Block, BlockType,
    AraVoiceParams, AraSongParams, VideoJobParams,
)
from ara_core.interaction.executor import (
    InteractionExecutor, BlockResult, BlockStatus, ExecutionContext,
)

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Available modality types."""
    AUDIO = "audio"      # Voice, music, sound effects
    VIDEO = "video"      # Generated video, animations
    TEXT = "text"        # Chat, captions, UI text
    IMAGE = "image"      # Static images, thumbnails
    MEMORY = "memory"    # Episode storage/retrieval


@dataclass
class ModalityOutput:
    """Output from a modality."""
    modality: ModalityType
    data: Any
    duration_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationState:
    """Current state of the multimodal simulation."""
    # Emotional state
    emotion: str = "neutral"
    energy: float = 0.5

    # Goals and context
    active_goals: List[str] = field(default_factory=list)
    current_project: Optional[str] = None

    # Resource tracking
    gpu_budget_remaining: float = 1.0  # 0-1 fraction of daily budget
    power_budget_remaining: float = 1.0
    compute_cost_today: float = 0.0

    # Session info
    session_start: float = field(default_factory=time.time)
    interactions_count: int = 0
    last_interaction: Optional[str] = None

    # Sensor states
    mic_active: bool = False
    camera_active: bool = False


@dataclass
class SimulationConfig:
    """Configuration for the multimodal simulation."""
    sample_rate: int = 48000

    # Resource limits
    max_gpu_hours_daily: float = 4.0
    max_power_kwh_daily: float = 2.0
    max_video_duration_s: float = 60.0

    # Behavior settings
    enable_voice: bool = True
    enable_video: bool = True
    enable_memory: bool = True

    # Hardware preferences
    preferred_gpu_vendor: str = "nvidia"
    min_vram_gb: float = 8.0

    # DB connection
    hive_dsn: Optional[str] = None


class MultimodalSimulator:
    """
    The main multimodal simulation controller.

    Coordinates all of Ara's sensory and output modalities:
    - Receives inputs (audio, video, text, system metrics)
    - Determines emotional state and goals
    - Plans and executes interactions using InteractionSpec
    - Produces outputs (voice, video, UI, memory)
    - Tracks costs and respects budgets

    This is the "brain" that ties everything together.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.state = SimulationState()

        # Initialize subsystems
        self.executor = InteractionExecutor(sample_rate=self.config.sample_rate)

        # Audio subsystem (lazy-loaded)
        self._voice_synth = None
        self._song_player = None

        # Parts picker integration
        self._parts_picker_available = False
        self._check_hive_connection()

        # Output callbacks
        self._audio_callback: Optional[Callable[[np.ndarray], None]] = None
        self._video_callback: Optional[Callable[[str], None]] = None
        self._text_callback: Optional[Callable[[str], None]] = None

        # Session history
        self.session_outputs: List[ModalityOutput] = []

        logger.info("MultimodalSimulator initialized")

    def _check_hive_connection(self):
        """Check if we can connect to ara-hive."""
        try:
            import psycopg2
            dsn = self.config.hive_dsn or os.getenv("ARA_HIVE_DSN")
            if dsn:
                conn = psycopg2.connect(dsn)
                conn.close()
                self._parts_picker_available = True
                logger.info("Connected to ara-hive")
        except Exception as e:
            logger.warning(f"ara-hive not available: {e}")

    @property
    def voice_synth(self):
        """Lazy-load voice synthesizer."""
        if self._voice_synth is None:
            from arasong.synth.formant_voice import FormantVoiceSynth, VoiceConfig
            voice_config = VoiceConfig(sample_rate=self.config.sample_rate)
            self._voice_synth = FormantVoiceSynth(voice_config)
        return self._voice_synth

    @property
    def song_player(self):
        """Lazy-load song player."""
        if self._song_player is None:
            from arasong.engine.song_player_v2 import AraSongPlayerV2
            self._song_player = AraSongPlayerV2(sample_rate=self.config.sample_rate)
        return self._song_player

    # =========================================================================
    # Output Callbacks
    # =========================================================================

    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for audio output."""
        self._audio_callback = callback

    def set_video_callback(self, callback: Callable[[str], None]):
        """Set callback for video job completion."""
        self._video_callback = callback

    def set_text_callback(self, callback: Callable[[str], None]):
        """Set callback for text output."""
        self._text_callback = callback

    # =========================================================================
    # State Management
    # =========================================================================

    def update_emotion(self, emotion: str, energy: float):
        """Update the current emotional state."""
        self.state.emotion = emotion
        self.state.energy = np.clip(energy, 0.0, 1.0)
        logger.debug(f"Emotion updated: {emotion} (energy={energy:.2f})")

    def add_goal(self, goal: str):
        """Add an active goal."""
        if goal not in self.state.active_goals:
            self.state.active_goals.append(goal)

    def remove_goal(self, goal: str):
        """Remove a goal."""
        if goal in self.state.active_goals:
            self.state.active_goals.remove(goal)

    def check_budget(self, estimated_cost: float = 0.0) -> bool:
        """Check if we have budget for an operation."""
        return self.state.gpu_budget_remaining > estimated_cost

    def deduct_budget(self, cost: float, power_kwh: float = 0.0):
        """Deduct from daily budget."""
        self.state.compute_cost_today += cost
        daily_budget = self.config.max_gpu_hours_daily  # Simplified
        self.state.gpu_budget_remaining = max(0, 1.0 - self.state.compute_cost_today / daily_budget)

        if power_kwh > 0:
            self.state.power_budget_remaining = max(
                0, 1.0 - power_kwh / self.config.max_power_kwh_daily
            )

    # =========================================================================
    # Hardware Integration
    # =========================================================================

    def request_gpu(self, min_vram_gb: float = None, job_type: str = "inference") -> Optional[int]:
        """
        Request GPU allocation through parts picker.

        Returns parts_job_id or None if not available.
        """
        if not self._parts_picker_available:
            logger.warning("Parts picker not available")
            return None

        try:
            from ara_hive.src.parts_picker import request_hardware, get_allocation

            job_id = request_hardware(
                requester="multimodal_sim",
                job_type=job_type,
                min_vram_gb=min_vram_gb or self.config.min_vram_gb,
                device_types=["gpu"],
                preferred_vendor=self.config.preferred_gpu_vendor,
            )

            # Wait for allocation (with timeout)
            result = get_allocation(job_id, timeout_s=10.0)
            if result and result.get("status") == "allocated":
                logger.info(f"GPU allocated: {result['hardware_ids']}")
                return job_id
            else:
                logger.warning(f"GPU allocation failed: {result}")
                return None

        except Exception as e:
            logger.error(f"GPU request error: {e}")
            return None

    def release_gpu(self, job_id: int):
        """Release a GPU allocation."""
        if not self._parts_picker_available:
            return

        try:
            from ara_hive.src.parts_picker import release_hardware
            release_hardware(job_id)
            logger.info(f"GPU released: job_id={job_id}")
        except Exception as e:
            logger.error(f"GPU release error: {e}")

    # =========================================================================
    # Core Interaction Processing
    # =========================================================================

    def run_interaction(self, spec: InteractionSpec,
                        guards: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Execute a complete multimodal interaction.

        This is the main entry point for running behavior.
        """
        logger.info(f"Running interaction: {spec.interaction_id}")
        start_time = time.time()

        # Update state from spec
        self.update_emotion(spec.state.emotion, spec.state.energy)

        # Build guard context
        guard_context = guards or {}
        guard_context["budget_ok"] = self.check_budget(0.1)
        guard_context["gpu_ok"] = self._parts_picker_available

        # Execute the interaction
        result = self.executor.execute(spec, guard_context)

        # Process outputs
        outputs = {
            "interaction_id": spec.interaction_id,
            "success": result.success,
            "duration_ms": result.total_duration_ms,
            "cost": result.total_cost,
            "modalities": [],
        }

        # Handle audio output
        if result.audio_output is not None and len(result.audio_output) > 0:
            audio_output = ModalityOutput(
                modality=ModalityType.AUDIO,
                data=result.audio_output,
                duration_s=len(result.audio_output) / self.config.sample_rate,
                metadata={"sample_rate": self.config.sample_rate}
            )
            self.session_outputs.append(audio_output)
            outputs["modalities"].append("audio")

            if self._audio_callback:
                self._audio_callback(result.audio_output)

        # Handle text output
        if spec.outputs.voice:
            text_output = ModalityOutput(
                modality=ModalityType.TEXT,
                data=spec.outputs.voice,
            )
            self.session_outputs.append(text_output)
            outputs["modalities"].append("text")

            if self._text_callback:
                self._text_callback(spec.outputs.voice)

        # Update session state
        self.state.interactions_count += 1
        self.state.last_interaction = spec.interaction_id
        self.deduct_budget(result.total_cost)

        # Log results
        elapsed = time.time() - start_time
        logger.info(f"Interaction complete: {result.success} "
                    f"({result.total_duration_ms:.0f}ms, cost={result.total_cost:.3f})")

        return outputs

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def speak(self, text: str, emotion: str = None, warmth: float = 0.5) -> np.ndarray:
        """
        Generate speech audio for text.

        Quick helper that bypasses full InteractionSpec for simple speech.
        """
        from arasong.synth.dynamics import get_emotion_params, scale_params_by_energy

        emotion = emotion or self.state.emotion
        params = get_emotion_params(emotion)
        params = scale_params_by_energy(params, self.state.energy)

        # Estimate duration from text length
        duration = len(text) * 0.08

        # Generate using voice synth
        samples = self.voice_synth.render_note(
            "A", 4, duration, text[:20],
            amplitude=0.6, emotion_params=params
        )

        # Log output
        output = ModalityOutput(
            modality=ModalityType.AUDIO,
            data=samples,
            duration_s=len(samples) / self.config.sample_rate,
            metadata={"text": text, "emotion": emotion}
        )
        self.session_outputs.append(output)

        if self._audio_callback:
            self._audio_callback(samples)

        return samples

    def play_song(self, song_path: str) -> np.ndarray:
        """
        Render and return a song.
        """
        self.song_player.load_song(song_path)
        samples = self.song_player.render_song()

        output = ModalityOutput(
            modality=ModalityType.AUDIO,
            data=samples,
            duration_s=len(samples) / self.config.sample_rate,
            metadata={"song_path": song_path}
        )
        self.session_outputs.append(output)

        if self._audio_callback:
            self._audio_callback(samples)

        return samples

    def submit_video_job(self, prompt: str, duration_s: float = 10.0,
                         resolution: str = "720p") -> Optional[int]:
        """
        Submit a video generation job to the hive.

        Returns job_id or None.
        """
        if not self._parts_picker_available:
            logger.warning("Hive not available for video jobs")
            return None

        if duration_s > self.config.max_video_duration_s:
            logger.warning(f"Video duration {duration_s}s exceeds limit")
            duration_s = self.config.max_video_duration_s

        # TODO: Implement actual video job submission
        # This would INSERT into video_jobs table
        logger.info(f"Video job submitted: {prompt[:50]}... ({duration_s}s, {resolution})")
        return None  # Placeholder

    # =========================================================================
    # Session Management
    # =========================================================================

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            "session_duration_s": time.time() - self.state.session_start,
            "interactions_count": self.state.interactions_count,
            "outputs_count": len(self.session_outputs),
            "total_audio_duration_s": sum(
                o.duration_s for o in self.session_outputs
                if o.modality == ModalityType.AUDIO
            ),
            "emotion": self.state.emotion,
            "energy": self.state.energy,
            "budget_remaining": self.state.gpu_budget_remaining,
            "compute_cost": self.state.compute_cost_today,
        }

    def reset_session(self):
        """Reset session state."""
        self.state = SimulationState()
        self.session_outputs = []
        logger.info("Session reset")


# =============================================================================
# Factory Functions
# =============================================================================

def create_simulator(config: Optional[Dict] = None) -> MultimodalSimulator:
    """Create a configured multimodal simulator."""
    if config:
        sim_config = SimulationConfig(**config)
    else:
        sim_config = SimulationConfig()

    return MultimodalSimulator(sim_config)


def run_interaction_from_yaml(yaml_str: str, guards: Optional[Dict[str, bool]] = None) -> Dict:
    """
    Quick helper to run an interaction from YAML.
    """
    spec = InteractionSpec.from_yaml(yaml_str)
    sim = create_simulator()
    return sim.run_interaction(spec, guards)
