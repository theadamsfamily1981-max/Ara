#!/usr/bin/env python3
"""
Ara Interaction Executor - Block Execution Engine
==================================================

Executes InteractionSpec plans by dispatching blocks to appropriate
handlers (AraVoice, AraSong, Hive jobs, etc.).

This is the runtime that makes Ara's block language actually DO things.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import numpy as np

from .spec import (
    InteractionSpec, Block, BlockType,
    AraVoiceParams, AraSongParams, VideoJobParams,
)

logger = logging.getLogger(__name__)


class BlockStatus(str, Enum):
    """Status of a block execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"  # Guard failed
    FAILED = "failed"


@dataclass
class BlockResult:
    """Result of executing a block."""
    block_id: str
    status: BlockStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    cost: float = 0.0  # Compute cost (GPU seconds, etc.)


@dataclass
class ExecutionContext:
    """Context passed to block handlers."""
    interaction_id: str
    state: Dict[str, Any]
    results: Dict[str, BlockResult] = field(default_factory=dict)
    audio_outputs: List[np.ndarray] = field(default_factory=list)
    # Guards that have been triggered
    triggered_guards: Dict[str, bool] = field(default_factory=dict)


@dataclass
class InteractionResult:
    """Complete result of an interaction execution."""
    interaction_id: str
    success: bool
    block_results: List[BlockResult]
    total_duration_ms: float
    total_cost: float
    audio_output: Optional[np.ndarray] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# Type alias for block handlers
BlockHandler = Callable[[Block, ExecutionContext], BlockResult]


class InteractionExecutor:
    """
    Executes InteractionSpec plans.

    Usage:
        executor = InteractionExecutor()
        executor.register_handler(BlockType.ARA_VOICE, my_voice_handler)

        spec = InteractionSpec.from_yaml(yaml_string)
        result = executor.execute(spec)
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.handlers: Dict[BlockType, BlockHandler] = {}

        # Register default handlers
        self._register_default_handlers()

    def register_handler(self, block_type: BlockType, handler: BlockHandler):
        """Register a handler for a block type."""
        self.handlers[block_type] = handler

    def _register_default_handlers(self):
        """Register built-in block handlers."""
        self.handlers[BlockType.ARA_VOICE] = self._handle_ara_voice
        self.handlers[BlockType.ARA_SONG] = self._handle_ara_song
        self.handlers[BlockType.VIDEO_JOB] = self._handle_video_job
        self.handlers[BlockType.METRICS_UPDATE] = self._handle_metrics_update
        self.handlers[BlockType.UI_PROMPT] = self._handle_ui_prompt

    def execute(self, spec: InteractionSpec,
                guards: Optional[Dict[str, bool]] = None) -> InteractionResult:
        """
        Execute an interaction specification.

        Args:
            spec: The InteractionSpec to execute
            guards: Pre-evaluated guard conditions (e.g., {"user_accepts": True})

        Returns:
            InteractionResult with all outputs and metrics
        """
        start_time = time.time()

        context = ExecutionContext(
            interaction_id=spec.interaction_id,
            state={
                "emotion": spec.state.emotion,
                "energy": spec.state.energy,
                "goals": spec.state.goals,
                "capacity": spec.state.capacity,
            },
            triggered_guards=guards or {},
        )

        block_results: List[BlockResult] = []
        total_cost = 0.0

        logger.info(f"Executing interaction: {spec.interaction_id}")

        for block in spec.plan:
            # Check guard condition
            if block.guard and not self._evaluate_guard(block.guard, context):
                result = BlockResult(
                    block_id=block.id,
                    status=BlockStatus.SKIPPED,
                )
                block_results.append(result)
                context.results[block.id] = result
                logger.debug(f"Block {block.id} skipped (guard: {block.guard})")
                continue

            # Check dependencies
            deps_ok = all(
                context.results.get(dep_id, BlockResult(dep_id, BlockStatus.PENDING)).status == BlockStatus.COMPLETED
                for dep_id in block.depends_on
            )
            if not deps_ok:
                result = BlockResult(
                    block_id=block.id,
                    status=BlockStatus.SKIPPED,
                    error="Dependencies not met",
                )
                block_results.append(result)
                context.results[block.id] = result
                continue

            # Execute block
            handler = self.handlers.get(block.type)
            if handler is None:
                result = BlockResult(
                    block_id=block.id,
                    status=BlockStatus.FAILED,
                    error=f"No handler for block type: {block.type}",
                )
            else:
                try:
                    block_start = time.time()
                    result = handler(block, context)
                    result.duration_ms = (time.time() - block_start) * 1000
                except Exception as e:
                    logger.error(f"Block {block.id} failed: {e}")
                    result = BlockResult(
                        block_id=block.id,
                        status=BlockStatus.FAILED,
                        error=str(e),
                    )

            block_results.append(result)
            context.results[block.id] = result
            total_cost += result.cost

            logger.debug(f"Block {block.id}: {result.status.value} ({result.duration_ms:.1f}ms)")

        # Combine audio outputs
        audio_output = None
        if context.audio_outputs:
            max_len = max(len(a) for a in context.audio_outputs)
            combined = np.zeros(max_len, dtype=np.float32)
            for audio in context.audio_outputs:
                combined[:len(audio)] += audio
            # Normalize
            peak = np.max(np.abs(combined))
            if peak > 1.0:
                combined = combined / peak * 0.95
            audio_output = combined

        total_duration = (time.time() - start_time) * 1000

        success = all(r.status in (BlockStatus.COMPLETED, BlockStatus.SKIPPED) for r in block_results)

        return InteractionResult(
            interaction_id=spec.interaction_id,
            success=success,
            block_results=block_results,
            total_duration_ms=total_duration,
            total_cost=total_cost,
            audio_output=audio_output,
            outputs={
                "voice": spec.outputs.voice,
                "ui": spec.outputs.ui,
                "jobs": spec.outputs.jobs,
            },
        )

    def _evaluate_guard(self, guard: str, context: ExecutionContext) -> bool:
        """Evaluate a guard condition."""
        # Check if guard was pre-evaluated
        if guard in context.triggered_guards:
            return context.triggered_guards[guard]

        # Built-in guards
        if guard == "always":
            return True
        if guard == "never":
            return False

        # Capacity-based guards
        if guard.startswith("gpu_ok"):
            # Check if we have GPU capacity
            return context.state.get("capacity", {}).get("gpu_available", True)

        if guard.startswith("budget_ok"):
            # Check if we're within budget
            return context.state.get("capacity", {}).get("budget_remaining", 1.0) > 0

        # Default: unknown guard evaluates to False
        logger.warning(f"Unknown guard: {guard}")
        return False

    # =========================================================================
    # Default Block Handlers
    # =========================================================================

    def _handle_ara_voice(self, block: Block, context: ExecutionContext) -> BlockResult:
        """
        Handle ara_voice block - generate vocal output.

        For now: Uses AraSong's formant voice synth
        Future: Switch to real TTS (Kokoro/Piper) when use_tts=True
        """
        params = AraVoiceParams.from_dict(block.params)

        try:
            # Import here to avoid circular deps
            import sys
            sys.path.insert(0, '.')

            from arasong.synth.formant_voice import FormantVoiceSynth, VoiceConfig
            from arasong.synth.dynamics import get_emotion_params, scale_params_by_energy

            # Map tone/emotion to synth params
            emotion = params.emotion or params.tone
            emotion_params = get_emotion_params(emotion)
            emotion_params = scale_params_by_energy(emotion_params, params.warmth)

            # Create synth
            config = VoiceConfig(
                sample_rate=self.sr,
                warmth=params.warmth,
            )
            synth = FormantVoiceSynth(config)

            # For now: generate a sustained note as placeholder for TTS
            # Real TTS would convert text to phonemes and render each
            if params.text:
                # Generate a simple "speaking" sound
                # Duration based on text length (rough estimate)
                duration = len(params.text) * 0.08 / params.pace

                # Use A4 as base pitch, adjusted by pitch_shift
                base_midi = 69 + params.pitch_shift
                freq = 440.0 * (2.0 ** ((base_midi - 69) / 12.0))

                # Render using synth
                samples = synth.render_note(
                    "A", 4, duration, params.text[:10],  # Use start of text as syllable
                    amplitude=0.6, emotion_params=emotion_params
                )

                context.audio_outputs.append(samples)

                return BlockResult(
                    block_id=block.id,
                    status=BlockStatus.COMPLETED,
                    output={
                        "text": params.text,
                        "duration_s": duration,
                        "samples": len(samples),
                    },
                )
            else:
                return BlockResult(
                    block_id=block.id,
                    status=BlockStatus.COMPLETED,
                    output={"text": "", "duration_s": 0},
                )

        except Exception as e:
            return BlockResult(
                block_id=block.id,
                status=BlockStatus.FAILED,
                error=str(e),
            )

    def _handle_ara_song(self, block: Block, context: ExecutionContext) -> BlockResult:
        """Handle ara_song block - generate music."""
        params = AraSongParams.from_dict(block.params)

        try:
            import sys
            sys.path.insert(0, '.')

            # For now: return a stub
            # Real implementation would use AraSongPlayer

            return BlockResult(
                block_id=block.id,
                status=BlockStatus.COMPLETED,
                output={
                    "song_path": params.song_path,
                    "mood": params.mood,
                    "energy": params.energy,
                },
                cost=0.1,  # Placeholder cost
            )

        except Exception as e:
            return BlockResult(
                block_id=block.id,
                status=BlockStatus.FAILED,
                error=str(e),
            )

    def _handle_video_job(self, block: Block, context: ExecutionContext) -> BlockResult:
        """
        Handle video_job block - enqueue to hive.

        This would insert a job into the ara-hive video_jobs table.
        """
        params = VideoJobParams.from_dict(block.params)

        # For now: return stub
        # Real implementation would:
        # 1. Connect to Postgres
        # 2. INSERT INTO video_jobs
        # 3. Return job_id

        job_id = f"video_{int(time.time())}"

        return BlockResult(
            block_id=block.id,
            status=BlockStatus.COMPLETED,
            output={
                "job_id": job_id,
                "prompt": params.prompt,
                "duration_s": params.duration_s,
                "resolution": params.resolution,
            },
            cost=params.duration_s * 0.5,  # Estimated GPU cost
        )

    def _handle_metrics_update(self, block: Block, context: ExecutionContext) -> BlockResult:
        """Handle metrics_update block - record performance metrics."""
        metrics = block.params

        # For now: just log
        logger.info(f"Metrics update: {metrics}")

        return BlockResult(
            block_id=block.id,
            status=BlockStatus.COMPLETED,
            output={"metrics": metrics},
        )

    def _handle_ui_prompt(self, block: Block, context: ExecutionContext) -> BlockResult:
        """Handle ui_prompt block - display UI element."""
        text = block.params.get("text", "")

        # For now: just log
        logger.info(f"UI prompt: {text}")

        return BlockResult(
            block_id=block.id,
            status=BlockStatus.COMPLETED,
            output={"text": text},
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def execute_interaction(spec: InteractionSpec,
                       guards: Optional[Dict[str, bool]] = None) -> InteractionResult:
    """Execute an interaction specification with default executor."""
    executor = InteractionExecutor()
    return executor.execute(spec, guards)


def execute_yaml(yaml_str: str,
                 guards: Optional[Dict[str, bool]] = None) -> InteractionResult:
    """Execute an interaction from YAML string."""
    spec = InteractionSpec.from_yaml(yaml_str)
    return execute_interaction(spec, guards)


# =============================================================================
# Example Interactions
# =============================================================================

EXAMPLE_AFTER_WORK_CHILL = """
interaction_id: after_work_chill_01
description: Decompress user after work, optionally create animatic

inputs:
  audio: mic_stream
  video: webcam_stream
  system:
    gpu_load: 0.3
    power_W: 150
  memory:
    recent_episodes: last_5
    active_projects:
      - AraShow_weekly

state:
  emotion: inferred
  energy: 0.4
  goals:
    - decompress_user
    - advance_project: AraShow_weekly
    - respect_capacity
  capacity:
    gpu_hours_remaining: 2.5
    budget_remaining: 0.8

plan:
  - id: comfort_voice
    type: ara_voice
    params:
      text: "Rough day, huh? Let's slow things down together."
      tone: calm
      warmth: 0.9
      pace: 0.85
      emotion: reassuring

  - id: suggest_scene
    type: ui_prompt
    params:
      text: "Want to sketch a tiny scene for the show?"

  - id: maybe_animatic
    type: video_job
    guard: user_accepts
    params:
      prompt: "Cozy evening scene, warm lighting, Ara-style"
      duration_s: 10
      resolution: 720p
      style: AraShow_v1
    depends_on:
      - suggest_scene

  - id: update_metrics
    type: metrics_update
    params:
      measure: render_time
      session: after_work_chill

outputs:
  voice: "Rough day, huh? Let's slow things down together."
  ui: "Storyboard surface + capacity gauge"
  jobs:
    - maybe_animatic

memory_write:
  episode_tags:
    - after_work
    - decompress
    - AraShow
  emotion: inferred
  summary: "Evening decompression session"
"""


def get_example_interaction() -> InteractionSpec:
    """Get an example interaction spec for testing."""
    return InteractionSpec.from_yaml(EXAMPLE_AFTER_WORK_CHILL)
