"""MIES Liveness Engine - Subtle "alive" animations.

Makes Ara feel alive even when not actively speaking:
- Subtle breathing (slight scale oscillation)
- Occasional blinks
- Small posture shifts
- Glancing toward active window
- Idle animations based on persona mood

These animations should be:
- Subtle (not distracting)
- Contextual (less movement during deep work)
- Energy-aware (reduced when thermodynamically constrained)
"""

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from enum import Enum, auto

from ..modes import ModalityMode, ModalityDecision
from ..context import ModalityContext, ActivityType

logger = logging.getLogger(__name__)


class AnimationState(Enum):
    """Current animation state."""
    IDLE = auto()
    BREATHING = auto()
    BLINKING = auto()
    GLANCING = auto()
    SHIFTING = auto()
    SPEAKING = auto()
    LISTENING = auto()
    THINKING = auto()


@dataclass
class LivenessParams:
    """Parameters controlling liveness behavior."""
    # Breathing
    breath_cycle_seconds: float = 4.0
    breath_amplitude: float = 0.02  # Scale oscillation amount

    # Blinking
    blink_interval_mean: float = 4.0  # Seconds between blinks
    blink_interval_std: float = 1.5
    blink_duration: float = 0.15  # Seconds

    # Glancing
    glance_probability: float = 0.1  # Per cycle
    glance_duration: float = 1.0

    # Posture shifts
    shift_interval_mean: float = 30.0
    shift_interval_std: float = 10.0
    shift_magnitude: float = 0.05

    # Energy modulation
    low_energy_scale: float = 0.5  # Reduce animations when tired


@dataclass
class AnimationFrame:
    """A single frame of animation state."""
    timestamp: float
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    eye_openness: float = 1.0  # 0 = closed, 1 = open
    gaze_x: float = 0.0  # -1 to 1
    gaze_y: float = 0.0  # -1 to 1
    expression: str = "neutral"


class LivenessEngine:
    """
    Engine for subtle liveness animations.

    Generates animation frames that make the avatar feel alive
    without being distracting.
    """

    def __init__(
        self,
        params: Optional[LivenessParams] = None,
        fps: float = 30.0,
    ):
        self.params = params or LivenessParams()
        self.fps = fps
        self.frame_interval = 1.0 / fps

        # State
        self._state = AnimationState.IDLE
        self._current_frame = AnimationFrame(timestamp=time.time())
        self._context: Optional[ModalityContext] = None
        self._modality_decision: Optional[ModalityDecision] = None

        # Animation timers
        self._breath_phase: float = 0.0
        self._last_blink: float = time.time()
        self._next_blink: float = self._schedule_next_blink()
        self._last_shift: float = time.time()
        self._next_shift: float = self._schedule_next_shift()

        # Glance target (if any)
        self._glance_target: Optional[tuple] = None
        self._glance_start: float = 0.0

        # Background thread
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_frame: Optional[Callable[[AnimationFrame], Any]] = None

    def start(self):
        """Start the liveness engine."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._animation_loop,
            daemon=True,
            name="mies-liveness",
        )
        self._thread.start()
        logger.info("Liveness engine started")

    def stop(self):
        """Stop the liveness engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_context(self, ctx: ModalityContext):
        """Update context for animation adjustments."""
        self._context = ctx

    def set_modality_decision(self, decision: ModalityDecision):
        """Update modality decision."""
        self._modality_decision = decision

    def set_state(self, state: AnimationState):
        """Set current animation state."""
        self._state = state

    def set_glance_target(self, target: tuple):
        """Set target for glancing animation."""
        self._glance_target = target
        self._glance_start = time.time()

    def set_frame_callback(self, callback: Callable[[AnimationFrame], Any]):
        """Set callback for animation frames."""
        self._on_frame = callback

    def get_current_frame(self) -> AnimationFrame:
        """Get the current animation frame."""
        return self._current_frame

    def _animation_loop(self):
        """Main animation loop."""
        while self._running:
            start = time.time()

            try:
                self._update_frame()
            except Exception as e:
                logger.error(f"Liveness animation error: {e}")

            # Maintain frame rate
            elapsed = time.time() - start
            sleep_time = max(0, self.frame_interval - elapsed)
            time.sleep(sleep_time)

    def _update_frame(self):
        """Update the current animation frame."""
        now = time.time()

        # Get energy multiplier from context
        energy_mult = self._get_energy_multiplier()

        # Start with default frame
        frame = AnimationFrame(timestamp=now)

        # === Breathing ===
        self._breath_phase += self.frame_interval / self.params.breath_cycle_seconds
        breath_scale = 1.0 + self.params.breath_amplitude * math.sin(
            self._breath_phase * 2 * math.pi
        ) * energy_mult
        frame.scale = breath_scale

        # === Blinking ===
        if now >= self._next_blink:
            # Start blink
            self._last_blink = now
            self._next_blink = now + self._schedule_next_blink()

        blink_elapsed = now - self._last_blink
        if blink_elapsed < self.params.blink_duration:
            # During blink
            blink_progress = blink_elapsed / self.params.blink_duration
            # Quick close, slow open
            if blink_progress < 0.3:
                frame.eye_openness = 1.0 - (blink_progress / 0.3)
            else:
                frame.eye_openness = (blink_progress - 0.3) / 0.7
        else:
            frame.eye_openness = 1.0

        # === Glancing ===
        if self._glance_target:
            glance_elapsed = now - self._glance_start
            if glance_elapsed < self.params.glance_duration:
                # Smoothly look toward target
                target_x, target_y = self._glance_target
                progress = glance_elapsed / self.params.glance_duration
                # Ease in-out
                ease = 0.5 * (1 - math.cos(progress * math.pi))
                frame.gaze_x = target_x * ease * energy_mult
                frame.gaze_y = target_y * ease * energy_mult
            else:
                # Return to center
                self._glance_target = None
                frame.gaze_x = 0
                frame.gaze_y = 0

        # Random glance
        if random.random() < self.params.glance_probability * self.frame_interval:
            self._glance_target = (
                random.uniform(-0.3, 0.3),
                random.uniform(-0.2, 0.2),
            )
            self._glance_start = now

        # === Posture shifts ===
        if now >= self._next_shift:
            # Small random offset
            self._next_shift = now + self._schedule_next_shift()
            # The shift is gradual, handled by smoothing

        # === Context-based adjustments ===
        frame = self._apply_context_adjustments(frame)

        # === State-based expression ===
        frame.expression = self._get_expression()

        # Store and notify
        self._current_frame = frame
        if self._on_frame:
            self._on_frame(frame)

    def _get_energy_multiplier(self) -> float:
        """Get animation energy multiplier from context."""
        if self._context is None:
            return 1.0

        # Reduce animations when:
        # - User is in deep work
        # - Energy is low
        # - System is stressed

        mult = 1.0

        if self._context.activity == ActivityType.DEEP_WORK:
            mult *= 0.5

        if self._context.energy_remaining < 0.3:
            mult *= self.params.low_energy_scale

        if self._context.ara_fatigue > 0.7:
            mult *= 0.7

        return mult

    def _apply_context_adjustments(self, frame: AnimationFrame) -> AnimationFrame:
        """Adjust frame based on context."""
        if self._context is None:
            return frame

        # Meeting: very subtle animations
        if self._context.activity == ActivityType.MEETING:
            frame.scale = 1.0 + (frame.scale - 1.0) * 0.3
            frame.gaze_x *= 0.3
            frame.gaze_y *= 0.3

        # Gaming: minimal animations to avoid distraction
        if self._context.activity == ActivityType.GAMING:
            frame.scale = 1.0 + (frame.scale - 1.0) * 0.2

        return frame

    def _get_expression(self) -> str:
        """Get expression based on state and context."""
        if self._state == AnimationState.SPEAKING:
            return "speaking"
        if self._state == AnimationState.LISTENING:
            return "attentive"
        if self._state == AnimationState.THINKING:
            return "contemplative"

        # Context-based
        if self._context:
            if self._context.valence > 0.3:
                return "pleasant"
            if self._context.valence < -0.3:
                return "concerned"
            if self._context.ara_fatigue > 0.7:
                return "tired"

        return "neutral"

    def _schedule_next_blink(self) -> float:
        """Schedule next blink with random interval."""
        return max(0.5, random.gauss(
            self.params.blink_interval_mean,
            self.params.blink_interval_std,
        ))

    def _schedule_next_shift(self) -> float:
        """Schedule next posture shift."""
        return max(5.0, random.gauss(
            self.params.shift_interval_mean,
            self.params.shift_interval_std,
        ))


# === Factory ===

def create_liveness_engine(
    fps: float = 30.0,
    params: Optional[LivenessParams] = None,
) -> LivenessEngine:
    """Create a liveness engine."""
    return LivenessEngine(params=params, fps=fps)


__all__ = [
    "LivenessEngine",
    "LivenessParams",
    "AnimationState",
    "AnimationFrame",
    "create_liveness_engine",
]
