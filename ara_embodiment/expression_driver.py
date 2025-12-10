"""
Expression Driver
=================

Maps high-level intent to low-level animation/expression commands:
- Lip sync from speech audio/phonemes
- Gaze direction from attention targets
- Gestures from conversation intent
- Facial expressions from emotional state

Outputs commands for the rendering layer (quilt renderer, avatar engine).
"""

from __future__ import annotations

import threading
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class Viseme(str, Enum):
    """Viseme (mouth shape) for lip sync."""
    SILENCE = "silence"   # Mouth closed
    AA = "aa"             # "father"
    AE = "ae"             # "cat"
    AH = "ah"             # "but"
    AO = "ao"             # "bought"
    AW = "aw"             # "cow"
    AY = "ay"             # "bite"
    B_M_P = "bmp"         # Bilabial
    CH_J = "chj"          # "church", "judge"
    D_T_N = "dtn"         # Alveolar
    EH = "eh"             # "bet"
    ER = "er"             # "bird"
    EY = "ey"             # "bait"
    F_V = "fv"            # Labiodental
    G_K = "gk"            # Velar
    IH = "ih"             # "bit"
    IY = "iy"             # "beat"
    L = "l"               # Lateral
    OW = "ow"             # "boat"
    OY = "oy"             # "boy"
    R = "r"               # Rhotic
    S_Z = "sz"            # Sibilant
    SH_ZH = "shzh"        # Postalveolar
    TH = "th"             # Dental fricative
    UH = "uh"             # "book"
    UW = "uw"             # "boot"
    W = "w"               # Labial-velar


class Expression(str, Enum):
    """Facial expressions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"
    THINKING = "thinking"
    CONFUSED = "confused"
    INTERESTED = "interested"
    CONCERNED = "concerned"
    PLAYFUL = "playful"


@dataclass
class LipSyncFrame:
    """A single frame of lip sync data."""
    timestamp: float
    viseme: Viseme
    weight: float = 1.0  # 0-1, how strongly to apply
    transition_ms: float = 50.0  # Blend time from previous

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "viseme": self.viseme.value,
            "weight": self.weight,
            "transition_ms": self.transition_ms,
        }


@dataclass
class GazeCommand:
    """Command to look at a target."""
    target_type: str  # "position", "node", "user", "screen"
    target_value: Any  # Vec3 for position, node_id for node
    urgency: float = 0.5  # 0 = slow/lazy, 1 = snap
    hold_sec: float = 0.0  # How long to hold (0 = until next command)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_type": self.target_type,
            "target_value": str(self.target_value),
            "urgency": self.urgency,
            "hold_sec": self.hold_sec,
        }


@dataclass
class GestureCommand:
    """Command to play a gesture animation."""
    gesture_name: str
    intensity: float = 1.0  # 0-1
    speed: float = 1.0      # Playback speed multiplier
    loop: bool = False
    blend_in_ms: float = 200.0
    blend_out_ms: float = 200.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gesture_name": self.gesture_name,
            "intensity": self.intensity,
            "speed": self.speed,
            "loop": self.loop,
            "blend_in_ms": self.blend_in_ms,
            "blend_out_ms": self.blend_out_ms,
        }


@dataclass
class ExpressionState:
    """Current expression state."""
    base_expression: Expression = Expression.NEUTRAL
    expression_intensity: float = 1.0

    # Blend shapes (0-1 each)
    brow_up: float = 0.0
    brow_down: float = 0.0
    brow_furrow: float = 0.0
    eye_wide: float = 0.0
    eye_squint: float = 0.0
    mouth_smile: float = 0.0
    mouth_frown: float = 0.0
    mouth_open: float = 0.0

    # Current viseme
    current_viseme: Viseme = Viseme.SILENCE
    viseme_weight: float = 0.0

    # Gaze
    gaze_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    gaze_target_id: Optional[str] = None

    # Active gesture
    active_gesture: Optional[str] = None
    gesture_progress: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_expression": self.base_expression.value,
            "expression_intensity": self.expression_intensity,
            "blend_shapes": {
                "brow_up": self.brow_up,
                "brow_down": self.brow_down,
                "brow_furrow": self.brow_furrow,
                "eye_wide": self.eye_wide,
                "eye_squint": self.eye_squint,
                "mouth_smile": self.mouth_smile,
                "mouth_frown": self.mouth_frown,
                "mouth_open": self.mouth_open,
            },
            "viseme": self.current_viseme.value,
            "viseme_weight": self.viseme_weight,
            "gaze_direction": self.gaze_direction,
            "gaze_target_id": self.gaze_target_id,
            "active_gesture": self.active_gesture,
            "gesture_progress": self.gesture_progress,
        }


class ExpressionDriver:
    """
    Drives avatar expression based on high-level commands.

    Thread-safe: state updates protected by lock.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._state = ExpressionState()

        # Animation queues
        self._lip_sync_queue: List[LipSyncFrame] = []
        self._gesture_queue: List[GestureCommand] = []

        # Current commands
        self._current_gaze: Optional[GazeCommand] = None
        self._gaze_start_time: float = 0.0

        # Expression presets
        self._expression_presets: Dict[Expression, Dict[str, float]] = {
            Expression.NEUTRAL: {},
            Expression.HAPPY: {"mouth_smile": 0.7, "eye_squint": 0.2},
            Expression.SAD: {"mouth_frown": 0.5, "brow_up": 0.3},
            Expression.SURPRISED: {"eye_wide": 0.8, "brow_up": 0.6, "mouth_open": 0.5},
            Expression.ANGRY: {"brow_down": 0.6, "brow_furrow": 0.7, "eye_squint": 0.3},
            Expression.THINKING: {"brow_furrow": 0.3, "eye_squint": 0.2},
            Expression.CONFUSED: {"brow_furrow": 0.5, "brow_up": 0.3},
            Expression.INTERESTED: {"brow_up": 0.4, "eye_wide": 0.3},
            Expression.CONCERNED: {"brow_furrow": 0.4, "mouth_frown": 0.2},
            Expression.PLAYFUL: {"mouth_smile": 0.5, "brow_up": 0.2, "eye_squint": 0.1},
        }

        # Timing
        self._last_update = time.time()

    def get_state(self) -> ExpressionState:
        """Get current expression state."""
        with self._lock:
            return ExpressionState(
                base_expression=self._state.base_expression,
                expression_intensity=self._state.expression_intensity,
                brow_up=self._state.brow_up,
                brow_down=self._state.brow_down,
                brow_furrow=self._state.brow_furrow,
                eye_wide=self._state.eye_wide,
                eye_squint=self._state.eye_squint,
                mouth_smile=self._state.mouth_smile,
                mouth_frown=self._state.mouth_frown,
                mouth_open=self._state.mouth_open,
                current_viseme=self._state.current_viseme,
                viseme_weight=self._state.viseme_weight,
                gaze_direction=self._state.gaze_direction,
                gaze_target_id=self._state.gaze_target_id,
                active_gesture=self._state.active_gesture,
                gesture_progress=self._state.gesture_progress,
            )

    def set_expression(self, expression: str, intensity: float = 1.0) -> None:
        """Set the base facial expression."""
        with self._lock:
            try:
                expr = Expression(expression)
            except ValueError:
                expr = Expression.NEUTRAL

            self._state.base_expression = expr
            self._state.expression_intensity = max(0.0, min(1.0, intensity))

            # Apply preset blend shapes
            preset = self._expression_presets.get(expr, {})
            for key, value in preset.items():
                setattr(self._state, key, value * intensity)

    def queue_lip_sync(self, frames: List[LipSyncFrame]) -> None:
        """Queue lip sync frames for playback."""
        with self._lock:
            self._lip_sync_queue.extend(frames)
            # Sort by timestamp
            self._lip_sync_queue.sort(key=lambda f: f.timestamp)

    def clear_lip_sync(self) -> None:
        """Clear pending lip sync."""
        with self._lock:
            self._lip_sync_queue.clear()
            self._state.current_viseme = Viseme.SILENCE
            self._state.viseme_weight = 0.0

    def look_at(self, target: Any, urgency: float = 0.5) -> None:
        """
        Set gaze target.

        target can be:
        - "user": look at primary user
        - "screen": look at screen/camera
        - Vec3 or tuple: look at position
        - str: look at node by ID
        """
        with self._lock:
            if target == "user":
                cmd = GazeCommand("user", None, urgency)
            elif target == "screen":
                cmd = GazeCommand("screen", None, urgency)
            elif isinstance(target, (tuple, list)):
                cmd = GazeCommand("position", target, urgency)
            else:
                cmd = GazeCommand("node", str(target), urgency)

            self._current_gaze = cmd
            self._gaze_start_time = time.time()
            self._state.gaze_target_id = str(target)

    def play_gesture(self, gesture_name: str, intensity: float = 1.0) -> None:
        """Play a gesture animation."""
        with self._lock:
            cmd = GestureCommand(gesture_name, intensity)
            self._gesture_queue.append(cmd)
            self._state.active_gesture = gesture_name
            self._state.gesture_progress = 0.0

    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update expression state.

        Called each frame by the renderer.
        Returns the current state as dict for rendering.
        """
        with self._lock:
            now = time.time()

            # Update lip sync
            self._update_lip_sync(now)

            # Update gaze
            self._update_gaze(dt)

            # Update gesture
            self._update_gesture(dt)

            self._last_update = now
            return self._state.to_dict()

    def _update_lip_sync(self, now: float) -> None:
        """Update lip sync from queue."""
        if not self._lip_sync_queue:
            # Decay to silence
            self._state.viseme_weight *= 0.9
            if self._state.viseme_weight < 0.01:
                self._state.current_viseme = Viseme.SILENCE
                self._state.viseme_weight = 0.0
            return

        # Find current frame
        while self._lip_sync_queue and self._lip_sync_queue[0].timestamp < now:
            frame = self._lip_sync_queue.pop(0)
            self._state.current_viseme = frame.viseme
            self._state.viseme_weight = frame.weight

    def _update_gaze(self, dt: float) -> None:
        """Update gaze direction."""
        if not self._current_gaze:
            return

        # Calculate target direction based on gaze command
        target_dir = (0.0, 0.0, 1.0)  # Default forward

        if self._current_gaze.target_type == "screen":
            target_dir = (0.0, 0.0, 1.0)  # Look at camera
        elif self._current_gaze.target_type == "position":
            pos = self._current_gaze.target_value
            if pos:
                # Normalize direction to position
                length = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                if length > 0.001:
                    target_dir = (pos[0]/length, pos[1]/length, pos[2]/length)

        # Interpolate gaze direction
        urgency = self._current_gaze.urgency
        blend = min(1.0, dt * (2.0 + urgency * 8.0))  # Faster with higher urgency

        current = self._state.gaze_direction
        self._state.gaze_direction = (
            current[0] + (target_dir[0] - current[0]) * blend,
            current[1] + (target_dir[1] - current[1]) * blend,
            current[2] + (target_dir[2] - current[2]) * blend,
        )

        # Check if hold expired
        if self._current_gaze.hold_sec > 0:
            elapsed = time.time() - self._gaze_start_time
            if elapsed > self._current_gaze.hold_sec:
                self._current_gaze = None

    def _update_gesture(self, dt: float) -> None:
        """Update gesture progress."""
        if not self._state.active_gesture:
            return

        # Advance gesture progress
        self._state.gesture_progress += dt * 2.0  # Assume ~0.5s gestures

        if self._state.gesture_progress >= 1.0:
            # Check for queued gestures
            if self._gesture_queue:
                next_gesture = self._gesture_queue.pop(0)
                self._state.active_gesture = next_gesture.gesture_name
                self._state.gesture_progress = 0.0
            else:
                self._state.active_gesture = None
                self._state.gesture_progress = 0.0

    def generate_lip_sync_from_text(
        self,
        text: str,
        start_time: float,
        words_per_minute: float = 150.0,
    ) -> List[LipSyncFrame]:
        """
        Generate approximate lip sync from text.

        This is a simple approximation; real systems use phoneme analysis.
        """
        frames = []

        # Simple mapping of letters to visemes
        letter_visemes = {
            'a': Viseme.AA, 'e': Viseme.EH, 'i': Viseme.IY,
            'o': Viseme.OW, 'u': Viseme.UW,
            'b': Viseme.B_M_P, 'm': Viseme.B_M_P, 'p': Viseme.B_M_P,
            'f': Viseme.F_V, 'v': Viseme.F_V,
            'd': Viseme.D_T_N, 't': Viseme.D_T_N, 'n': Viseme.D_T_N,
            's': Viseme.S_Z, 'z': Viseme.S_Z,
            'l': Viseme.L, 'r': Viseme.R, 'w': Viseme.W,
        }

        # Calculate timing
        words = text.split()
        total_duration = (len(words) / words_per_minute) * 60.0
        chars = [c.lower() for c in text if c.isalpha()]

        if not chars:
            return frames

        time_per_char = total_duration / len(chars)
        current_time = start_time

        for char in chars:
            viseme = letter_visemes.get(char, Viseme.SILENCE)
            frames.append(LipSyncFrame(
                timestamp=current_time,
                viseme=viseme,
                weight=0.8,
                transition_ms=time_per_char * 500,  # Half the duration as transition
            ))
            current_time += time_per_char

        # End with silence
        frames.append(LipSyncFrame(
            timestamp=current_time,
            viseme=Viseme.SILENCE,
            weight=0.0,
            transition_ms=100,
        ))

        return frames

    def get_stats(self) -> Dict[str, Any]:
        """Get driver statistics."""
        with self._lock:
            return {
                "current_expression": self._state.base_expression.value,
                "current_viseme": self._state.current_viseme.value,
                "lip_sync_queue_size": len(self._lip_sync_queue),
                "gesture_queue_size": len(self._gesture_queue),
                "active_gesture": self._state.active_gesture,
                "gaze_target": self._state.gaze_target_id,
            }
