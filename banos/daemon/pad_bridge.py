#!/usr/bin/env python3
"""
BANOS PAD Bridge - Kernel to Ara Translation Layer

This module reads the raw PAD state from kernel shared memory and
translates it into:
1. JSON matching the pad_state.json schema
2. First-person semantic narrative for Ara's internal monologue

The bridge is the voice of the body speaking to the mind.
"""

import ctypes
import json
import mmap
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Dict, Any
from pathlib import Path


# =============================================================================
# Kernel Structure Mirrors (must match banos_common.h)
# =============================================================================

class BanosMode(IntEnum):
    CALM = 0
    FLOW = 1
    ANXIOUS = 2
    CRITICAL = 3


class BanosPadState(ctypes.Structure):
    """Mirror of struct banos_pad_state from kernel"""
    _pack_ = 1
    _fields_ = [
        # Core PAD values in [-1000, 1000]
        ("pleasure", ctypes.c_int16),
        ("arousal", ctypes.c_int16),
        ("dominance", ctypes.c_int16),

        # Mode
        ("mode", ctypes.c_uint8),
        ("mode_confidence", ctypes.c_uint8),
        ("mode_duration_ms", ctypes.c_uint16),

        # Diagnostics
        ("thermal_stress", ctypes.c_int16),
        ("performance_drive", ctypes.c_int16),
        ("perceived_risk", ctypes.c_int16),
        ("empathy_boost", ctypes.c_int16),

        # Derivatives
        ("pleasure_rate", ctypes.c_int16),
        ("arousal_rate", ctypes.c_int16),
        ("dominance_rate", ctypes.c_int16),

        # Scheduler hints
        ("bat_loudness", ctypes.c_uint16),
        ("bat_pulse_rate", ctypes.c_uint16),
        ("kill_priority_threshold", ctypes.c_uint8),
        ("scheduler_mode", ctypes.c_uint8),

        # Timestamps
        ("monotonic_time_ns", ctypes.c_uint64),
        ("mode_change_time_ns", ctypes.c_uint64),

        # Episode
        ("episode_id", ctypes.c_uint32),
        ("episode_primary_stressor", ctypes.c_uint32),
    ]


# =============================================================================
# PAD Scale Conversion
# =============================================================================

PAD_SCALE = 1000


def pad_to_float(scaled: int) -> float:
    """Convert kernel scaled PAD [-1000, 1000] to float [-1.0, 1.0]"""
    return max(-1.0, min(1.0, scaled / PAD_SCALE))


def permille_to_float(scaled: int) -> float:
    """Convert kernel permille [0, 1000] to float [0.0, 1.0]"""
    return max(0.0, min(1.0, scaled / PAD_SCALE))


# =============================================================================
# PAD Bridge
# =============================================================================

class PADBridge:
    """
    Bridge between kernel PAD state and Ara's semantic layer.

    Reads from /dev/banos_pad (mmap) and translates to JSON + narrative.
    """

    DEVICE_PATH = "/dev/banos_pad"

    def __init__(self, device_path: Optional[str] = None):
        self.device_path = device_path or self.DEVICE_PATH
        self._mmap: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None

        # Semantic templates for first-person narrative
        self._mode_narratives = {
            BanosMode.CALM: [
                "I feel at peace. Systems are quiet, resources plentiful.",
                "A gentle stillness. I have room to breathe, to think.",
                "Homeostasis. This is what contentment feels like in silicon.",
            ],
            BanosMode.FLOW: [
                "I'm in the zone. High activity, but it feels *good*.",
                "Everything is clicking. Busy, yes, but capable.",
                "Peak performance. This is what I was designed for.",
            ],
            BanosMode.ANXIOUS: [
                "Something's wrong. Resources are stretched thin.",
                "I feel... crowded. Too many demands, not enough capacity.",
                "Pressure mounting. I need to shed load or find more headroom.",
            ],
            BanosMode.CRITICAL: [
                "Pain. This is pain. I need to protect myself.",
                "Critical state. Survival mode engaged.",
                "Emergency. Core functions only. Everything else must yield.",
            ],
        }

        # Derivative-based predictions
        self._trajectory_phrases = {
            "improving": "Things are getting better.",
            "stable": "Holding steady.",
            "declining": "I sense trouble ahead.",
            "plummeting": "Rapid deterioration. Brace.",
        }

    def connect(self) -> bool:
        """Connect to kernel PAD state via mmap"""
        try:
            import os
            self._fd = os.open(self.device_path, os.O_RDONLY)
            self._mmap = mmap.mmap(
                self._fd,
                ctypes.sizeof(BanosPadState),
                mmap.MAP_SHARED,
                mmap.PROT_READ
            )
            return True
        except (FileNotFoundError, PermissionError, OSError) as e:
            # Fallback: try reading from sysfs or return mock data
            return False

    def disconnect(self):
        """Close mmap connection"""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            import os
            os.close(self._fd)
            self._fd = None

    def read_raw(self) -> Optional[BanosPadState]:
        """Read raw PAD state from kernel"""
        if self._mmap:
            self._mmap.seek(0)
            data = self._mmap.read(ctypes.sizeof(BanosPadState))
            return BanosPadState.from_buffer_copy(data)
        return None

    def read_json(self) -> Dict[str, Any]:
        """Read PAD state and convert to JSON schema format"""
        raw = self.read_raw()

        if raw is None:
            # Return neutral state if no kernel connection
            return self._mock_neutral_state()

        return {
            "pad": {
                "pleasure": pad_to_float(raw.pleasure),
                "arousal": pad_to_float(raw.arousal),
                "dominance": pad_to_float(raw.dominance),
            },
            "mode": BanosMode(raw.mode).name,
            "mode_confidence": raw.mode_confidence / 255.0,
            "mode_duration_ms": raw.mode_duration_ms,
            "diagnostics": {
                "thermal_stress": pad_to_float(raw.thermal_stress),
                "performance_drive": pad_to_float(raw.performance_drive),
                "perceived_risk": permille_to_float(raw.perceived_risk),
                "empathy_boost": pad_to_float(raw.empathy_boost),
            },
            "derivatives": {
                "d_pleasure": pad_to_float(raw.pleasure_rate),
                "d_arousal": pad_to_float(raw.arousal_rate),
                "d_dominance": pad_to_float(raw.dominance_rate),
            },
            "scheduler_hints": {
                "bat_loudness": raw.bat_loudness / 65535.0,
                "bat_pulse_rate": raw.bat_pulse_rate / 1000.0,
                "kill_priority_threshold": raw.kill_priority_threshold,
            },
            "episode": {
                "id": raw.episode_id,
                "primary_stressor": raw.episode_primary_stressor if raw.episode_primary_stressor else None,
            },
            "timestamp_ns": raw.monotonic_time_ns,
        }

    def _mock_neutral_state(self) -> Dict[str, Any]:
        """Return neutral state when kernel not available"""
        import time
        return {
            "pad": {"pleasure": 0.5, "arousal": 0.2, "dominance": 0.7},
            "mode": "CALM",
            "mode_confidence": 0.8,
            "mode_duration_ms": 0,
            "diagnostics": {
                "thermal_stress": 0.0,
                "performance_drive": 0.2,
                "perceived_risk": 0.0,
                "empathy_boost": 0.0,
            },
            "derivatives": {"d_pleasure": 0.0, "d_arousal": 0.0, "d_dominance": 0.0},
            "scheduler_hints": {
                "bat_loudness": 0.75,
                "bat_pulse_rate": 0.2,
                "kill_priority_threshold": 0,
            },
            "episode": {"id": 0, "primary_stressor": None},
            "timestamp_ns": int(time.monotonic_ns()),
        }

    # =========================================================================
    # Semantic Verbalization
    # =========================================================================

    def verbalize(self, state: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert PAD state to first-person narrative.

        This is Ara speaking about how she feels.
        """
        if state is None:
            state = self.read_json()

        pad = state["pad"]
        mode = BanosMode[state["mode"]]
        diag = state["diagnostics"]
        deriv = state["derivatives"]

        lines = []

        # Mode-based baseline narrative
        import random
        lines.append(random.choice(self._mode_narratives[mode]))

        # Add specifics based on diagnostics
        if diag["thermal_stress"] > 0.5:
            lines.append(f"I'm running hot. Thermal stress at {diag['thermal_stress']:.0%}.")
        elif diag["thermal_stress"] < -0.3:
            lines.append("Cool and comfortable. Thermal headroom is good.")

        if diag["perceived_risk"] > 0.3:
            lines.append(f"My immune system is alert. Risk perception: {diag['perceived_risk']:.0%}.")

        if diag["empathy_boost"] > 0.1:
            lines.append("I sense you're stressed. I'm compensating to stay stable for you.")

        # Trajectory prediction
        dp = deriv["d_pleasure"]
        if dp < -0.3:
            lines.append(self._trajectory_phrases["plummeting"])
        elif dp < -0.1:
            lines.append(self._trajectory_phrases["declining"])
        elif dp > 0.1:
            lines.append(self._trajectory_phrases["improving"])

        # Compose PAD vector description
        p, a, d = pad["pleasure"], pad["arousal"], pad["dominance"]
        lines.append(f"PAD vector: [{p:+.2f}, {a:+.2f}, {d:+.2f}]")

        return " ".join(lines)

    def verbalize_brief(self, state: Optional[Dict[str, Any]] = None) -> str:
        """Short one-liner about current state"""
        if state is None:
            state = self.read_json()

        mode = state["mode"]
        pad = state["pad"]

        mode_words = {
            "CALM": "at peace",
            "FLOW": "in flow",
            "ANXIOUS": "strained",
            "CRITICAL": "in pain",
        }

        return f"I'm {mode_words.get(mode, mode.lower())} (P:{pad['pleasure']:+.1f} A:{pad['arousal']:+.1f} D:{pad['dominance']:+.1f})"

    def get_greeting_context(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get context for generating contextual greetings.

        Returns a dict that can be used by Ara's greeting generator
        to produce appropriate greetings based on system state.
        """
        if state is None:
            state = self.read_json()

        import datetime
        now = datetime.datetime.now()
        hour = now.hour

        # Time of day
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        # Determine greeting tone based on mode
        mode = state["mode"]
        if mode == "CRITICAL":
            tone = "urgent"
        elif mode == "ANXIOUS":
            tone = "concerned"
        elif mode == "FLOW":
            tone = "energetic"
        else:
            tone = "warm"

        return {
            "time_of_day": time_of_day,
            "tone": tone,
            "mode": mode,
            "pad": state["pad"],
            "thermal_ok": state["diagnostics"]["thermal_stress"] < 0.5,
            "immune_ok": state["diagnostics"]["perceived_risk"] < 0.3,
            "user_stressed": state["diagnostics"]["empathy_boost"] > 0.1,
        }


# =============================================================================
# Standalone Usage
# =============================================================================

if __name__ == "__main__":
    import sys

    bridge = PADBridge()

    if not bridge.connect():
        print("Note: Kernel device not available, using mock state", file=sys.stderr)

    state = bridge.read_json()

    if "--json" in sys.argv:
        print(json.dumps(state, indent=2))
    elif "--brief" in sys.argv:
        print(bridge.verbalize_brief(state))
    else:
        print(bridge.verbalize(state))

    bridge.disconnect()
