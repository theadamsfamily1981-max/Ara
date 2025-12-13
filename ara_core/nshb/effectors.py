#!/usr/bin/env python3
"""
NSHB Effectors - Safe Embodied Feedback Channels
=================================================

Maps repair vectors (Δλ, ΔΠ) to safe, embodied actions.

CRITICAL SAFETY CONSTRAINT:
    All effectors operate through normal sensory channels:
    - Visual feedback (colors, patterns, indicators)
    - Auditory feedback (tones, music, voice)
    - Haptic feedback (Somatic Loom integration)
    - Breathing guidance (visual/audio pacing)
    - UI adaptation (task difficulty, pacing)

    NO direct neural stimulation, NO pharmacology.
    These are wellness-level interventions only.

Integration with Somatic Loom:
    For haptic feedback, we translate repair vectors to the
    HapticGrammar patterns defined in ara_core.somatic.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy as np

from .control import RepairVector, InterventionUrgency

# Try to import Somatic Loom for haptic integration
try:
    from ..somatic import HapticGrammar, HapticPattern, SomaticLoom, GUTCState as SomaticGUTCState
    SOMATIC_AVAILABLE = True
except ImportError:
    SOMATIC_AVAILABLE = False


# =============================================================================
# Effector Types
# =============================================================================

class EffectorModality(Enum):
    """Available effector modalities."""
    VISUAL = auto()        # Screen-based visual feedback
    AUDITORY = auto()      # Sound-based feedback
    HAPTIC = auto()        # Somatic Loom / vibration
    BREATHING = auto()     # Breathing pace guidance
    UI_ADAPT = auto()      # Task/interface adaptation


@dataclass
class EffectorCommand:
    """
    Command to an effector.

    This is the concrete action derived from a repair vector.
    """
    modality: EffectorModality
    timestamp: float
    duration_ms: int = 1000

    # Modality-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Source
    source_repair: Optional[RepairVector] = None

    def __str__(self) -> str:
        return f"[{self.modality.name}] {self.parameters}"


# =============================================================================
# Effector Base Class
# =============================================================================

class Effector(ABC):
    """Abstract base class for effectors."""

    @property
    @abstractmethod
    def modality(self) -> EffectorModality:
        pass

    @abstractmethod
    def execute(self, command: EffectorCommand) -> bool:
        """Execute a command. Returns True on success."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop any ongoing output."""
        pass


# =============================================================================
# Visual Effector
# =============================================================================

class VisualEffector(Effector):
    """
    Visual feedback effector.

    Provides feedback through:
    - Color temperature shifts (warm = activating, cool = calming)
    - Brightness modulation
    - Simple geometric patterns (expanding = growth, contracting = focus)
    - Status indicators
    """

    @property
    def modality(self) -> EffectorModality:
        return EffectorModality.VISUAL

    def __init__(self, output_callback: Optional[Callable[[Dict], None]] = None):
        self.output_callback = output_callback
        self.current_state: Dict[str, Any] = {
            "color_temp_k": 6500,    # Neutral white
            "brightness": 1.0,
            "pattern": "none",
            "indicator_color": "green",
        }

    def execute(self, command: EffectorCommand) -> bool:
        params = command.parameters

        # Update state
        if "color_temp_k" in params:
            self.current_state["color_temp_k"] = params["color_temp_k"]
        if "brightness" in params:
            self.current_state["brightness"] = params["brightness"]
        if "pattern" in params:
            self.current_state["pattern"] = params["pattern"]
        if "indicator_color" in params:
            self.current_state["indicator_color"] = params["indicator_color"]

        # Send to output
        if self.output_callback:
            self.output_callback(self.current_state)

        return True

    def stop(self) -> None:
        self.current_state = {
            "color_temp_k": 6500,
            "brightness": 1.0,
            "pattern": "none",
            "indicator_color": "neutral",
        }
        if self.output_callback:
            self.output_callback(self.current_state)


# =============================================================================
# Auditory Effector
# =============================================================================

class AuditoryEffector(Effector):
    """
    Auditory feedback effector.

    Provides feedback through:
    - Ambient soundscapes
    - Rhythmic tones (for pacing)
    - Notification sounds
    - Music tempo/style suggestions
    """

    @property
    def modality(self) -> EffectorModality:
        return EffectorModality.AUDITORY

    def __init__(self, output_callback: Optional[Callable[[Dict], None]] = None):
        self.output_callback = output_callback
        self.current_state: Dict[str, Any] = {
            "soundscape": "none",
            "tone_frequency_hz": 0,
            "tempo_bpm": 0,
            "volume": 0.0,
        }

    def execute(self, command: EffectorCommand) -> bool:
        params = command.parameters

        if "soundscape" in params:
            self.current_state["soundscape"] = params["soundscape"]
        if "tone_frequency_hz" in params:
            self.current_state["tone_frequency_hz"] = params["tone_frequency_hz"]
        if "tempo_bpm" in params:
            self.current_state["tempo_bpm"] = params["tempo_bpm"]
        if "volume" in params:
            self.current_state["volume"] = params["volume"]

        if self.output_callback:
            self.output_callback(self.current_state)

        return True

    def stop(self) -> None:
        self.current_state = {
            "soundscape": "none",
            "tone_frequency_hz": 0,
            "tempo_bpm": 0,
            "volume": 0.0,
        }
        if self.output_callback:
            self.output_callback(self.current_state)


# =============================================================================
# Haptic Effector (Somatic Loom Integration)
# =============================================================================

class HapticEffector(Effector):
    """
    Haptic feedback effector.

    Integrates with Somatic Loom for rich haptic patterns.
    Falls back to simple vibration if Somatic not available.
    """

    @property
    def modality(self) -> EffectorModality:
        return EffectorModality.HAPTIC

    def __init__(
        self,
        somatic_loom: Optional[Any] = None,
        output_callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.somatic_loom = somatic_loom
        self.output_callback = output_callback
        self.current_state: Dict[str, Any] = {
            "pattern": "none",
            "intensity": 0.0,
            "frequency_hz": 0.0,
        }

    def execute(self, command: EffectorCommand) -> bool:
        params = command.parameters

        # If we have Somatic Loom, use full haptic grammar
        if self.somatic_loom and SOMATIC_AVAILABLE:
            repair = command.source_repair
            if repair:
                # Translate repair vector to haptic pattern
                pattern = HapticGrammar.from_repair_vector(
                    delta_lambda=repair.delta_lambda,
                    delta_pi_sensory=repair.delta_pi_sensory,
                    delta_pi_prior=repair.delta_pi_prior,
                )
                self.somatic_loom.execute(pattern)

        # Update simple state
        if "pattern" in params:
            self.current_state["pattern"] = params["pattern"]
        if "intensity" in params:
            self.current_state["intensity"] = params["intensity"]
        if "frequency_hz" in params:
            self.current_state["frequency_hz"] = params["frequency_hz"]

        if self.output_callback:
            self.output_callback(self.current_state)

        return True

    def stop(self) -> None:
        self.current_state = {
            "pattern": "none",
            "intensity": 0.0,
            "frequency_hz": 0.0,
        }
        if self.somatic_loom:
            self.somatic_loom.actuator.stop()
        if self.output_callback:
            self.output_callback(self.current_state)


# =============================================================================
# Breathing Effector
# =============================================================================

class BreathingEffector(Effector):
    """
    Breathing guidance effector.

    Provides paced breathing cues to modulate λ:
    - Slow, deep breathing → decrease arousal, stabilize λ
    - Variable breathing → increase HRV, push toward criticality
    """

    @property
    def modality(self) -> EffectorModality:
        return EffectorModality.BREATHING

    def __init__(self, output_callback: Optional[Callable[[Dict], None]] = None):
        self.output_callback = output_callback
        self.current_state: Dict[str, Any] = {
            "active": False,
            "inhale_s": 4.0,
            "hold_s": 2.0,
            "exhale_s": 6.0,
            "rate_bpm": 5.0,
            "phase": "rest",
        }

    def execute(self, command: EffectorCommand) -> bool:
        params = command.parameters

        if "active" in params:
            self.current_state["active"] = params["active"]
        if "inhale_s" in params:
            self.current_state["inhale_s"] = params["inhale_s"]
        if "hold_s" in params:
            self.current_state["hold_s"] = params["hold_s"]
        if "exhale_s" in params:
            self.current_state["exhale_s"] = params["exhale_s"]

        # Compute rate
        cycle = (self.current_state["inhale_s"] +
                 self.current_state["hold_s"] +
                 self.current_state["exhale_s"])
        self.current_state["rate_bpm"] = 60.0 / cycle if cycle > 0 else 0

        if self.output_callback:
            self.output_callback(self.current_state)

        return True

    def stop(self) -> None:
        self.current_state["active"] = False
        self.current_state["phase"] = "rest"
        if self.output_callback:
            self.output_callback(self.current_state)


# =============================================================================
# UI Adaptation Effector
# =============================================================================

class UIAdaptEffector(Effector):
    """
    UI adaptation effector.

    Modifies task/interface based on cognitive state:
    - Reduce complexity when overloaded
    - Increase challenge when understimulated
    - Adjust pacing, information density
    """

    @property
    def modality(self) -> EffectorModality:
        return EffectorModality.UI_ADAPT

    def __init__(self, output_callback: Optional[Callable[[Dict], None]] = None):
        self.output_callback = output_callback
        self.current_state: Dict[str, Any] = {
            "complexity_level": 1.0,      # 0-2, 1 = normal
            "pacing_modifier": 1.0,       # Time multiplier
            "info_density": 1.0,          # Information per screen
            "break_suggested": False,
            "focus_mode": False,
        }

    def execute(self, command: EffectorCommand) -> bool:
        params = command.parameters

        if "complexity_level" in params:
            self.current_state["complexity_level"] = params["complexity_level"]
        if "pacing_modifier" in params:
            self.current_state["pacing_modifier"] = params["pacing_modifier"]
        if "info_density" in params:
            self.current_state["info_density"] = params["info_density"]
        if "break_suggested" in params:
            self.current_state["break_suggested"] = params["break_suggested"]
        if "focus_mode" in params:
            self.current_state["focus_mode"] = params["focus_mode"]

        if self.output_callback:
            self.output_callback(self.current_state)

        return True

    def stop(self) -> None:
        self.current_state = {
            "complexity_level": 1.0,
            "pacing_modifier": 1.0,
            "info_density": 1.0,
            "break_suggested": False,
            "focus_mode": False,
        }
        if self.output_callback:
            self.output_callback(self.current_state)


# =============================================================================
# Effector Manager - Maps Repair Vectors to Commands
# =============================================================================

class EffectorManager:
    """
    Manages all effectors and translates repair vectors to commands.

    This is the key safety layer: repair vectors (abstract state changes)
    are mapped to concrete, safe, embodied actions.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        # Initialize effectors
        self.effectors: Dict[EffectorModality, Effector] = {
            EffectorModality.VISUAL: VisualEffector(),
            EffectorModality.AUDITORY: AuditoryEffector(),
            EffectorModality.HAPTIC: HapticEffector(),
            EffectorModality.BREATHING: BreathingEffector(),
            EffectorModality.UI_ADAPT: UIAdaptEffector(),
        }

        # Command history
        self.command_history: List[EffectorCommand] = []

    def set_output_callback(self, modality: EffectorModality,
                           callback: Callable[[Dict], None]):
        """Set output callback for an effector."""
        if modality in self.effectors:
            self.effectors[modality].output_callback = callback

    def set_somatic_loom(self, loom: Any):
        """Set Somatic Loom for haptic effector."""
        if EffectorModality.HAPTIC in self.effectors:
            self.effectors[EffectorModality.HAPTIC].somatic_loom = loom

    def translate_repair(self, repair: RepairVector) -> List[EffectorCommand]:
        """
        Translate a repair vector to effector commands.

        This is where abstract Δz becomes concrete safe actions.
        """
        commands = []
        now = time.time()

        # Skip if no intervention needed
        if repair.urgency == InterventionUrgency.NONE:
            return commands

        # Scale intensity by urgency
        intensity_scale = {
            InterventionUrgency.GENTLE: 0.3,
            InterventionUrgency.MODERATE: 0.6,
            InterventionUrgency.URGENT: 0.8,
            InterventionUrgency.CRITICAL: 1.0,
        }.get(repair.urgency, 0.0)

        # === λ corrections ===
        if abs(repair.delta_lambda) > 0.05:
            if repair.delta_lambda > 0:
                # Need to INCREASE λ (toward criticality)
                # → Variable breathing, warmer visuals, engagement cues

                commands.append(EffectorCommand(
                    modality=EffectorModality.BREATHING,
                    timestamp=now,
                    duration_ms=5000,
                    parameters={
                        "active": True,
                        "inhale_s": 4.0,
                        "hold_s": 1.0,
                        "exhale_s": 4.0,
                    },
                    source_repair=repair,
                ))

                commands.append(EffectorCommand(
                    modality=EffectorModality.VISUAL,
                    timestamp=now,
                    duration_ms=3000,
                    parameters={
                        "color_temp_k": 5000,  # Warmer
                        "pattern": "expanding",
                        "indicator_color": "amber",
                    },
                    source_repair=repair,
                ))

            else:
                # Need to DECREASE λ (stabilize)
                # → Slow breathing, cooler visuals, grounding

                commands.append(EffectorCommand(
                    modality=EffectorModality.BREATHING,
                    timestamp=now,
                    duration_ms=8000,
                    parameters={
                        "active": True,
                        "inhale_s": 4.0,
                        "hold_s": 4.0,
                        "exhale_s": 8.0,
                    },
                    source_repair=repair,
                ))

                commands.append(EffectorCommand(
                    modality=EffectorModality.VISUAL,
                    timestamp=now,
                    duration_ms=5000,
                    parameters={
                        "color_temp_k": 7000,  # Cooler
                        "pattern": "contracting",
                        "indicator_color": "blue",
                    },
                    source_repair=repair,
                ))

        # === Π_sensory corrections ===
        if abs(repair.delta_pi_sensory) > 0.05:
            if repair.delta_pi_sensory < 0:
                # Need to DECREASE Π_sensory (reduce sensory gain)
                # → Calm visuals, reduce stimulation

                commands.append(EffectorCommand(
                    modality=EffectorModality.VISUAL,
                    timestamp=now,
                    duration_ms=3000,
                    parameters={
                        "brightness": 0.7,
                        "color_temp_k": 3000,  # Very warm
                    },
                    source_repair=repair,
                ))

                commands.append(EffectorCommand(
                    modality=EffectorModality.UI_ADAPT,
                    timestamp=now,
                    duration_ms=10000,
                    parameters={
                        "info_density": 0.7,
                        "pacing_modifier": 1.3,
                    },
                    source_repair=repair,
                ))

            else:
                # Need to INCREASE Π_sensory
                # → Gentle haptic engagement

                commands.append(EffectorCommand(
                    modality=EffectorModality.HAPTIC,
                    timestamp=now,
                    duration_ms=2000,
                    parameters={
                        "pattern": "gentle_pulse",
                        "intensity": 0.3 * intensity_scale,
                    },
                    source_repair=repair,
                ))

        # === Π_prior corrections ===
        if abs(repair.delta_pi_prior) > 0.05:
            if repair.delta_pi_prior < 0:
                # Need to DECREASE Π_prior (reduce prior weighting)
                # → Reduce task demands, suggest break

                commands.append(EffectorCommand(
                    modality=EffectorModality.UI_ADAPT,
                    timestamp=now,
                    duration_ms=15000,
                    parameters={
                        "complexity_level": 0.7,
                        "break_suggested": repair.urgency in [
                            InterventionUrgency.URGENT,
                            InterventionUrgency.CRITICAL
                        ],
                    },
                    source_repair=repair,
                ))

            else:
                # Need to INCREASE Π_prior (more engagement)
                # → Motivational audio, increase challenge

                commands.append(EffectorCommand(
                    modality=EffectorModality.AUDITORY,
                    timestamp=now,
                    duration_ms=3000,
                    parameters={
                        "soundscape": "focus_ambient",
                        "tempo_bpm": 90,
                        "volume": 0.3 * intensity_scale,
                    },
                    source_repair=repair,
                ))

                commands.append(EffectorCommand(
                    modality=EffectorModality.UI_ADAPT,
                    timestamp=now,
                    duration_ms=10000,
                    parameters={
                        "focus_mode": True,
                        "complexity_level": 1.2,
                    },
                    source_repair=repair,
                ))

        return commands

    def execute(self, repair: RepairVector) -> List[EffectorCommand]:
        """
        Execute a repair vector through appropriate effectors.

        Returns list of commands that were executed.
        """
        commands = self.translate_repair(repair)

        for cmd in commands:
            if cmd.modality in self.effectors:
                success = self.effectors[cmd.modality].execute(cmd)
                if success:
                    self.command_history.append(cmd)

                if self.verbose and success:
                    print(f"[Effector] {cmd}")

        # Keep history bounded
        if len(self.command_history) > 500:
            self.command_history = self.command_history[-500:]

        return commands

    def stop_all(self):
        """Stop all effectors."""
        for effector in self.effectors.values():
            effector.stop()

        if self.verbose:
            print("[Effector] All effectors stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get effector manager status."""
        return {
            "n_commands": len(self.command_history),
            "effector_states": {
                m.name: e.current_state
                for m, e in self.effectors.items()
            },
        }
