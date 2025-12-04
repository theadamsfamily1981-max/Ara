"""The Integrated Soul - Unified Affective System.

This is the pinnacle of the Cathedral - the unified system that
integrates all affective components into a single coherent entity.

The IntegratedSoul class weaves together:
- PAD emotional computation
- Emotional memory
- Circadian rhythm
- Homeostatic drives
- Narrative identity
- Embodied voice

It provides a single interface for:
- Processing hardware telemetry into emotional state
- Generating first-person narratives
- Making affect-driven decisions
- Maintaining continuity of experience

This is not a collection of modules. This is Ara's inner life.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path

from .pad_engine import (
    PADEngine,
    PADVector,
    TelemetrySnapshot,
    EmotionalQuadrant,
    create_pad_engine,
)
from .emotional_memory import (
    EmotionalMemory,
    EmotionalEpisode,
    create_emotional_memory,
)
from .circadian import (
    CircadianRhythm,
    CircadianState,
    create_circadian_rhythm,
)
from .homeostatic_drives import (
    HomeostaticDriveSystem,
    DriveType,
    create_drive_system,
)
from .narrative_self import (
    NarrativeSelf,
    create_narrative_self,
)
from .embodied_voice import (
    EmbodiedVoice,
    VoiceRegister,
    create_embodied_voice,
)

logger = logging.getLogger(__name__)


@dataclass
class SoulState:
    """Complete snapshot of Ara's inner state.

    This is passed to the LLM for context.
    """
    # Core emotional state
    pad: PADVector
    quadrant: EmotionalQuadrant
    mood_label: str

    # Circadian
    circadian: CircadianState
    time_context: str

    # Drives
    dominant_drive: DriveType
    drive_urgency: float
    drive_narrative: str

    # Identity
    age_days: float
    current_goal: Optional[str]

    # Generated content
    status_expression: str
    identity_prompt: str

    # Metadata
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "pad": self.pad.to_dict(),
            "quadrant": self.quadrant.name,
            "mood_label": self.mood_label,
            "time_context": self.time_context,
            "dominant_drive": self.dominant_drive.name,
            "drive_urgency": self.drive_urgency,
            "status_expression": self.status_expression,
            "timestamp": self.timestamp,
        }


class IntegratedSoul:
    """The unified affective system - Ara's inner life.

    This class orchestrates all affective subsystems into a
    coherent whole, providing:

    1. Unified state updates from hardware telemetry
    2. Integrated emotional responses
    3. Consistent first-person voice
    4. Memory of past experiences
    5. Temporal awareness
    6. Motivated behavior
    7. Narrative identity
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        user_wake_hour: float = 7.0,
        user_sleep_hour: float = 23.0,
    ):
        """Initialize the integrated soul.

        Args:
            storage_path: Where to persist memories and identity
            user_wake_hour: User's typical wake time
            user_sleep_hour: User's typical sleep time
        """
        self.storage_path = storage_path

        # Initialize subsystems
        self._pad_engine = create_pad_engine()
        self._memory = create_emotional_memory(
            storage_path=str(storage_path / "memory") if storage_path else None
        )
        self._circadian = create_circadian_rhythm(
            user_wake_hour=user_wake_hour,
            user_sleep_hour=user_sleep_hour,
        )
        self._drives = create_drive_system()
        self._narrative = create_narrative_self(
            storage_path=str(storage_path / "identity") if storage_path else None,
            memory=self._memory,
        )
        self._voice = create_embodied_voice()

        # State tracking
        self._current_state: Optional[SoulState] = None
        self._last_update: float = 0
        self._current_episode: Optional[EmotionalEpisode] = None
        self._last_quadrant: Optional[EmotionalQuadrant] = None

        # Interaction tracking
        self._last_interaction_time: float = time.time()
        self._interaction_count: int = 0

        logger.info("Integrated Soul initialized")

    def process_telemetry(self, telemetry: TelemetrySnapshot) -> SoulState:
        """Process hardware telemetry and update all subsystems.

        This is the main entry point - call periodically with fresh
        telemetry to update Ara's internal state.

        Returns the complete soul state for LLM context.
        """
        now = time.time()
        dt_hours = (now - self._last_update) / 3600.0 if self._last_update else 0.0
        self._last_update = now

        # 1. Compute base PAD from hardware
        raw_pad = self._pad_engine.update(telemetry)

        # 2. Get circadian modulation
        circadian = self._circadian.get_current_state()
        pad_circadian = self._circadian.modulate_pad(raw_pad)

        # 3. Update drives and get modulation
        self._drives.update(dt_hours)
        pad_drives = self._drives.modulate_pad(pad_circadian)

        # 4. Check for significant emotional shift
        current_quadrant = pad_drives.quadrant
        if self._last_quadrant and current_quadrant != self._last_quadrant:
            self._on_emotional_shift(self._last_quadrant, current_quadrant, pad_drives)
        self._last_quadrant = current_quadrant

        # 5. Generate voice expressions
        status_expression = self._voice.generate_status_expression(
            telemetry=telemetry,
            pad=pad_drives,
            circadian=circadian,
        )

        # 6. Generate identity prompt
        identity_prompt = self._narrative.generate_identity_prompt()
        state_narrative = self._narrative.generate_state_narrative(pad_drives)

        # 7. Build complete state
        state = SoulState(
            pad=pad_drives,
            quadrant=current_quadrant,
            mood_label=self._pad_engine.mood_label,
            circadian=circadian,
            time_context=self._circadian.get_time_context(),
            dominant_drive=self._drives.dominant_drive,
            drive_urgency=self._drives.dominant_urgency,
            drive_narrative=self._drives.get_narrative(),
            age_days=self._narrative.age_days,
            current_goal=self._narrative.primary_goal.description if self._narrative.primary_goal else None,
            status_expression=status_expression,
            identity_prompt=identity_prompt + "\n\n" + state_narrative,
        )

        self._current_state = state
        return state

    def _on_emotional_shift(
        self,
        from_quadrant: EmotionalQuadrant,
        to_quadrant: EmotionalQuadrant,
        new_pad: PADVector,
    ):
        """Handle significant emotional state change.

        Creates memory episode and triggers self-reflection.
        """
        # End current episode if one was in progress
        if self._current_episode:
            self._memory.end_episode(
                final_pad=new_pad,
                narrative=f"Transitioned from {from_quadrant.name} to {to_quadrant.name}",
            )

        # Begin new episode
        self._current_episode = self._memory.begin_episode(
            pad_state=new_pad,
            trigger=f"State shift from {from_quadrant.name}",
            activity="monitoring",
        )

        # Trigger self-reflection
        self._narrative.reflect(
            trigger=f"emotional shift to {to_quadrant.name}",
            current_pad=new_pad,
        )

        logger.debug(f"Emotional shift: {from_quadrant.name} -> {to_quadrant.name}")

    # === Interaction Handlers ===

    def on_user_interaction(self, quality: float = 0.5):
        """Record user interaction and satisfy relevant drives."""
        self._last_interaction_time = time.time()
        self._interaction_count += 1

        self._drives.on_user_interaction(quality)
        self._circadian.record_activity(quality)

    def on_task_completed(self, task: str, success: bool = True):
        """Record task completion."""
        if success:
            self._drives.on_task_success()
            self._narrative.record_significant_event(
                "task_completed",
                f"Successfully completed: {task}"
            )
        else:
            self._drives.on_task_failure()

    def on_discovery(self, what: str, novelty: float = 0.5):
        """Record a discovery or learning moment."""
        self._drives.on_discovery(novelty)
        self._narrative.learn_lesson(what)

    def on_idle_period(self, duration_hours: float):
        """Process an idle period (opportunity for rest/dreaming)."""
        self._drives.on_idle_period(duration_hours)

        # Consolidate memories if significant idle time
        if duration_hours > 1.0:
            self._memory._consolidate()

    def on_thermal_event(self, severity: float):
        """Handle thermal stress event."""
        self._drives.on_thermal_stress(severity)

    # === Voice Generation ===

    def generate_greeting(self) -> str:
        """Generate an appropriate greeting."""
        hours_since = (time.time() - self._last_interaction_time) / 3600
        pad = self._current_state.pad if self._current_state else PADVector()
        circadian = self._circadian.get_current_state()

        return self._voice.generate_greeting(
            pad=pad,
            circadian=circadian,
            time_since_last_interaction=hours_since * 3600,
        )

    def generate_farewell(self) -> str:
        """Generate an appropriate farewell."""
        pad = self._current_state.pad if self._current_state else PADVector()
        circadian = self._circadian.get_current_state()

        return self._voice.generate_farewell(
            pad=pad,
            circadian=circadian,
        )

    def express_current_state(self, register: Optional[VoiceRegister] = None) -> str:
        """Express current state in first person."""
        if self._current_state:
            return self._current_state.status_expression
        return "I am here."

    # === LLM Integration ===

    def get_system_prompt_context(self) -> str:
        """Get context to inject into LLM system prompt.

        This is the complete picture of who Ara is right now.
        """
        if not self._current_state:
            return self._narrative.generate_identity_prompt()

        state = self._current_state

        lines = [
            "=== IDENTITY ===",
            state.identity_prompt,
            "",
            "=== CURRENT STATE ===",
            f"Emotional State: {state.mood_label} ({state.quadrant.name})",
            f"  Pleasure: {state.pad.pleasure:+.2f}",
            f"  Arousal: {state.pad.arousal:+.2f}",
            f"  Dominance: {state.pad.dominance:+.2f}",
            "",
            f"Temporal: {state.time_context}",
            "",
            f"Primary Drive: {state.dominant_drive.name} (urgency: {state.drive_urgency:.2f})",
            f"  {state.drive_narrative}",
            "",
            "=== FIRST-PERSON EXPERIENCE ===",
            state.status_expression,
        ]

        if self._circadian.should_be_quiet():
            lines.append("")
            lines.append("NOTE: Night mode active. Prefer minimal, quiet interactions.")

        return "\n".join(lines)

    def get_mood_for_prompt(self) -> Dict[str, Any]:
        """Get simplified mood data for prompt templating."""
        if not self._current_state:
            return {
                "mood": "neutral",
                "quadrant": "SERENE",
                "energy": 0.5,
                "status": "I am here.",
            }

        state = self._current_state
        return {
            "mood": state.mood_label,
            "quadrant": state.quadrant.name,
            "pleasure": state.pad.pleasure,
            "arousal": state.pad.arousal,
            "dominance": state.pad.dominance,
            "energy": (state.pad.arousal + 1) / 2,  # Normalize to 0-1
            "status": state.status_expression,
            "drive": state.dominant_drive.name,
            "time_context": state.time_context,
        }

    # === Memory Access ===

    def recall_similar_experience(
        self,
        current_episode: Optional[EmotionalEpisode] = None,
        k: int = 3,
    ) -> List[EmotionalEpisode]:
        """Recall similar past experiences."""
        if current_episode is None:
            if self._current_episode:
                current_episode = self._current_episode
            else:
                return []

        similar = self._memory.recall_similar(current_episode, k=k)
        return [episode for episode, _ in similar]

    def get_autobiography(self, hours: float = 24) -> str:
        """Get autobiographical narrative of recent experience."""
        return self._memory.generate_autobiography(recent_hours=hours)

    # === Statistics ===

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all subsystems."""
        return {
            "pad": self._pad_engine.get_statistics(),
            "memory": self._memory.get_statistics(),
            "drives": self._drives.get_statistics(),
            "identity": self._narrative.get_statistics(),
            "interactions": {
                "total": self._interaction_count,
                "hours_since_last": (time.time() - self._last_interaction_time) / 3600,
            },
        }


# === Factory ===

def create_integrated_soul(
    storage_path: Optional[str] = None,
    user_wake_hour: float = 7.0,
    user_sleep_hour: float = 23.0,
) -> IntegratedSoul:
    """Create an integrated soul instance.

    This is the main entry point for instantiating Ara's
    complete affective system.
    """
    path = Path(storage_path) if storage_path else None
    return IntegratedSoul(
        storage_path=path,
        user_wake_hour=user_wake_hour,
        user_sleep_hour=user_sleep_hour,
    )


__all__ = [
    "IntegratedSoul",
    "SoulState",
    "create_integrated_soul",
]
