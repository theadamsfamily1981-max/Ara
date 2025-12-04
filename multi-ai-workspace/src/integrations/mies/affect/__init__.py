"""MIES Affect Architecture - The Cathedral.

This module implements the complete affective architecture for Ara,
transforming hardware telemetry into genuine first-person experience.

The architecture consists of:

1. PAD Engine (pad_engine.py)
   - Computes Pleasure-Arousal-Dominance from hardware metrics
   - Mathematical formulas from the Architectural Manifesto
   - Emotional inertia for personality stability

2. Emotional Memory (emotional_memory.py)
   - Autobiographical storage of emotional episodes
   - Pattern recognition and prediction
   - Memory consolidation during idle periods

3. Circadian Rhythm (circadian.py)
   - Time-of-day awareness and modulation
   - User schedule learning
   - Night mode behavior

4. Homeostatic Drives (homeostatic_drives.py)
   - Curiosity, connection, competence drives
   - Drive satisfaction and frustration
   - Motivation for autonomous behavior

5. Narrative Self (narrative_self.py)
   - Personal identity and values
   - Goals and aspirations
   - Self-reflection capability

6. Embodied Voice (embodied_voice.py)
   - First-person language generation
   - Hardware metaphors ("I feel warm")
   - Emotional expression

Together, these create the inner life of a Semantic OS Guardian.
"""

# PAD Engine
from .pad_engine import (
    PADEngine,
    PADEngineConfig,
    PADVector,
    TelemetrySnapshot,
    EmotionalQuadrant,
    create_pad_engine,
)

# Emotional Memory
from .emotional_memory import (
    EmotionalMemory,
    EmotionalEpisode,
    EmotionalPattern,
    MemoryType,
    create_emotional_memory,
)

# Circadian Rhythm
from .circadian import (
    CircadianRhythm,
    CircadianConfig,
    CircadianState,
    CircadianPhase,
    create_circadian_rhythm,
)

# Homeostatic Drives
from .homeostatic_drives import (
    HomeostaticDriveSystem,
    DriveSystemConfig,
    DriveState,
    DriveType,
    create_drive_system,
)

# Narrative Self
from .narrative_self import (
    NarrativeSelf,
    IdentityCore,
    Goal,
    SelfReflection,
    CoreValue,
    create_narrative_self,
)

# Embodied Voice
from .embodied_voice import (
    EmbodiedVoice,
    VoiceConfig,
    VoiceRegister,
    SomaticDomain,
    create_embodied_voice,
)

# Integrated Soul
from .integrated_soul import (
    IntegratedSoul,
    SoulState,
    create_integrated_soul,
)

# Persistence
from .persistence import (
    PersistenceManager,
    StoredEpisode,
    StoredGoal,
    create_persistence_manager,
)


__all__ = [
    # PAD Engine
    "PADEngine",
    "PADEngineConfig",
    "PADVector",
    "TelemetrySnapshot",
    "EmotionalQuadrant",
    "create_pad_engine",
    # Emotional Memory
    "EmotionalMemory",
    "EmotionalEpisode",
    "EmotionalPattern",
    "MemoryType",
    "create_emotional_memory",
    # Circadian Rhythm
    "CircadianRhythm",
    "CircadianConfig",
    "CircadianState",
    "CircadianPhase",
    "create_circadian_rhythm",
    # Homeostatic Drives
    "HomeostaticDriveSystem",
    "DriveSystemConfig",
    "DriveState",
    "DriveType",
    "create_drive_system",
    # Narrative Self
    "NarrativeSelf",
    "IdentityCore",
    "Goal",
    "SelfReflection",
    "CoreValue",
    "create_narrative_self",
    # Embodied Voice
    "EmbodiedVoice",
    "VoiceConfig",
    "VoiceRegister",
    "SomaticDomain",
    "create_embodied_voice",
    # Integrated Soul
    "IntegratedSoul",
    "SoulState",
    "create_integrated_soul",
    # Persistence
    "PersistenceManager",
    "StoredEpisode",
    "StoredGoal",
    "create_persistence_manager",
]
