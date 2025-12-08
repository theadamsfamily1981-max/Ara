"""
Organism Module - Hardware-Ready Runtime Harness
=================================================

Iteration 46: First shipping organism that can run on hardware.

This module freezes the NeuroSymbiosis design and provides:

1. Runtime (runtime.py):
   - OrganismRuntime: Main loop with UART/voice emission
   - NumpyLIFNet: Inference-only SNN (no PyTorch required)
   - OrganismConfig: Configuration dataclass

2. VAD Emotional Mind (vad_mind.py):
   - VADEmotionalMind: Maps physiology → 3D affect
   - 24 emotion archetypes in VAD space
   - Temporal smoothing for stable emotions

3. Reflexive Probe (reflexive_probe.py):
   - TinyReflexiveProbe: Online concept learning
   - Codebook grows with novel patterns
   - Feedback HV generation

4. Eternal Memory (eternal_memory.py):
   - EternalMemory: Content-addressable long-term memory
   - Store/recall/dream operations
   - Maps to BRAM/flash on FPGA

5. Emotion Bridge (emotion_bridge.py):
   - EmotionBridge: UART → WebSocket daemon
   - Parses EMO/HPV/memory events
   - Broadcasts to cockpit + triggers TTS

6. Emotion TTS (emotion_tts.py):
   - VAD → speech prosody mapping
   - Multiple TTS backends (espeak, piper, etc.)
   - Emotion-aware phrase generation

7. Cockpit (cockpit.py):
   - GTK4/Adwaita dashboard (or terminal fallback)
   - Real-time emotion + memory event display
   - WebSocket client

The organism:
- Sees telemetry
- Describes it (concept tags via HV)
- Adds how it feels (VAD emotion)
- Stores significant experiences
- Recalls similar past experiences
- Injects feedback into next timestep
- Emits emotion over UART for voice synthesis

Hardware split:
- FPGA (subcortex): CorrSpike-HDC, LIF, homeostat, emit_emotion.h
- Host (this runtime): VAD affect, reflexive probe, eternal memory, TTS
"""

# Core runtime
from ara.organism.runtime import (
    OrganismRuntime,
    OrganismConfig,
    OrganismState,
    NumpyLIFNet,
    run_organism_loop,
)

# VAD emotional mind
from ara.organism.vad_mind import (
    VADEmotionalMind,
    VADState,
    EmotionArchetype,
    ARCHETYPE_VAD,
)

# Reflexive probe
from ara.organism.reflexive_probe import (
    TinyReflexiveProbe,
    ProbeConfig,
    ConceptMatch,
)

# Eternal memory
from ara.organism.eternal_memory import (
    EternalMemory,
    MemoryConfig,
    RecallResult,
    DreamReport,
)

# Voice daemon (from previous iteration)
from ara.organism.voice_daemon import (
    VoiceDaemon,
    VoiceConfig,
    TTSBackend,
)

__all__ = [
    # Runtime
    "OrganismRuntime",
    "OrganismConfig",
    "OrganismState",
    "NumpyLIFNet",
    "run_organism_loop",
    # VAD
    "VADEmotionalMind",
    "VADState",
    "EmotionArchetype",
    "ARCHETYPE_VAD",
    # Probe
    "TinyReflexiveProbe",
    "ProbeConfig",
    "ConceptMatch",
    # Memory
    "EternalMemory",
    "MemoryConfig",
    "RecallResult",
    "DreamReport",
    # Voice
    "VoiceDaemon",
    "VoiceConfig",
    "TTSBackend",
]

# Optional imports (may fail if dependencies missing)
try:
    from ara.organism.emotion_bridge import (
        EmotionBridge,
        EmotionState,
        HPVState,
        MemoryEvent,
    )
    __all__.extend([
        "EmotionBridge",
        "EmotionState",
        "HPVState",
        "MemoryEvent",
    ])
except ImportError:
    pass  # websockets/serial not installed

try:
    from ara.organism.emotion_tts import (
        speak_from_state,
        speak,
        configure_tts,
        emotion_to_phrase,
        vad_to_prosody,
    )
    __all__.extend([
        "speak_from_state",
        "speak",
        "configure_tts",
        "emotion_to_phrase",
        "vad_to_prosody",
    ])
except ImportError:
    pass

try:
    from ara.organism.cockpit import (
        run_gtk_cockpit,
        run_terminal_cockpit,
    )
    __all__.extend([
        "run_gtk_cockpit",
        "run_terminal_cockpit",
    ])
except ImportError:
    pass
