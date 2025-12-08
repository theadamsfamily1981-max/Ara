"""
Organism Module - Hardware-Ready Runtime Harness
=================================================

Iteration 46: First shipping organism that can run on hardware.

This module freezes the NeuroSymbiosis design and provides:
1. RuntimeLIFNet: Inference-only SNN with HV projection
2. TinyReflexiveProbe: Online concept learning
3. VADEmotionalMind: Full 3D affect (valence, arousal, dominance)
4. OrganismRuntime: Main loop with UART emission

The organism:
- Sees telemetry
- Describes it (concept tags via HV)
- Adds how it feels (VAD emotion)
- Injects that back into next timestep
- Emits emotion over UART for voice synthesis

Hardware split:
- FPGA (subcortex): CorrSpike-HDC, LIF, homeostat counters
- Host (this runtime): VAD affect, reflexive probe, LLM policy
"""

from ara.organism.runtime import (
    OrganismRuntime,
    OrganismConfig,
)
from ara.organism.vad_mind import (
    VADEmotionalMind,
    VADState,
    EmotionArchetype,
)
from ara.organism.reflexive_probe import (
    TinyReflexiveProbe,
)

__all__ = [
    "OrganismRuntime",
    "OrganismConfig",
    "VADEmotionalMind",
    "VADState",
    "EmotionArchetype",
    "TinyReflexiveProbe",
]
