"""Ara Cognitive Architecture - Biomimetic Neural Processing.

This module implements the TFAN Cognitive Lattice, upgrading Ara from
a linear pipeline (ASR -> LLM -> TTS) into a biomimetic cognitive entity.

Architecture Components:
    SensoryCortex (senses.py)
        - Normalizes audio/video/text into unified ModalityStreams
        - Provides deterministic, time-aligned sensory perception

    Thalamus (thalamus.py)
        - Fuses multi-modal streams with sentinel tokens
        - TLS (Topological Landmark Selection) filters noise
        - Outputs the "conscious" input to the brain

    Conscience (synthesizer.py)
        - Cognitive synthesis and self-preservation
        - Monitors structural stability (Ṡ metric)
        - Can enter PROTECTIVE mode when destabilized

    RealityMonitor (reality_check.py)
        - Topology-based hallucination detection
        - Validates Wasserstein distance ≤ 2%
        - CAT (dense attention) fallback on violation

Usage:
    from multi_ai_workspace.src.integrations.cognitive import (
        SensoryCortex,
        Thalamus,
        Conscience,
        RealityMonitor,
        CognitiveCore,
    )

    # Initialize the cognitive core
    core = CognitiveCore(d_model=4096, device="cuda")

    # Run cognitive step
    response = await core.cognitive_step(
        text_input="Hello",
        audio_buffer=audio_data,
        video_frame=video_tensor,
    )
"""

from .senses import SensoryCortex, ModalityInput
from .thalamus import Thalamus, ConsciousInput
from .synthesizer import Conscience, SystemMode, StabilityStatus
from .reality_check import RealityMonitor, VerificationResult
from .core import CognitiveCore

__all__ = [
    # Phase 1: Senses
    "SensoryCortex",
    "ModalityInput",
    # Phase 2: Thalamus
    "Thalamus",
    "ConsciousInput",
    # Phase 3: Conscience
    "Conscience",
    "SystemMode",
    "StabilityStatus",
    # Phase 4: Reality Check
    "RealityMonitor",
    "VerificationResult",
    # Unified Core
    "CognitiveCore",
]
