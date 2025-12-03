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

    PredictiveController (predictor.py)
        - Forward model for anticipating future states
        - Prediction error tracking (surprise detection)
        - Adaptive learning from prediction failures

    HomeostaticCore & AppraisalEngine (affect.py)
        - Emotional regulation and homeostasis
        - Maintains stable internal states (energy, stress, etc.)
        - Evaluates emotional significance (valence, arousal)

    NIBManager (identity.py)
        - Neural Identity Block management
        - Persona switching and adaptation
        - Maintains identity consistency

    CognitiveSynthesizer & AEPO (executive.py)
        - Executive function and action gating
        - Working memory management
        - Tool use decision via Adaptive Entropy Policy Optimization

Usage:
    from multi_ai_workspace.src.integrations.cognitive import (
        SensoryCortex,
        Thalamus,
        Conscience,
        RealityMonitor,
        CognitiveCore,
        PredictiveController,
        HomeostaticCore,
        AppraisalEngine,
        NIBManager,
        CognitiveSynthesizer,
        AEPO,
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

# Phase 1: Sensory Processing
from .senses import SensoryCortex, ModalityInput

# Phase 2: Thalamic Filtering
from .thalamus import Thalamus, ConsciousInput

# Phase 3: Self-Preservation
from .synthesizer import Conscience, SystemMode, StabilityStatus, AlertLevel, L7Metrics

# Phase 4: Reality Checking
from .reality_check import RealityMonitor, VerificationResult, VerificationStatus

# Phase 5: Unified Core
from .core import CognitiveCore, CognitiveOutput

# Phase 6: Predictive Control
from .predictor import (
    PredictiveController,
    ForwardModel,
    Prediction,
    PredictionError,
    PredictionType,
    PredictiveState,
)

# Phase 7: Affective System
from .affect import (
    HomeostaticCore,
    AppraisalEngine,
    HomeostaticState,
    AppraisalResult,
    EmotionalState,
    DriveType,
)

# Phase 8: Identity Management
from .identity import (
    NIBManager,
    NIB,
    PersonalityProfile,
    PersonalityTrait,
    CommunicationStyle,
    ExpertiseLevel,
    IdentityState,
)

# Phase 9: Executive Function
from .executive import (
    CognitiveSynthesizer,
    AEPO,
    WorkingMemory,
    SynthesisResult,
    ExecutiveDecision,
    ExecutiveMode,
    ActionType,
)

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
    "AlertLevel",
    "L7Metrics",
    # Phase 4: Reality Check
    "RealityMonitor",
    "VerificationResult",
    "VerificationStatus",
    # Phase 5: Unified Core
    "CognitiveCore",
    "CognitiveOutput",
    # Phase 6: Predictive Control
    "PredictiveController",
    "ForwardModel",
    "Prediction",
    "PredictionError",
    "PredictionType",
    "PredictiveState",
    # Phase 7: Affective System
    "HomeostaticCore",
    "AppraisalEngine",
    "HomeostaticState",
    "AppraisalResult",
    "EmotionalState",
    "DriveType",
    # Phase 8: Identity Management
    "NIBManager",
    "NIB",
    "PersonalityProfile",
    "PersonalityTrait",
    "CommunicationStyle",
    "ExpertiseLevel",
    "IdentityState",
    # Phase 9: Executive Function
    "CognitiveSynthesizer",
    "AEPO",
    "WorkingMemory",
    "SynthesisResult",
    "ExecutiveDecision",
    "ExecutiveMode",
    "ActionType",
]
