"""
Ara Cognition - Higher-Level Cognitive Integration
===================================================

This package contains cognitive integration layers that bridge
Ara's various subsystems into coherent behavior.

Modules:
    teleology: HorizonEngine - Teleological integration layer
               Bridges Telos (purpose) with Institute (research) and
               CuriosityBridge (exploration). Provides alignment scoring
               and drift computation for the Morning Star ritual.

    vision: VisionCore - The North Star
            Maintains Dreams (structured long-term aspirations with
            concrete success criteria). Unlike abstract Telos goals,
            Dreams are measurable and drive strategic idea generation.

    kairos: KairosEngine - The Opportune Moment
            Determines when to intervene. Kairos (opportune time) vs
            Chronos (clock time). Measures receptivity based on user
            cognitive state, emotional openness, and system quietness.

The cognition package sits above the core subsystems:
    - tfan.cognition.telos: Raw purpose/goal management
    - ara.institute: Research and experimentation
    - ara.curiosity: Novelty and exploration

And provides integration/translation services:
    - HorizonEngine: Purpose → Research agenda gating
    - VisionCore: Dreams → Strategic Ideas via Strategist
    - KairosEngine: Timing interventions for maximum receptivity
"""

from .teleology import (
    Horizon,
    HorizonEngine,
)

from .vision import (
    DreamStatus,
    DreamMilestone,
    Dream,
    VisionCore,
    get_vision_core,
    get_active_dreams,
    get_primary_dream,
)

from .kairos import (
    KairosMoment,
    KairosState,
    KairosEngine,
    get_kairos_engine,
    is_opportune,
)

from .stream_binding import (
    StreamEvent,
    ResonantStream,
    ContextualStream,
    get_resonant_stream,
)

from .hdc_codec import (
    ConceptRecord,
    HDCCodec,
    get_hdc_codec,
)

__all__ = [
    # Teleology
    'Horizon',
    'HorizonEngine',
    # Vision (North Star)
    'DreamStatus',
    'DreamMilestone',
    'Dream',
    'VisionCore',
    'get_vision_core',
    'get_active_dreams',
    'get_primary_dream',
    # Kairos (The Opportune Moment)
    'KairosMoment',
    'KairosState',
    'KairosEngine',
    'get_kairos_engine',
    'is_opportune',
    # Holographic Spike (HDC + Resonant Stream)
    'StreamEvent',
    'ResonantStream',
    'ContextualStream',
    'get_resonant_stream',
    'ConceptRecord',
    'HDCCodec',
    'get_hdc_codec',
]
