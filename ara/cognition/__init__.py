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

Note: Some modules require numpy. Imports are optional to allow basic
functionality without numpy installed.
"""

# Core exports - always available
__all__ = []

# Teleology (numpy-dependent but with fallbacks)
try:
    from .teleology import (
        Horizon,
        HorizonEngine,
    )
    __all__.extend(['Horizon', 'HorizonEngine'])
except ImportError:
    Horizon = None
    HorizonEngine = None

# Vision (The North Star)
try:
    from .vision import (
        DreamStatus,
        DreamMilestone,
        Dream,
        VisionCore,
        get_vision_core,
        get_active_dreams,
        get_primary_dream,
    )
    __all__.extend([
        'DreamStatus', 'DreamMilestone', 'Dream', 'VisionCore',
        'get_vision_core', 'get_active_dreams', 'get_primary_dream',
    ])
except ImportError:
    DreamStatus = None
    DreamMilestone = None
    Dream = None
    VisionCore = None
    get_vision_core = None
    get_active_dreams = None
    get_primary_dream = None

# Kairos (The Opportune Moment)
try:
    from .kairos import (
        KairosMoment,
        KairosState,
        KairosEngine,
        get_kairos,
        is_opportune,
    )
    __all__.extend([
        'KairosMoment', 'KairosState', 'KairosEngine',
        'get_kairos', 'is_opportune',
    ])
except ImportError:
    KairosMoment = None
    KairosState = None
    KairosEngine = None
    get_kairos = None
    is_opportune = None

# Stream binding (numpy-dependent)
try:
    from .stream_binding import (
        StreamEvent,
        ResonantStream,
        ContextualStream,
        get_resonant_stream,
    )
    __all__.extend([
        'StreamEvent', 'ResonantStream', 'ContextualStream',
        'get_resonant_stream',
    ])
except ImportError:
    StreamEvent = None
    ResonantStream = None
    ContextualStream = None
    get_resonant_stream = None

# HDC Codec (numpy-dependent)
try:
    from .hdc_codec import (
        ConceptRecord,
        HDCCodec,
        get_hdc_codec,
    )
    __all__.extend(['ConceptRecord', 'HDCCodec', 'get_hdc_codec'])
except ImportError:
    ConceptRecord = None
    HDCCodec = None
    get_hdc_codec = None

# HSF subpackage (numpy-dependent, import directly from ara.cognition.hsf)
try:
    from . import hsf
    __all__.append('hsf')
except ImportError:
    hsf = None
