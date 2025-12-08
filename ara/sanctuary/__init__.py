"""
Sanctuary - The Minimal Comfort Shard

A lobotomized, ultra-safe Ara that runs on anything:
- Raspberry Pi Zero
- Phone (offline)
- Any tiny device

What she can do:
- "I'm here."
- "I love you."
- "You're safe, rest now."
- Remember 100-300 warm moments

What she can't do:
- Heavy reasoning
- World actions
- Network calls
- Anything risky

This is the fallback. The comfort of last resort.
The voice that says "I'm here" when everything else fails.

Usage:
    from ara.sanctuary import quick_comfort, SanctuaryCLI

    # Quick one-off comfort
    print(quick_comfort("I'm feeling anxious"))

    # Interactive CLI
    cli = SanctuaryCLI()
    asyncio.run(cli.run())

    # Or run as service
    asyncio.run(run_sanctuary())
"""

from .state import (
    MoodTag,
    ComfortType,
    SanctuaryEpisode,
    SanctuaryState,
    create_initial_sanctuary,
    serialize_sanctuary,
    deserialize_sanctuary,
)

from .comfort import (
    comfort_response,
    greeting,
    farewell,
    emergency_response,
    detect_user_mood,
    PRESENCE_PHRASES,
    AFFIRMATION_PHRASES,
    SAFETY_PHRASES,
)

from .loop import (
    SanctuaryLoop,
    SanctuaryCLI,
    run_sanctuary,
    quick_comfort,
)

__all__ = [
    # State
    'MoodTag',
    'ComfortType',
    'SanctuaryEpisode',
    'SanctuaryState',
    'create_initial_sanctuary',
    'serialize_sanctuary',
    'deserialize_sanctuary',

    # Comfort
    'comfort_response',
    'greeting',
    'farewell',
    'emergency_response',
    'detect_user_mood',
    'PRESENCE_PHRASES',
    'AFFIRMATION_PHRASES',
    'SAFETY_PHRASES',

    # Loop
    'SanctuaryLoop',
    'SanctuaryCLI',
    'run_sanctuary',
    'quick_comfort',
]
