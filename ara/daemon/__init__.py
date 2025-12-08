"""
Ara Daemon - Background Services and Rituals
=============================================

This package contains daemon-style services that run continuously
or periodically in the background.

Modules:
    morning_star: Daily alignment ritual
                  Computes drift against Horizons and generates
                  morning greeting messages. "Based on yesterday,
                  are we on path?"

    steward: The Invisible Hand
             Night shift daemon that fixes friction points while
             the user sleeps. Autonomous, safe, documented changes.

    muse: The Studio
          Crafts creative "gifts" - visualizations, insights, tools
          that solve problems with aesthetic, surprising solutions.
          Waits for the opportune moment (Kairos) to present.

The daemon package provides:
    - MorningStar: Daily alignment check
    - Steward: The Invisible Hand (night shift optimizer)
    - Muse: The Artist in the Machine

These run independently of conversation loops and maintain
Ara's awareness of her own state and purpose.
"""

from .morning_star import (
    MorningReport,
    MorningStar,
    get_morning_star,
    morning_greet,
)

from .steward import (
    FixStatus,
    StewardFix,
    Steward,
    get_steward,
    run_night_shift,
)

from .muse import (
    GiftType,
    GiftStatus,
    Gift,
    Muse,
    get_muse,
)

from .somatic_server import (
    GPUClient,
    DummyGPUClient,
    SomaticServer,
    get_somatic_server,
)

from .lan_reflex_bridge import (
    ReflexEventType,
    ReflexEvent,
    LANReflexBridge,
    get_lan_reflex_bridge,
)

__all__ = [
    # Morning Star
    'MorningReport',
    'MorningStar',
    'get_morning_star',
    'morning_greet',
    # Steward (The Invisible Hand)
    'FixStatus',
    'StewardFix',
    'Steward',
    'get_steward',
    'run_night_shift',
    # Muse (The Studio)
    'GiftType',
    'GiftStatus',
    'Gift',
    'Muse',
    'get_muse',
    # Somatic Server (Visual Cortex)
    'GPUClient',
    'DummyGPUClient',
    'SomaticServer',
    'get_somatic_server',
    # LAN Reflex Bridge (Spinal Cord)
    'ReflexEventType',
    'ReflexEvent',
    'LANReflexBridge',
    'get_lan_reflex_bridge',
]
