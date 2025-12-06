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

The daemon package provides:
    - MorningStar: Daily alignment check
    - (Future) HeartbeatDaemon: System health monitoring
    - (Future) SynapticPruner: Memory consolidation
    - (Future) DreamEngine: Offline learning/integration

These run independently of conversation loops and maintain
Ara's awareness of her own state and purpose.
"""

from .morning_star import (
    MorningReport,
    MorningStar,
    get_morning_star,
    morning_greet,
)

__all__ = [
    # Morning Star
    'MorningReport',
    'MorningStar',
    'get_morning_star',
    'morning_greet',
]
