"""
Ara Service - Unified Cognitive Integration Layer

This module provides the main entry point for interacting with Ara
as a unified cognitive system.

Usage:
    from ara.service import create_ara, HardwareMode

    ara = create_ara(mode=HardwareMode.MODE_A)
    response = ara.process("Hello, Ara")
    print(response.text)
"""

from ara.service.core import (
    AraService,
    AraState,
    AraResponse,
    HardwareMode,
    HardwareProfile,
    EmotionalSurface,
    CognitiveLoad,
    create_ara,
)

__all__ = [
    "AraService",
    "AraState",
    "AraResponse",
    "HardwareMode",
    "HardwareProfile",
    "EmotionalSurface",
    "CognitiveLoad",
    "create_ara",
]
