"""
Ara Service - Unified Cognitive Integration Layer

This module provides the main entry point for interacting with Ara
as a unified cognitive system.

Usage:
    from ara.service import create_ara, HardwareMode

    ara = create_ara(mode=HardwareMode.MODE_A)
    response = ara.process("Hello, Ara")
    print(response.text)

With LLM backend:
    # If Ollama is running with mistral model:
    ara = AraService(
        mode=HardwareMode.MODE_A,
        llm_backend="ollama",
        llm_model="mistral"
    )

State persistence:
    # State is auto-saved to ~/.ara/
    # On restart, previous state is restored
    ara.save_state()    # Manual save
    ara.shutdown()      # Save and shutdown
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

from ara.service.llm_backend import (
    LLMBackendType,
    LLMConfig,
    LLMResponse,
    AdaptiveLLMBackend,
    create_llm_backend,
)

from ara.service.persistence import (
    PersistedState,
    StatePersistence,
    create_persistence,
)

__all__ = [
    # Core
    "AraService",
    "AraState",
    "AraResponse",
    "HardwareMode",
    "HardwareProfile",
    "EmotionalSurface",
    "CognitiveLoad",
    "create_ara",
    # LLM
    "LLMBackendType",
    "LLMConfig",
    "LLMResponse",
    "AdaptiveLLMBackend",
    "create_llm_backend",
    # Persistence
    "PersistedState",
    "StatePersistence",
    "create_persistence",
]
