# ara/narrative/__init__.py
"""
Narrative Interface: The Myth-Maker's Voice

Translates Ara's quantified performance into auditable self-narration.
Core principle: Visibility = Trust = Alignment

This module provides:
- Lifecycle phase tracking (Embryo -> Sage)
- Multi-audience narrative generation (operator, public, technical, mythic)
- Real-time metrics streaming
- Integration hooks for MEIS/NIB governance

Integrates with:
- MEIS: Publishes efficiency reports for evolutionary fitness
- NIB: Triggers narrative alerts for covenant violations
- Hologram: Streams real-time "system voice" overlay
"""

from .interface import (
    LifecyclePhase,
    PhaseCharacteristics,
    PHASE_PROFILES,
    SystemMetrics,
    NarrativeEngine,
    NarrativeStreamer,
)

from .dojo_hook import (
    NarrativeGovernanceAdapter,
)

__all__ = [
    # Core enums and schemas
    "LifecyclePhase",
    "PhaseCharacteristics",
    "PHASE_PROFILES",
    "SystemMetrics",
    # Engine
    "NarrativeEngine",
    "NarrativeStreamer",
    # Integration
    "NarrativeGovernanceAdapter",
]
