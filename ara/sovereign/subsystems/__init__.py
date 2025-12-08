"""
Sovereign Subsystems: Field Ownership Architecture

Each subsystem owns specific branches of the SovereignState tree.
This prevents race conditions and makes debugging trivial.

Ownership Matrix:
    BANOS       → hardware.*, safety.kill_switch_engaged
    MindReader  → user.*
    Covenant    → safety.autonomy_level, safety.trust_*, safety.founder_protection_active
    ChiefOfStaff→ work.*, safety.risk_assessment
    HTC         → soul.axis_*, soul.memory_*, soul.global_coherence
    Teleology   → teleology.*
    Avatar      → avatar.*
    Tracer      → trace.*
    Clock       → time.*

Usage:
    from ara.sovereign.subsystems import (
        OwnershipRegistry,
        GuardedStateWriter,
        Subsystem,
        BANOSSubsystem,
        MindReaderSubsystem,
        CovenantSubsystem,
    )

    # Create registry and writer
    registry = OwnershipRegistry(strict=True)
    writer = GuardedStateWriter(state, registry)

    # Create subsystems
    banos = BANOSSubsystem(writer)
    mind_reader = MindReaderSubsystem(writer)
    covenant = CovenantSubsystem(writer)

    # Run sense phase
    banos.sense()
    mind_reader.sense(last_message="Hello")
    covenant.evaluate()
"""

from .ownership import (
    Subsystem,
    FieldOwnership,
    OWNERSHIP_MATRIX,
    OwnershipRegistry,
    WriteRecord,
    GuardedStateWriter,
    SubsystemBase,
    get_ownership_registry,
    reset_ownership_registry,
)

from .banos import BANOSSubsystem
from .mind_reader import MindReaderSubsystem, CognitiveMode
from .covenant import CovenantSubsystem, AutonomyLevel, TrustTransaction

__all__ = [
    # Ownership
    'Subsystem',
    'FieldOwnership',
    'OWNERSHIP_MATRIX',
    'OwnershipRegistry',
    'WriteRecord',
    'GuardedStateWriter',
    'SubsystemBase',
    'get_ownership_registry',
    'reset_ownership_registry',

    # Subsystems
    'BANOSSubsystem',
    'MindReaderSubsystem',
    'CognitiveMode',
    'CovenantSubsystem',
    'AutonomyLevel',
    'TrustTransaction',
]
