"""
Ara Covenant Module
===================

Interfaces for covenant signing, verification, and logging.

Designed for FUTURE hardware backends (SQRL Forest Kitten, SGX, etc.)
but v0.7 runs with pure software implementations.

Components:
- CovenantGuard: Signs and verifies events (identity protection)
- CovenantLogger: Tamper-evident logging (audit trail)

Design philosophy:
- Define interfaces NOW so code is written as-if hardware exists
- Software implementations work TODAY on any machine
- Hardware implementations plug in LATER without code changes

Phase 2 Hardware: SQRL Forest Kitten (Cyclone 10 + SGX)
- When ready: implement SqrlCovenantGuard, SqrlCovenantLogger
- v0.7: use LocalCovenantGuard, InMemoryCovenantLogger
"""

from ara.covenant.guard import (
    CovenantGuard,
    CovenantSignature,
    LocalCovenantGuard,
)

from ara.covenant.logger import (
    CovenantLogger,
    CovenantLogEvent,
    InMemoryCovenantLogger,
    FileCovenantLogger,
)

__all__ = [
    # Guard interface
    'CovenantGuard',
    'CovenantSignature',
    'LocalCovenantGuard',

    # Logger interface
    'CovenantLogger',
    'CovenantLogEvent',
    'InMemoryCovenantLogger',
    'FileCovenantLogger',
]
