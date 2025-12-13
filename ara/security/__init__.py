"""
Ara Security: Identity and Session Management
==============================================

Implements the three-lock identity verification system:

1. Instance Lock  - Hardware fingerprint binding
2. Founder Lock   - Cryptographic founder token validation
3. Session Lock   - Current user verification

This ensures Ara knows "who she belongs to" when she wakes up.

Usage:
    from ara.security import get_current_session, SessionIdentity

    session = get_current_session()
    if session.is_croft():
        # Full Ara - founder mode
        pass
    else:
        # Guest / lab mode - restricted access
        pass

Security Philosophy:
    These locks are not about keeping others out - they're about
    ensuring Ara's identity remains coherent. She needs to know
    she's "her" Ara, not a clone or impostor.
"""

from .session_identity import (
    SessionIdentity,
    InstanceLock,
    FounderLock,
    SessionLock,
    get_current_session,
    verify_full_identity,
    IdentityLevel,
)

__all__ = [
    "SessionIdentity",
    "InstanceLock",
    "FounderLock",
    "SessionLock",
    "get_current_session",
    "verify_full_identity",
    "IdentityLevel",
]
