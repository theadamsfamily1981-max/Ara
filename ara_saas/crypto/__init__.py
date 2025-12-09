"""
Ara SaaS Crypto
================

Client-side cryptographic operations.

Key Hierarchy:
- UserRootSK: Ed25519 signing key (never leaves device)
- UserMemKEK: AES-256 key for wrapping memory record keys
- DeviceKeys: Per-device X25519 + ML-KEM hybrid
- SessionKeys: Ephemeral session keys derived via HKDF

All encryption happens on the client. Server only sees ciphertext.
"""

from .keys import (
    UserKeys,
    DeviceKeys,
    Session,
    KeyPair,
    SymmetricKey,
    WrappedKey,
    aead_encrypt,
    aead_decrypt,
    wrap_key,
    unwrap_key,
    derive_key,
    derive_session_key,
)

__all__ = [
    "UserKeys",
    "DeviceKeys",
    "Session",
    "KeyPair",
    "SymmetricKey",
    "WrappedKey",
    "aead_encrypt",
    "aead_decrypt",
    "wrap_key",
    "unwrap_key",
    "derive_key",
    "derive_session_key",
]
