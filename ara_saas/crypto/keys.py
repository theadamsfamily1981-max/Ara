"""
Crypto Keys & Sessions
=======================

Key hierarchy for the Ara memory system:

Per User:
- UserRootSK/PK: Long-term signature key
- UserMemKEK: Master key to wrap per-record data keys

Per Device:
- DeviceSK/PK: Device identity keypair
- Device cert signed by UserRootSK

Per Session:
- SessionKey: Derived from X25519 + (future) ML-KEM
- Used for AEAD (AES-256-GCM)

v0.1 Implementation:
- Uses standard library where possible
- Placeholder for PQ algorithms (swap later)
"""

from __future__ import annotations

import os
import base64
import hashlib
import secrets
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import cryptography library, fallback to basic stdlib
# First check if cffi backend is available (prevents pyo3 panic)
HAS_CRYPTO = False
AESGCM = None
HKDF = None
hashes = None

try:
    import _cffi_backend  # noqa: F401 - test if cffi is working
    _cffi_ok = True
except (ImportError, ModuleNotFoundError):
    _cffi_ok = False
    logger.warning("cffi backend not available, skipping cryptography library")

if _cffi_ok:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        HAS_CRYPTO = True
    except (ImportError, Exception) as e:
        logger.warning(f"cryptography library not available ({type(e).__name__}), using basic fallback")


# =============================================================================
# Constants
# =============================================================================

KEY_SIZE = 32  # 256 bits for AES-256
NONCE_SIZE = 12  # 96 bits for AES-GCM
SALT_SIZE = 16


# =============================================================================
# Key Types
# =============================================================================

@dataclass
class KeyPair:
    """A generic keypair."""
    public_key: bytes
    secret_key: bytes
    algorithm: str = "x25519"  # or "ml-kem-768" later

    def public_key_b64(self) -> str:
        return base64.b64encode(self.public_key).decode()

    def secret_key_b64(self) -> str:
        return base64.b64encode(self.secret_key).decode()


@dataclass
class SymmetricKey:
    """A symmetric key (AES-256)."""
    key: bytes
    algorithm: str = "aes-256-gcm"

    def to_b64(self) -> str:
        return base64.b64encode(self.key).decode()

    @classmethod
    def from_b64(cls, b64: str) -> SymmetricKey:
        return cls(key=base64.b64decode(b64))

    @classmethod
    def generate(cls) -> SymmetricKey:
        return cls(key=secrets.token_bytes(KEY_SIZE))


@dataclass
class WrappedKey:
    """A key wrapped (encrypted) with another key."""
    ciphertext: bytes
    nonce: bytes
    algorithm: str = "aes-256-gcm-wrap"

    def to_b64(self) -> str:
        return base64.b64encode(self.nonce + self.ciphertext).decode()

    @classmethod
    def from_b64(cls, b64: str) -> WrappedKey:
        data = base64.b64decode(b64)
        return cls(
            nonce=data[:NONCE_SIZE],
            ciphertext=data[NONCE_SIZE:],
        )


# =============================================================================
# AEAD Operations (AES-256-GCM)
# =============================================================================

def aead_encrypt(
    key: bytes,
    plaintext: bytes,
    associated_data: Optional[bytes] = None,
) -> Tuple[bytes, bytes]:
    """
    Encrypt with AES-256-GCM.

    Returns (nonce, ciphertext).
    """
    nonce = secrets.token_bytes(NONCE_SIZE)

    if HAS_CRYPTO:
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
    else:
        # Fallback: XOR with key hash (NOT SECURE - dev only!)
        logger.warning("Using insecure fallback encryption!")
        key_stream = hashlib.sha256(key + nonce).digest()
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, key_stream * 100))

    return nonce, ciphertext


def aead_decrypt(
    key: bytes,
    nonce: bytes,
    ciphertext: bytes,
    associated_data: Optional[bytes] = None,
) -> bytes:
    """
    Decrypt with AES-256-GCM.

    Returns plaintext.
    """
    if HAS_CRYPTO:
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data)
    else:
        # Fallback: XOR with key hash (NOT SECURE - dev only!)
        key_stream = hashlib.sha256(key + nonce).digest()
        plaintext = bytes(c ^ k for c, k in zip(ciphertext, key_stream * 100))

    return plaintext


# =============================================================================
# Key Derivation
# =============================================================================

def derive_key(
    secret: bytes,
    info: bytes,
    salt: Optional[bytes] = None,
    length: int = KEY_SIZE,
) -> bytes:
    """
    Derive a key using HKDF.

    Args:
        secret: Input key material
        info: Context info (e.g., "session_key")
        salt: Optional salt
        length: Output length

    Returns:
        Derived key bytes
    """
    if salt is None:
        salt = b"\x00" * SALT_SIZE

    if HAS_CRYPTO:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            info=info,
        )
        return hkdf.derive(secret)
    else:
        # Fallback: Simple hash-based derivation
        return hashlib.sha256(salt + secret + info).digest()[:length]


def derive_session_key(
    x25519_shared: bytes,
    pq_shared: Optional[bytes],
    client_id: str,
    service_id: str,
) -> bytes:
    """
    Derive a session key from key exchange results.

    Combines classical (X25519) and post-quantum (ML-KEM) shares.
    """
    # Combine shares
    if pq_shared:
        combined = x25519_shared + pq_shared
    else:
        combined = x25519_shared

    # Derive with context
    info = f"session:{client_id}:{service_id}".encode()
    return derive_key(combined, info)


# =============================================================================
# Key Wrapping
# =============================================================================

def wrap_key(kek: bytes, key_to_wrap: bytes) -> WrappedKey:
    """
    Wrap a key with a key-encrypting key (KEK).

    Uses AEAD so the wrapped key is authenticated.
    """
    nonce, ciphertext = aead_encrypt(kek, key_to_wrap)
    return WrappedKey(ciphertext=ciphertext, nonce=nonce)


def unwrap_key(kek: bytes, wrapped: WrappedKey) -> bytes:
    """
    Unwrap a key using a KEK.
    """
    return aead_decrypt(kek, wrapped.nonce, wrapped.ciphertext)


# =============================================================================
# User Keys
# =============================================================================

@dataclass
class UserKeys:
    """Complete set of user keys."""
    user_id: str
    root_keypair: KeyPair  # For signing
    mem_kek: SymmetricKey  # For wrapping memory record keys

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "root_pk": self.root_keypair.public_key_b64(),
            "root_sk": self.root_keypair.secret_key_b64(),
            "mem_kek": self.mem_kek.to_b64(),
        }

    def save(self, path: Path) -> None:
        """Save keys to file (KEEP SECURE!)."""
        path.write_text(json.dumps(self.to_dict()))
        logger.info(f"Saved user keys to {path}")

    @classmethod
    def load(cls, path: Path) -> UserKeys:
        """Load keys from file."""
        data = json.loads(path.read_text())
        return cls(
            user_id=data["user_id"],
            root_keypair=KeyPair(
                public_key=base64.b64decode(data["root_pk"]),
                secret_key=base64.b64decode(data["root_sk"]),
            ),
            mem_kek=SymmetricKey.from_b64(data["mem_kek"]),
        )

    @classmethod
    def generate(cls, user_id: str) -> UserKeys:
        """Generate new user keys."""
        # For v0.1: random bytes as keypair (placeholder for real Ed25519/Dilithium)
        root_kp = KeyPair(
            public_key=secrets.token_bytes(32),
            secret_key=secrets.token_bytes(64),
            algorithm="ed25519-placeholder",
        )
        mem_kek = SymmetricKey.generate()

        return cls(
            user_id=user_id,
            root_keypair=root_kp,
            mem_kek=mem_kek,
        )


@dataclass
class DeviceKeys:
    """Device-specific keys."""
    device_id: str
    keypair: KeyPair
    user_id: str  # Associated user

    @classmethod
    def generate(cls, device_id: str, user_id: str) -> DeviceKeys:
        """Generate device keys."""
        kp = KeyPair(
            public_key=secrets.token_bytes(32),
            secret_key=secrets.token_bytes(32),
            algorithm="x25519-placeholder",
        )
        return cls(device_id=device_id, keypair=kp, user_id=user_id)


# =============================================================================
# Session
# =============================================================================

@dataclass
class Session:
    """An encrypted session between client and service."""
    session_id: str
    client_id: str
    service_id: str
    session_key: bytes
    created_at: float = 0.0

    def encrypt(self, plaintext: bytes) -> Tuple[bytes, bytes]:
        """Encrypt data for this session."""
        return aead_encrypt(self.session_key, plaintext)

    def decrypt(self, nonce: bytes, ciphertext: bytes) -> bytes:
        """Decrypt data from this session."""
        return aead_decrypt(self.session_key, nonce, ciphertext)

    @classmethod
    def establish(
        cls,
        session_id: str,
        client_id: str,
        service_id: str,
        shared_secret: bytes,
    ) -> Session:
        """Establish a session from a shared secret."""
        import time
        session_key = derive_session_key(
            x25519_shared=shared_secret,
            pq_shared=None,  # Add PQ later
            client_id=client_id,
            service_id=service_id,
        )
        return cls(
            session_id=session_id,
            client_id=client_id,
            service_id=service_id,
            session_key=session_key,
            created_at=time.time(),
        )
