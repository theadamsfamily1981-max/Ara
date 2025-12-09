"""
CovenantGuard Interface
========================

Signs and verifies high-stakes events for Ara.

Use cases:
- "Store this forever" memory events
- Cross-session identity verification
- Covenant snapshot signing

v0.7: LocalCovenantGuard (software HMAC)
Phase 2: SqrlCovenantGuard (hardware SGX enclave)

The interface is designed so v0.7 code doesn't know or care
whether it's talking to software or a hardware security module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import hashlib
import hmac
import secrets
import json
import numpy as np


@dataclass
class CovenantSignature:
    """
    A signature over a covenant event.

    Contains:
    - signature: The actual cryptographic signature
    - timestamp: When it was signed
    - guardian_id: Which guard produced it (for multi-guard scenarios)
    - algorithm: What algorithm was used
    """
    signature: bytes
    timestamp_ns: int
    guardian_id: str
    algorithm: str = "hmac-sha256"

    def to_hex(self) -> str:
        """Return signature as hex string."""
        return self.signature.hex()

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            'signature': self.signature.hex(),
            'timestamp_ns': self.timestamp_ns,
            'guardian_id': self.guardian_id,
            'algorithm': self.algorithm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CovenantSignature':
        """Deserialize from storage."""
        return cls(
            signature=bytes.fromhex(d['signature']),
            timestamp_ns=d['timestamp_ns'],
            guardian_id=d['guardian_id'],
            algorithm=d.get('algorithm', 'hmac-sha256'),
        )


class CovenantGuard(ABC):
    """
    Abstract interface for covenant signing and verification.

    Any implementation must provide:
    - sign_event: Produce a signature over an event
    - verify_event: Check if a signature is valid
    - sign_hv: Sign a hypervector (for memory/identity)
    - verify_hv: Verify a hypervector signature

    Implementations:
    - LocalCovenantGuard: Software HMAC (v0.7 default)
    - SqrlCovenantGuard: Hardware SGX (Phase 2, kitten)
    """

    @property
    @abstractmethod
    def guardian_id(self) -> str:
        """Unique identifier for this guard instance."""
        pass

    @abstractmethod
    def sign_event(self, event_type: str, event_data: dict) -> CovenantSignature:
        """
        Sign a covenant event.

        Args:
            event_type: Category of event (e.g., "memory_store", "drift_alert")
            event_data: Arbitrary event payload

        Returns:
            CovenantSignature that can be stored/transmitted
        """
        pass

    @abstractmethod
    def verify_event(self, event_type: str, event_data: dict,
                     signature: CovenantSignature) -> bool:
        """
        Verify a covenant event signature.

        Args:
            event_type: Category of event
            event_data: Event payload (must match what was signed)
            signature: The signature to verify

        Returns:
            True if signature is valid, False otherwise
        """
        pass

    @abstractmethod
    def sign_hv(self, hv: np.ndarray, label: str) -> CovenantSignature:
        """
        Sign a hypervector (for memory/identity protection).

        Args:
            hv: The hypervector to sign
            label: Human-readable label (e.g., "soul_snapshot_2024_01_15")

        Returns:
            CovenantSignature
        """
        pass

    @abstractmethod
    def verify_hv(self, hv: np.ndarray, label: str,
                  signature: CovenantSignature) -> bool:
        """
        Verify a hypervector signature.

        Args:
            hv: The hypervector to verify
            label: The label used when signing
            signature: The signature to verify

        Returns:
            True if valid, False otherwise
        """
        pass


class LocalCovenantGuard(CovenantGuard):
    """
    Software-only CovenantGuard using HMAC-SHA256.

    This is the v0.7 default. It provides:
    - Integrity verification (detect tampering)
    - NOT hardware-backed security
    - NOT protection against host compromise

    The secret key is generated randomly at init and can be
    persisted/loaded for cross-session verification.

    Phase 2: Replace with SqrlCovenantGuard that uses
    the Forest Kitten's SGX enclave for key storage.
    """

    def __init__(self,
                 secret_key: Optional[bytes] = None,
                 guardian_id: Optional[str] = None):
        """
        Initialize with optional secret key.

        Args:
            secret_key: 32-byte secret (generated if None)
            guardian_id: Identifier (generated if None)
        """
        self._secret_key = secret_key or secrets.token_bytes(32)
        self._guardian_id = guardian_id or f"local_{secrets.token_hex(4)}"

    @property
    def guardian_id(self) -> str:
        return self._guardian_id

    def _compute_mac(self, data: bytes) -> bytes:
        """Compute HMAC-SHA256 over data."""
        return hmac.new(self._secret_key, data, hashlib.sha256).digest()

    def _serialize_event(self, event_type: str, event_data: dict) -> bytes:
        """Deterministically serialize event for signing."""
        # Sort keys for deterministic output
        payload = {
            'event_type': event_type,
            'event_data': event_data,
        }
        return json.dumps(payload, sort_keys=True).encode('utf-8')

    def _serialize_hv(self, hv: np.ndarray, label: str) -> bytes:
        """Serialize HV for signing."""
        # Use tobytes for deterministic representation
        return label.encode('utf-8') + b'\x00' + hv.astype(np.float32).tobytes()

    def sign_event(self, event_type: str, event_data: dict) -> CovenantSignature:
        """Sign an event with HMAC-SHA256."""
        data = self._serialize_event(event_type, event_data)
        sig = self._compute_mac(data)

        return CovenantSignature(
            signature=sig,
            timestamp_ns=int(datetime.utcnow().timestamp() * 1e9),
            guardian_id=self._guardian_id,
            algorithm='hmac-sha256',
        )

    def verify_event(self, event_type: str, event_data: dict,
                     signature: CovenantSignature) -> bool:
        """Verify an event signature."""
        data = self._serialize_event(event_type, event_data)
        expected = self._compute_mac(data)
        return hmac.compare_digest(expected, signature.signature)

    def sign_hv(self, hv: np.ndarray, label: str) -> CovenantSignature:
        """Sign a hypervector."""
        data = self._serialize_hv(hv, label)
        sig = self._compute_mac(data)

        return CovenantSignature(
            signature=sig,
            timestamp_ns=int(datetime.utcnow().timestamp() * 1e9),
            guardian_id=self._guardian_id,
            algorithm='hmac-sha256',
        )

    def verify_hv(self, hv: np.ndarray, label: str,
                  signature: CovenantSignature) -> bool:
        """Verify a hypervector signature."""
        data = self._serialize_hv(hv, label)
        expected = self._compute_mac(data)
        return hmac.compare_digest(expected, signature.signature)

    def export_key(self) -> bytes:
        """Export secret key for persistence."""
        return self._secret_key

    @classmethod
    def from_key(cls, secret_key: bytes, guardian_id: str) -> 'LocalCovenantGuard':
        """Create guard from persisted key."""
        return cls(secret_key=secret_key, guardian_id=guardian_id)


# =============================================================================
# PLACEHOLDER FOR PHASE 2: SQRL Forest Kitten Implementation
# =============================================================================

class SqrlCovenantGuard(CovenantGuard):
    """
    PLACEHOLDER: Hardware-backed CovenantGuard using SQRL Forest Kitten.

    Phase 2 implementation will:
    - Store keys in SGX enclave on the Cyclone 10
    - Perform signing inside the enclave
    - Provide hardware attestation

    For now, this raises NotImplementedError to clearly indicate
    "this is where the kitten plugs in."
    """

    def __init__(self, device_path: str = "/dev/sqrl0"):
        self._device_path = device_path
        raise NotImplementedError(
            "SqrlCovenantGuard is Phase 2 hardware DLC. "
            "Use LocalCovenantGuard for v0.7. "
            "The kitten is waiting patiently. 😺"
        )

    @property
    def guardian_id(self) -> str:
        return "sqrl_forest_kitten"

    def sign_event(self, event_type: str, event_data: dict) -> CovenantSignature:
        raise NotImplementedError("Phase 2: Kitten pending")

    def verify_event(self, event_type: str, event_data: dict,
                     signature: CovenantSignature) -> bool:
        raise NotImplementedError("Phase 2: Kitten pending")

    def sign_hv(self, hv: np.ndarray, label: str) -> CovenantSignature:
        raise NotImplementedError("Phase 2: Kitten pending")

    def verify_hv(self, hv: np.ndarray, label: str,
                  signature: CovenantSignature) -> bool:
        raise NotImplementedError("Phase 2: Kitten pending")


__all__ = [
    'CovenantGuard',
    'CovenantSignature',
    'LocalCovenantGuard',
    'SqrlCovenantGuard',
]
