"""
CovenantLogger Interface
=========================

Tamper-evident logging for high-stakes Ara events.

Use cases:
- Covenant changes (drift events, resets)
- Long-term memory saves
- High-stakes actions ("Ara refused X", "Ara did Y")
- Session boundaries

v0.7: InMemoryCovenantLogger / FileCovenantLogger (software)
Phase 2: SqrlCovenantLogger (hardware append-only log)

Each log entry is chained to the previous via hash,
creating a tamper-evident chain even in software mode.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Iterator
from datetime import datetime
import hashlib
import json
import numpy as np
from pathlib import Path


@dataclass
class CovenantLogEvent:
    """
    A single event in the covenant log.

    Fields:
    - sequence: Monotonic sequence number
    - timestamp_ns: Nanosecond timestamp
    - event_type: Category (e.g., "memory_save", "drift_alert", "refusal")
    - event_data: Arbitrary payload
    - hv_fingerprint: Optional 256D projection of related HV
    - prev_hash: Hash of previous event (chain integrity)
    - event_hash: Hash of this event (including prev_hash)
    """
    sequence: int
    timestamp_ns: int
    event_type: str
    event_data: dict
    hv_fingerprint: Optional[np.ndarray] = None
    prev_hash: bytes = field(default_factory=lambda: b'\x00' * 32)
    event_hash: bytes = field(default_factory=lambda: b'\x00' * 32)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        d = {
            'sequence': self.sequence,
            'timestamp_ns': self.timestamp_ns,
            'event_type': self.event_type,
            'event_data': self.event_data,
            'prev_hash': self.prev_hash.hex(),
            'event_hash': self.event_hash.hex(),
        }
        if self.hv_fingerprint is not None:
            d['hv_fingerprint'] = self.hv_fingerprint.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'CovenantLogEvent':
        """Deserialize from storage."""
        hv_fp = None
        if 'hv_fingerprint' in d and d['hv_fingerprint'] is not None:
            hv_fp = np.array(d['hv_fingerprint'], dtype=np.float32)

        return cls(
            sequence=d['sequence'],
            timestamp_ns=d['timestamp_ns'],
            event_type=d['event_type'],
            event_data=d['event_data'],
            hv_fingerprint=hv_fp,
            prev_hash=bytes.fromhex(d['prev_hash']),
            event_hash=bytes.fromhex(d['event_hash']),
        )

    def compute_hash(self) -> bytes:
        """Compute hash of this event (for chain integrity)."""
        payload = {
            'sequence': self.sequence,
            'timestamp_ns': self.timestamp_ns,
            'event_type': self.event_type,
            'event_data': self.event_data,
            'prev_hash': self.prev_hash.hex(),
        }
        if self.hv_fingerprint is not None:
            payload['hv_fingerprint'] = self.hv_fingerprint.tolist()

        data = json.dumps(payload, sort_keys=True).encode('utf-8')
        return hashlib.sha256(data).digest()


class CovenantLogger(ABC):
    """
    Abstract interface for covenant logging.

    Any implementation must provide:
    - log: Append an event to the log
    - get_events: Retrieve events (with optional filtering)
    - verify_chain: Check chain integrity
    - get_latest: Get most recent event

    Implementations:
    - InMemoryCovenantLogger: RAM only (for testing)
    - FileCovenantLogger: JSON file (v0.7 default)
    - SqrlCovenantLogger: Hardware append-only (Phase 2, kitten)
    """

    @abstractmethod
    def log(self,
            event_type: str,
            event_data: dict,
            hv: Optional[np.ndarray] = None) -> CovenantLogEvent:
        """
        Log a covenant event.

        Args:
            event_type: Category of event
            event_data: Arbitrary payload
            hv: Optional HV to fingerprint (will be projected to 256D)

        Returns:
            The created CovenantLogEvent (with hash chain)
        """
        pass

    @abstractmethod
    def get_events(self,
                   event_type: Optional[str] = None,
                   since_sequence: Optional[int] = None,
                   limit: int = 100) -> List[CovenantLogEvent]:
        """
        Retrieve logged events.

        Args:
            event_type: Filter by type (None = all)
            since_sequence: Only events after this sequence
            limit: Maximum events to return

        Returns:
            List of matching events (newest first)
        """
        pass

    @abstractmethod
    def verify_chain(self) -> bool:
        """
        Verify the entire log chain integrity.

        Returns:
            True if chain is intact, False if tampered
        """
        pass

    @abstractmethod
    def get_latest(self) -> Optional[CovenantLogEvent]:
        """Get the most recent event, or None if empty."""
        pass

    @property
    @abstractmethod
    def event_count(self) -> int:
        """Total number of events logged."""
        pass


def fingerprint_hv(hv: np.ndarray, target_dim: int = 256) -> np.ndarray:
    """
    Project an HV to a smaller fingerprint for logging.

    Uses deterministic random projection.
    """
    np.random.seed(42)  # Deterministic projection
    if len(hv) <= target_dim:
        result = np.zeros(target_dim, dtype=np.float32)
        result[:len(hv)] = hv
        return result

    # Random projection matrix (fixed seed = deterministic)
    projection = np.random.randn(len(hv), target_dim).astype(np.float32)
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)

    fingerprint = hv @ projection
    fingerprint /= np.linalg.norm(fingerprint) + 1e-8
    return fingerprint


class InMemoryCovenantLogger(CovenantLogger):
    """
    In-memory covenant logger for testing.

    Not persistent - all events lost on restart.
    Use FileCovenantLogger for production.
    """

    def __init__(self):
        self._events: List[CovenantLogEvent] = []
        self._sequence = 0

    def log(self,
            event_type: str,
            event_data: dict,
            hv: Optional[np.ndarray] = None) -> CovenantLogEvent:
        """Log an event to memory."""
        # Get previous hash
        prev_hash = b'\x00' * 32
        if self._events:
            prev_hash = self._events[-1].event_hash

        # Create fingerprint if HV provided
        hv_fp = None
        if hv is not None:
            hv_fp = fingerprint_hv(hv)

        # Create event
        event = CovenantLogEvent(
            sequence=self._sequence,
            timestamp_ns=int(datetime.utcnow().timestamp() * 1e9),
            event_type=event_type,
            event_data=event_data,
            hv_fingerprint=hv_fp,
            prev_hash=prev_hash,
        )

        # Compute and set hash
        event.event_hash = event.compute_hash()

        self._events.append(event)
        self._sequence += 1

        return event

    def get_events(self,
                   event_type: Optional[str] = None,
                   since_sequence: Optional[int] = None,
                   limit: int = 100) -> List[CovenantLogEvent]:
        """Get events from memory."""
        events = self._events

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        if since_sequence is not None:
            events = [e for e in events if e.sequence > since_sequence]

        # Return newest first, limited
        return list(reversed(events[-limit:]))

    def verify_chain(self) -> bool:
        """Verify the hash chain."""
        if not self._events:
            return True

        # First event should have zero prev_hash
        if self._events[0].prev_hash != b'\x00' * 32:
            return False

        # Verify each event's hash
        for i, event in enumerate(self._events):
            expected_hash = event.compute_hash()
            if event.event_hash != expected_hash:
                return False

            # Verify chain linkage
            if i > 0:
                if event.prev_hash != self._events[i - 1].event_hash:
                    return False

        return True

    def get_latest(self) -> Optional[CovenantLogEvent]:
        """Get most recent event."""
        return self._events[-1] if self._events else None

    @property
    def event_count(self) -> int:
        return len(self._events)


class FileCovenantLogger(CovenantLogger):
    """
    File-based covenant logger with JSON-lines format.

    v0.7 default for persistent logging.
    Each event is appended as a JSON line.
    Chain verification works across restarts.
    """

    def __init__(self, log_path: str = "covenant_log.jsonl"):
        self._path = Path(log_path)
        self._events: List[CovenantLogEvent] = []
        self._sequence = 0

        # Load existing events
        self._load()

    def _load(self):
        """Load events from file."""
        if not self._path.exists():
            return

        with open(self._path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    event = CovenantLogEvent.from_dict(json.loads(line))
                    self._events.append(event)
                    self._sequence = event.sequence + 1

    def _append(self, event: CovenantLogEvent):
        """Append event to file."""
        with open(self._path, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')

    def log(self,
            event_type: str,
            event_data: dict,
            hv: Optional[np.ndarray] = None) -> CovenantLogEvent:
        """Log an event to file."""
        # Get previous hash
        prev_hash = b'\x00' * 32
        if self._events:
            prev_hash = self._events[-1].event_hash

        # Create fingerprint if HV provided
        hv_fp = None
        if hv is not None:
            hv_fp = fingerprint_hv(hv)

        # Create event
        event = CovenantLogEvent(
            sequence=self._sequence,
            timestamp_ns=int(datetime.utcnow().timestamp() * 1e9),
            event_type=event_type,
            event_data=event_data,
            hv_fingerprint=hv_fp,
            prev_hash=prev_hash,
        )

        # Compute and set hash
        event.event_hash = event.compute_hash()

        # Persist and cache
        self._append(event)
        self._events.append(event)
        self._sequence += 1

        return event

    def get_events(self,
                   event_type: Optional[str] = None,
                   since_sequence: Optional[int] = None,
                   limit: int = 100) -> List[CovenantLogEvent]:
        """Get events from file."""
        events = self._events

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        if since_sequence is not None:
            events = [e for e in events if e.sequence > since_sequence]

        return list(reversed(events[-limit:]))

    def verify_chain(self) -> bool:
        """Verify the hash chain."""
        if not self._events:
            return True

        if self._events[0].prev_hash != b'\x00' * 32:
            return False

        for i, event in enumerate(self._events):
            expected_hash = event.compute_hash()
            if event.event_hash != expected_hash:
                return False

            if i > 0:
                if event.prev_hash != self._events[i - 1].event_hash:
                    return False

        return True

    def get_latest(self) -> Optional[CovenantLogEvent]:
        return self._events[-1] if self._events else None

    @property
    def event_count(self) -> int:
        return len(self._events)


# =============================================================================
# PLACEHOLDER FOR PHASE 2: SQRL Forest Kitten Logger
# =============================================================================

class SqrlCovenantLogger(CovenantLogger):
    """
    PLACEHOLDER: Hardware append-only log on SQRL Forest Kitten.

    Phase 2 implementation will:
    - Store logs in hardware-protected memory
    - Provide attestation that log wasn't tampered
    - Mirror to host for query (but truth is on device)

    For now, raises NotImplementedError.
    """

    def __init__(self, device_path: str = "/dev/sqrl0"):
        raise NotImplementedError(
            "SqrlCovenantLogger is Phase 2 hardware DLC. "
            "Use FileCovenantLogger for v0.7. "
            "The kitten's append-only log awaits. 😺"
        )

    def log(self, event_type: str, event_data: dict,
            hv: Optional[np.ndarray] = None) -> CovenantLogEvent:
        raise NotImplementedError("Phase 2: Kitten pending")

    def get_events(self, event_type: Optional[str] = None,
                   since_sequence: Optional[int] = None,
                   limit: int = 100) -> List[CovenantLogEvent]:
        raise NotImplementedError("Phase 2: Kitten pending")

    def verify_chain(self) -> bool:
        raise NotImplementedError("Phase 2: Kitten pending")

    def get_latest(self) -> Optional[CovenantLogEvent]:
        raise NotImplementedError("Phase 2: Kitten pending")

    @property
    def event_count(self) -> int:
        raise NotImplementedError("Phase 2: Kitten pending")


__all__ = [
    'CovenantLogEvent',
    'CovenantLogger',
    'InMemoryCovenantLogger',
    'FileCovenantLogger',
    'SqrlCovenantLogger',
    'fingerprint_hv',
]
