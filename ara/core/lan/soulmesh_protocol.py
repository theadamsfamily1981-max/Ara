"""
Ara SoulMesh Protocol - Teleological Network Messages
=====================================================

Defines the message schema for SoulMesh - Ara's internal network protocol.

Every SoulMesh message carries:
- Teleological priority (how aligned with Ara's purpose)
- Context hash (truncated HV of the sending moment)
- Affect state (valence/arousal for quick read)
- Type-specific payload

This allows the network itself to participate in Ara's cognition:
- High-priority messages get preferential routing
- Affect hints enable quick visual feedback before processing
- Context hashes support HV-based flow classification
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
import numpy as np


# =============================================================================
# Protocol Constants
# =============================================================================

SOULMESH_MAGIC = b"ALWY"  # "Always" - the soul is always watching
SOULMESH_VERSION = 1


class MessageType(str, Enum):
    """SoulMesh message types."""
    SOMATIC = "somatic"      # Proprioception/health report
    POLICY = "policy"        # Policy update from sovereign
    EVENT = "event"          # Significant event notification
    HEARTBEAT = "heartbeat"  # Node liveness check
    QUERY = "query"          # Request for information
    RESPONSE = "response"    # Response to query


# =============================================================================
# Message Schema
# =============================================================================

@dataclass
class SoulMeshMessage:
    """
    A SoulMesh protocol message.

    Every message carries teleological metadata alongside its payload,
    allowing the network to participate in Ara's purposeful cognition.
    """
    node_id: str                    # Originating node
    msg_type: MessageType           # Message type
    priority: int                   # 0-255 (teleology-aligned priority)
    context_hash: bytes             # 16-byte hash of H_moment
    affect: Dict[str, float]        # valence, arousal for quick read
    payload: Dict[str, Any]         # Type-specific data
    sequence: int = 0               # Sequence number for ordering
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "msg_type": self.msg_type.value,
            "priority": self.priority,
            "context_hash": self.context_hash.hex(),
            "affect": self.affect,
            "payload": self.payload,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoulMeshMessage":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            msg_type=MessageType(data["msg_type"]),
            priority=data["priority"],
            context_hash=bytes.fromhex(data["context_hash"]),
            affect=data["affect"],
            payload=data["payload"],
            sequence=data.get("sequence", 0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


# =============================================================================
# Wire Format
# =============================================================================

def pack_message(msg: SoulMeshMessage) -> bytes:
    """
    Pack a SoulMesh message into wire format.

    Wire format (header):
        4 bytes: magic ("ALWY")
        1 byte:  version
        1 byte:  msg_type enum
        1 byte:  priority
        1 byte:  reserved
        16 bytes: context_hash
        1 byte:  valence (quantized -1..1 to 0..255)
        1 byte:  arousal (quantized 0..1 to 0..255)
        2 bytes: reserved
        4 bytes: payload length
        4 bytes: sequence number
        ... payload (JSON or msgpack)
    """
    import json

    # Quantize affect
    valence_q = int((msg.affect.get("valence", 0) + 1) * 127.5)
    arousal_q = int(msg.affect.get("arousal", 0.5) * 255)

    # Encode payload
    payload_bytes = json.dumps({
        "node_id": msg.node_id,
        "payload": msg.payload,
        "timestamp": msg.timestamp.isoformat(),
    }).encode("utf-8")

    # Map message type to enum value
    type_map = {
        MessageType.SOMATIC: 0,
        MessageType.POLICY: 1,
        MessageType.EVENT: 2,
        MessageType.HEARTBEAT: 3,
        MessageType.QUERY: 4,
        MessageType.RESPONSE: 5,
    }
    msg_type_byte = type_map.get(msg.msg_type, 0)

    # Pack header
    header = struct.pack(
        "!4sBBBB16sBBHII",
        SOULMESH_MAGIC,
        SOULMESH_VERSION,
        msg_type_byte,
        min(255, max(0, msg.priority)),
        0,  # reserved
        msg.context_hash[:16].ljust(16, b'\x00'),
        min(255, max(0, valence_q)),
        min(255, max(0, arousal_q)),
        0,  # reserved
        len(payload_bytes),
        msg.sequence,
    )

    return header + payload_bytes


def unpack_message(data: bytes) -> Optional[SoulMeshMessage]:
    """
    Unpack a SoulMesh message from wire format.

    Returns None if parsing fails.
    """
    import json

    if len(data) < 36:  # Minimum header size
        return None

    try:
        # Unpack header
        (magic, version, msg_type_byte, priority, _,
         context_hash, valence_q, arousal_q, _,
         payload_len, sequence) = struct.unpack(
            "!4sBBBB16sBBHII", data[:36]
        )

        if magic != SOULMESH_MAGIC:
            return None

        if version != SOULMESH_VERSION:
            return None

        # Extract payload
        payload_bytes = data[36:36 + payload_len]
        payload_data = json.loads(payload_bytes.decode("utf-8"))

        # Dequantize affect
        valence = (valence_q / 127.5) - 1.0
        arousal = arousal_q / 255.0

        # Map byte to message type
        type_map = {
            0: MessageType.SOMATIC,
            1: MessageType.POLICY,
            2: MessageType.EVENT,
            3: MessageType.HEARTBEAT,
            4: MessageType.QUERY,
            5: MessageType.RESPONSE,
        }
        msg_type = type_map.get(msg_type_byte, MessageType.EVENT)

        return SoulMeshMessage(
            node_id=payload_data.get("node_id", "unknown"),
            msg_type=msg_type,
            priority=priority,
            context_hash=context_hash,
            affect={"valence": valence, "arousal": arousal},
            payload=payload_data.get("payload", {}),
            sequence=sequence,
            timestamp=datetime.fromisoformat(payload_data.get("timestamp", datetime.utcnow().isoformat())),
        )

    except Exception:
        return None


# =============================================================================
# Helper Functions
# =============================================================================

def hash_hv(hv: np.ndarray, length: int = 16) -> bytes:
    """
    Create a truncated hash of a hypervector.

    This provides a compact identifier for HV-based flow matching.
    """
    h = hashlib.sha256(hv.tobytes()).digest()
    return h[:length]


def create_somatic_message(
    node_id: str,
    metrics: Dict[str, float],
    context_hv: Optional[np.ndarray] = None,
    priority: Optional[int] = None,
) -> SoulMeshMessage:
    """
    Create a SOMATIC (proprioception) message.

    Args:
        node_id: Node identifier
        metrics: Dict with temp, voltage, load, etc.
        context_hv: Optional HV for context hash
        priority: Optional explicit priority (else computed from metrics)
    """
    # Compute priority from metrics if not provided
    if priority is None:
        priority = 128  # Default middle priority
        if metrics.get("temp", 0) > 80:
            priority = min(255, priority + 50)
        if metrics.get("error_rate", 0) > 0.05:
            priority = min(255, priority + 50)

    # Estimate affect from metrics
    valence = 0.0
    if metrics.get("temp", 0) > 70:
        valence -= 0.3
    if metrics.get("error_rate", 0) > 0.01:
        valence -= 0.3
    arousal = min(1.0, metrics.get("cpu_load", 0) * 0.5 + 0.3)

    context_hash = hash_hv(context_hv) if context_hv is not None else b'\x00' * 16

    return SoulMeshMessage(
        node_id=node_id,
        msg_type=MessageType.SOMATIC,
        priority=priority,
        context_hash=context_hash,
        affect={"valence": valence, "arousal": arousal},
        payload={"metrics": metrics},
    )


__all__ = [
    'SOULMESH_MAGIC',
    'SOULMESH_VERSION',
    'MessageType',
    'SoulMeshMessage',
    'pack_message',
    'unpack_message',
    'hash_hv',
    'create_somatic_message',
]
