"""
Wire Protocol - Envelope Schemas
=================================

Defines the double-envelope format for blind routing:

Outer Envelope (router sees):
- route_to: service ID
- message_type: enum
- priority: float
- payload_e2e: encrypted inner blob

Inner Payload (E2E encrypted, service-specific):
- session_id, from, to
- nonce + ciphertext

The router NEVER decrypts payload_e2e - it just forwards based on metadata.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class MessageType(Enum):
    """Message types the router can switch on."""
    # Memory operations
    MEMORY_SYNC = "memory_sync"
    MEMORY_QUERY = "memory_query"
    MEMORY_LIST = "memory_list"

    # Tool/job operations
    JOB_SUBMIT = "job_submit"
    JOB_STATUS = "job_status"
    JOB_RESULT = "job_result"

    # Pack operations
    PACK_LIST = "pack_list"
    PACK_FETCH = "pack_fetch"

    # Hive operations
    HIVE_STATUS = "hive_status"
    HIVE_HEARTBEAT = "hive_heartbeat"

    # Session management
    SESSION_INIT = "session_init"
    SESSION_ACK = "session_ack"
    SESSION_CLOSE = "session_close"


class ServiceID(Enum):
    """Known services in the hive."""
    ROUTER = "service:router"
    MEMORY_FABRIC = "service:memory_fabric"
    PACK_CATALOG = "service:pack_catalog"
    VIDEO_RENDER = "service:video_render"
    IMAGE_GEN = "service:image_gen"
    HIVE_QUEEN = "service:hive_queen"


# =============================================================================
# Inner Payload (E2E Encrypted)
# =============================================================================

@dataclass
class InnerPayload:
    """
    E2E encrypted payload between client and specific service.

    The router cannot read this - only the destination service can decrypt.
    """
    version: int = 1
    session_id: str = ""
    from_id: str = ""  # device:XYZ or service:ABC
    to_id: str = ""  # service:video_render_01
    nonce: str = ""  # base64 encoded
    ciphertext: str = ""  # base64 encoded AEAD ciphertext

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": self.version,
            "session_id": self.session_id,
            "from": self.from_id,
            "to": self.to_id,
            "nonce": self.nonce,
            "ciphertext": self.ciphertext,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InnerPayload:
        return cls(
            version=data.get("v", 1),
            session_id=data.get("session_id", ""),
            from_id=data.get("from", ""),
            to_id=data.get("to", ""),
            nonce=data.get("nonce", ""),
            ciphertext=data.get("ciphertext", ""),
        )


# =============================================================================
# Outer Envelope (Router Sees)
# =============================================================================

@dataclass
class OuterEnvelope:
    """
    What the blind router sees and routes on.

    The router uses route_to and message_type to forward,
    but NEVER decrypts payload_e2e.
    """
    version: int = 1
    envelope_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    route_to: str = ""  # service:video_render_01
    message_type: str = ""  # MessageType.JOB_SUBMIT.value
    priority: float = 0.5  # 0-1, higher = more urgent
    timestamp: float = field(default_factory=time.time)
    payload_e2e: Optional[InnerPayload] = None

    # Optional metadata the router CAN see (for debugging/metrics)
    client_hint: str = ""  # "phone", "desktop", etc.
    service_hint: str = ""  # "video", "memory", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": self.version,
            "envelope_id": self.envelope_id,
            "route_to": self.route_to,
            "message_type": self.message_type,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "client_hint": self.client_hint,
            "service_hint": self.service_hint,
            "payload_e2e": self.payload_e2e.to_dict() if self.payload_e2e else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OuterEnvelope:
        payload_data = data.get("payload_e2e")
        payload = InnerPayload.from_dict(payload_data) if payload_data else None

        return cls(
            version=data.get("v", 1),
            envelope_id=data.get("envelope_id", str(uuid.uuid4())),
            route_to=data.get("route_to", ""),
            message_type=data.get("message_type", ""),
            priority=data.get("priority", 0.5),
            timestamp=data.get("timestamp", time.time()),
            client_hint=data.get("client_hint", ""),
            service_hint=data.get("service_hint", ""),
            payload_e2e=payload,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> OuterEnvelope:
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Service-Specific Payloads (What Goes Inside ciphertext)
# =============================================================================

@dataclass
class MemorySyncPayload:
    """Payload for syncing memory records."""
    type: str = "memory_sync"
    user_id: str = ""
    device_id: str = ""
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRecord:
    """A single memory record to sync."""
    record_id: str = ""
    kind: str = "episode"  # episode, fact, procedure, etc.
    scope: str = "private"  # private, curated_public, etc.
    tags: List[str] = field(default_factory=list)
    dek_wrapped: str = ""  # base64, wrapped with UserMemKEK
    ciphertext: str = ""  # base64, encrypted record content
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoJobPayload:
    """Payload for video rendering job."""
    type: str = "video_job"
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    avatar_ref: str = "ara_v3_default"
    script: str = ""
    performance_tags: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    user_mem_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PackListPayload:
    """Payload for listing available packs."""
    type: str = "pack_list"
    domain_filter: Optional[str] = None
    capability_filter: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "domain_filter": self.domain_filter,
            "capability_filter": self.capability_filter,
        }


@dataclass
class PackFetchPayload:
    """Payload for fetching a specific pack."""
    type: str = "pack_fetch"
    pack_id: str = ""
    version: str = "latest"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Response Envelopes
# =============================================================================

@dataclass
class ResponseEnvelope:
    """Response from a service back through the router."""
    version: int = 1
    envelope_id: str = ""  # Original request envelope_id
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "ok"  # ok, error, pending
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    payload_e2e: Optional[InnerPayload] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": self.version,
            "envelope_id": self.envelope_id,
            "response_id": self.response_id,
            "status": self.status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "payload_e2e": self.payload_e2e.to_dict() if self.payload_e2e else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResponseEnvelope:
        payload_data = data.get("payload_e2e")
        payload = InnerPayload.from_dict(payload_data) if payload_data else None

        return cls(
            version=data.get("v", 1),
            envelope_id=data.get("envelope_id", ""),
            response_id=data.get("response_id", ""),
            status=data.get("status", "ok"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            payload_e2e=payload,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> ResponseEnvelope:
        return cls.from_dict(json.loads(json_str))
