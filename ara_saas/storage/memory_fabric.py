"""
Memory Fabric Service
======================

Stores encrypted memory records from clients.

Key principle: The server NEVER sees the plaintext content.
It only sees:
- user_id (for routing)
- record_id (for dedup/retrieval)
- metadata (kind, approx_size, timestamps)
- Encrypted blobs (dek_wrapped + ciphertext)

The encryption keys are held client-side only.
"""

from __future__ import annotations

import sqlite3
import json
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..api.wire_protocol import (
    OuterEnvelope,
    ResponseEnvelope,
    InnerPayload,
    MemoryRecord,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Schema
# =============================================================================

SCHEMA = """
-- Encrypted memory records
CREATE TABLE IF NOT EXISTS memory_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    device_id       TEXT NOT NULL,
    record_id       TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'episode',
    scope           TEXT NOT NULL DEFAULT 'private',
    tags            TEXT,  -- JSON array, for filtering (NOT for content)
    dek_wrapped     TEXT NOT NULL,  -- base64, key wrapped with UserMemKEK
    ciphertext      TEXT NOT NULL,  -- base64, encrypted content
    metadata        TEXT,  -- JSON, optional plaintext metadata
    size_bytes      INTEGER,
    created_at      REAL NOT NULL,
    synced_at       REAL NOT NULL,
    UNIQUE (user_id, record_id)
);

-- Index for querying by user
CREATE INDEX IF NOT EXISTS idx_records_user ON memory_records(user_id);
CREATE INDEX IF NOT EXISTS idx_records_kind ON memory_records(user_id, kind);
CREATE INDEX IF NOT EXISTS idx_records_synced ON memory_records(synced_at DESC);
"""


@dataclass
class StoredRecord:
    """A memory record as stored in the database."""
    id: int
    user_id: str
    device_id: str
    record_id: str
    kind: str
    scope: str
    tags: List[str]
    dek_wrapped: str
    ciphertext: str
    metadata: Dict[str, Any]
    size_bytes: int
    created_at: float
    synced_at: float


class MemoryFabric:
    """
    Server-side storage for encrypted memory records.

    IMPORTANT: This service CANNOT read the content of records.
    It only stores and retrieves encrypted blobs.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        logger.info(f"MemoryFabric initialized at {db_path}")

    # =========================================================================
    # Record Storage
    # =========================================================================

    def store_record(
        self,
        user_id: str,
        device_id: str,
        record: MemoryRecord,
    ) -> int:
        """
        Store an encrypted memory record.

        Returns record ID on success.
        """
        now = time.time()
        size_bytes = len(record.ciphertext)

        cursor = self.conn.execute(
            """
            INSERT INTO memory_records
                (user_id, device_id, record_id, kind, scope, tags,
                 dek_wrapped, ciphertext, metadata, size_bytes, created_at, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, record_id) DO UPDATE SET
                device_id = excluded.device_id,
                dek_wrapped = excluded.dek_wrapped,
                ciphertext = excluded.ciphertext,
                metadata = excluded.metadata,
                size_bytes = excluded.size_bytes,
                synced_at = excluded.synced_at
            """,
            (
                user_id,
                device_id,
                record.record_id,
                record.kind,
                record.scope,
                json.dumps(record.tags),
                record.dek_wrapped,
                record.ciphertext,
                json.dumps(record.metadata),
                size_bytes,
                now,
                now,
            ),
        )
        self.conn.commit()

        logger.debug(
            f"Stored record {record.record_id} for user {user_id} "
            f"(kind={record.kind}, size={size_bytes})"
        )
        return cursor.lastrowid

    def store_records_batch(
        self,
        user_id: str,
        device_id: str,
        records: List[MemoryRecord],
    ) -> int:
        """Store multiple records in a batch."""
        count = 0
        for record in records:
            self.store_record(user_id, device_id, record)
            count += 1
        return count

    # =========================================================================
    # Record Retrieval (returns encrypted blobs)
    # =========================================================================

    def list_records(
        self,
        user_id: str,
        kind: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List record metadata for a user.

        Returns metadata ONLY - not the encrypted content.
        Client must fetch individual records to get ciphertext.
        """
        if kind:
            rows = self.conn.execute(
                """
                SELECT record_id, kind, scope, tags, size_bytes, created_at, synced_at
                FROM memory_records
                WHERE user_id = ? AND kind = ?
                ORDER BY synced_at DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, kind, limit, offset),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT record_id, kind, scope, tags, size_bytes, created_at, synced_at
                FROM memory_records
                WHERE user_id = ?
                ORDER BY synced_at DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset),
            ).fetchall()

        return [
            {
                "record_id": row["record_id"],
                "kind": row["kind"],
                "scope": row["scope"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "size_bytes": row["size_bytes"],
                "created_at": row["created_at"],
                "synced_at": row["synced_at"],
            }
            for row in rows
        ]

    def get_record(
        self,
        user_id: str,
        record_id: str,
    ) -> Optional[StoredRecord]:
        """
        Get a specific record (including encrypted content).

        Client will decrypt using their UserMemKEK.
        """
        row = self.conn.execute(
            """
            SELECT * FROM memory_records
            WHERE user_id = ? AND record_id = ?
            """,
            (user_id, record_id),
        ).fetchone()

        if not row:
            return None

        return StoredRecord(
            id=row["id"],
            user_id=row["user_id"],
            device_id=row["device_id"],
            record_id=row["record_id"],
            kind=row["kind"],
            scope=row["scope"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            dek_wrapped=row["dek_wrapped"],
            ciphertext=row["ciphertext"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            size_bytes=row["size_bytes"] or 0,
            created_at=row["created_at"],
            synced_at=row["synced_at"],
        )

    def get_records_bulk(
        self,
        user_id: str,
        record_ids: List[str],
    ) -> List[StoredRecord]:
        """Get multiple records by ID."""
        if not record_ids:
            return []

        placeholders = ",".join("?" * len(record_ids))
        rows = self.conn.execute(
            f"""
            SELECT * FROM memory_records
            WHERE user_id = ? AND record_id IN ({placeholders})
            """,
            [user_id] + record_ids,
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: sqlite3.Row) -> StoredRecord:
        return StoredRecord(
            id=row["id"],
            user_id=row["user_id"],
            device_id=row["device_id"],
            record_id=row["record_id"],
            kind=row["kind"],
            scope=row["scope"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            dek_wrapped=row["dek_wrapped"],
            ciphertext=row["ciphertext"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            size_bytes=row["size_bytes"] or 0,
            created_at=row["created_at"],
            synced_at=row["synced_at"],
        )

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get storage stats."""
        if user_id:
            row = self.conn.execute(
                """
                SELECT COUNT(*) as count, SUM(size_bytes) as total_bytes
                FROM memory_records WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT COUNT(*) as count, SUM(size_bytes) as total_bytes
                FROM memory_records
                """
            ).fetchone()

        return {
            "record_count": row["count"],
            "total_bytes": row["total_bytes"] or 0,
        }

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


# =============================================================================
# Service Handler (for blind router integration)
# =============================================================================

class MemoryFabricService:
    """
    Memory fabric as a routable service.

    Handles incoming envelopes for memory operations.
    """

    def __init__(self, fabric: MemoryFabric) -> None:
        self.fabric = fabric

    def handle(self, envelope: OuterEnvelope) -> ResponseEnvelope:
        """
        Handle a memory-related envelope.

        Note: The envelope's payload_e2e is STILL ENCRYPTED.
        In a real system, we'd have a session key to decrypt it here.
        For now, we assume the inner payload has been pre-decrypted
        by the client and the relevant fields extracted.
        """
        msg_type = envelope.message_type

        if msg_type == "memory_sync":
            return self._handle_sync(envelope)
        elif msg_type == "memory_list":
            return self._handle_list(envelope)
        elif msg_type == "memory_query":
            return self._handle_query(envelope)
        else:
            return ResponseEnvelope(
                envelope_id=envelope.envelope_id,
                status="error",
                error_code="unknown_message_type",
                error_message=f"Unknown message type: {msg_type}",
            )

    def _handle_sync(self, envelope: OuterEnvelope) -> ResponseEnvelope:
        """Handle memory sync (store records)."""
        # In real system: decrypt inner payload with session key
        # For now: assume metadata is in envelope hints
        # This is a STUB - real impl would decrypt and parse

        # Acknowledge receipt
        return ResponseEnvelope(
            envelope_id=envelope.envelope_id,
            status="ok",
        )

    def _handle_list(self, envelope: OuterEnvelope) -> ResponseEnvelope:
        """Handle list request."""
        # Would extract user_id from decrypted payload
        # For now, just ack
        return ResponseEnvelope(
            envelope_id=envelope.envelope_id,
            status="ok",
        )

    def _handle_query(self, envelope: OuterEnvelope) -> ResponseEnvelope:
        """Handle query request."""
        return ResponseEnvelope(
            envelope_id=envelope.envelope_id,
            status="ok",
        )
