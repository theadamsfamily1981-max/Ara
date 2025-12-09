"""
Episodic Memory Store
======================

SQLite-backed episodic memory for Ara.
Stores conversation turns, events, and experiences.
"""

from __future__ import annotations

import sqlite3
import json
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    domain TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    visibility TEXT DEFAULT 'private',
    importance REAL DEFAULT 0.5,
    meta TEXT
);

CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(domain);
CREATE INDEX IF NOT EXISTS idx_episodes_visibility ON episodes(visibility);
"""


@dataclass
class Episode:
    """A single episodic memory."""
    ts: float
    domain: str
    role: str
    text: str
    visibility: str = "private"  # private, curated_public, deep_cut, vault
    importance: float = 0.5  # 0.0 - 1.0
    meta: Dict[str, Any] = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple) -> Episode:
        """Create Episode from database row."""
        ts, domain, role, text, visibility, importance, meta_json = row
        return cls(
            ts=ts,
            domain=domain,
            role=role,
            text=text,
            visibility=visibility or "private",
            importance=importance or 0.5,
            meta=json.loads(meta_json) if meta_json else {},
        )


class EpisodeStore:
    """
    SQLite-backed episodic memory store.

    Supports:
    - Adding episodes with visibility and importance
    - Retrieving recent episodes by domain
    - Filtering by visibility level
    - Compaction (future)
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        logger.info(f"EpisodeStore initialized at {db_path}")

    def add(self, ep: Episode) -> int:
        """Add an episode and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO episodes (ts, domain, role, text, visibility, importance, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ep.ts,
                ep.domain,
                ep.role,
                ep.text,
                ep.visibility,
                ep.importance,
                json.dumps(ep.meta),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_event(
        self,
        domain: str,
        role: str,
        text: str,
        visibility: str = "private",
        importance: float = 0.5,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Convenience method to add an episode."""
        ep = Episode(
            ts=time.time(),
            domain=domain,
            role=role,
            text=text,
            visibility=visibility,
            importance=importance,
            meta=meta or {},
        )
        return self.add(ep)

    def recent(
        self,
        domain: Optional[str] = None,
        limit: int = 20,
        min_visibility: str = "private",
    ) -> List[Episode]:
        """
        Get recent episodes.

        Args:
            domain: Filter by domain (None = all)
            limit: Max episodes to return
            min_visibility: Minimum visibility level
        """
        visibility_order = ["private", "curated_public", "deep_cut", "vault"]
        min_level = visibility_order.index(min_visibility) if min_visibility in visibility_order else 0

        # Filter to allowed visibilities
        allowed = visibility_order[:min_level + 1]
        placeholders = ",".join("?" * len(allowed))

        if domain:
            query = f"""
                SELECT ts, domain, role, text, visibility, importance, meta
                FROM episodes
                WHERE domain = ? AND visibility IN ({placeholders})
                ORDER BY ts DESC
                LIMIT ?
            """
            params = [domain] + allowed + [limit]
        else:
            query = f"""
                SELECT ts, domain, role, text, visibility, importance, meta
                FROM episodes
                WHERE visibility IN ({placeholders})
                ORDER BY ts DESC
                LIMIT ?
            """
            params = allowed + [limit]

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()
        return [Episode.from_row(row) for row in rows]

    def search_text(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """Simple text search in episodes."""
        if domain:
            cursor = self.conn.execute(
                """
                SELECT ts, domain, role, text, visibility, importance, meta
                FROM episodes
                WHERE domain = ? AND text LIKE ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (domain, f"%{query}%", limit),
            )
        else:
            cursor = self.conn.execute(
                """
                SELECT ts, domain, role, text, visibility, importance, meta
                FROM episodes
                WHERE text LIKE ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            )

        rows = cursor.fetchall()
        return [Episode.from_row(row) for row in rows]

    def count(self, domain: Optional[str] = None) -> int:
        """Count episodes."""
        if domain:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE domain = ?",
                (domain,),
            )
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM episodes")
        return cursor.fetchone()[0]

    def domains(self) -> List[str]:
        """List all domains with episodes."""
        cursor = self.conn.execute(
            "SELECT DISTINCT domain FROM episodes ORDER BY domain"
        )
        return [row[0] for row in cursor.fetchall()]

    def compact(self, max_age_days: int = 90, keep_important: bool = True) -> int:
        """
        Compact old episodes.

        Args:
            max_age_days: Delete episodes older than this
            keep_important: Keep episodes with importance > 0.8

        Returns:
            Number of episodes deleted
        """
        cutoff = time.time() - (max_age_days * 86400)

        if keep_important:
            cursor = self.conn.execute(
                """
                DELETE FROM episodes
                WHERE ts < ? AND importance < 0.8
                """,
                (cutoff,),
            )
        else:
            cursor = self.conn.execute(
                "DELETE FROM episodes WHERE ts < ?",
                (cutoff,),
            )

        deleted = cursor.rowcount
        self.conn.commit()
        logger.info(f"Compacted {deleted} episodes older than {max_age_days} days")
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
