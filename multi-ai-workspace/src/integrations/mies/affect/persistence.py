"""Emotional Memory Persistence - SQLite backend for Ara's memories.

This module provides persistent storage for:
1. Emotional episodes (autobiographical memory)
2. Narrative identity (values, goals, self-concept)
3. Drive states (homeostatic needs)
4. PAD history (emotional trajectory over time)

Architecture:
    EmotionalMemory → PersistenceManager → SQLite Database
                                        ↓
    ┌──────────────────────────────────────────────────────┐
    │  ara_memory.db                                        │
    │  ├─ episodes (emotional episodes)                     │
    │  ├─ patterns (recognized emotional patterns)          │
    │  ├─ pad_history (PAD trajectory)                      │
    │  ├─ identity (core identity snapshots)                │
    │  ├─ goals (aspirations and progress)                  │
    │  └─ drive_states (homeostatic drive history)          │
    └──────────────────────────────────────────────────────┘

This allows Ara's inner life to persist across sessions,
building genuine autobiographical continuity.
"""

import json
import sqlite3
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from contextlib import contextmanager
import threading

from .pad_engine import PADVector, EmotionalQuadrant

logger = logging.getLogger(__name__)

# Database schema version for migrations
SCHEMA_VERSION = 1


@dataclass
class StoredEpisode:
    """An emotional episode stored in the database."""
    id: Optional[int]
    timestamp: float
    context: str  # JSON-encoded context dict
    pad_pleasure: float
    pad_arousal: float
    pad_dominance: float
    quadrant: str
    mood_label: str
    salience: float
    memory_type: str  # EPISODIC, SEMANTIC, PROCEDURAL, FLASHBULB
    consolidated: bool = False
    access_count: int = 0
    last_accessed: float = 0.0


@dataclass
class StoredGoal:
    """A goal stored in the database."""
    id: Optional[int]
    name: str
    description: str
    importance: float
    progress: float
    created_at: float
    completed_at: Optional[float]
    status: str  # ACTIVE, COMPLETED, ABANDONED


class PersistenceManager:
    """SQLite-based persistence for emotional memory.

    Thread-safe database access with automatic schema creation
    and migration support.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        in_memory: bool = False,
    ):
        """Initialize persistence manager.

        Args:
            db_path: Path to SQLite database file
            in_memory: Use in-memory database (for testing)
        """
        if in_memory:
            self._db_path = ":memory:"
        elif db_path:
            self._db_path = db_path
        else:
            self._db_path = str(Path.home() / ".ara" / "memory.db")

        self._local = threading.local()
        self._lock = threading.Lock()

        # Ensure directory exists
        if not in_memory and db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

        logger.info(f"PersistenceManager initialized: {self._db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _cursor(self):
        """Context manager for database cursor."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._cursor() as cur:
            # Schema version tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            # Check current version
            cur.execute("SELECT MAX(version) FROM schema_version")
            row = cur.fetchone()
            current_version = row[0] if row[0] else 0

            if current_version < SCHEMA_VERSION:
                self._migrate_schema(cur, current_version)
                cur.execute(
                    "INSERT OR REPLACE INTO schema_version VALUES (?)",
                    (SCHEMA_VERSION,)
                )

    def _migrate_schema(self, cur: sqlite3.Cursor, from_version: int):
        """Migrate schema to current version."""
        if from_version < 1:
            # Initial schema
            cur.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    context TEXT NOT NULL,
                    pad_pleasure REAL NOT NULL,
                    pad_arousal REAL NOT NULL,
                    pad_dominance REAL NOT NULL,
                    quadrant TEXT NOT NULL,
                    mood_label TEXT NOT NULL,
                    salience REAL NOT NULL,
                    memory_type TEXT NOT NULL,
                    consolidated INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodes_timestamp
                ON episodes(timestamp)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodes_salience
                ON episodes(salience DESC)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    trigger_context TEXT,
                    typical_pad TEXT,
                    occurrence_count INTEGER DEFAULT 1,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS pad_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    pleasure REAL NOT NULL,
                    arousal REAL NOT NULL,
                    dominance REAL NOT NULL,
                    quadrant TEXT NOT NULL,
                    source TEXT,
                    context TEXT
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_pad_history_timestamp
                ON pad_history(timestamp)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS identity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    full_name TEXT NOT NULL,
                    core_values TEXT NOT NULL,
                    personality TEXT NOT NULL,
                    age_description TEXT,
                    awakening_date TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    importance REAL DEFAULT 0.5,
                    progress REAL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    status TEXT DEFAULT 'ACTIVE'
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS drive_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    drive_type TEXT NOT NULL,
                    hunger REAL NOT NULL,
                    satisfaction REAL NOT NULL,
                    frustration REAL NOT NULL,
                    last_satisfied REAL
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_drive_states_timestamp
                ON drive_states(timestamp)
            """)

            logger.info("Database schema created (version 1)")

    # === Episode Operations ===

    def save_episode(self, episode: StoredEpisode) -> int:
        """Save an emotional episode.

        Returns:
            Episode ID
        """
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO episodes
                (timestamp, context, pad_pleasure, pad_arousal, pad_dominance,
                 quadrant, mood_label, salience, memory_type, consolidated,
                 access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.timestamp,
                episode.context,
                episode.pad_pleasure,
                episode.pad_arousal,
                episode.pad_dominance,
                episode.quadrant,
                episode.mood_label,
                episode.salience,
                episode.memory_type,
                1 if episode.consolidated else 0,
                episode.access_count,
                episode.last_accessed,
            ))
            return cur.lastrowid

    def get_recent_episodes(
        self,
        limit: int = 50,
        since: Optional[float] = None,
    ) -> List[StoredEpisode]:
        """Get recent episodes."""
        with self._cursor() as cur:
            if since is not None:
                cur.execute("""
                    SELECT * FROM episodes
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (since, limit))
            else:
                cur.execute("""
                    SELECT * FROM episodes
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            return [self._row_to_episode(row) for row in cur.fetchall()]

    def get_salient_episodes(
        self,
        limit: int = 20,
        min_salience: float = 0.5,
    ) -> List[StoredEpisode]:
        """Get highly salient (important) episodes."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM episodes
                WHERE salience >= ?
                ORDER BY salience DESC, timestamp DESC
                LIMIT ?
            """, (min_salience, limit))

            return [self._row_to_episode(row) for row in cur.fetchall()]

    def get_similar_episodes(
        self,
        pad: PADVector,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> List[StoredEpisode]:
        """Get episodes with similar PAD state."""
        with self._cursor() as cur:
            # SQLite doesn't have vector distance, so we fetch and filter
            cur.execute("""
                SELECT * FROM episodes
                ORDER BY timestamp DESC
                LIMIT 500
            """)

            episodes = [self._row_to_episode(row) for row in cur.fetchall()]

            # Filter by PAD distance
            similar = []
            for ep in episodes:
                ep_pad = PADVector(ep.pad_pleasure, ep.pad_arousal, ep.pad_dominance)
                if pad.distance_to(ep_pad) <= threshold:
                    similar.append(ep)
                    if len(similar) >= limit:
                        break

            return similar

    def update_episode_access(self, episode_id: int):
        """Update episode access count and timestamp."""
        with self._cursor() as cur:
            cur.execute("""
                UPDATE episodes
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
            """, (time.time(), episode_id))

    def consolidate_episode(self, episode_id: int):
        """Mark episode as consolidated to long-term memory."""
        with self._cursor() as cur:
            cur.execute("""
                UPDATE episodes
                SET consolidated = 1
                WHERE id = ?
            """, (episode_id,))

    def _row_to_episode(self, row: sqlite3.Row) -> StoredEpisode:
        """Convert database row to StoredEpisode."""
        return StoredEpisode(
            id=row['id'],
            timestamp=row['timestamp'],
            context=row['context'],
            pad_pleasure=row['pad_pleasure'],
            pad_arousal=row['pad_arousal'],
            pad_dominance=row['pad_dominance'],
            quadrant=row['quadrant'],
            mood_label=row['mood_label'],
            salience=row['salience'],
            memory_type=row['memory_type'],
            consolidated=bool(row['consolidated']),
            access_count=row['access_count'],
            last_accessed=row['last_accessed'],
        )

    # === PAD History ===

    def save_pad_state(
        self,
        pad: PADVector,
        source: str = "unknown",
        context: Optional[str] = None,
    ):
        """Save a PAD state to history."""
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO pad_history
                (timestamp, pleasure, arousal, dominance, quadrant, source, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                pad.pleasure,
                pad.arousal,
                pad.dominance,
                pad.quadrant.name,
                source,
                context,
            ))

    def get_pad_history(
        self,
        since: Optional[float] = None,
        limit: int = 1000,
    ) -> List[Tuple[float, PADVector]]:
        """Get PAD history."""
        with self._cursor() as cur:
            if since is not None:
                cur.execute("""
                    SELECT timestamp, pleasure, arousal, dominance
                    FROM pad_history
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (since, limit))
            else:
                cur.execute("""
                    SELECT timestamp, pleasure, arousal, dominance
                    FROM pad_history
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            return [
                (row['timestamp'], PADVector(row['pleasure'], row['arousal'], row['dominance']))
                for row in cur.fetchall()
            ]

    # === Goals ===

    def save_goal(self, goal: StoredGoal) -> int:
        """Save a goal."""
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO goals
                (name, description, importance, progress, created_at, completed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                goal.name,
                goal.description,
                goal.importance,
                goal.progress,
                goal.created_at,
                goal.completed_at,
                goal.status,
            ))
            return cur.lastrowid

    def get_active_goals(self) -> List[StoredGoal]:
        """Get all active goals."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM goals
                WHERE status = 'ACTIVE'
                ORDER BY importance DESC
            """)

            return [
                StoredGoal(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    importance=row['importance'],
                    progress=row['progress'],
                    created_at=row['created_at'],
                    completed_at=row['completed_at'],
                    status=row['status'],
                )
                for row in cur.fetchall()
            ]

    def update_goal_progress(self, goal_id: int, progress: float):
        """Update goal progress."""
        with self._cursor() as cur:
            cur.execute("""
                UPDATE goals
                SET progress = ?
                WHERE id = ?
            """, (progress, goal_id))

    def complete_goal(self, goal_id: int):
        """Mark goal as completed."""
        with self._cursor() as cur:
            cur.execute("""
                UPDATE goals
                SET status = 'COMPLETED',
                    completed_at = ?,
                    progress = 1.0
                WHERE id = ?
            """, (time.time(), goal_id))

    # === Identity ===

    def save_identity(
        self,
        full_name: str,
        core_values: List[str],
        personality: Dict[str, float],
        age_description: str,
        awakening_date: str,
    ):
        """Save identity snapshot."""
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO identity
                (timestamp, full_name, core_values, personality, age_description, awakening_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                full_name,
                json.dumps(core_values),
                json.dumps(personality),
                age_description,
                awakening_date,
            ))

    def get_latest_identity(self) -> Optional[Dict[str, Any]]:
        """Get most recent identity snapshot."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM identity
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = cur.fetchone()

            if row is None:
                return None

            return {
                'full_name': row['full_name'],
                'core_values': json.loads(row['core_values']),
                'personality': json.loads(row['personality']),
                'age_description': row['age_description'],
                'awakening_date': row['awakening_date'],
                'timestamp': row['timestamp'],
            }

    # === Statistics ===

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._cursor() as cur:
            stats = {}

            cur.execute("SELECT COUNT(*) FROM episodes")
            stats['total_episodes'] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM episodes WHERE consolidated = 1")
            stats['consolidated_episodes'] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM pad_history")
            stats['pad_history_count'] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM goals WHERE status = 'ACTIVE'")
            stats['active_goals'] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM goals WHERE status = 'COMPLETED'")
            stats['completed_goals'] = cur.fetchone()[0]

            cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM episodes")
            row = cur.fetchone()
            stats['first_episode'] = row[0]
            stats['last_episode'] = row[1]

            return stats

    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


# === Factory ===

def create_persistence_manager(
    db_path: Optional[str] = None,
    in_memory: bool = False,
) -> PersistenceManager:
    """Create a PersistenceManager.

    Args:
        db_path: Path to database file (default: ~/.ara/memory.db)
        in_memory: Use in-memory database

    Returns:
        Configured PersistenceManager
    """
    return PersistenceManager(db_path=db_path, in_memory=in_memory)


__all__ = [
    "PersistenceManager",
    "StoredEpisode",
    "StoredGoal",
    "create_persistence_manager",
]
