"""
Waggle Board - Distributed Job Coordination
============================================

Database models for the bee colony job scheduler.

Tables:
- nodes: Physical/virtual machines in the hive
- sites: (task_type, node) combinations with performance metrics
- tasks: Jobs to be executed

The "waggle board" is the shared state that bees use to:
1. Advertise good sites (high Q̂ = high intensity)
2. Find work (pick sites weighted by intensity)
3. Coordinate load (congestion tracking)
"""

from __future__ import annotations

import sqlite3
import json
import time
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator
from enum import Enum

logger = logging.getLogger(__name__)

# SQLite busy timeout (ms) - wait before raising SQLITE_BUSY
SQLITE_BUSY_TIMEOUT_MS = 5000

# Max retries for database operations under contention
MAX_DB_RETRIES = 3
DB_RETRY_DELAY_SEC = 0.1


# =============================================================================
# Schema
# =============================================================================

SCHEMA = """
-- Nodes in the hive (physical/virtual machines)
CREATE TABLE IF NOT EXISTS nodes (
    id              TEXT PRIMARY KEY,
    role            TEXT NOT NULL DEFAULT 'worker',
    hostname        TEXT,
    ip_address      TEXT,
    capabilities    TEXT,  -- JSON list of task_types this node can handle
    last_heartbeat  REAL,
    cpu_load        REAL DEFAULT 0.0,
    mem_used_pct    REAL DEFAULT 0.0,
    gpu_load        REAL DEFAULT 0.0,
    status          TEXT DEFAULT 'online',
    meta            TEXT
);

-- Sites = (task_type, node) combos with performance metrics
CREATE TABLE IF NOT EXISTS sites (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type       TEXT NOT NULL,
    node_id         TEXT NOT NULL REFERENCES nodes(id),
    q_hat           REAL NOT NULL DEFAULT 0.0,
    intensity       REAL NOT NULL DEFAULT 0.1,
    congestion      INTEGER NOT NULL DEFAULT 0,
    visit_count     INTEGER NOT NULL DEFAULT 0,
    success_count   INTEGER NOT NULL DEFAULT 0,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    avg_duration_ms REAL DEFAULT 0.0,
    last_update     REAL NOT NULL,
    meta            TEXT,
    UNIQUE (task_type, node_id)
);

-- Tasks = jobs waiting / in-progress / done
CREATE TABLE IF NOT EXISTS tasks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    priority        REAL DEFAULT 0.5,
    payload         TEXT NOT NULL,
    result          TEXT,
    reward          REAL,
    assigned_node   TEXT REFERENCES nodes(id),
    assigned_site   INTEGER REFERENCES sites(id),
    created_at      REAL NOT NULL,
    started_at      REAL,
    completed_at    REAL,
    error           TEXT,
    meta            TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_type_status ON tasks(task_type, status);
CREATE INDEX IF NOT EXISTS idx_sites_type ON sites(task_type);
CREATE INDEX IF NOT EXISTS idx_sites_intensity ON sites(intensity DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
"""


# =============================================================================
# Data Classes
# =============================================================================

class NodeRole(Enum):
    WORKER = "worker"
    GPU = "gpu"
    PI = "pi"
    MONITOR = "monitor"
    QUEEN = "queen"


class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    DRAINING = "draining"


class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Node:
    """A machine in the hive."""
    id: str
    role: str = "worker"
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    last_heartbeat: float = 0.0
    cpu_load: float = 0.0
    mem_used_pct: float = 0.0
    gpu_load: float = 0.0
    status: str = "online"
    meta: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if node is available for work."""
        return self.status == "online" and self.cpu_load < 0.9

    def is_stale(self, timeout_sec: float = 60.0) -> bool:
        """Check if heartbeat is stale."""
        return time.time() - self.last_heartbeat > timeout_sec


@dataclass
class Site:
    """A (task_type, node) combination with performance metrics."""
    id: int
    task_type: str
    node_id: str
    q_hat: float = 0.0  # Performance estimate
    intensity: float = 0.1  # Waggle dance strength
    congestion: int = 0  # Active agents on this site
    visit_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_duration_ms: float = 0.0
    last_update: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Neutral for new sites
        return self.success_count / total


@dataclass
class Task:
    """A job to be executed."""
    id: int
    task_type: str
    status: str = "queued"
    priority: float = 0.5
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None
    assigned_node: Optional[str] = None
    assigned_site: Optional[int] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate task duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None


# =============================================================================
# Waggle Board (Database Interface)
# =============================================================================

class WaggleBoard:
    """
    The waggle board - shared state for bee coordination.

    This is where bees:
    1. Register themselves (nodes)
    2. Report site quality (sites)
    3. Pick up and complete work (tasks)

    Thread Safety:
    All database operations are protected by an RLock. The lock is
    automatically acquired via the _db_op() context manager which also
    handles SQLITE_BUSY retries.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety: RLock allows recursive acquisition from same thread
        self._lock = threading.RLock()

        # Create connection with thread safety disabled (we manage it ourselves)
        # and set busy timeout to handle concurrent access
        self.conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0,
        )
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrent read/write performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout={}".format(SQLITE_BUSY_TIMEOUT_MS))

        self.conn.executescript(SCHEMA)
        self.conn.commit()
        logger.info(f"WaggleBoard initialized at {db_path} (thread-safe)")

    @contextmanager
    def _db_op(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for thread-safe database operations.

        Acquires lock, handles retries on SQLITE_BUSY, and ensures
        proper cleanup on error.
        """
        with self._lock:
            retries = 0
            while True:
                try:
                    yield self.conn
                    return
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and retries < MAX_DB_RETRIES:
                        retries += 1
                        logger.warning(
                            f"Database locked, retry {retries}/{MAX_DB_RETRIES}"
                        )
                        time.sleep(DB_RETRY_DELAY_SEC * retries)
                    else:
                        raise

    # =========================================================================
    # Nodes
    # =========================================================================

    def register_node(self, node: Node) -> None:
        """Register or update a node."""
        with self._db_op() as conn:
            conn.execute(
                """
                INSERT INTO nodes (id, role, hostname, ip_address, capabilities,
                                   last_heartbeat, cpu_load, mem_used_pct, gpu_load, status, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    role = excluded.role,
                    hostname = excluded.hostname,
                    ip_address = excluded.ip_address,
                    capabilities = excluded.capabilities,
                    last_heartbeat = excluded.last_heartbeat,
                    cpu_load = excluded.cpu_load,
                    mem_used_pct = excluded.mem_used_pct,
                    gpu_load = excluded.gpu_load,
                    status = excluded.status,
                    meta = excluded.meta
                """,
                (
                    node.id,
                    node.role,
                    node.hostname,
                    node.ip_address,
                    json.dumps(node.capabilities),
                    node.last_heartbeat or time.time(),
                    node.cpu_load,
                    node.mem_used_pct,
                    node.gpu_load,
                    node.status,
                    json.dumps(node.meta),
                ),
            )
            conn.commit()

    def heartbeat(
        self,
        node_id: str,
        cpu_load: float = 0.0,
        mem_used_pct: float = 0.0,
        gpu_load: float = 0.0,
    ) -> None:
        """Update node heartbeat and metrics."""
        with self._db_op() as conn:
            conn.execute(
                """
                UPDATE nodes
                SET last_heartbeat = ?, cpu_load = ?, mem_used_pct = ?, gpu_load = ?
                WHERE id = ?
                """,
                (time.time(), cpu_load, mem_used_pct, gpu_load, node_id),
            )
            conn.commit()

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        with self._db_op() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            return self._row_to_node(row) if row else None

    def list_nodes(self, status: Optional[str] = None) -> List[Node]:
        """List all nodes, optionally filtered by status."""
        with self._db_op() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM nodes WHERE status = ?", (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM nodes").fetchall()
            return [self._row_to_node(r) for r in rows]

    def mark_stale_nodes_offline(self, timeout_sec: float = 60.0) -> int:
        """Mark nodes with stale heartbeats as offline."""
        with self._db_op() as conn:
            cutoff = time.time() - timeout_sec
            cursor = conn.execute(
                """
                UPDATE nodes SET status = 'offline'
                WHERE status = 'online' AND last_heartbeat < ?
                """,
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        return Node(
            id=row["id"],
            role=row["role"],
            hostname=row["hostname"],
            ip_address=row["ip_address"],
            capabilities=json.loads(row["capabilities"]) if row["capabilities"] else [],
            last_heartbeat=row["last_heartbeat"] or 0.0,
            cpu_load=row["cpu_load"] or 0.0,
            mem_used_pct=row["mem_used_pct"] or 0.0,
            gpu_load=row["gpu_load"] or 0.0,
            status=row["status"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )

    # =========================================================================
    # Sites
    # =========================================================================

    def create_site(self, task_type: str, node_id: str) -> Site:
        """Create a new site for (task_type, node)."""
        with self._db_op() as conn:
            now = time.time()
            conn.execute(
                """
                INSERT INTO sites (task_type, node_id, q_hat, intensity, last_update)
                VALUES (?, ?, 0.0, 0.1, ?)
                ON CONFLICT(task_type, node_id) DO UPDATE SET last_update = excluded.last_update
                """,
                (task_type, node_id, now),
            )
            conn.commit()

            # Fetch the created/updated site
            row = conn.execute(
                "SELECT * FROM sites WHERE task_type = ? AND node_id = ?",
                (task_type, node_id),
            ).fetchone()
            return self._row_to_site(row)

    def get_sites_for_task_type(self, task_type: str) -> List[Site]:
        """Get all sites for a task type, ordered by intensity."""
        with self._db_op() as conn:
            rows = conn.execute(
                """
                SELECT s.* FROM sites s
                JOIN nodes n ON s.node_id = n.id
                WHERE s.task_type = ? AND n.status = 'online'
                ORDER BY s.intensity DESC
                """,
                (task_type,),
            ).fetchall()
            return [self._row_to_site(r) for r in rows]

    def get_sites_for_node(self, node_id: str) -> List[Site]:
        """Get all sites for a node."""
        with self._db_op() as conn:
            rows = conn.execute(
                "SELECT * FROM sites WHERE node_id = ? ORDER BY intensity DESC",
                (node_id,),
            ).fetchall()
            return [self._row_to_site(r) for r in rows]

    def update_site(
        self,
        site_id: int,
        q_hat: Optional[float] = None,
        intensity: Optional[float] = None,
        congestion_delta: int = 0,
        success: Optional[bool] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Update site metrics."""
        with self._db_op() as conn:
            updates = ["last_update = ?"]
            params: List[Any] = [time.time()]

            if q_hat is not None:
                updates.append("q_hat = ?")
                params.append(q_hat)

            if intensity is not None:
                updates.append("intensity = ?")
                params.append(max(0.01, intensity))  # Floor at 0.01

            if congestion_delta != 0:
                updates.append("congestion = MAX(0, congestion + ?)")
                params.append(congestion_delta)

            if success is not None:
                updates.append("visit_count = visit_count + 1")
                if success:
                    updates.append("success_count = success_count + 1")
                else:
                    updates.append("failure_count = failure_count + 1")

            if duration_ms is not None:
                # Running average
                updates.append(
                    "avg_duration_ms = (avg_duration_ms * visit_count + ?) / (visit_count + 1)"
                )
                params.append(duration_ms)

            params.append(site_id)
            conn.execute(
                f"UPDATE sites SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

    def evaporate_intensity(self, decay_factor: float = 0.95, min_intensity: float = 0.01) -> int:
        """Apply evaporation to all site intensities."""
        with self._db_op() as conn:
            cursor = conn.execute(
                """
                UPDATE sites
                SET intensity = MAX(?, intensity * ?)
                """,
                (min_intensity, decay_factor),
            )
            conn.commit()
            return cursor.rowcount

    def cool_node_sites(self, node_id: str, factor: float = 0.5) -> int:
        """Reduce intensity for all sites on a node (cooling pheromone)."""
        with self._db_op() as conn:
            cursor = conn.execute(
                """
                UPDATE sites
                SET intensity = intensity * ?
                WHERE node_id = ?
                """,
                (factor, node_id),
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_site(self, row: sqlite3.Row) -> Site:
        return Site(
            id=row["id"],
            task_type=row["task_type"],
            node_id=row["node_id"],
            q_hat=row["q_hat"],
            intensity=row["intensity"],
            congestion=row["congestion"],
            visit_count=row["visit_count"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            avg_duration_ms=row["avg_duration_ms"] or 0.0,
            last_update=row["last_update"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )

    # =========================================================================
    # Tasks
    # =========================================================================

    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: float = 0.5,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Submit a new task to the queue."""
        with self._db_op() as conn:
            now = time.time()
            cursor = conn.execute(
                """
                INSERT INTO tasks (task_type, status, priority, payload, created_at, meta)
                VALUES (?, 'queued', ?, ?, ?, ?)
                """,
                (task_type, priority, json.dumps(payload), now, json.dumps(meta or {})),
            )
            task_id = cursor.lastrowid
            conn.commit()

            # Fetch the created task
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return self._row_to_task(row)

    def claim_task(
        self,
        task_type: str,
        node_id: str,
        site_id: Optional[int] = None,
    ) -> Optional[Task]:
        """
        Claim a queued task for execution.

        Uses SELECT FOR UPDATE semantics via a transaction.
        Thread-safe: entire claim operation is atomic under lock.
        """
        with self._db_op() as conn:
            # Find oldest queued task of this type
            row = conn.execute(
                """
                SELECT * FROM tasks
                WHERE task_type = ? AND status = 'queued'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (task_type,),
            ).fetchone()

            if not row:
                return None

            task_id = row["id"]
            now = time.time()

            # Claim it
            conn.execute(
                """
                UPDATE tasks
                SET status = 'running', assigned_node = ?, assigned_site = ?, started_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (node_id, site_id, now, task_id),
            )
            conn.commit()

            # Re-fetch to confirm
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()

            if row and row["status"] == "running" and row["assigned_node"] == node_id:
                return self._row_to_task(row)

            return None

    def complete_task(
        self,
        task_id: int,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a task as complete."""
        with self._db_op() as conn:
            now = time.time()
            status = "done" if success else "failed"
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, result = ?, reward = ?, completed_at = ?, error = ?
                WHERE id = ?
                """,
                (status, json.dumps(result) if result else None, reward, now, error, task_id),
            )
            conn.commit()

    def get_task(self, task_id: int) -> Optional[Task]:
        """Get a task by ID."""
        with self._db_op() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return self._row_to_task(row) if row else None

    def count_queued(self, task_type: Optional[str] = None) -> int:
        """Count queued tasks."""
        with self._db_op() as conn:
            if task_type:
                row = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status = 'queued' AND task_type = ?",
                    (task_type,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status = 'queued'"
                ).fetchone()
            return row[0]

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            task_type=row["task_type"],
            status=row["status"],
            priority=row["priority"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            result=json.loads(row["result"]) if row["result"] else None,
            reward=row["reward"],
            assigned_node=row["assigned_node"],
            assigned_site=row["assigned_site"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error=row["error"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get overall hive statistics."""
        with self._db_op() as conn:
            nodes_online = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE status = 'online'"
            ).fetchone()[0]

            tasks_queued = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'queued'"
            ).fetchone()[0]

            tasks_running = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]

            tasks_completed = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'done'"
            ).fetchone()[0]

            avg_intensity = conn.execute(
                "SELECT AVG(intensity) FROM sites"
            ).fetchone()[0] or 0.0

            return {
                "nodes_online": nodes_online,
                "tasks_queued": tasks_queued,
                "tasks_running": tasks_running,
                "tasks_completed": tasks_completed,
                "avg_site_intensity": avg_intensity,
            }

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            self.conn.close()
