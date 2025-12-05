"""Idea Board - The governance layer for Ara's proposals.

This module provides the IdeaBoard service that manages the lifecycle
of ideas from creation through execution and completion.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

from .models import (
    Idea,
    IdeaCategory,
    IdeaRisk,
    IdeaStatus,
    IdeaOutcome,
    Signal,
)

logger = logging.getLogger(__name__)


@dataclass
class IdeaBoardConfig:
    """Configuration for the Idea Board."""

    # Storage
    db_path: Optional[Path] = None  # None = in-memory

    # Governance
    auto_approve_none_risk: bool = False  # Auto-approve NONE risk ideas
    max_running_ideas: int = 3
    max_inbox_age_hours: float = 72.0  # Auto-park old inbox items

    # Rate limits
    max_ideas_per_hour: int = 10
    max_ideas_per_day: int = 50

    # Notifications
    notify_on_new: bool = True
    notify_on_complete: bool = True


class IdeaBoard:
    """The Idea Board - Ara's lab notebook and governance system.

    This class manages:
    - Storage of ideas in SQLite
    - Lifecycle transitions (draft → inbox → approved → running → done)
    - Queries and filtering
    - Rate limiting
    - Notifications via callbacks
    """

    def __init__(self, config: Optional[IdeaBoardConfig] = None):
        """Initialize the Idea Board.

        Args:
            config: Board configuration. If None, uses defaults with
                   in-memory database.
        """
        self.config = config or IdeaBoardConfig()

        # Database
        db_path = self.config.db_path
        if db_path:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(db_path)
        else:
            self._db_path = ":memory:"

        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

        # Callbacks
        self._on_new_idea: Optional[Callable[[Idea], None]] = None
        self._on_status_change: Optional[Callable[[Idea, IdeaStatus, IdeaStatus], None]] = None
        self._on_complete: Optional[Callable[[Idea], None]] = None

        # Rate limiting
        self._ideas_this_hour = 0
        self._ideas_today = 0
        self._hour_start = time.time()
        self._day_start = time.time()

        # Initialize database
        self._init_db()

        logger.info(f"IdeaBoard initialized (db={self._db_path})")

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection (creates if needed)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ideas (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    created_by TEXT NOT NULL,
                    category TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tags TEXT,
                    hypothesis TEXT,
                    plan TEXT,
                    rollback_plan TEXT,
                    signals TEXT,
                    related_objects TEXT,
                    sandbox_status TEXT,
                    sandbox_results TEXT,
                    sandbox_logs TEXT,
                    started_at REAL,
                    completed_at REAL,
                    outcome TEXT,
                    outcome_notes TEXT,
                    outcome_signals TEXT,
                    human_decision TEXT,
                    human_notes TEXT,
                    reviewed_by TEXT,
                    reviewed_at REAL,
                    thread TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status);
                CREATE INDEX IF NOT EXISTS idx_ideas_category ON ideas(category);
                CREATE INDEX IF NOT EXISTS idx_ideas_created_at ON ideas(created_at);
            """)
            conn.commit()

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()

        # Reset hourly counter
        if now - self._hour_start > 3600:
            self._hour_start = now
            self._ideas_this_hour = 0

        # Reset daily counter
        if now - self._day_start > 86400:
            self._day_start = now
            self._ideas_today = 0

        return (
            self._ideas_this_hour < self.config.max_ideas_per_hour and
            self._ideas_today < self.config.max_ideas_per_day
        )

    def _idea_to_row(self, idea: Idea) -> Dict[str, Any]:
        """Convert idea to database row."""
        return {
            "id": idea.id,
            "title": idea.title,
            "created_at": idea.created_at,
            "updated_at": idea.updated_at,
            "created_by": idea.created_by,
            "category": idea.category.value,
            "risk": idea.risk.value,
            "status": idea.status.value,
            "tags": json.dumps(idea.tags),
            "hypothesis": idea.hypothesis,
            "plan": json.dumps(idea.plan),
            "rollback_plan": json.dumps(idea.rollback_plan),
            "signals": json.dumps([s.to_dict() for s in idea.signals]),
            "related_objects": json.dumps(idea.related_objects),
            "sandbox_status": idea.sandbox_status.value,
            "sandbox_results": json.dumps(idea.sandbox_results) if idea.sandbox_results else None,
            "sandbox_logs": json.dumps(idea.sandbox_logs),
            "started_at": idea.started_at,
            "completed_at": idea.completed_at,
            "outcome": idea.outcome.value if idea.outcome else None,
            "outcome_notes": idea.outcome_notes,
            "outcome_signals": json.dumps([s.to_dict() for s in idea.outcome_signals]),
            "human_decision": idea.human_decision,
            "human_notes": idea.human_notes,
            "reviewed_by": idea.reviewed_by,
            "reviewed_at": idea.reviewed_at,
            "thread": json.dumps(idea.thread),
        }

    def _row_to_idea(self, row: sqlite3.Row) -> Idea:
        """Convert database row to idea."""
        return Idea(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"],
            category=IdeaCategory(row["category"]),
            risk=IdeaRisk(row["risk"]),
            status=IdeaStatus(row["status"]),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            hypothesis=row["hypothesis"] or "",
            plan=json.loads(row["plan"]) if row["plan"] else [],
            rollback_plan=json.loads(row["rollback_plan"]) if row["rollback_plan"] else [],
            signals=[Signal.from_dict(s) for s in json.loads(row["signals"])] if row["signals"] else [],
            related_objects=json.loads(row["related_objects"]) if row["related_objects"] else [],
            sandbox_status=row["sandbox_status"],
            sandbox_results=json.loads(row["sandbox_results"]) if row["sandbox_results"] else None,
            sandbox_logs=json.loads(row["sandbox_logs"]) if row["sandbox_logs"] else [],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            outcome=IdeaOutcome(row["outcome"]) if row["outcome"] else None,
            outcome_notes=row["outcome_notes"] or "",
            outcome_signals=[Signal.from_dict(s) for s in json.loads(row["outcome_signals"])] if row["outcome_signals"] else [],
            human_decision=row["human_decision"],
            human_notes=row["human_notes"] or "",
            reviewed_by=row["reviewed_by"],
            reviewed_at=row["reviewed_at"],
            thread=json.loads(row["thread"]) if row["thread"] else [],
        )

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create(self, idea: Idea) -> bool:
        """Create a new idea.

        Args:
            idea: The idea to create

        Returns:
            True if created, False if rate limited or duplicate
        """
        if not self._check_rate_limit():
            logger.warning("Idea creation rate limited")
            return False

        with self._lock:
            conn = self._get_conn()
            try:
                row = self._idea_to_row(idea)
                cols = ", ".join(row.keys())
                placeholders = ", ".join("?" * len(row))
                conn.execute(
                    f"INSERT INTO ideas ({cols}) VALUES ({placeholders})",
                    list(row.values())
                )
                conn.commit()

                self._ideas_this_hour += 1
                self._ideas_today += 1

                logger.info(f"Created idea: {idea.id} - {idea.title}")

                if self._on_new_idea:
                    self._on_new_idea(idea)

                return True

            except sqlite3.IntegrityError:
                logger.warning(f"Duplicate idea ID: {idea.id}")
                return False

    def get(self, idea_id: str) -> Optional[Idea]:
        """Get an idea by ID."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_idea(row)
            return None

    def update(self, idea: Idea) -> bool:
        """Update an existing idea."""
        idea.touch()

        with self._lock:
            conn = self._get_conn()
            row = self._idea_to_row(idea)

            # Build UPDATE statement
            set_clause = ", ".join(f"{k} = ?" for k in row.keys() if k != "id")
            values = [v for k, v in row.items() if k != "id"]
            values.append(idea.id)

            cursor = conn.execute(
                f"UPDATE ideas SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.debug(f"Updated idea: {idea.id}")
                return True
            return False

    def delete(self, idea_id: str) -> bool:
        """Delete an idea."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM ideas WHERE id = ?", (idea_id,))
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Queries
    # =========================================================================

    def list_all(self, limit: int = 100) -> List[Idea]:
        """List all ideas."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT * FROM ideas ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            return [self._row_to_idea(row) for row in cursor.fetchall()]

    def list_by_status(self, status: IdeaStatus, limit: int = 100) -> List[Idea]:
        """List ideas by status."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT * FROM ideas WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status.value, limit)
            )
            return [self._row_to_idea(row) for row in cursor.fetchall()]

    def list_by_category(self, category: IdeaCategory, limit: int = 100) -> List[Idea]:
        """List ideas by category."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT * FROM ideas WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                (category.value, limit)
            )
            return [self._row_to_idea(row) for row in cursor.fetchall()]

    def list_inbox(self) -> List[Idea]:
        """Get all ideas in the inbox (new, awaiting review)."""
        return self.list_by_status(IdeaStatus.INBOX) + self.list_by_status(IdeaStatus.NEEDS_REVIEW)

    def list_approved(self) -> List[Idea]:
        """Get all approved ideas waiting for execution."""
        return self.list_by_status(IdeaStatus.APPROVED)

    def list_running(self) -> List[Idea]:
        """Get all currently running ideas."""
        return self.list_by_status(IdeaStatus.RUNNING)

    def list_completed(self, limit: int = 50) -> List[Idea]:
        """Get completed ideas."""
        return self.list_by_status(IdeaStatus.COMPLETED, limit)

    def count_by_status(self) -> Dict[str, int]:
        """Get counts per status."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT status, COUNT(*) as count FROM ideas GROUP BY status"
            )
            return {row["status"]: row["count"] for row in cursor.fetchall()}

    def search(self, query: str, limit: int = 20) -> List[Idea]:
        """Search ideas by title or hypothesis."""
        with self._lock:
            conn = self._get_conn()
            pattern = f"%{query}%"
            cursor = conn.execute(
                """SELECT * FROM ideas
                   WHERE title LIKE ? OR hypothesis LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (pattern, pattern, limit)
            )
            return [self._row_to_idea(row) for row in cursor.fetchall()]

    # =========================================================================
    # Lifecycle Operations
    # =========================================================================

    def submit_to_inbox(self, idea: Idea) -> bool:
        """Move a draft idea to the inbox for review.

        Returns:
            True if moved, False if not allowed
        """
        if idea.status != IdeaStatus.DRAFT:
            return False

        old_status = idea.status
        idea.status = IdeaStatus.INBOX
        idea.touch()

        if self.update(idea):
            if self._on_status_change:
                self._on_status_change(idea, old_status, idea.status)

            # Auto-approve if configured
            if self.config.auto_approve_none_risk and idea.risk == IdeaRisk.NONE:
                idea.approve(by="auto", notes="Auto-approved: zero risk")
                self.update(idea)

            return True
        return False

    def approve(self, idea_id: str, by: str = "human", notes: str = "") -> bool:
        """Approve an idea for execution."""
        idea = self.get(idea_id)
        if not idea or not idea.is_actionable():
            return False

        old_status = idea.status
        idea.approve(by=by, notes=notes)

        if self.update(idea):
            if self._on_status_change:
                self._on_status_change(idea, old_status, idea.status)
            return True
        return False

    def reject(self, idea_id: str, by: str = "human", notes: str = "") -> bool:
        """Reject an idea."""
        idea = self.get(idea_id)
        if not idea or idea.is_terminal():
            return False

        old_status = idea.status
        idea.reject(by=by, notes=notes)

        if self.update(idea):
            if self._on_status_change:
                self._on_status_change(idea, old_status, idea.status)
            return True
        return False

    def park(self, idea_id: str, by: str = "human", notes: str = "") -> bool:
        """Park an idea for later consideration."""
        idea = self.get(idea_id)
        if not idea or idea.is_terminal():
            return False

        old_status = idea.status
        idea.park(by=by, notes=notes)

        if self.update(idea):
            if self._on_status_change:
                self._on_status_change(idea, old_status, idea.status)
            return True
        return False

    def start_execution(self, idea_id: str) -> bool:
        """Start executing an approved idea."""
        idea = self.get(idea_id)
        if not idea or not idea.is_executable():
            return False

        # Check running limit
        running = self.list_running()
        if len(running) >= self.config.max_running_ideas:
            logger.warning(f"Max running ideas reached ({self.config.max_running_ideas})")
            return False

        old_status = idea.status
        idea.start_execution()

        if self.update(idea):
            if self._on_status_change:
                self._on_status_change(idea, old_status, idea.status)
            return True
        return False

    def complete_execution(
        self,
        idea_id: str,
        outcome: IdeaOutcome,
        notes: str = "",
        signals: Optional[List[Signal]] = None
    ) -> bool:
        """Complete execution of an idea."""
        idea = self.get(idea_id)
        if not idea or idea.status != IdeaStatus.RUNNING:
            return False

        old_status = idea.status
        idea.complete(outcome, notes)
        if signals:
            idea.outcome_signals = signals

        if self.update(idea):
            if self._on_status_change:
                self._on_status_change(idea, old_status, idea.status)
            if self._on_complete:
                self._on_complete(idea)
            return True
        return False

    def revert(self, idea_id: str, notes: str = "") -> bool:
        """Revert a running idea."""
        idea = self.get(idea_id)
        if not idea or idea.status != IdeaStatus.RUNNING:
            return False

        old_status = idea.status
        idea.revert(notes)

        if self.update(idea):
            if self._on_status_change:
                self._on_status_change(idea, old_status, idea.status)
            if self._on_complete:
                self._on_complete(idea)
            return True
        return False

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_new_idea(self, callback: Callable[[Idea], None]) -> None:
        """Register callback for new ideas."""
        self._on_new_idea = callback

    def on_status_change(self, callback: Callable[[Idea, IdeaStatus, IdeaStatus], None]) -> None:
        """Register callback for status changes."""
        self._on_status_change = callback

    def on_complete(self, callback: Callable[[Idea], None]) -> None:
        """Register callback for completed ideas."""
        self._on_complete = callback

    # =========================================================================
    # Maintenance
    # =========================================================================

    def cleanup_old_inbox(self) -> int:
        """Park ideas that have been in inbox too long.

        Returns:
            Number of ideas parked
        """
        cutoff = time.time() - (self.config.max_inbox_age_hours * 3600)
        parked = 0

        for idea in self.list_inbox():
            if idea.created_at < cutoff:
                if self.park(idea.id, by="auto", notes="Auto-parked: exceeded inbox age limit"):
                    parked += 1

        return parked

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the board state."""
        counts = self.count_by_status()
        return {
            "total_ideas": sum(counts.values()),
            "by_status": counts,
            "inbox_count": counts.get("inbox", 0) + counts.get("needs_review", 0),
            "approved_count": counts.get("approved", 0),
            "running_count": counts.get("running", 0),
            "completed_count": counts.get("completed", 0),
            "ideas_this_hour": self._ideas_this_hour,
            "ideas_today": self._ideas_today,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
