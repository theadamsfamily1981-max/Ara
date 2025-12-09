"""
Pheromone Store
================

Shared state for all agents in the hive.

The store:
- Holds all active pheromones
- Handles TTL-based expiry
- Provides filtered views for agents
- Can persist to disk/Redis for crash recovery

For v0.1, this is an in-memory store with optional JSON persistence.
Later: Redis pub/sub, NATS, or custom mesh transport.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Callable, Any

from .pheromones import Pheromone, PheromoneKind

logger = logging.getLogger(__name__)


class PheromoneStore:
    """
    Central store for pheromones.

    Thread-safe, with optional persistence.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._lock = Lock()
        self._pheromones: List[Pheromone] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._subscribers: List[Callable[[Pheromone], None]] = []

        # Load persisted state if available
        if self._persist_path and self._persist_path.exists():
            self._load_from_disk()

    # =========================================================================
    # Core Operations
    # =========================================================================

    def emit(
        self,
        kind: PheromoneKind,
        key: str,
        strength: float = 1.0,
        ttl: float = 3600,
        emitter: str = "unknown",
        meta: Optional[Dict] = None,
    ) -> Pheromone:
        """
        Emit a new pheromone into the store.

        Args:
            kind: Type of pheromone
            key: Identifier (e.g., "KDP_QUANTUM_GAP")
            strength: Signal strength (0.0–1.0)
            ttl: Time to live in seconds
            emitter: Who emitted this (e.g., "agent:worker_01")
            meta: Additional metadata

        Returns:
            The created Pheromone
        """
        pheromone = Pheromone.create(
            kind=kind,
            key=key,
            strength=strength,
            ttl=ttl,
            emitter=emitter,
            meta=meta,
        )

        with self._lock:
            self._pheromones.append(pheromone)

        # Notify subscribers
        for callback in self._subscribers:
            try:
                callback(pheromone)
            except Exception as e:
                logger.warning(f"Subscriber error: {e}")

        # Persist if configured
        if self._persist_path:
            self._persist_to_disk()

        return pheromone

    def snapshot(self, include_expired: bool = False) -> List[Pheromone]:
        """
        Get a snapshot of all pheromones.

        By default, filters out expired pheromones.
        """
        now = datetime.utcnow()

        with self._lock:
            if not include_expired:
                # Prune expired
                self._pheromones = [p for p in self._pheromones if not p.is_expired(now)]
            return list(self._pheromones)

    def get_by_kind(self, kind: PheromoneKind) -> List[Pheromone]:
        """Get all pheromones of a specific kind."""
        return [p for p in self.snapshot() if p.kind == kind]

    def get_by_key(self, key: str) -> List[Pheromone]:
        """Get all pheromones with a specific key."""
        return [p for p in self.snapshot() if p.key == key]

    def get_strongest(self, kind: Optional[PheromoneKind] = None, n: int = 5) -> List[Pheromone]:
        """Get the strongest pheromones, optionally filtered by kind."""
        pheromones = self.snapshot()
        if kind:
            pheromones = [p for p in pheromones if p.kind == kind]

        # Sort by decayed strength
        now = datetime.utcnow()
        pheromones.sort(key=lambda p: p.decayed_strength(now), reverse=True)
        return pheromones[:n]

    def clear(self):
        """Clear all pheromones."""
        with self._lock:
            self._pheromones = []
        if self._persist_path:
            self._persist_to_disk()

    # =========================================================================
    # Subscriptions
    # =========================================================================

    def subscribe(self, callback: Callable[[Pheromone], None]):
        """Subscribe to new pheromone emissions."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Pheromone], None]):
        """Unsubscribe from emissions."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    # =========================================================================
    # Persistence
    # =========================================================================

    def _persist_to_disk(self):
        """Save current state to disk."""
        if not self._persist_path:
            return

        try:
            data = [p.to_dict() for p in self._pheromones]
            with open(self._persist_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to persist pheromones: {e}")

    def _load_from_disk(self):
        """Load state from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return

        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            self._pheromones = [Pheromone.from_dict(d) for d in data]
            # Prune expired on load
            now = datetime.utcnow()
            self._pheromones = [p for p in self._pheromones if not p.is_expired(now)]
            logger.info(f"Loaded {len(self._pheromones)} pheromones from disk")
        except Exception as e:
            logger.warning(f"Failed to load pheromones: {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        pheromones = self.snapshot()
        by_kind = {}
        for kind in PheromoneKind:
            by_kind[kind.value] = len([p for p in pheromones if p.kind == kind])

        return {
            "total": len(pheromones),
            "by_kind": by_kind,
            "unique_keys": len(set(p.key for p in pheromones)),
            "unique_emitters": len(set(p.emitter for p in pheromones)),
        }

    def to_status_dict(self) -> Dict[str, Any]:
        """Get a status dict for display/debugging."""
        pheromones = self.snapshot()
        now = datetime.utcnow()

        return {
            "timestamp": now.isoformat(),
            "stats": self.stats(),
            "global_mode": self._get_current_global_mode(),
            "top_priorities": [
                {"key": p.key, "strength": p.decayed_strength(now)}
                for p in self.get_strongest(PheromoneKind.PRIORITY, 5)
            ],
            "active_alarms": [
                {"key": p.key, "emitter": p.emitter}
                for p in self.get_by_kind(PheromoneKind.ALARM)
            ],
        }

    def _get_current_global_mode(self) -> Optional[str]:
        """Get the current global mode (strongest global pheromone)."""
        globals = self.get_strongest(PheromoneKind.GLOBAL, 1)
        if globals:
            return globals[0].key
        return None
