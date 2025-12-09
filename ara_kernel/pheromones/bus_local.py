"""
Local Pheromone Bus
====================

In-process pheromone bus for coordinating agents.
Gives agents a shared sense of "what's hot" without central micromanagement.

For distributed setups, this can be backed by Redis/NATS instead.
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PheromoneScope(Enum):
    """Signal propagation scope."""
    LOCAL = "local"       # This process only
    CLUSTER = "cluster"   # All nodes in cluster
    REMOTE = "remote"     # Cross-cluster


@dataclass
class Pheromone:
    """A single pheromone signal."""
    topic: str
    intensity: float
    scope: PheromoneScope
    ttl_ms: int
    payload: Dict[str, Any]
    emitter: str
    created_ts: float = field(default_factory=time.time)

    def is_alive(self) -> bool:
        """Check if pheromone is still active."""
        age_ms = (time.time() - self.created_ts) * 1000
        return age_ms <= self.ttl_ms

    def remaining_ttl_ms(self) -> float:
        """Remaining time-to-live in milliseconds."""
        age_ms = (time.time() - self.created_ts) * 1000
        return max(0, self.ttl_ms - age_ms)

    def decayed_intensity(self) -> float:
        """Get intensity with time decay applied."""
        if self.ttl_ms <= 0:
            return self.intensity

        remaining = self.remaining_ttl_ms()
        decay_factor = remaining / self.ttl_ms
        return self.intensity * decay_factor


class LocalPheromoneBus:
    """
    In-process pheromone bus.

    Features:
    - Thread-safe signal emission and reading
    - Automatic expiration of old signals
    - Aggregation by topic
    - Intensity decay over time
    """

    def __init__(self, max_signals: int = 1000) -> None:
        self._lock = threading.RLock()
        self._signals: List[Pheromone] = []
        self._max_signals = max_signals

    def emit(
        self,
        topic: str,
        intensity: float,
        scope: str = "local",
        ttl_ms: int = 30_000,
        payload: Optional[Dict[str, Any]] = None,
        emitter: str = "unknown",
    ) -> Pheromone:
        """
        Emit a pheromone signal.

        Args:
            topic: Signal category (e.g., "success/publishing")
            intensity: Strength 0.0-1.0
            scope: "local", "cluster", or "remote"
            ttl_ms: Time-to-live in milliseconds
            payload: Optional structured data
            emitter: ID of emitting agent

        Returns:
            The created pheromone
        """
        p = Pheromone(
            topic=topic,
            intensity=max(0.0, min(1.0, intensity)),
            scope=PheromoneScope(scope),
            ttl_ms=ttl_ms,
            payload=payload or {},
            emitter=emitter,
            created_ts=time.time(),
        )

        with self._lock:
            self._signals.append(p)
            # Prune if over limit
            if len(self._signals) > self._max_signals:
                self._prune()

        logger.debug(f"Pheromone emitted: {topic} (intensity={intensity:.2f})")
        return p

    def snapshot(self) -> Dict[str, Any]:
        """
        Get current state of all live pheromones.

        Returns aggregated view by topic.
        """
        live = self._get_live()

        # Aggregate by topic
        topics: Dict[str, Dict[str, Any]] = {}
        for p in live:
            t = topics.setdefault(p.topic, {
                "max_intensity": 0.0,
                "count": 0,
                "scope": p.scope.value,
                "emitters": [],
            })
            decayed = p.decayed_intensity()
            if decayed > t["max_intensity"]:
                t["max_intensity"] = decayed
            t["count"] += 1
            if p.emitter not in t["emitters"]:
                t["emitters"].append(p.emitter)

        return {
            "topics": topics,
            "total_signals": len(live),
            "snapshot_ts": time.time(),
        }

    def get_topic(self, topic: str) -> List[Pheromone]:
        """Get all live pheromones for a specific topic."""
        return [p for p in self._get_live() if p.topic == topic]

    def get_by_prefix(self, prefix: str) -> List[Pheromone]:
        """Get all live pheromones with topic starting with prefix."""
        return [p for p in self._get_live() if p.topic.startswith(prefix)]

    def max_intensity(self, topic: str) -> float:
        """Get maximum intensity for a topic (with decay)."""
        pheromones = self.get_topic(topic)
        if not pheromones:
            return 0.0
        return max(p.decayed_intensity() for p in pheromones)

    def has_alarm(self) -> bool:
        """Check if any alarm signals are active."""
        return any(p.topic.startswith("alarm/") for p in self._get_live())

    def emit_from_result(
        self,
        event: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """
        Emit pheromones based on event result.

        Simple heuristic:
        - Success → emit success pheromone
        - Failure → emit alarm pheromone
        """
        domain = event.get("domain", "default")
        result_data = result.get("result", {})
        actions_executed = result_data.get("actions_executed", 0)
        actions_failed = result_data.get("actions_failed", 0)

        if actions_executed > 0:
            self.emit(
                topic=f"success/{domain}",
                intensity=0.5,
                ttl_ms=10_000,
                payload={"event_type": event.get("type"), "actions": actions_executed},
                emitter=event.get("emitter", "kernel"),
            )

        if actions_failed > 0:
            self.emit(
                topic=f"failure/{domain}",
                intensity=0.7,
                ttl_ms=30_000,
                payload={"event_type": event.get("type"), "failures": actions_failed},
                emitter=event.get("emitter", "kernel"),
            )

    def _get_live(self) -> List[Pheromone]:
        """Get all live (non-expired) pheromones."""
        with self._lock:
            live = [p for p in self._signals if p.is_alive()]
            self._signals = live  # Prune dead ones
            return live

    def _prune(self) -> None:
        """Remove expired and oldest signals to stay under limit."""
        # Remove expired first
        self._signals = [p for p in self._signals if p.is_alive()]

        # If still over limit, remove oldest
        if len(self._signals) > self._max_signals:
            self._signals = self._signals[-self._max_signals:]

    def clear(self) -> None:
        """Clear all pheromones."""
        with self._lock:
            self._signals = []

    def count(self) -> int:
        """Number of live pheromones."""
        return len(self._get_live())
