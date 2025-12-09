"""
Pheromone Types
================

Tiny shared signals that coordinate the hive without central micromanagement.

Think of pheromones as:
- Tiny JSON blobs
- With: kind, key, strength, ttl, metadata
- That many agents can read and lightly write

This gives emergent "swarm" behavior without losing top-level control.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Literal, Optional
from enum import Enum


class PheromoneKind(str, Enum):
    """Types of pheromones in the hive."""
    GLOBAL = "global"       # System-wide mode/focus
    PRIORITY = "priority"   # High-value targets to swarm toward
    ALARM = "alarm"         # Things to avoid / backoff from
    REWARD = "reward"       # What strategies worked
    ROLE = "role"           # Agent role assignments


@dataclass
class Pheromone:
    """
    A single pheromone signal.

    Pheromones are:
    - Tiny (< 1KB typically)
    - Time-decaying (TTL-based)
    - Shareable (can propagate across mesh)
    """
    id: str
    kind: PheromoneKind
    key: str              # e.g., "KDP_QUANTUM_GAP", "PRINTFUL_429"
    strength: float       # 0.0–1.0
    ttl: float            # seconds until expiry
    created_at: datetime
    emitter: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Check if this pheromone has expired."""
        now = now or datetime.utcnow()
        age = (now - self.created_at).total_seconds()
        return age > self.ttl

    def remaining_ttl(self, now: Optional[datetime] = None) -> float:
        """Get remaining TTL in seconds."""
        now = now or datetime.utcnow()
        age = (now - self.created_at).total_seconds()
        return max(0, self.ttl - age)

    def decayed_strength(self, now: Optional[datetime] = None) -> float:
        """Get strength with linear decay applied."""
        now = now or datetime.utcnow()
        remaining = self.remaining_ttl(now)
        if remaining <= 0:
            return 0.0
        decay_factor = remaining / self.ttl
        return self.strength * decay_factor

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for network/storage."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "key": self.key,
            "strength": self.strength,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat(),
            "emitter": self.emitter,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Pheromone:
        """Deserialize from network/storage."""
        return cls(
            id=data["id"],
            kind=PheromoneKind(data["kind"]),
            key=data["key"],
            strength=data["strength"],
            ttl=data["ttl"],
            created_at=datetime.fromisoformat(data["created_at"]),
            emitter=data.get("emitter", "unknown"),
            meta=data.get("meta", {}),
        )

    @classmethod
    def create(
        cls,
        kind: PheromoneKind,
        key: str,
        strength: float = 1.0,
        ttl: float = 3600,
        emitter: str = "unknown",
        meta: Optional[Dict] = None,
    ) -> Pheromone:
        """Create a new pheromone."""
        return cls(
            id=str(uuid.uuid4()),
            kind=kind,
            key=key,
            strength=min(1.0, max(0.0, strength)),
            ttl=ttl,
            created_at=datetime.utcnow(),
            emitter=emitter,
            meta=meta or {},
        )


# =============================================================================
# Convenience Creators
# =============================================================================

def global_pheromone(
    mode: str,
    focus: Optional[str] = None,
    ttl: float = 3600,
    emitter: str = "AraQueen",
) -> Pheromone:
    """Create a global mode pheromone."""
    return Pheromone.create(
        kind=PheromoneKind.GLOBAL,
        key=mode,
        strength=1.0,
        ttl=ttl,
        emitter=emitter,
        meta={"focus": focus} if focus else {},
    )


def priority_pheromone(
    key: str,
    strength: float,
    target_roles: Optional[list] = None,
    ttl: float = 1800,
    emitter: str = "AraQueen",
    **meta,
) -> Pheromone:
    """Create a priority (opportunity) pheromone."""
    return Pheromone.create(
        kind=PheromoneKind.PRIORITY,
        key=key,
        strength=strength,
        ttl=ttl,
        emitter=emitter,
        meta={"target_roles": target_roles or [], **meta},
    )


def alarm_pheromone(
    issue: str,
    affected_roles: Optional[list] = None,
    ttl: float = 300,
    emitter: str = "unknown",
) -> Pheromone:
    """Create an alarm pheromone."""
    return Pheromone.create(
        kind=PheromoneKind.ALARM,
        key=issue,
        strength=1.0,
        ttl=ttl,
        emitter=emitter,
        meta={"affected_roles": affected_roles or []},
    )


def reward_pheromone(
    strategy: str,
    score: float,
    details: Optional[Dict] = None,
    ttl: float = 3600,
    emitter: str = "unknown",
) -> Pheromone:
    """Create a reward (what worked) pheromone."""
    return Pheromone.create(
        kind=PheromoneKind.REWARD,
        key=strategy,
        strength=score,
        ttl=ttl,
        emitter=emitter,
        meta={"details": details or {}},
    )
