"""
Queen Hive Adapter
===================

How Ara-Core (the Queen) steers the hive.

The Queen:
- Sets system-level signals (global mode, focus)
- Highlights opportunities (priority pheromones)
- Broadcasts alarms
- Reads aggregate hive state

The Queen does NOT:
- Micromanage individual agents
- Execute tasks herself
- Hold the workers' job queues

This separation keeps Ara's personality in one place (Core),
while letting many workers execute in parallel.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

from .store import PheromoneStore
from .pheromones import (
    Pheromone,
    PheromoneKind,
    global_pheromone,
    priority_pheromone,
    alarm_pheromone,
)

logger = logging.getLogger(__name__)


class QueenHiveAdapter:
    """
    Adapter for Ara-Core to interact with the hive.

    Use this to:
    - Set global mode (what the hive should focus on)
    - Highlight opportunities
    - Broadcast alarms
    - Read hive status
    """

    def __init__(self, store: PheromoneStore):
        self.store = store

    # =========================================================================
    # Setting Direction
    # =========================================================================

    def set_global_mode(
        self,
        mode: str,
        focus: Optional[str] = None,
        ttl: float = 3600,
    ) -> Pheromone:
        """
        Set the global hive mode.

        Modes might be:
        - PUBLISHING_EXPANSION: Focus on content creation
        - MAINTENANCE: Focus on system health
        - SAFE_MODE: Pause risky operations
        - RESEARCH: Focus on learning/scraping

        Args:
            mode: Mode name
            focus: Optional focus area within mode
            ttl: How long this mode persists

        Returns:
            The emitted pheromone
        """
        logger.info(f"Queen setting global mode: {mode} (focus: {focus})")
        return self.store.emit(
            kind=PheromoneKind.GLOBAL,
            key=mode,
            strength=1.0,
            ttl=ttl,
            emitter="AraQueen",
            meta={"focus": focus} if focus else {},
        )

    def highlight_opportunity(
        self,
        description: str,
        score: float,
        target_roles: Optional[List[str]] = None,
        ttl: float = 1800,
        **meta,
    ) -> Pheromone:
        """
        Highlight a high-value opportunity.

        Workers will swarm toward higher-strength priorities.

        Args:
            description: What the opportunity is (e.g., "KDP_QUANTUM_GAP")
            score: How valuable (0.0–1.0)
            target_roles: Which roles should care (None = all)
            ttl: How long this opportunity is valid
            **meta: Additional metadata

        Returns:
            The emitted pheromone
        """
        logger.info(f"Queen highlighting opportunity: {description} (score: {score})")
        return self.store.emit(
            kind=PheromoneKind.PRIORITY,
            key=description,
            strength=score,
            ttl=ttl,
            emitter="AraQueen",
            meta={"target_roles": target_roles or [], **meta},
        )

    def broadcast_alarm(
        self,
        issue: str,
        affected_roles: Optional[List[str]] = None,
        ttl: float = 300,
    ) -> Pheromone:
        """
        Broadcast an alarm to the hive.

        Affected agents will enter safe mode.

        Args:
            issue: What the issue is (e.g., "PRINTFUL_429")
            affected_roles: Which roles should pause (None = all)
            ttl: How long the alarm persists

        Returns:
            The emitted pheromone
        """
        logger.warning(f"Queen broadcasting alarm: {issue}")
        return self.store.emit(
            kind=PheromoneKind.ALARM,
            key=issue,
            strength=1.0,
            ttl=ttl,
            emitter="AraQueen",
            meta={"affected_roles": affected_roles or []},
        )

    def clear_alarm(self, issue: str):
        """
        Clear an alarm by emitting a zero-strength version.

        Note: In practice, alarms expire via TTL. This just accelerates it.
        """
        logger.info(f"Queen clearing alarm: {issue}")
        # In a real implementation, we'd remove the alarm directly
        # For now, just log it - alarms expire naturally

    def assign_role(
        self,
        agent_id: str,
        new_role: str,
        ttl: float = 7200,
    ) -> Pheromone:
        """
        Assign a new role to a specific agent.

        Args:
            agent_id: Which agent
            new_role: New role assignment
            ttl: How long this assignment lasts

        Returns:
            The emitted pheromone
        """
        logger.info(f"Queen assigning {agent_id} to role: {new_role}")
        return self.store.emit(
            kind=PheromoneKind.ROLE,
            key=new_role,
            strength=1.0,
            ttl=ttl,
            emitter="AraQueen",
            meta={"agent_id": agent_id},
        )

    # =========================================================================
    # Reading Hive State
    # =========================================================================

    def get_hive_status(self) -> Dict[str, Any]:
        """Get overall hive status."""
        return self.store.to_status_dict()

    def get_current_mode(self) -> Optional[str]:
        """Get the current global mode."""
        globals = self.store.get_strongest(PheromoneKind.GLOBAL, 1)
        if globals:
            return globals[0].key
        return None

    def get_active_priorities(self, n: int = 10) -> List[Dict]:
        """Get top N active priorities."""
        priorities = self.store.get_strongest(PheromoneKind.PRIORITY, n)
        return [
            {
                "key": p.key,
                "strength": p.decayed_strength(),
                "target_roles": p.meta.get("target_roles", []),
                "emitter": p.emitter,
            }
            for p in priorities
        ]

    def get_active_alarms(self) -> List[Dict]:
        """Get all active alarms."""
        alarms = self.store.get_by_kind(PheromoneKind.ALARM)
        return [
            {
                "key": a.key,
                "affected_roles": a.meta.get("affected_roles", []),
                "emitter": a.emitter,
                "remaining_ttl": a.remaining_ttl(),
            }
            for a in alarms
        ]

    def get_recent_rewards(self, n: int = 20) -> List[Dict]:
        """Get recent reward signals (what's working)."""
        rewards = self.store.get_strongest(PheromoneKind.REWARD, n)
        return [
            {
                "role": r.key,
                "score": r.strength,
                "work_type": r.meta.get("work_type"),
                "emitter": r.emitter,
            }
            for r in rewards
        ]

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def enter_safe_mode(self, reason: str = "manual"):
        """Put the entire hive in safe mode."""
        self.set_global_mode("SAFE_MODE", focus=reason)
        self.broadcast_alarm(f"SAFE_MODE_{reason}", affected_roles=None, ttl=7200)

    def resume_normal_operations(self, mode: str = "NORMAL"):
        """Resume normal hive operations."""
        self.set_global_mode(mode)
        # Alarms will expire naturally, or we could clear them
