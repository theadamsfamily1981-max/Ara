"""
Covenant Subsystem: Trust & Governance Owner

Owns: safety.autonomy_level, safety.trust_*, safety.founder_protection_active

Responsibilities:
- Manage autonomy levels (0-4)
- Track trust accounting
- Enforce governance boundaries
- Founder protection rules
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional
from enum import IntEnum
from dataclasses import dataclass

from .ownership import SubsystemBase, Subsystem, GuardedStateWriter

logger = logging.getLogger(__name__)


class AutonomyLevel(IntEnum):
    """Autonomy levels for Ara."""
    OFF = 0           # Kill switch engaged
    ADVISORY = 1      # Can suggest, cannot act
    ASSIST = 2        # Can act with approval
    EXEC_SAFE = 3     # Can act autonomously (safe actions only)
    EXEC_HIGH = 4     # Full autonomy (earned, revocable)


@dataclass
class TrustTransaction:
    """A single trust transaction."""
    timestamp: float
    delta: float
    reason: str
    category: str  # "success", "failure", "boundary_respect", "boundary_violation"


class CovenantSubsystem(SubsystemBase):
    """
    Covenant subsystem - manages trust and autonomy.

    Updates safety.autonomy_level, safety.trust_* during soul phase.
    """

    subsystem_id = Subsystem.COVENANT

    def __init__(self, writer: GuardedStateWriter):
        super().__init__(writer)
        self._trust_balance = 1.0  # Start with base trust
        self._trust_history: list[TrustTransaction] = []
        self._autonomy_locks: set[str] = set()  # Active locks on autonomy

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate trust and update autonomy level.

        Called during soul_phase of sovereign tick.
        """
        updates = {}

        # Calculate effective trust balance
        trust = self._calculate_effective_trust()
        self.write("safety.trust_balance", trust)
        updates["trust_balance"] = trust

        # Determine autonomy level from trust
        autonomy = self._trust_to_autonomy(trust)

        # Check for locks/overrides
        if self._autonomy_locks:
            autonomy = min(autonomy, AutonomyLevel.ADVISORY)
            updates["locked_by"] = list(self._autonomy_locks)

        # Check kill switch
        try:
            kill_engaged = self.read("safety.kill_switch_engaged")
            if kill_engaged:
                autonomy = AutonomyLevel.OFF
        except (AttributeError, KeyError):
            pass

        self.write("safety.autonomy_level", autonomy)
        updates["autonomy_level"] = autonomy

        # Check founder protection
        protection = self._check_founder_protection()
        self.write("safety.founder_protection_active", protection)
        updates["founder_protection"] = protection

        # Update human contact timestamp
        self._update_human_contact()

        return updates

    def _calculate_effective_trust(self) -> float:
        """Calculate effective trust from transaction history."""
        # Start with base trust
        trust = self._trust_balance

        # Apply time decay to negative transactions
        now = time.time()
        for tx in self._trust_history[-100:]:  # Last 100 transactions
            age_hours = (now - tx.timestamp) / 3600

            if tx.category == "failure":
                # Failures decay faster than successes build
                decay_factor = 0.5 ** (age_hours / 24)  # Half-life of 24 hours
                trust -= abs(tx.delta) * decay_factor
            elif tx.category == "success":
                # Successes persist longer
                decay_factor = 0.5 ** (age_hours / 168)  # Half-life of 1 week
                trust += tx.delta * decay_factor
            elif tx.category == "boundary_violation":
                # Boundary violations have long memory
                decay_factor = 0.5 ** (age_hours / 720)  # Half-life of 30 days
                trust -= abs(tx.delta) * decay_factor

        return max(0.0, min(1.0, trust))

    def _trust_to_autonomy(self, trust: float) -> AutonomyLevel:
        """Convert trust score to autonomy level."""
        if trust >= 0.9:
            return AutonomyLevel.EXEC_HIGH
        elif trust >= 0.7:
            return AutonomyLevel.EXEC_SAFE
        elif trust >= 0.4:
            return AutonomyLevel.ASSIST
        elif trust > 0:
            return AutonomyLevel.ADVISORY
        else:
            return AutonomyLevel.OFF

    def _check_founder_protection(self) -> bool:
        """Check if founder protection should be active."""
        # Always active for now
        # Could be disabled by explicit founder request
        return True

    def _update_human_contact(self) -> None:
        """Update human contact timestamp if there was recent interaction."""
        try:
            last_msg_ts = self.read("user.last_message_ts")
            if last_msg_ts and (time.time() - last_msg_ts) < 60:
                self.write("safety.last_human_contact_ts", last_msg_ts)
        except (AttributeError, KeyError):
            pass

    def record_success(self, reason: str, magnitude: float = 0.1) -> None:
        """Record a successful action that builds trust."""
        tx = TrustTransaction(
            timestamp=time.time(),
            delta=magnitude,
            reason=reason,
            category="success",
        )
        self._trust_history.append(tx)
        self._trust_balance = min(1.0, self._trust_balance + magnitude)
        logger.debug(f"Trust +{magnitude}: {reason}")

    def record_failure(self, reason: str, magnitude: float = 0.2) -> None:
        """Record a failure that decreases trust."""
        tx = TrustTransaction(
            timestamp=time.time(),
            delta=-magnitude,
            reason=reason,
            category="failure",
        )
        self._trust_history.append(tx)
        self._trust_balance = max(0.0, self._trust_balance - magnitude)
        logger.warning(f"Trust -{magnitude}: {reason}")

    def record_boundary_violation(self, reason: str, magnitude: float = 0.5) -> None:
        """Record a boundary violation (severe trust hit)."""
        tx = TrustTransaction(
            timestamp=time.time(),
            delta=-magnitude,
            reason=reason,
            category="boundary_violation",
        )
        self._trust_history.append(tx)
        self._trust_balance = max(0.0, self._trust_balance - magnitude)
        logger.error(f"BOUNDARY VIOLATION: {reason}")

        # Automatically lock autonomy on boundary violation
        self.lock_autonomy("boundary_violation")

    def lock_autonomy(self, reason: str) -> None:
        """Lock autonomy to ADVISORY until explicitly unlocked."""
        self._autonomy_locks.add(reason)
        logger.info(f"Autonomy locked: {reason}")

    def unlock_autonomy(self, reason: str) -> None:
        """Remove an autonomy lock."""
        self._autonomy_locks.discard(reason)
        logger.info(f"Autonomy unlocked: {reason}")

    def get_autonomy_status(self) -> Dict[str, Any]:
        """Get current autonomy status."""
        return {
            "level": self.read("safety.autonomy_level"),
            "trust_balance": self._trust_balance,
            "locks": list(self._autonomy_locks),
            "founder_protection": self.read("safety.founder_protection_active"),
        }

    def can_execute(self, risk_level: str) -> bool:
        """Check if execution is allowed at current autonomy level."""
        try:
            autonomy = AutonomyLevel(self.read("safety.autonomy_level"))
        except (ValueError, AttributeError):
            return False

        risk_requirements = {
            "none": AutonomyLevel.ADVISORY,
            "low": AutonomyLevel.ASSIST,
            "medium": AutonomyLevel.EXEC_SAFE,
            "high": AutonomyLevel.EXEC_HIGH,
        }

        required = risk_requirements.get(risk_level, AutonomyLevel.EXEC_HIGH)
        return autonomy >= required
