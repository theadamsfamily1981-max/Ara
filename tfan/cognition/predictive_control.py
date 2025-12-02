"""
Predictive Self-Healing Control: L7 → L3/GUF/AEPO Direct Wiring

This module implements "predictive anxiety" - the ability to:
1. Detect that Ṡ (structural velocity) is rising
2. BEFORE failure manifests:
   - Flip L3 policy to CONSERVATIVE
   - Shift GUF toward INTERNAL/RECOVERY mode
   - Reserve AEPO time slot for structural fix

This is anticipatory intelligence, not reactive control.

The key insight:
- Traditional: "topo_gap bad" → "fix it"
- L7 Predictive: "Ṡ trending up" → "prepare to fix it before it breaks"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
import logging

logger = logging.getLogger("tfan.cognition.predictive_control")


# ============================================================
# L3 Policy States (what we're switching between)
# ============================================================

class L3Policy(str, Enum):
    """L3 Metacontrol policy states."""
    EXPLORATORY = "exploratory"     # High temperature, allow entropy
    BALANCED = "balanced"           # Normal operation
    CONSERVATIVE = "conservative"   # Low temperature, strict routing
    PROTECTIVE = "protective"       # Minimal risk, verified paths only
    EMERGENCY = "emergency"         # Survival mode, full lockdown


# ============================================================
# GUF Scheduler Modes (self vs world allocation)
# ============================================================

class GUFSchedulerMode(str, Enum):
    """GUF scheduler focus modes."""
    EXTERNAL = "external"       # 90% external, 10% internal
    BALANCED = "balanced"       # 50% each
    INTERNAL = "internal"       # 70% internal, 30% external
    RECOVERY = "recovery"       # 90% internal, emergency self-repair


# ============================================================
# AEPO Reservation System
# ============================================================

@dataclass
class AEPOSlot:
    """A reserved time slot for AEPO structural optimization."""
    slot_id: str
    requested_at: datetime
    execute_within_ms: int
    task_type: str  # "structural_fix", "proactive_redundancy", "emergency_repair"
    priority: int   # 1 = highest, 5 = lowest
    executed: bool = False
    executed_at: Optional[datetime] = None
    result: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Has this slot expired without execution?"""
        if self.executed:
            return False
        deadline = self.requested_at + timedelta(milliseconds=self.execute_within_ms)
        return datetime.now() > deadline

    @property
    def time_remaining_ms(self) -> int:
        """Milliseconds remaining before deadline."""
        deadline = self.requested_at + timedelta(milliseconds=self.execute_within_ms)
        remaining = (deadline - datetime.now()).total_seconds() * 1000
        return max(0, int(remaining))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "requested_at": self.requested_at.isoformat(),
            "execute_within_ms": self.execute_within_ms,
            "task_type": self.task_type,
            "priority": self.priority,
            "executed": self.executed,
            "time_remaining_ms": self.time_remaining_ms,
            "is_expired": self.is_expired
        }


class AEPOReservationQueue:
    """
    Queue for AEPO time slot reservations.

    When L7 predicts instability, we reserve future AEPO bandwidth
    before the crisis actually hits.
    """

    def __init__(self, max_slots: int = 10):
        self._slots: List[AEPOSlot] = []
        self._max_slots = max_slots
        self._slot_counter = 0
        self._executed_count = 0
        self._expired_count = 0

    def reserve(
        self,
        task_type: str,
        execute_within_ms: int = 500,
        priority: int = 3
    ) -> AEPOSlot:
        """
        Reserve an AEPO slot.

        Returns the slot for tracking.
        """
        self._slot_counter += 1
        slot = AEPOSlot(
            slot_id=f"aepo_{self._slot_counter:04d}",
            requested_at=datetime.now(),
            execute_within_ms=execute_within_ms,
            task_type=task_type,
            priority=priority
        )

        # Insert by priority (lower number = higher priority)
        inserted = False
        for i, existing in enumerate(self._slots):
            if slot.priority < existing.priority:
                self._slots.insert(i, slot)
                inserted = True
                break
        if not inserted:
            self._slots.append(slot)

        # Trim if over capacity
        if len(self._slots) > self._max_slots:
            self._slots = self._slots[:self._max_slots]

        logger.info(f"Reserved AEPO slot {slot.slot_id}: {task_type} within {execute_within_ms}ms")
        return slot

    def get_next_slot(self) -> Optional[AEPOSlot]:
        """Get the highest priority pending slot."""
        self._cleanup_expired()
        for slot in self._slots:
            if not slot.executed and not slot.is_expired:
                return slot
        return None

    def execute_slot(self, slot_id: str, result: str = "completed") -> bool:
        """Mark a slot as executed."""
        for slot in self._slots:
            if slot.slot_id == slot_id:
                slot.executed = True
                slot.executed_at = datetime.now()
                slot.result = result
                self._executed_count += 1
                logger.info(f"Executed AEPO slot {slot_id}: {result}")
                return True
        return False

    def _cleanup_expired(self) -> int:
        """Remove expired slots and count them."""
        expired = [s for s in self._slots if s.is_expired]
        self._expired_count += len(expired)
        self._slots = [s for s in self._slots if not s.is_expired]
        return len(expired)

    @property
    def pending_count(self) -> int:
        """Count of pending (non-executed, non-expired) slots."""
        return len([s for s in self._slots if not s.executed and not s.is_expired])

    @property
    def has_urgent(self) -> bool:
        """Is there a high-priority slot pending?"""
        for slot in self._slots:
            if not slot.executed and not slot.is_expired and slot.priority <= 2:
                return True
        return False

    def get_state(self) -> Dict[str, Any]:
        """Get queue state."""
        return {
            "pending": self.pending_count,
            "has_urgent": self.has_urgent,
            "total_reserved": self._slot_counter,
            "total_executed": self._executed_count,
            "total_expired": self._expired_count,
            "slots": [s.to_dict() for s in self._slots if not s.executed]
        }


# ============================================================
# Predictive Control State
# ============================================================

@dataclass
class PredictiveControlState:
    """Current state of the predictive control system."""
    # L3 Policy
    current_policy: L3Policy = L3Policy.BALANCED
    policy_locked: bool = False  # If true, don't auto-switch
    policy_lock_reason: str = ""

    # GUF Scheduler
    guf_mode: GUFSchedulerMode = GUFSchedulerMode.BALANCED
    internal_allocation: float = 0.5  # 0-1, fraction for internal work

    # Predictive State
    structural_rate: float = 0.0  # Ṡ
    alert_level: str = "stable"
    predicted_steps_to_failure: int = -1  # -1 = no failure predicted
    prediction_confidence: float = 0.0

    # Intervention History
    last_intervention: Optional[datetime] = None
    intervention_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l3_policy": {
                "current": self.current_policy.value,
                "locked": self.policy_locked,
                "lock_reason": self.policy_lock_reason
            },
            "guf": {
                "mode": self.guf_mode.value,
                "internal_allocation": self.internal_allocation
            },
            "prediction": {
                "structural_rate": self.structural_rate,
                "alert_level": self.alert_level,
                "steps_to_failure": self.predicted_steps_to_failure,
                "confidence": self.prediction_confidence
            },
            "interventions": {
                "count": self.intervention_count,
                "last": self.last_intervention.isoformat() if self.last_intervention else None
            }
        }


# ============================================================
# Predictive Controller
# ============================================================

class PredictiveController:
    """
    The tight integration layer for predictive self-healing.

    This is where Ṡ meets policy meets scheduling.

    Wiring:
        L7 TemporalTopologyTracker
            ↓ structural_rate (Ṡ)
        PredictiveController
            ├→ L3 Policy (CONSERVATIVE when Ṡ > threshold)
            ├→ GUF Mode (INTERNAL/RECOVERY when Ṡ high)
            └→ AEPO Queue (reserve slot when Ṡ rising)
    """

    def __init__(
        self,
        # Thresholds
        rate_elevated: float = 0.15,     # Ṡ for "elevated" - start watching
        rate_warning: float = 0.30,      # Ṡ for "warning" - take action
        rate_critical: float = 0.50,     # Ṡ for "critical" - emergency
        # Timing
        aepo_default_deadline_ms: int = 500,
        # Callbacks (for integration)
        on_policy_change: Optional[Callable[[L3Policy, L3Policy], None]] = None,
        on_guf_change: Optional[Callable[[GUFSchedulerMode, float], None]] = None,
    ):
        # Thresholds
        self.rate_elevated = rate_elevated
        self.rate_warning = rate_warning
        self.rate_critical = rate_critical
        self.aepo_default_deadline_ms = aepo_default_deadline_ms

        # Callbacks
        self._on_policy_change = on_policy_change
        self._on_guf_change = on_guf_change

        # State
        self._state = PredictiveControlState()
        self._aepo_queue = AEPOReservationQueue()

        # Intervention history for analysis
        self._interventions: List[Dict[str, Any]] = []
        self._max_history = 100

        # Statistics
        self._stats = {
            "updates": 0,
            "policy_changes": 0,
            "guf_changes": 0,
            "aepo_reservations": 0,
            "prevented_crises": 0  # Times we went WARNING → STABLE without hitting CRITICAL
        }

        # High-water mark tracking for prevented crisis detection
        self._high_water_alert = "stable"  # Worst alert seen since last stable

    @property
    def state(self) -> PredictiveControlState:
        """Current control state."""
        return self._state

    @property
    def aepo_queue(self) -> AEPOReservationQueue:
        """AEPO reservation queue."""
        return self._aepo_queue

    # --------------------------------------------------------
    # Core Update Method: Feed me Ṡ
    # --------------------------------------------------------

    def update(
        self,
        structural_rate: float,
        alert_level: str = "stable",
        predicted_steps: int = -1,
        confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Update from L7 tracker.

        This is the main integration point. Every time L7 computes Ṡ,
        feed it here and we'll decide what to do.

        Args:
            structural_rate: Ṡ from L7
            alert_level: L7's alert assessment
            predicted_steps: Steps until predicted failure (-1 = none)
            confidence: Confidence in prediction

        Returns:
            Dict with actions taken and recommendations
        """
        self._stats["updates"] += 1

        # Record previous state for change detection
        old_policy = self._state.current_policy
        old_guf_mode = self._state.guf_mode
        old_alert = self._state.alert_level

        # Update state
        self._state.structural_rate = structural_rate
        self._state.alert_level = alert_level
        self._state.predicted_steps_to_failure = predicted_steps
        self._state.prediction_confidence = confidence

        # Compute interventions
        actions_taken = []

        # 1. Determine target policy based on Ṡ
        if not self._state.policy_locked:
            new_policy = self._compute_target_policy(structural_rate, alert_level)
            if new_policy != old_policy:
                self._transition_policy(old_policy, new_policy)
                actions_taken.append({
                    "action": "policy_change",
                    "from": old_policy.value,
                    "to": new_policy.value,
                    "reason": f"Ṡ={structural_rate:.3f}, alert={alert_level}"
                })

        # 2. Determine GUF mode
        new_guf_mode, new_allocation = self._compute_guf_mode(structural_rate, alert_level)
        if new_guf_mode != old_guf_mode:
            self._transition_guf(old_guf_mode, new_guf_mode, new_allocation)
            actions_taken.append({
                "action": "guf_change",
                "from": old_guf_mode.value,
                "to": new_guf_mode.value,
                "allocation": new_allocation,
                "reason": f"Ṡ={structural_rate:.3f}"
            })

        # 3. Consider AEPO reservation if rising
        aepo_slot = None
        if self._should_reserve_aepo(structural_rate, old_alert, alert_level):
            aepo_slot = self._reserve_aepo_slot(structural_rate, alert_level)
            actions_taken.append({
                "action": "aepo_reserved",
                "slot_id": aepo_slot.slot_id,
                "task_type": aepo_slot.task_type,
                "deadline_ms": aepo_slot.execute_within_ms
            })

        # 4. Track if we prevented a crisis using high-water mark
        alert_order = ["stable", "elevated", "warning", "critical"]
        if alert_level in alert_order:
            current_idx = alert_order.index(alert_level)
            high_idx = alert_order.index(self._high_water_alert)

            # Update high-water mark if worse
            if current_idx > high_idx:
                self._high_water_alert = alert_level

            # Check if we returned to stable after being in warning (but not critical)
            if alert_level == "stable" and self._high_water_alert == "warning":
                self._stats["prevented_crises"] += 1
                self._high_water_alert = "stable"  # Reset high-water mark
            elif alert_level == "stable":
                self._high_water_alert = "stable"  # Reset on any return to stable

        # Build result
        result = {
            "structural_rate": structural_rate,
            "alert_level": alert_level,
            "current_policy": self._state.current_policy.value,
            "current_guf_mode": self._state.guf_mode.value,
            "actions_taken": actions_taken,
            "aepo_pending": self._aepo_queue.pending_count,
            "has_urgent_aepo": self._aepo_queue.has_urgent,
            "recommendation": self._generate_recommendation()
        }

        # Record intervention if we took action
        if actions_taken:
            self._record_intervention(result)

        return result

    def _compute_target_policy(
        self,
        structural_rate: float,
        alert_level: str
    ) -> L3Policy:
        """Compute the target L3 policy based on current Ṡ."""
        if alert_level == "critical" or structural_rate >= self.rate_critical:
            return L3Policy.EMERGENCY
        elif alert_level == "warning" or structural_rate >= self.rate_warning:
            return L3Policy.PROTECTIVE
        elif alert_level == "elevated" or structural_rate >= self.rate_elevated:
            return L3Policy.CONSERVATIVE
        else:
            return L3Policy.BALANCED

    def _compute_guf_mode(
        self,
        structural_rate: float,
        alert_level: str
    ) -> Tuple[GUFSchedulerMode, float]:
        """Compute GUF mode and internal allocation."""
        if alert_level == "critical" or structural_rate >= self.rate_critical:
            return GUFSchedulerMode.RECOVERY, 0.9
        elif alert_level == "warning" or structural_rate >= self.rate_warning:
            return GUFSchedulerMode.INTERNAL, 0.7
        elif alert_level == "elevated" or structural_rate >= self.rate_elevated:
            return GUFSchedulerMode.BALANCED, 0.5
        else:
            return GUFSchedulerMode.EXTERNAL, 0.2

    def _should_reserve_aepo(
        self,
        structural_rate: float,
        old_alert: str,
        new_alert: str
    ) -> bool:
        """Decide if we should reserve an AEPO slot."""
        # Reserve when transitioning to worse state
        alert_order = ["stable", "elevated", "warning", "critical"]

        if new_alert not in alert_order or old_alert not in alert_order:
            return False

        old_idx = alert_order.index(old_alert)
        new_idx = alert_order.index(new_alert)

        # Reserve if alert is escalating
        if new_idx > old_idx:
            return True

        # Also reserve if rate is high but stable (proactive)
        if structural_rate >= self.rate_elevated and self._aepo_queue.pending_count < 2:
            return True

        return False

    def _reserve_aepo_slot(
        self,
        structural_rate: float,
        alert_level: str
    ) -> AEPOSlot:
        """Reserve an AEPO slot based on current state."""
        # Determine task type and priority
        if alert_level == "critical":
            task_type = "emergency_repair"
            priority = 1
            deadline_ms = 200
        elif alert_level == "warning":
            task_type = "structural_fix"
            priority = 2
            deadline_ms = 500
        else:
            task_type = "proactive_redundancy"
            priority = 3
            deadline_ms = 1000

        slot = self._aepo_queue.reserve(
            task_type=task_type,
            execute_within_ms=deadline_ms,
            priority=priority
        )

        self._stats["aepo_reservations"] += 1
        return slot

    def _transition_policy(self, old: L3Policy, new: L3Policy) -> None:
        """Execute policy transition."""
        self._state.current_policy = new
        self._stats["policy_changes"] += 1
        self._state.last_intervention = datetime.now()
        self._state.intervention_count += 1

        logger.info(f"L3 Policy: {old.value} → {new.value}")

        if self._on_policy_change:
            self._on_policy_change(old, new)

    def _transition_guf(
        self,
        old: GUFSchedulerMode,
        new: GUFSchedulerMode,
        allocation: float
    ) -> None:
        """Execute GUF mode transition."""
        self._state.guf_mode = new
        self._state.internal_allocation = allocation
        self._stats["guf_changes"] += 1

        logger.info(f"GUF Mode: {old.value} → {new.value} (internal={allocation:.0%})")

        if self._on_guf_change:
            self._on_guf_change(new, allocation)

    def _generate_recommendation(self) -> str:
        """Generate human-readable recommendation."""
        s = self._state

        if s.alert_level == "critical":
            return "CRITICAL: Focus all resources on structural stability. External requests paused."
        elif s.alert_level == "warning":
            return f"WARNING: Ṡ={s.structural_rate:.3f} rising. Protective measures active. Monitor closely."
        elif s.alert_level == "elevated":
            return f"ELEVATED: Ṡ={s.structural_rate:.3f} above baseline. Conservative policy engaged."
        else:
            return "STABLE: Normal operation. System healthy."

    def _record_intervention(self, result: Dict[str, Any]) -> None:
        """Record intervention for analysis."""
        record = {
            "timestamp": datetime.now().isoformat(),
            **result
        }
        self._interventions.append(record)
        if len(self._interventions) > self._max_history:
            self._interventions = self._interventions[-self._max_history:]

    # --------------------------------------------------------
    # Query Methods
    # --------------------------------------------------------

    def get_policy_parameters(self) -> Dict[str, Any]:
        """Get current policy parameters for L3 integration."""
        policy = self._state.current_policy

        params = {
            L3Policy.EXPLORATORY: {
                "temperature_mult": 1.2,
                "entropy_level": 0.8,
                "safety_mode": "relaxed",
                "allow_speculation": True
            },
            L3Policy.BALANCED: {
                "temperature_mult": 1.0,
                "entropy_level": 0.5,
                "safety_mode": "normal",
                "allow_speculation": True
            },
            L3Policy.CONSERVATIVE: {
                "temperature_mult": 0.8,
                "entropy_level": 0.3,
                "safety_mode": "careful",
                "allow_speculation": False
            },
            L3Policy.PROTECTIVE: {
                "temperature_mult": 0.6,
                "entropy_level": 0.2,
                "safety_mode": "strict",
                "allow_speculation": False
            },
            L3Policy.EMERGENCY: {
                "temperature_mult": 0.4,
                "entropy_level": 0.1,
                "safety_mode": "lockdown",
                "allow_speculation": False
            }
        }

        return {
            "policy": policy.value,
            "parameters": params.get(policy, params[L3Policy.BALANCED])
        }

    def get_guf_allocation(self) -> Dict[str, float]:
        """Get current GUF focus allocation."""
        internal = self._state.internal_allocation
        return {
            "internal": internal,
            "external": 1.0 - internal,
            "mode": self._state.guf_mode.value
        }

    def lock_policy(self, policy: L3Policy, reason: str) -> None:
        """Lock policy to a specific state (manual override)."""
        self._state.policy_locked = True
        self._state.policy_lock_reason = reason
        if self._state.current_policy != policy:
            old = self._state.current_policy
            self._transition_policy(old, policy)
        logger.info(f"Policy locked to {policy.value}: {reason}")

    def unlock_policy(self) -> None:
        """Unlock policy for automatic control."""
        self._state.policy_locked = False
        self._state.policy_lock_reason = ""
        logger.info("Policy unlocked")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get complete state summary."""
        return {
            "state": self._state.to_dict(),
            "aepo_queue": self._aepo_queue.get_state(),
            "stats": self._stats.copy(),
            "policy_params": self.get_policy_parameters(),
            "guf_allocation": self.get_guf_allocation()
        }

    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """Get history of interventions."""
        return self._interventions.copy()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            **self._stats,
            "aepo_queue": self._aepo_queue.get_state()
        }


# ============================================================
# Integration Helper: Wire L7 Tracker to Predictive Controller
# ============================================================

def wire_l7_to_predictive_controller(
    tracker,  # TemporalTopologyTracker
    controller: PredictiveController
) -> Callable:
    """
    Create a callback that wires L7 tracker updates to PredictiveController.

    Usage:
        from tfan.l7 import TemporalTopologyTracker
        from tfan.cognition.predictive_control import PredictiveController, wire_l7_to_predictive_controller

        tracker = TemporalTopologyTracker()
        controller = PredictiveController()

        # This creates the wiring
        update_callback = wire_l7_to_predictive_controller(tracker, controller)

        # Now when you update the tracker, call the callback
        dynamics = tracker.update(betti_0=5, spectral_gap=0.8)
        result = update_callback(dynamics)
    """
    def on_l7_update(dynamics) -> Dict[str, Any]:
        """Called after each L7 tracker update."""
        return controller.update(
            structural_rate=dynamics.structural_rate,
            alert_level=dynamics.alert_level.value if hasattr(dynamics.alert_level, 'value') else str(dynamics.alert_level),
            predicted_steps=dynamics.predicted_instability_steps,
            confidence=dynamics.confidence
        )

    return on_l7_update


# ============================================================
# Factory Functions
# ============================================================

def create_predictive_controller(
    rate_elevated: float = 0.15,
    rate_warning: float = 0.30,
    rate_critical: float = 0.50
) -> PredictiveController:
    """Create a predictive controller with specified thresholds."""
    return PredictiveController(
        rate_elevated=rate_elevated,
        rate_warning=rate_warning,
        rate_critical=rate_critical
    )


__all__ = [
    "L3Policy",
    "GUFSchedulerMode",
    "AEPOSlot",
    "AEPOReservationQueue",
    "PredictiveControlState",
    "PredictiveController",
    "wire_l7_to_predictive_controller",
    "create_predictive_controller"
]
