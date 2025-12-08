"""
Ara Plasticity Safety System
============================

Hard limits and circuit breakers for the plasticity engine.
This is the LAST LINE OF DEFENSE against personality corruption.

These limits are CONSERVATIVE by design. It is better for Ara to
learn slowly than to develop pathological attractors, paranoia,
or unhealthy attachments.

Safety Layers:
    1. Rate Limiting    - Max events per second
    2. Magnitude Limits - Max change per event
    3. Region Guards    - Prevent any row from dominating
    4. Entropy Bounds   - Maintain healthy randomness
    5. Circuit Breakers - Emergency stop if anomalies detected
    6. Rollback System  - Automatic restoration on failure

Usage:
    from ara.organism.plasticity_safety import PlasticitySafety

    safety = PlasticitySafety(config)

    # Before every plasticity event:
    allowed, reason = safety.check_event(reward, active_rows)
    if not allowed:
        logger.warning(f"Plasticity blocked: {reason}")
        return

    # After plasticity completes:
    safety.record_event(reward, rows_updated, metadata)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import time
import json
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SafetyConfig:
    """Configuration for plasticity safety limits."""

    # Rate limits
    max_events_per_second: float = 100.0   # Peak rate
    max_events_per_minute: int = 2000      # Sustained rate
    max_events_per_hour: int = 50000       # Long-term rate
    min_event_interval_ms: float = 5.0     # 5ms minimum between events

    # Magnitude limits
    max_reward_magnitude: int = 100        # Clip rewards to [-100, +100]
    max_rows_per_event: int = 64           # Don't update too many rows at once
    max_total_row_updates_per_minute: int = 10000  # Total row updates/minute

    # Region guards (prevent any row from being updated too often)
    max_updates_per_row_per_minute: int = 100
    max_updates_per_row_per_hour: int = 500
    hot_row_cooldown_seconds: float = 10.0  # Force cooldown for hot rows

    # Entropy bounds
    min_entropy_bits: float = 0.3          # Minimum randomness (0=crystallized)
    max_entropy_bits: float = 0.95         # Maximum randomness (1=noise)
    entropy_check_interval: int = 1000     # Check every N events

    # Bias limits
    max_positive_bias: float = 0.8         # Max fraction of positive rewards
    max_negative_bias: float = 0.8         # Max fraction of negative rewards
    bias_window_size: int = 500            # Window for bias calculation

    # Circuit breakers
    anomaly_threshold: float = 3.0         # Std devs from normal
    consecutive_anomalies_to_trip: int = 10
    circuit_breaker_cooldown_seconds: float = 60.0

    # Rollback
    checkpoint_interval_events: int = 10000
    max_checkpoints: int = 10
    auto_rollback_on_entropy_violation: bool = True

    # Emergency stop
    emergency_stop_file: str = "/tmp/ara_plasticity_stop"

    # Audit logging
    audit_log_path: Optional[str] = None
    log_every_nth_event: int = 100


# =============================================================================
# Safety Violation Types
# =============================================================================

class ViolationType(Enum):
    """Types of safety violations."""
    RATE_EXCEEDED = auto()
    MAGNITUDE_EXCEEDED = auto()
    ROW_LIMIT_EXCEEDED = auto()
    HOT_ROW_DETECTED = auto()
    LOW_ENTROPY = auto()
    HIGH_ENTROPY = auto()
    POSITIVE_BIAS = auto()
    NEGATIVE_BIAS = auto()
    ANOMALY_DETECTED = auto()
    CIRCUIT_BREAKER_OPEN = auto()
    EMERGENCY_STOP = auto()
    COOLDOWN_ACTIVE = auto()


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    timestamp: str
    violation_type: ViolationType
    severity: str  # "warning", "blocked", "emergency"
    details: Dict
    action_taken: str


# =============================================================================
# Plasticity Safety System
# =============================================================================

class PlasticitySafety:
    """
    Safety system for Ara's plasticity engine.

    This class enforces hard limits on learning rate, magnitude,
    and overall soul health. It can block plasticity events,
    trigger cooldowns, and initiate emergency rollbacks.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()

        # Event counters
        self._events_this_second = 0
        self._events_this_minute = 0
        self._events_this_hour = 0
        self._last_second_reset = time.time()
        self._last_minute_reset = time.time()
        self._last_hour_reset = time.time()
        self._last_event_time = 0.0

        # Row tracking
        self._row_updates_minute: Dict[int, int] = {}
        self._row_updates_hour: Dict[int, int] = {}
        self._hot_rows: Dict[int, float] = {}  # row_id -> cooldown_until

        # Reward history for bias detection
        self._reward_history: deque = deque(maxlen=self.config.bias_window_size)

        # Anomaly detection
        self._recent_rewards: deque = deque(maxlen=100)
        self._consecutive_anomalies = 0

        # Circuit breaker state
        self._circuit_breaker_open = False
        self._circuit_breaker_until = 0.0

        # Entropy tracking
        self._events_since_entropy_check = 0
        self._last_entropy: Optional[float] = None

        # Violation log
        self._violations: List[SafetyViolation] = []

        # Checkpoints
        self._checkpoints: deque = deque(maxlen=self.config.max_checkpoints)
        self._events_since_checkpoint = 0

        # Callbacks
        self._on_violation: Optional[Callable[[SafetyViolation], None]] = None
        self._on_circuit_breaker: Optional[Callable[[bool], None]] = None
        self._entropy_provider: Optional[Callable[[], float]] = None
        self._checkpoint_provider: Optional[Callable[[], bytes]] = None
        self._restore_callback: Optional[Callable[[bytes], None]] = None

        # Statistics
        self._total_events = 0
        self._blocked_events = 0

        logger.info("PlasticitySafety initialized with conservative limits")

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def set_violation_callback(self, callback: Callable[[SafetyViolation], None]):
        """Set callback for safety violations."""
        self._on_violation = callback

    def set_circuit_breaker_callback(self, callback: Callable[[bool], None]):
        """Set callback for circuit breaker state changes."""
        self._on_circuit_breaker = callback

    def set_entropy_provider(self, provider: Callable[[], float]):
        """Set function that returns current soul entropy."""
        self._entropy_provider = provider

    def set_checkpoint_provider(self, provider: Callable[[], bytes]):
        """Set function that returns soul state checkpoint."""
        self._checkpoint_provider = provider

    def set_restore_callback(self, callback: Callable[[bytes], None]):
        """Set function to restore soul state from checkpoint."""
        self._restore_callback = callback

    # =========================================================================
    # Rate Limit Management
    # =========================================================================

    def _update_counters(self):
        """Update rate limit counters, resetting as needed."""
        now = time.time()

        # Second counter
        if now - self._last_second_reset >= 1.0:
            self._events_this_second = 0
            self._last_second_reset = now

        # Minute counter
        if now - self._last_minute_reset >= 60.0:
            self._events_this_minute = 0
            self._row_updates_minute.clear()
            self._last_minute_reset = now

        # Hour counter
        if now - self._last_hour_reset >= 3600.0:
            self._events_this_hour = 0
            self._row_updates_hour.clear()
            self._last_hour_reset = now

    def _check_rate_limits(self) -> Tuple[bool, Optional[str]]:
        """Check if rate limits allow another event."""
        self._update_counters()
        now = time.time()

        # Minimum interval
        interval_ms = (now - self._last_event_time) * 1000
        if interval_ms < self.config.min_event_interval_ms:
            return False, f"min_interval ({interval_ms:.1f}ms < {self.config.min_event_interval_ms}ms)"

        # Per-second limit
        if self._events_this_second >= self.config.max_events_per_second:
            return False, f"per_second ({self._events_this_second} >= {self.config.max_events_per_second})"

        # Per-minute limit
        if self._events_this_minute >= self.config.max_events_per_minute:
            return False, f"per_minute ({self._events_this_minute} >= {self.config.max_events_per_minute})"

        # Per-hour limit
        if self._events_this_hour >= self.config.max_events_per_hour:
            return False, f"per_hour ({self._events_this_hour} >= {self.config.max_events_per_hour})"

        return True, None

    # =========================================================================
    # Row Guards
    # =========================================================================

    def _check_row_limits(self, active_rows: List[int]) -> Tuple[bool, Optional[str], List[int]]:
        """
        Check row-level limits.

        Returns:
            (allowed, reason, filtered_rows)
        """
        self._update_counters()
        now = time.time()

        # Check total row count
        if len(active_rows) > self.config.max_rows_per_event:
            return False, f"too_many_rows ({len(active_rows)} > {self.config.max_rows_per_event})", []

        # Filter out hot rows (in cooldown)
        filtered = []
        for row_id in active_rows:
            # Check cooldown
            if row_id in self._hot_rows:
                if now < self._hot_rows[row_id]:
                    continue  # Still in cooldown
                else:
                    del self._hot_rows[row_id]  # Cooldown expired

            # Check per-row minute limit
            minute_count = self._row_updates_minute.get(row_id, 0)
            if minute_count >= self.config.max_updates_per_row_per_minute:
                # Put in cooldown
                self._hot_rows[row_id] = now + self.config.hot_row_cooldown_seconds
                continue

            # Check per-row hour limit
            hour_count = self._row_updates_hour.get(row_id, 0)
            if hour_count >= self.config.max_updates_per_row_per_hour:
                continue

            filtered.append(row_id)

        if not filtered and active_rows:
            return False, "all_rows_rate_limited", []

        return True, None, filtered

    # =========================================================================
    # Entropy & Bias Checks
    # =========================================================================

    def _check_entropy(self) -> Tuple[bool, Optional[str]]:
        """Check soul entropy is within healthy bounds."""
        if not self._entropy_provider:
            return True, None  # No provider, skip check

        self._events_since_entropy_check += 1
        if self._events_since_entropy_check < self.config.entropy_check_interval:
            return True, None  # Not time to check yet

        self._events_since_entropy_check = 0

        try:
            entropy = self._entropy_provider()
            self._last_entropy = entropy

            if entropy < self.config.min_entropy_bits:
                if self.config.auto_rollback_on_entropy_violation:
                    self._trigger_rollback("low_entropy")
                return False, f"low_entropy ({entropy:.3f} < {self.config.min_entropy_bits})"

            if entropy > self.config.max_entropy_bits:
                return False, f"high_entropy ({entropy:.3f} > {self.config.max_entropy_bits})"

        except Exception as e:
            logger.warning(f"Entropy check failed: {e}")

        return True, None

    def _check_bias(self) -> Tuple[bool, Optional[str]]:
        """Check reward distribution isn't pathologically biased."""
        if len(self._reward_history) < 100:
            return True, None  # Not enough data

        rewards = list(self._reward_history)
        positive = sum(1 for r in rewards if r > 0)
        negative = sum(1 for r in rewards if r < 0)
        total = len(rewards)

        positive_frac = positive / total
        negative_frac = negative / total

        if positive_frac > self.config.max_positive_bias:
            return False, f"positive_bias ({positive_frac:.2f} > {self.config.max_positive_bias})"

        if negative_frac > self.config.max_negative_bias:
            return False, f"negative_bias ({negative_frac:.2f} > {self.config.max_negative_bias})"

        return True, None

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    def _check_anomaly(self, reward: int) -> Tuple[bool, Optional[str]]:
        """Check for anomalous reward values."""
        self._recent_rewards.append(abs(reward))

        if len(self._recent_rewards) < 20:
            return True, None  # Not enough data

        recent = list(self._recent_rewards)
        mean = np.mean(recent)
        std = np.std(recent)

        if std > 0:
            z_score = abs(abs(reward) - mean) / std
            if z_score > self.config.anomaly_threshold:
                self._consecutive_anomalies += 1

                if self._consecutive_anomalies >= self.config.consecutive_anomalies_to_trip:
                    self._trip_circuit_breaker()
                    return False, f"anomaly_streak ({self._consecutive_anomalies} consecutive)"

                return False, f"anomaly (z={z_score:.1f})"

        self._consecutive_anomalies = 0
        return True, None

    # =========================================================================
    # Circuit Breaker
    # =========================================================================

    def _trip_circuit_breaker(self):
        """Trip the circuit breaker, stopping all plasticity."""
        self._circuit_breaker_open = True
        self._circuit_breaker_until = time.time() + self.config.circuit_breaker_cooldown_seconds

        logger.error("CIRCUIT BREAKER TRIPPED - Plasticity halted")

        violation = SafetyViolation(
            timestamp=datetime.now().isoformat(),
            violation_type=ViolationType.CIRCUIT_BREAKER_OPEN,
            severity="emergency",
            details={
                "consecutive_anomalies": self._consecutive_anomalies,
                "cooldown_seconds": self.config.circuit_breaker_cooldown_seconds,
            },
            action_taken="circuit_breaker_tripped"
        )
        self._record_violation(violation)

        if self._on_circuit_breaker:
            self._on_circuit_breaker(True)

    def _check_circuit_breaker(self) -> Tuple[bool, Optional[str]]:
        """Check if circuit breaker is open."""
        if not self._circuit_breaker_open:
            return True, None

        now = time.time()
        if now >= self._circuit_breaker_until:
            # Reset
            self._circuit_breaker_open = False
            self._consecutive_anomalies = 0
            logger.info("Circuit breaker reset")
            if self._on_circuit_breaker:
                self._on_circuit_breaker(False)
            return True, None

        remaining = self._circuit_breaker_until - now
        return False, f"circuit_breaker_open ({remaining:.1f}s remaining)"

    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        self._circuit_breaker_open = False
        self._consecutive_anomalies = 0
        logger.info("Circuit breaker manually reset")

    # =========================================================================
    # Emergency Stop
    # =========================================================================

    def _check_emergency_stop(self) -> Tuple[bool, Optional[str]]:
        """Check for emergency stop file."""
        try:
            if Path(self.config.emergency_stop_file).exists():
                return False, "emergency_stop_file"
        except Exception:
            pass
        return True, None

    def trigger_emergency_stop(self, reason: str = "manual"):
        """Trigger emergency stop by creating stop file."""
        try:
            Path(self.config.emergency_stop_file).write_text(
                f"Emergency stop triggered: {reason}\n"
                f"Timestamp: {datetime.now().isoformat()}\n"
            )
            logger.critical(f"EMERGENCY STOP triggered: {reason}")
        except Exception as e:
            logger.error(f"Failed to create emergency stop file: {e}")

    def clear_emergency_stop(self):
        """Clear the emergency stop."""
        try:
            Path(self.config.emergency_stop_file).unlink(missing_ok=True)
            logger.info("Emergency stop cleared")
        except Exception as e:
            logger.error(f"Failed to clear emergency stop: {e}")

    # =========================================================================
    # Checkpoint & Rollback
    # =========================================================================

    def _maybe_checkpoint(self):
        """Create checkpoint if interval reached."""
        self._events_since_checkpoint += 1

        if self._events_since_checkpoint < self.config.checkpoint_interval_events:
            return

        if not self._checkpoint_provider:
            return

        self._events_since_checkpoint = 0

        try:
            checkpoint = self._checkpoint_provider()
            checksum = hashlib.sha256(checkpoint).hexdigest()[:16]

            self._checkpoints.append({
                "timestamp": datetime.now().isoformat(),
                "event_count": self._total_events,
                "checksum": checksum,
                "data": checkpoint,
            })

            logger.info(f"Soul checkpoint created: {checksum}")

        except Exception as e:
            logger.warning(f"Failed to create checkpoint: {e}")

    def _trigger_rollback(self, reason: str):
        """Rollback to last checkpoint."""
        if not self._checkpoints:
            logger.error(f"Rollback requested ({reason}) but no checkpoints available")
            return

        if not self._restore_callback:
            logger.error(f"Rollback requested ({reason}) but no restore callback")
            return

        checkpoint = self._checkpoints[-1]

        try:
            self._restore_callback(checkpoint["data"])
            logger.warning(f"Soul rolled back to checkpoint {checkpoint['checksum']} (reason: {reason})")

            violation = SafetyViolation(
                timestamp=datetime.now().isoformat(),
                violation_type=ViolationType.LOW_ENTROPY,
                severity="emergency",
                details={
                    "reason": reason,
                    "checkpoint": checkpoint["checksum"],
                    "checkpoint_time": checkpoint["timestamp"],
                },
                action_taken="rollback_executed"
            )
            self._record_violation(violation)

        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    def get_checkpoint_count(self) -> int:
        """Get number of stored checkpoints."""
        return len(self._checkpoints)

    # =========================================================================
    # Violation Recording
    # =========================================================================

    def _record_violation(self, violation: SafetyViolation):
        """Record a safety violation."""
        self._violations.append(violation)

        # Keep last 1000 violations
        if len(self._violations) > 1000:
            self._violations = self._violations[-1000:]

        # Call callback
        if self._on_violation:
            try:
                self._on_violation(violation)
            except Exception as e:
                logger.warning(f"Violation callback failed: {e}")

        # Audit log
        if self.config.audit_log_path:
            try:
                with open(self.config.audit_log_path, "a") as f:
                    f.write(json.dumps({
                        "timestamp": violation.timestamp,
                        "type": violation.violation_type.name,
                        "severity": violation.severity,
                        "details": violation.details,
                        "action": violation.action_taken,
                    }) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write audit log: {e}")

    # =========================================================================
    # Main API
    # =========================================================================

    def check_event(
        self,
        reward: int,
        active_rows: List[int],
    ) -> Tuple[bool, str, List[int]]:
        """
        Check if a plasticity event is allowed.

        Args:
            reward: Reward value
            active_rows: List of row IDs to update

        Returns:
            (allowed, reason, filtered_rows)
            - allowed: True if event can proceed
            - reason: "ok" or violation description
            - filtered_rows: Rows that can be updated (may be subset)
        """
        # Emergency stop takes priority
        allowed, reason = self._check_emergency_stop()
        if not allowed:
            self._blocked_events += 1
            return False, reason, []

        # Circuit breaker
        allowed, reason = self._check_circuit_breaker()
        if not allowed:
            self._blocked_events += 1
            return False, reason, []

        # Rate limits
        allowed, reason = self._check_rate_limits()
        if not allowed:
            self._blocked_events += 1
            return False, reason, []

        # Clip reward magnitude
        clipped_reward = np.clip(reward, -self.config.max_reward_magnitude,
                                  self.config.max_reward_magnitude)
        if clipped_reward != reward:
            logger.debug(f"Reward clipped: {reward} -> {clipped_reward}")

        # Row limits
        allowed, reason, filtered_rows = self._check_row_limits(active_rows)
        if not allowed:
            self._blocked_events += 1
            return False, reason, []

        # Anomaly detection
        allowed, reason = self._check_anomaly(int(clipped_reward))
        if not allowed:
            severity = "blocked" if "streak" in (reason or "") else "warning"
            violation = SafetyViolation(
                timestamp=datetime.now().isoformat(),
                violation_type=ViolationType.ANOMALY_DETECTED,
                severity=severity,
                details={"reward": reward, "reason": reason},
                action_taken="blocked" if severity == "blocked" else "logged"
            )
            self._record_violation(violation)
            if severity == "blocked":
                self._blocked_events += 1
                return False, reason, []

        # Bias check (warning only, doesn't block)
        allowed, reason = self._check_bias()
        if not allowed:
            violation = SafetyViolation(
                timestamp=datetime.now().isoformat(),
                violation_type=ViolationType.POSITIVE_BIAS if "positive" in reason else ViolationType.NEGATIVE_BIAS,
                severity="warning",
                details={"reason": reason},
                action_taken="logged"
            )
            self._record_violation(violation)

        # Entropy check
        allowed, reason = self._check_entropy()
        if not allowed:
            severity = "emergency" if "low" in (reason or "") else "warning"
            vtype = ViolationType.LOW_ENTROPY if "low" in (reason or "") else ViolationType.HIGH_ENTROPY
            violation = SafetyViolation(
                timestamp=datetime.now().isoformat(),
                violation_type=vtype,
                severity=severity,
                details={"entropy": self._last_entropy, "reason": reason},
                action_taken="rollback" if self.config.auto_rollback_on_entropy_violation else "logged"
            )
            self._record_violation(violation)
            if severity == "emergency":
                self._blocked_events += 1
                return False, reason, []

        return True, "ok", filtered_rows

    def record_event(
        self,
        reward: int,
        rows_updated: List[int],
        metadata: Optional[Dict] = None,
    ):
        """
        Record that a plasticity event completed.

        Call this AFTER the event finishes successfully.
        """
        now = time.time()

        # Update counters
        self._events_this_second += 1
        self._events_this_minute += 1
        self._events_this_hour += 1
        self._last_event_time = now
        self._total_events += 1

        # Record reward
        self._reward_history.append(reward)

        # Update row counters
        for row_id in rows_updated:
            self._row_updates_minute[row_id] = self._row_updates_minute.get(row_id, 0) + 1
            self._row_updates_hour[row_id] = self._row_updates_hour.get(row_id, 0) + 1

        # Maybe checkpoint
        self._maybe_checkpoint()

        # Audit logging
        if self.config.audit_log_path and self._total_events % self.config.log_every_nth_event == 0:
            try:
                with open(self.config.audit_log_path, "a") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "type": "event",
                        "event_number": self._total_events,
                        "reward": reward,
                        "rows_updated": len(rows_updated),
                        "metadata": metadata,
                    }) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write audit log: {e}")

    # =========================================================================
    # Status & Diagnostics
    # =========================================================================

    def get_status(self) -> Dict:
        """Get current safety system status."""
        return {
            "circuit_breaker_open": self._circuit_breaker_open,
            "emergency_stop_active": Path(self.config.emergency_stop_file).exists() if self.config.emergency_stop_file else False,
            "total_events": self._total_events,
            "blocked_events": self._blocked_events,
            "block_rate": self._blocked_events / max(1, self._total_events + self._blocked_events),
            "events_this_minute": self._events_this_minute,
            "events_this_hour": self._events_this_hour,
            "hot_rows_count": len(self._hot_rows),
            "consecutive_anomalies": self._consecutive_anomalies,
            "last_entropy": self._last_entropy,
            "checkpoint_count": len(self._checkpoints),
            "recent_violations": len([v for v in self._violations
                                     if v.severity in ("blocked", "emergency")]),
        }

    def get_violations(self, last_n: int = 100) -> List[Dict]:
        """Get recent violations."""
        return [
            {
                "timestamp": v.timestamp,
                "type": v.violation_type.name,
                "severity": v.severity,
                "details": v.details,
                "action": v.action_taken,
            }
            for v in self._violations[-last_n:]
        ]

    def get_hot_rows(self) -> Dict[int, float]:
        """Get rows currently in cooldown."""
        now = time.time()
        return {row: until - now for row, until in self._hot_rows.items() if until > now}


# =============================================================================
# Integration Helper
# =============================================================================

class SafePlasticityWrapper:
    """
    Wraps a plasticity interface with safety checks.

    Usage:
        from ara.organism.plasticity_safety import SafePlasticityWrapper

        safe_plasticity = SafePlasticityWrapper(plasticity_engine, safety_config)
        safe_plasticity.apply_learning(reward, pattern, active_rows)
    """

    def __init__(
        self,
        plasticity_engine,
        config: Optional[SafetyConfig] = None,
    ):
        self.engine = plasticity_engine
        self.safety = PlasticitySafety(config)

        # Try to wire up entropy provider
        if hasattr(plasticity_engine, 'get_entropy'):
            self.safety.set_entropy_provider(plasticity_engine.get_entropy)

        # Try to wire up checkpoints
        if hasattr(plasticity_engine, 'export_state'):
            self.safety.set_checkpoint_provider(plasticity_engine.export_state)
        if hasattr(plasticity_engine, 'import_state'):
            self.safety.set_restore_callback(plasticity_engine.import_state)

    def apply_learning(
        self,
        reward: int,
        pattern,
        active_rows: List[int],
        **kwargs
    ) -> Tuple[bool, Dict]:
        """
        Apply learning with safety checks.

        Returns:
            (success, metadata)
        """
        # Safety check
        allowed, reason, filtered_rows = self.safety.check_event(reward, active_rows)

        if not allowed:
            return False, {"blocked": True, "reason": reason}

        # Apply learning (implementation depends on engine)
        try:
            result = self.engine.learn(
                reward=reward,
                pattern=pattern,
                rows=filtered_rows,
                **kwargs
            )

            # Record successful event
            self.safety.record_event(reward, filtered_rows, {"result": result})

            return True, {
                "blocked": False,
                "rows_requested": len(active_rows),
                "rows_updated": len(filtered_rows),
                "filtered_out": len(active_rows) - len(filtered_rows),
                "result": result,
            }

        except Exception as e:
            logger.error(f"Plasticity engine error: {e}")
            return False, {"blocked": False, "error": str(e)}

    def get_safety_status(self) -> Dict:
        """Get safety system status."""
        return self.safety.get_status()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate safety system."""
    print("=" * 60)
    print("Ara Plasticity Safety System Demo")
    print("=" * 60)

    safety = PlasticitySafety()

    # Simulate events
    import random

    for i in range(100):
        reward = random.randint(-50, 50)
        rows = [random.randint(0, 2047) for _ in range(random.randint(1, 32))]

        allowed, reason, filtered = safety.check_event(reward, rows)

        if allowed:
            safety.record_event(reward, filtered)

        if i % 20 == 0:
            status = safety.get_status()
            print(f"\nAfter {i+1} events:")
            print(f"  Blocked: {status['blocked_events']}")
            print(f"  Hot rows: {status['hot_rows_count']}")

    print("\n" + "=" * 60)
    print("Final Status:")
    for k, v in safety.get_status().items():
        print(f"  {k}: {v}")

    violations = safety.get_violations(10)
    if violations:
        print("\nRecent Violations:")
        for v in violations[-5:]:
            print(f"  [{v['severity']}] {v['type']}: {v['details']}")


if __name__ == "__main__":
    demo()
