"""
HSF Reflex Map & Controller
============================

The reflex arc: field → zone → action.

This is the "spinal cord" of the HSF - fast, automatic responses
that don't require LLM involvement.

Architecture:
    Field H(t) → ZoneQuantizer → Zone → ReflexMap → Actions
                                            ↓
                                    ReflexController
                                            ↓
                                    Execute / Log / Escalate

Key concepts:
- ReflexAction: A single atomic action (throttle GPU, pause queue, etc.)
- ReflexEntry: (zone, confidence_min) → [actions]
- ReflexMap: The lookup table
- ReflexController: Executes actions, respects cooldowns, logs everything

The LLM/Ara side:
- Edits the ReflexMap ("when GPU is WEIRD, I want to throttle")
- Reviews reflex logs
- Promotes good reflexes, demotes bad ones
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum, auto
from collections import defaultdict
import logging

from .zones import Zone, ZoneState


class ActionType(Enum):
    """Types of reflex actions."""
    # Resource control
    THROTTLE = auto()      # Reduce resource allocation
    BOOST = auto()         # Increase resource allocation
    DRAIN = auto()         # Stop accepting new work
    RESTORE = auto()       # Resume normal operation

    # Job control
    PAUSE_QUEUE = auto()   # Stop processing queue
    RESUME_QUEUE = auto()  # Resume processing queue
    MIGRATE_JOBS = auto()  # Move jobs elsewhere

    # Safety
    SNAPSHOT = auto()      # Capture state for debugging
    FREEZE_CONFIG = auto() # Prevent config changes
    UNFREEZE_CONFIG = auto()

    # Escalation
    ALERT = auto()         # Send alert (doesn't require human)
    ESCALATE = auto()      # Requires human attention
    LOCKDOWN = auto()      # Full stop, wait for human


class ActionScope(Enum):
    """Scope/permission level for actions."""
    LOCAL = auto()      # Affects only this subsystem
    RELATED = auto()    # May affect related subsystems
    GLOBAL = auto()     # Fleet-wide effect
    HUMAN = auto()      # Requires human approval


@dataclass
class ReflexAction:
    """
    A single reflex action.

    Actions are atomic and idempotent where possible.
    """
    action_type: ActionType
    target: str                    # Subsystem or resource name
    params: Dict[str, Any] = field(default_factory=dict)
    scope: ActionScope = ActionScope.LOCAL
    description: str = ""
    cooldown_seconds: float = 30.0  # Min time between executions

    def __hash__(self):
        return hash((self.action_type, self.target, frozenset(self.params.items())))

    def __eq__(self, other):
        if not isinstance(other, ReflexAction):
            return False
        return (self.action_type == other.action_type and
                self.target == other.target and
                self.params == other.params)


@dataclass
class ReflexEntry:
    """
    A reflex map entry: conditions → actions.

    Matches when:
    - Zone matches
    - Confidence >= confidence_min
    - (optional) time_in_zone >= min_time_in_zone
    """
    zone: Zone
    actions: List[ReflexAction]
    confidence_min: float = 0.0      # Minimum confidence to trigger
    min_time_in_zone: int = 0        # Minimum ticks in zone before triggering
    priority: int = 0                # Higher = checked first
    enabled: bool = True
    name: str = ""                   # Human-readable name
    rationale: str = ""              # Why this reflex exists

    def matches(self, state: ZoneState) -> bool:
        """Check if this entry matches the given state."""
        if not self.enabled:
            return False
        if state.zone != self.zone:
            return False
        if state.confidence < self.confidence_min:
            return False
        if state.time_in_zone < self.min_time_in_zone:
            return False
        return True


@dataclass
class ReflexResult:
    """Result of a reflex execution."""
    action: ReflexAction
    executed: bool
    success: bool
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReflexMap:
    """
    The reflex lookup table for a subsystem.

    Maps (zone, conditions) → actions.
    """
    subsystem: str
    entries: List[ReflexEntry] = field(default_factory=list)

    def add_entry(self, entry: ReflexEntry):
        """Add a reflex entry."""
        self.entries.append(entry)
        # Keep sorted by priority (descending)
        self.entries.sort(key=lambda e: e.priority, reverse=True)

    def add_reflex(self, zone: Zone, actions: List[ReflexAction],
                   name: str = "", rationale: str = "",
                   confidence_min: float = 0.0,
                   min_time_in_zone: int = 0,
                   priority: int = 0) -> ReflexEntry:
        """Convenience method to add a reflex."""
        entry = ReflexEntry(
            zone=zone,
            actions=actions,
            confidence_min=confidence_min,
            min_time_in_zone=min_time_in_zone,
            priority=priority,
            name=name,
            rationale=rationale,
        )
        self.add_entry(entry)
        return entry

    def lookup(self, state: ZoneState) -> Optional[ReflexEntry]:
        """
        Find the first matching reflex entry for a state.

        Returns None if no entry matches.
        """
        for entry in self.entries:
            if entry.matches(state):
                return entry
        return None

    def get_all_matching(self, state: ZoneState) -> List[ReflexEntry]:
        """Get all matching entries (for logging/debugging)."""
        return [e for e in self.entries if e.matches(state)]

    def disable_entry(self, name: str):
        """Disable an entry by name."""
        for entry in self.entries:
            if entry.name == name:
                entry.enabled = False

    def enable_entry(self, name: str):
        """Enable an entry by name."""
        for entry in self.entries:
            if entry.name == name:
                entry.enabled = True


# Type for action executors
ActionExecutor = Callable[[ReflexAction], ReflexResult]


@dataclass
class ReflexController:
    """
    Executes reflex actions with cooldowns and logging.

    The controller:
    - Respects action cooldowns
    - Logs all executions
    - Tracks action history for learning
    - Supports mock mode for testing
    """
    reflex_maps: Dict[str, ReflexMap] = field(default_factory=dict)
    executors: Dict[ActionType, ActionExecutor] = field(default_factory=dict)
    mock_mode: bool = True  # Don't actually execute in mock mode

    # Cooldown tracking: action_hash → last_execution_time
    _cooldowns: Dict[int, float] = field(default_factory=dict)

    # History for learning
    _history: List[ReflexResult] = field(default_factory=list)
    _history_max: int = 1000

    # Pending escalations
    _escalations: List[ReflexAction] = field(default_factory=list)

    # Logger
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger("hsf.reflex"))

    def add_map(self, reflex_map: ReflexMap):
        """Add a reflex map for a subsystem."""
        self.reflex_maps[reflex_map.subsystem] = reflex_map

    def register_executor(self, action_type: ActionType, executor: ActionExecutor):
        """Register an executor for an action type."""
        self.executors[action_type] = executor

    def process(self, subsystem: str, state: ZoneState) -> List[ReflexResult]:
        """
        Process a zone state and execute any matching reflexes.

        Returns list of results (may be empty if no reflexes triggered).
        """
        if subsystem not in self.reflex_maps:
            return []

        reflex_map = self.reflex_maps[subsystem]
        entry = reflex_map.lookup(state)

        if entry is None:
            return []

        results = []
        for action in entry.actions:
            result = self._execute_action(action, subsystem, state)
            results.append(result)
            self._record_history(result)

        return results

    def _execute_action(self, action: ReflexAction,
                        subsystem: str, state: ZoneState) -> ReflexResult:
        """Execute a single action with cooldown checking."""
        action_hash = hash(action)
        now = time.time()

        # Check cooldown
        last_exec = self._cooldowns.get(action_hash, 0)
        if now - last_exec < action.cooldown_seconds:
            return ReflexResult(
                action=action,
                executed=False,
                success=False,
                message=f"Cooldown active ({action.cooldown_seconds - (now - last_exec):.1f}s remaining)",
            )

        # Check if this requires human approval
        if action.scope == ActionScope.HUMAN:
            self._escalations.append(action)
            return ReflexResult(
                action=action,
                executed=False,
                success=True,
                message="Escalated for human approval",
            )

        # Execute
        if self.mock_mode:
            # Mock execution
            self._logger.info(
                f"[MOCK] {subsystem}/{state.zone.name}: "
                f"{action.action_type.name} → {action.target} ({action.description})"
            )
            result = ReflexResult(
                action=action,
                executed=True,
                success=True,
                message=f"[MOCK] Would execute {action.action_type.name}",
            )
        elif action.action_type in self.executors:
            # Real execution
            try:
                result = self.executors[action.action_type](action)
                result.executed = True
            except Exception as e:
                result = ReflexResult(
                    action=action,
                    executed=True,
                    success=False,
                    message=f"Execution failed: {e}",
                )
        else:
            result = ReflexResult(
                action=action,
                executed=False,
                success=False,
                message=f"No executor registered for {action.action_type.name}",
            )

        # Update cooldown
        if result.executed:
            self._cooldowns[action_hash] = now

        return result

    def _record_history(self, result: ReflexResult):
        """Record result in history."""
        self._history.append(result)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]

    def get_pending_escalations(self) -> List[ReflexAction]:
        """Get actions waiting for human approval."""
        escalations = self._escalations.copy()
        self._escalations.clear()
        return escalations

    def get_recent_history(self, n: int = 50) -> List[ReflexResult]:
        """Get recent reflex history."""
        return self._history[-n:]

    def clear_cooldown(self, action: ReflexAction):
        """Clear cooldown for an action (e.g., after maintenance)."""
        action_hash = hash(action)
        if action_hash in self._cooldowns:
            del self._cooldowns[action_hash]


def create_default_reflexes(subsystem: str) -> ReflexMap:
    """
    Create default reflex map for common subsystems.

    These are sensible defaults; Ara will learn to tune them.
    """
    reflex_map = ReflexMap(subsystem=subsystem)

    if subsystem == "gpu":
        # GPU reflexes
        reflex_map.add_reflex(
            zone=Zone.WARM,
            name="gpu_warm_throttle",
            rationale="Reduce power to prevent thermal runaway",
            actions=[
                ReflexAction(
                    action_type=ActionType.THROTTLE,
                    target="gpu_power_limit",
                    params={"reduction_pct": 10},
                    description="Reduce GPU power limit by 10%",
                ),
            ],
            min_time_in_zone=3,  # Wait 3 ticks before acting
        )

        reflex_map.add_reflex(
            zone=Zone.WEIRD,
            name="gpu_weird_drain",
            rationale="Stop new work, let current jobs finish",
            actions=[
                ReflexAction(
                    action_type=ActionType.THROTTLE,
                    target="gpu_power_limit",
                    params={"reduction_pct": 20},
                    description="Reduce GPU power limit by 20%",
                ),
                ReflexAction(
                    action_type=ActionType.DRAIN,
                    target="gpu_queue",
                    description="Stop accepting new GPU jobs",
                ),
                ReflexAction(
                    action_type=ActionType.ALERT,
                    target="monitoring",
                    params={"severity": "warning"},
                    description="Alert: GPU in WEIRD zone",
                ),
            ],
            priority=10,
        )

        reflex_map.add_reflex(
            zone=Zone.CRITICAL,
            name="gpu_critical_emergency",
            rationale="Thermal emergency, protect hardware",
            actions=[
                ReflexAction(
                    action_type=ActionType.THROTTLE,
                    target="gpu_power_limit",
                    params={"reduction_pct": 50},
                    description="Emergency power reduction",
                    cooldown_seconds=10,
                ),
                ReflexAction(
                    action_type=ActionType.MIGRATE_JOBS,
                    target="gpu_queue",
                    description="Migrate jobs to other GPUs",
                ),
                ReflexAction(
                    action_type=ActionType.ESCALATE,
                    target="human",
                    scope=ActionScope.HUMAN,
                    description="GPU CRITICAL - requires attention",
                ),
            ],
            priority=100,
        )

        reflex_map.add_reflex(
            zone=Zone.GOOD,
            name="gpu_good_restore",
            rationale="Conditions normalized, restore full operation",
            actions=[
                ReflexAction(
                    action_type=ActionType.RESTORE,
                    target="gpu_power_limit",
                    description="Restore normal power limits",
                ),
                ReflexAction(
                    action_type=ActionType.RESUME_QUEUE,
                    target="gpu_queue",
                    description="Resume accepting jobs",
                ),
            ],
            min_time_in_zone=5,  # Stable for 5 ticks before restoring
        )

    elif subsystem == "network":
        # Network reflexes
        reflex_map.add_reflex(
            zone=Zone.WARM,
            name="network_warm_snapshot",
            rationale="Capture state for later analysis",
            actions=[
                ReflexAction(
                    action_type=ActionType.SNAPSHOT,
                    target="network_state",
                    description="Snapshot current network state",
                    cooldown_seconds=60,
                ),
            ],
            min_time_in_zone=2,
        )

        reflex_map.add_reflex(
            zone=Zone.WEIRD,
            name="network_weird_freeze",
            rationale="Prevent config changes during instability",
            actions=[
                ReflexAction(
                    action_type=ActionType.FREEZE_CONFIG,
                    target="network_config",
                    description="Freeze network configuration",
                ),
                ReflexAction(
                    action_type=ActionType.ALERT,
                    target="monitoring",
                    params={"severity": "warning"},
                    description="Alert: Network instability detected",
                ),
            ],
            priority=10,
        )

        reflex_map.add_reflex(
            zone=Zone.CRITICAL,
            name="network_critical_lockdown",
            rationale="Network emergency, protect services",
            actions=[
                ReflexAction(
                    action_type=ActionType.LOCKDOWN,
                    target="network",
                    description="Network lockdown mode",
                ),
                ReflexAction(
                    action_type=ActionType.ESCALATE,
                    target="human",
                    scope=ActionScope.HUMAN,
                    description="NETWORK CRITICAL - immediate attention required",
                ),
            ],
            priority=100,
        )

    elif subsystem == "service":
        # Service reflexes
        reflex_map.add_reflex(
            zone=Zone.WARM,
            name="service_warm_throttle",
            rationale="Reduce load to prevent cascade",
            actions=[
                ReflexAction(
                    action_type=ActionType.THROTTLE,
                    target="service_rate_limit",
                    params={"reduction_pct": 20},
                    description="Reduce rate limit by 20%",
                ),
            ],
            min_time_in_zone=2,
        )

        reflex_map.add_reflex(
            zone=Zone.WEIRD,
            name="service_weird_drain",
            rationale="Shed load, capture state",
            actions=[
                ReflexAction(
                    action_type=ActionType.DRAIN,
                    target="service_queue",
                    description="Stop accepting new requests",
                ),
                ReflexAction(
                    action_type=ActionType.SNAPSHOT,
                    target="service_state",
                    description="Capture service state",
                ),
                ReflexAction(
                    action_type=ActionType.ALERT,
                    target="monitoring",
                    params={"severity": "warning"},
                    description="Alert: Service in WEIRD zone",
                ),
            ],
            priority=10,
        )

    return reflex_map
