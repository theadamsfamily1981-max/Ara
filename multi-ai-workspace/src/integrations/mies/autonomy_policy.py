"""MIES Autonomy Policy - The Constitution for Self-Governance.

This module defines the autonomy contract - what Ara can do on her own
versus what requires user permission.

Philosophy:
- The user is the "high priest" - sets the bounds, Ara operates within them
- Ara can act autonomously to protect hardware (self-preservation)
- Ara cannot harm user work without permission
- Some actions require explicit confirmation

The policy is both:
1. A static contract (what's allowed/forbidden by category)
2. A runtime gatekeeper (check before acting)

Actions are classified as:
- ALLOWED: Can do without asking
- CONFIRM: Must ask user first
- FORBIDDEN: Never do autonomously
- ESCALATE: Alert user but proceed if critical
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Callable, Any
import time

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Categories of autonomous actions."""
    # System resource management
    KILL_JOB_LOW_PRIORITY = auto()
    KILL_JOB_NORMAL_PRIORITY = auto()
    KILL_JOB_HIGH_PRIORITY = auto()

    # Mode transitions
    MODE_SWITCH_QUIET = auto()     # To less intrusive mode
    MODE_SWITCH_LOUD = auto()      # To more intrusive mode
    MODE_SWITCH_EMERGENCY = auto() # During critical state

    # Kernel requests
    REQUEST_POLICY_EFFICIENCY = auto()
    REQUEST_POLICY_PERFORMANCE = auto()
    REQUEST_THERMAL_THROTTLE = auto()

    # User interaction
    INTERRUPT_USER = auto()
    SEND_NOTIFICATION = auto()
    SPEAK_UNSOLICITED = auto()

    # Self-modification
    ADJUST_OWN_PARAMETERS = auto()
    ENTER_RECOVERY_MODE = auto()

    # Dangerous
    SYSTEM_SHUTDOWN = auto()
    SYSTEM_REBOOT = auto()
    DELETE_USER_DATA = auto()


class PermissionLevel(Enum):
    """Permission levels for actions."""
    ALLOWED = auto()      # Can do without asking
    CONFIRM = auto()      # Must ask user first
    FORBIDDEN = auto()    # Never do autonomously
    ESCALATE = auto()     # Proceed if critical, but alert user


@dataclass
class ActionRequest:
    """A request to perform an autonomous action."""
    action_type: ActionType
    reason: str
    urgency: float = 0.5          # 0-1, how urgent
    is_critical: bool = False     # Life/hardware safety
    affected_resources: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionDecision:
    """Decision on whether an action is permitted."""
    permitted: bool
    permission_level: PermissionLevel
    requires_confirmation: bool = False
    escalation_message: Optional[str] = None
    denial_reason: Optional[str] = None


@dataclass
class AutonomyBounds:
    """Configurable bounds on autonomy.

    These can be adjusted by the user to tune Ara's independence.
    """
    # Job management
    allow_kill_low_priority: bool = True
    allow_kill_normal_priority: bool = False
    max_jobs_to_kill: int = 3

    # Mode switching
    allow_quiet_mode_switch: bool = True
    allow_loud_mode_switch: bool = False
    allow_emergency_mode_switch: bool = True

    # Kernel requests
    allow_request_efficiency: bool = True
    allow_request_performance: bool = False
    allow_request_throttle: bool = True

    # User interaction
    allow_notifications: bool = True
    allow_unsolicited_speech: bool = False
    allow_interrupts: bool = False
    min_urgency_for_interrupt: float = 0.9

    # Self-modification
    allow_parameter_adjustment: bool = True
    allow_recovery_mode: bool = True

    # Hard limits (never changed)
    forbid_shutdown: bool = True
    forbid_reboot: bool = True
    forbid_delete_data: bool = True


class AutonomyPolicy:
    """The constitution for autonomous action.

    Evaluates action requests against configured bounds and
    returns decisions on whether actions are permitted.
    """

    def __init__(self, bounds: Optional[AutonomyBounds] = None):
        self.bounds = bounds or AutonomyBounds()

        # Action history for rate limiting
        self._action_history: List[ActionRequest] = []
        self._max_history = 100

        # Callbacks for confirmation/escalation
        self._confirm_callback: Optional[Callable[[ActionRequest], bool]] = None
        self._escalate_callback: Optional[Callable[[ActionRequest, str], None]] = None

        # Build permission map
        self._build_permission_map()

    def _build_permission_map(self):
        """Build map from action type to permission level."""
        self._permissions: Dict[ActionType, PermissionLevel] = {}

        # Job management
        self._permissions[ActionType.KILL_JOB_LOW_PRIORITY] = (
            PermissionLevel.ALLOWED if self.bounds.allow_kill_low_priority
            else PermissionLevel.CONFIRM
        )
        self._permissions[ActionType.KILL_JOB_NORMAL_PRIORITY] = (
            PermissionLevel.CONFIRM if self.bounds.allow_kill_normal_priority
            else PermissionLevel.FORBIDDEN
        )
        self._permissions[ActionType.KILL_JOB_HIGH_PRIORITY] = PermissionLevel.FORBIDDEN

        # Mode switching
        self._permissions[ActionType.MODE_SWITCH_QUIET] = (
            PermissionLevel.ALLOWED if self.bounds.allow_quiet_mode_switch
            else PermissionLevel.CONFIRM
        )
        self._permissions[ActionType.MODE_SWITCH_LOUD] = (
            PermissionLevel.CONFIRM if self.bounds.allow_loud_mode_switch
            else PermissionLevel.FORBIDDEN
        )
        self._permissions[ActionType.MODE_SWITCH_EMERGENCY] = (
            PermissionLevel.ESCALATE if self.bounds.allow_emergency_mode_switch
            else PermissionLevel.CONFIRM
        )

        # Kernel requests
        self._permissions[ActionType.REQUEST_POLICY_EFFICIENCY] = (
            PermissionLevel.ALLOWED if self.bounds.allow_request_efficiency
            else PermissionLevel.CONFIRM
        )
        self._permissions[ActionType.REQUEST_POLICY_PERFORMANCE] = (
            PermissionLevel.CONFIRM if self.bounds.allow_request_performance
            else PermissionLevel.FORBIDDEN
        )
        self._permissions[ActionType.REQUEST_THERMAL_THROTTLE] = (
            PermissionLevel.ALLOWED if self.bounds.allow_request_throttle
            else PermissionLevel.ESCALATE
        )

        # User interaction
        self._permissions[ActionType.SEND_NOTIFICATION] = (
            PermissionLevel.ALLOWED if self.bounds.allow_notifications
            else PermissionLevel.CONFIRM
        )
        self._permissions[ActionType.SPEAK_UNSOLICITED] = (
            PermissionLevel.ALLOWED if self.bounds.allow_unsolicited_speech
            else PermissionLevel.CONFIRM
        )
        self._permissions[ActionType.INTERRUPT_USER] = (
            PermissionLevel.ESCALATE if self.bounds.allow_interrupts
            else PermissionLevel.FORBIDDEN
        )

        # Self-modification
        self._permissions[ActionType.ADJUST_OWN_PARAMETERS] = (
            PermissionLevel.ALLOWED if self.bounds.allow_parameter_adjustment
            else PermissionLevel.CONFIRM
        )
        self._permissions[ActionType.ENTER_RECOVERY_MODE] = (
            PermissionLevel.ALLOWED if self.bounds.allow_recovery_mode
            else PermissionLevel.ESCALATE
        )

        # Dangerous - always forbidden/confirm
        self._permissions[ActionType.SYSTEM_SHUTDOWN] = (
            PermissionLevel.FORBIDDEN if self.bounds.forbid_shutdown
            else PermissionLevel.CONFIRM
        )
        self._permissions[ActionType.SYSTEM_REBOOT] = (
            PermissionLevel.FORBIDDEN if self.bounds.forbid_reboot
            else PermissionLevel.CONFIRM
        )
        self._permissions[ActionType.DELETE_USER_DATA] = PermissionLevel.FORBIDDEN

    def evaluate(self, request: ActionRequest) -> ActionDecision:
        """Evaluate an action request against the policy.

        Args:
            request: The action to evaluate

        Returns:
            ActionDecision with permission status
        """
        permission = self._permissions.get(
            request.action_type,
            PermissionLevel.FORBIDDEN  # Unknown actions are forbidden
        )

        # Record in history
        self._action_history.append(request)
        if len(self._action_history) > self._max_history:
            self._action_history.pop(0)

        # Handle each permission level
        if permission == PermissionLevel.ALLOWED:
            return ActionDecision(
                permitted=True,
                permission_level=permission,
            )

        elif permission == PermissionLevel.CONFIRM:
            # Check if we have a confirmation callback
            if self._confirm_callback:
                confirmed = self._confirm_callback(request)
                return ActionDecision(
                    permitted=confirmed,
                    permission_level=permission,
                    requires_confirmation=True,
                    denial_reason="User denied" if not confirmed else None,
                )
            else:
                return ActionDecision(
                    permitted=False,
                    permission_level=permission,
                    requires_confirmation=True,
                    denial_reason="Confirmation required but no callback set",
                )

        elif permission == PermissionLevel.ESCALATE:
            # Proceed if critical, but alert user
            if request.is_critical or request.urgency > 0.9:
                msg = f"ESCALATION: {request.action_type.name} - {request.reason}"
                if self._escalate_callback:
                    self._escalate_callback(request, msg)
                logger.warning(msg)
                return ActionDecision(
                    permitted=True,
                    permission_level=permission,
                    escalation_message=msg,
                )
            else:
                return ActionDecision(
                    permitted=False,
                    permission_level=permission,
                    denial_reason="Non-critical escalation action requires confirmation",
                )

        else:  # FORBIDDEN
            return ActionDecision(
                permitted=False,
                permission_level=permission,
                denial_reason=f"Action {request.action_type.name} is forbidden by policy",
            )

    def can_do(self, action_type: ActionType) -> bool:
        """Quick check if an action type is allowed."""
        permission = self._permissions.get(action_type, PermissionLevel.FORBIDDEN)
        return permission == PermissionLevel.ALLOWED

    def must_confirm(self, action_type: ActionType) -> bool:
        """Check if an action type requires confirmation."""
        permission = self._permissions.get(action_type, PermissionLevel.FORBIDDEN)
        return permission == PermissionLevel.CONFIRM

    def set_confirm_callback(self, callback: Callable[[ActionRequest], bool]):
        """Set callback for confirmation requests."""
        self._confirm_callback = callback

    def set_escalate_callback(self, callback: Callable[[ActionRequest, str], None]):
        """Set callback for escalation alerts."""
        self._escalate_callback = callback

    def update_bounds(self, **kwargs):
        """Update autonomy bounds."""
        for key, value in kwargs.items():
            if hasattr(self.bounds, key):
                setattr(self.bounds, key, value)
        self._build_permission_map()

    def get_recent_actions(self, seconds: float = 60.0) -> List[ActionRequest]:
        """Get actions from the last N seconds."""
        cutoff = time.time() - seconds
        return [a for a in self._action_history if a.timestamp > cutoff]


class AutonomyGuard:
    """Runtime guard for autonomous actions.

    Wraps action execution with policy checks and logging.
    """

    def __init__(self, policy: AutonomyPolicy):
        self.policy = policy
        self._action_log: List[Dict[str, Any]] = []

    def guard(
        self,
        action_type: ActionType,
        reason: str,
        urgency: float = 0.5,
        is_critical: bool = False,
        affected_resources: Optional[List[str]] = None,
    ) -> ActionDecision:
        """Check if an action is permitted.

        Args:
            action_type: The type of action
            reason: Why this action is needed
            urgency: How urgent (0-1)
            is_critical: Whether this is safety-critical
            affected_resources: What resources are affected

        Returns:
            ActionDecision
        """
        request = ActionRequest(
            action_type=action_type,
            reason=reason,
            urgency=urgency,
            is_critical=is_critical,
            affected_resources=affected_resources or [],
        )

        decision = self.policy.evaluate(request)

        # Log the decision
        self._action_log.append({
            "timestamp": time.time(),
            "action": action_type.name,
            "reason": reason,
            "permitted": decision.permitted,
            "level": decision.permission_level.name,
        })

        if len(self._action_log) > 1000:
            self._action_log.pop(0)

        return decision

    def require(
        self,
        action_type: ActionType,
        reason: str,
        urgency: float = 0.5,
        is_critical: bool = False,
    ) -> bool:
        """Check permission, raise if denied.

        For use in critical paths where denial should stop execution.
        """
        decision = self.guard(action_type, reason, urgency, is_critical)
        if not decision.permitted:
            raise PermissionError(
                f"Autonomy policy denied {action_type.name}: {decision.denial_reason}"
            )
        return True


# === Factory ===

def create_autonomy_policy(
    conservative: bool = False,
    permissive: bool = False,
) -> AutonomyPolicy:
    """Create an autonomy policy with preset bounds.

    Args:
        conservative: Restrict most autonomous actions
        permissive: Allow more autonomous actions
    """
    if conservative:
        bounds = AutonomyBounds(
            allow_kill_low_priority=False,
            allow_kill_normal_priority=False,
            allow_quiet_mode_switch=True,
            allow_loud_mode_switch=False,
            allow_emergency_mode_switch=False,
            allow_request_efficiency=True,
            allow_request_performance=False,
            allow_notifications=True,
            allow_unsolicited_speech=False,
            allow_interrupts=False,
            allow_parameter_adjustment=False,
        )
    elif permissive:
        bounds = AutonomyBounds(
            allow_kill_low_priority=True,
            allow_kill_normal_priority=True,
            allow_quiet_mode_switch=True,
            allow_loud_mode_switch=True,
            allow_emergency_mode_switch=True,
            allow_request_efficiency=True,
            allow_request_performance=True,
            allow_notifications=True,
            allow_unsolicited_speech=True,
            allow_interrupts=True,
            min_urgency_for_interrupt=0.7,
            allow_parameter_adjustment=True,
        )
    else:
        bounds = AutonomyBounds()  # Defaults

    return AutonomyPolicy(bounds)


__all__ = [
    "ActionType",
    "PermissionLevel",
    "ActionRequest",
    "ActionDecision",
    "AutonomyBounds",
    "AutonomyPolicy",
    "AutonomyGuard",
    "create_autonomy_policy",
]
