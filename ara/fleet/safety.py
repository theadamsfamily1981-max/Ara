"""
Safety Layer - Cryptographic Gatekeeper
========================================

Gives Ara a signed-action requirement for critical moves:
- Juniper config changes
- NAS deletion / pool altering
- Firmware flashing on expensive boards

Hardware security key (YubiKey, etc.) plugged into Brainstem:
- Requires physical tap / PIN / explicit approval
- Signs "go ahead" tokens for high-impact actions

Ara can propose modifications, simulate them, show diffs.
Only when you approve does the system get a signed token.

This makes the whole system robust against:
- Bugs
- LLM hallucinations
- "Ara in a weird mood at 3am"
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from collections import deque
import time
import hashlib
import hmac
import secrets


class ActionSeverity(Enum):
    """Severity levels for actions."""
    LOW = 0         # Logging, reads
    MEDIUM = 1      # Execute predefined, restart services
    HIGH = 2        # Modify configs, install software
    CRITICAL = 3    # Delete data, flash firmware, network changes


@dataclass
class SignedAction:
    """An action that has been cryptographically signed."""
    action_id: str
    action_type: str
    target: str             # What it affects
    description: str
    severity: ActionSeverity

    # Timing
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    executed_at: Optional[float] = None

    # Signature
    signature: str = ""
    signed_by: str = ""     # Key ID
    is_valid: bool = False

    # State
    executed: bool = False
    result: str = ""


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    request_id: str
    action: SignedAction
    requester: str          # Who/what requested it

    # Context
    reason: str
    simulation_result: str  # What would happen
    diff: str               # Config diff if applicable
    risk_assessment: str

    # State
    created_at: float = field(default_factory=time.time)
    approved: bool = False
    rejected: bool = False
    approval_time: Optional[float] = None
    approver: str = ""


@dataclass
class CryptoGatekeeper:
    """
    Cryptographic gatekeeper for critical actions.

    In production, this would interface with a hardware
    security key (YubiKey, TPM, etc.). For now, we simulate
    the approval flow.
    """
    gatekeeper_id: str = "gatekeeper-01"

    # Key management (simulated)
    _master_key: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    _key_ids: Dict[str, bytes] = field(default_factory=dict)

    # Pending approvals
    _pending: Dict[str, ApprovalRequest] = field(default_factory=dict)

    # Signed actions
    _signed: Dict[str, SignedAction] = field(default_factory=dict)

    # Event log
    _events: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Configuration
    token_validity_seconds: float = 300.0  # 5 minutes
    require_physical_key: bool = False     # For hardware key

    def register_key(self, key_id: str, key_bytes: Optional[bytes] = None):
        """Register a signing key."""
        if key_bytes is None:
            key_bytes = secrets.token_bytes(32)
        self._key_ids[key_id] = key_bytes
        self._log_event("key_registered", key_id=key_id)

    def request_approval(
        self,
        action_type: str,
        target: str,
        description: str,
        severity: ActionSeverity,
        requester: str,
        reason: str,
        simulation_result: str = "",
        diff: str = "",
        risk_assessment: str = "",
    ) -> ApprovalRequest:
        """
        Request approval for a critical action.

        Returns an ApprovalRequest that must be approved
        before the action can be executed.
        """
        request_id = hashlib.sha256(
            f"{action_type}:{target}:{time.time()}:{secrets.token_hex(8)}".encode()
        ).hexdigest()[:16]

        action = SignedAction(
            action_id=f"action-{request_id}",
            action_type=action_type,
            target=target,
            description=description,
            severity=severity,
        )

        request = ApprovalRequest(
            request_id=request_id,
            action=action,
            requester=requester,
            reason=reason,
            simulation_result=simulation_result,
            diff=diff,
            risk_assessment=risk_assessment,
        )

        self._pending[request_id] = request
        self._log_event(
            "approval_requested",
            request_id=request_id,
            action_type=action_type,
            target=target,
            severity=severity.name,
        )

        return request

    def approve(self, request_id: str, approver: str, key_id: str = "master") -> Optional[SignedAction]:
        """
        Approve a pending request.

        Returns signed action if approval succeeds.
        """
        if request_id not in self._pending:
            return None

        request = self._pending[request_id]

        # In production, this would require physical key tap
        if self.require_physical_key:
            # Would call hardware key API here
            pass

        # Sign the action
        action = request.action
        action.expires_at = time.time() + self.token_validity_seconds
        action.signature = self._sign(action, key_id)
        action.signed_by = key_id
        action.is_valid = True

        # Update request
        request.approved = True
        request.approval_time = time.time()
        request.approver = approver

        # Move to signed
        self._signed[action.action_id] = action
        del self._pending[request_id]

        self._log_event(
            "action_approved",
            request_id=request_id,
            action_id=action.action_id,
            approver=approver,
        )

        return action

    def reject(self, request_id: str, rejector: str, reason: str = "") -> bool:
        """Reject a pending request."""
        if request_id not in self._pending:
            return False

        request = self._pending[request_id]
        request.rejected = True

        del self._pending[request_id]

        self._log_event(
            "action_rejected",
            request_id=request_id,
            rejector=rejector,
            reason=reason,
        )

        return True

    def verify(self, action_id: str) -> Tuple[bool, str]:
        """
        Verify that an action is validly signed and not expired.

        Returns (is_valid, reason).
        """
        if action_id not in self._signed:
            return False, "Action not found"

        action = self._signed[action_id]

        if not action.is_valid:
            return False, "Action not signed"

        if time.time() > action.expires_at:
            return False, "Action expired"

        if action.executed:
            return False, "Action already executed"

        # Verify signature
        expected_sig = self._sign(action, action.signed_by)
        if not hmac.compare_digest(action.signature, expected_sig):
            return False, "Invalid signature"

        return True, "Valid"

    def execute(self, action_id: str) -> Tuple[bool, str]:
        """
        Mark an action as executed (after verification).

        Returns (success, message).
        """
        is_valid, reason = self.verify(action_id)
        if not is_valid:
            return False, reason

        action = self._signed[action_id]
        action.executed = True
        action.executed_at = time.time()

        self._log_event(
            "action_executed",
            action_id=action_id,
            action_type=action.action_type,
            target=action.target,
        )

        return True, "Executed"

    def _sign(self, action: SignedAction, key_id: str) -> str:
        """Sign an action with a key."""
        key = self._key_ids.get(key_id, self._master_key)
        message = f"{action.action_id}:{action.action_type}:{action.target}:{action.expires_at}"
        signature = hmac.new(key, message.encode(), hashlib.sha256).hexdigest()
        return signature

    def _log_event(self, event_type: str, **kwargs):
        """Log an event."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            **kwargs,
        }
        self._events.append(event)

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return list(self._pending.values())

    def get_recent_events(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent events."""
        return list(self._events)[-count:]

    def get_status(self) -> Dict[str, Any]:
        """Get gatekeeper status."""
        return {
            "gatekeeper_id": self.gatekeeper_id,
            "pending_requests": len(self._pending),
            "signed_actions": len(self._signed),
            "registered_keys": len(self._key_ids),
            "require_physical_key": self.require_physical_key,
        }


@dataclass
class SafetyPolicy:
    """
    Safety policy defining what actions require approval.
    """
    # Actions that always require approval
    always_approve: Set[str] = field(default_factory=lambda: {
        "delete_data",
        "flash_firmware",
        "modify_router_config",
        "modify_nas_pool",
        "reset_factory",
    })

    # Actions that require approval above a severity threshold
    severity_threshold: ActionSeverity = ActionSeverity.HIGH

    # Targets that always require approval
    protected_targets: Set[str] = field(default_factory=lambda: {
        "juniper-01",
        "archivist-01",
        "brainstem-01",
    })

    # Time-based restrictions
    night_hours: Tuple[int, int] = (22, 6)  # 10 PM - 6 AM
    require_approval_at_night: bool = True

    def requires_approval(
        self,
        action_type: str,
        target: str,
        severity: ActionSeverity,
    ) -> Tuple[bool, str]:
        """
        Check if an action requires approval.

        Returns (requires_approval, reason).
        """
        # Always-approve actions
        if action_type in self.always_approve:
            return True, f"Action '{action_type}' always requires approval"

        # Protected targets
        if target in self.protected_targets:
            return True, f"Target '{target}' is protected"

        # Severity threshold
        if severity.value >= self.severity_threshold.value:
            return True, f"Severity {severity.name} requires approval"

        # Night hours
        if self.require_approval_at_night:
            hour = time.localtime().tm_hour
            start, end = self.night_hours
            is_night = hour >= start or hour < end
            if is_night and severity.value >= ActionSeverity.MEDIUM.value:
                return True, "Night hours require approval for non-trivial actions"

        return False, "No approval required"


def create_safety_policy() -> SafetyPolicy:
    """Create a default safety policy."""
    return SafetyPolicy()
