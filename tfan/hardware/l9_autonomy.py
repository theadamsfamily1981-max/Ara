"""
L9 Autonomy: Staged Hardware Self-Modification

This module extends Autosynth with explicit autonomy stages for
hardware self-modification. The key insight: autonomy must be earned
through demonstrated safety.

Autonomy Stages:
- Stage A (ADVISOR): Propose + verify only, human approves/deploys
- Stage B (SANDBOX): Auto-deploy non-critical accelerators, sandboxed
- Stage C (PARTIAL): Broader autonomy after proven track record

Stage Progression:
- Start at Stage A (always)
- Progress to Stage B after N successful proposals + M days
- Progress to Stage C after demonstrated safety at Stage B

Safety Gates:
- Every stage has explicit "veto" capabilities
- Rollback always possible
- PGU verification required at all stages
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Tuple


# ============================================================
# Autonomy Stages
# ============================================================

class AutonomyStage(str, Enum):
    """Autonomy levels for hardware self-modification."""
    ADVISOR = "advisor"    # Stage A: Propose only, human decides
    SANDBOX = "sandbox"    # Stage B: Auto-deploy non-critical, sandboxed
    PARTIAL = "partial"    # Stage C: Broader autonomy with track record


class KernelCriticality(str, Enum):
    """How critical is a kernel to system operation?"""
    NON_CRITICAL = "non_critical"   # Can fail without system impact
    STANDARD = "standard"           # Normal operation, has fallback
    CRITICAL = "critical"           # Essential for operation
    SAFETY = "safety"               # Safety-critical, never auto-deploy


@dataclass
class AutonomyPolicy:
    """Policy governing autonomy progression and limits."""
    # Current stage
    current_stage: AutonomyStage = AutonomyStage.ADVISOR

    # Stage A → Stage B requirements
    stage_b_min_proposals: int = 10      # Min verified proposals
    stage_b_min_days: int = 7            # Min days at Stage A
    stage_b_success_rate: float = 0.9    # Min success rate

    # Stage B → Stage C requirements
    stage_c_min_deployments: int = 5     # Min successful deployments
    stage_c_min_days: int = 30           # Min days at Stage B
    stage_c_incident_free_days: int = 14 # Days without incident

    # Stage B limits
    sandbox_max_kernels: int = 3         # Max simultaneous sandboxed kernels
    sandbox_max_lut_per_kernel: int = 10000  # Resource limit
    sandbox_allowed_criticality: Set[KernelCriticality] = field(
        default_factory=lambda: {KernelCriticality.NON_CRITICAL}
    )

    # Stage C limits
    partial_allowed_criticality: Set[KernelCriticality] = field(
        default_factory=lambda: {
            KernelCriticality.NON_CRITICAL,
            KernelCriticality.STANDARD
        }
    )

    # Global safety
    require_fallback_path: bool = True   # Always need CPU fallback
    max_daily_deployments: int = 5       # Rate limit
    veto_timeout_ms: int = 5000          # Time for human veto


@dataclass
class AutonomyState:
    """Current state of the autonomy system."""
    stage: AutonomyStage
    stage_since: datetime

    # Stage A stats
    proposals_created: int = 0
    proposals_verified: int = 0
    proposals_rejected: int = 0

    # Stage B stats
    sandbox_deployments: int = 0
    sandbox_successes: int = 0
    sandbox_failures: int = 0
    current_sandbox_kernels: int = 0

    # Stage C stats
    full_deployments: int = 0
    incidents: int = 0
    last_incident: Optional[datetime] = None

    # Rate limiting
    deployments_today: int = 0
    last_deployment_date: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        total = self.proposals_verified + self.proposals_rejected
        if total == 0:
            return 0.0
        return self.proposals_verified / total

    @property
    def days_at_stage(self) -> int:
        """Days at current stage."""
        return (datetime.now() - self.stage_since).days

    @property
    def days_since_incident(self) -> Optional[int]:
        """Days since last incident."""
        if self.last_incident is None:
            return None
        return (datetime.now() - self.last_incident).days

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "stage_since": self.stage_since.isoformat(),
            "days_at_stage": self.days_at_stage,
            "proposals": {
                "created": self.proposals_created,
                "verified": self.proposals_verified,
                "rejected": self.proposals_rejected,
                "success_rate": self.success_rate
            },
            "sandbox": {
                "deployments": self.sandbox_deployments,
                "successes": self.sandbox_successes,
                "failures": self.sandbox_failures,
                "current_kernels": self.current_sandbox_kernels
            },
            "full": {
                "deployments": self.full_deployments,
                "incidents": self.incidents,
                "days_since_incident": self.days_since_incident
            },
            "rate_limit": {
                "today": self.deployments_today,
                "max": 5  # Policy default
            }
        }


# ============================================================
# Autonomy Controller
# ============================================================

class AutonomyController:
    """
    Controls the autonomy level of the hardware self-modification system.

    Key responsibilities:
    1. Track autonomy state and statistics
    2. Decide if actions are allowed at current stage
    3. Progress between stages based on track record
    4. Enforce safety gates and veto capabilities
    """

    def __init__(
        self,
        policy: Optional[AutonomyPolicy] = None,
        start_stage: AutonomyStage = AutonomyStage.ADVISOR
    ):
        self.policy = policy or AutonomyPolicy()

        self._state = AutonomyState(
            stage=start_stage,
            stage_since=datetime.now()
        )

        # Callbacks
        self._veto_callback: Optional[Callable[[str], bool]] = None
        self._stage_change_callback: Optional[Callable[[AutonomyStage, AutonomyStage], None]] = None

        # History
        self._action_history: List[Dict[str, Any]] = []
        self._stage_history: List[Dict[str, Any]] = []

    @property
    def stage(self) -> AutonomyStage:
        """Current autonomy stage."""
        return self._state.stage

    @property
    def state(self) -> AutonomyState:
        """Current autonomy state."""
        return self._state

    def set_veto_callback(self, callback: Callable[[str], bool]) -> None:
        """Set callback for veto decisions. Returns True if vetoed."""
        self._veto_callback = callback

    def set_stage_change_callback(
        self,
        callback: Callable[[AutonomyStage, AutonomyStage], None]
    ) -> None:
        """Set callback for stage changes."""
        self._stage_change_callback = callback

    # ========================================
    # Permission Checks
    # ========================================

    def can_propose(self) -> bool:
        """Can the system propose new kernels?"""
        # Always allowed - proposal is just information
        return True

    def can_auto_deploy(
        self,
        criticality: KernelCriticality,
        resource_usage: Dict[str, int]
    ) -> Tuple[bool, str]:
        """
        Check if auto-deployment is allowed.

        Args:
            criticality: How critical is this kernel
            resource_usage: Resource estimates (LUT, BRAM, DSP)

        Returns:
            (allowed, reason)
        """
        # Stage A: Never auto-deploy
        if self._state.stage == AutonomyStage.ADVISOR:
            return False, "Stage A (ADVISOR): Auto-deploy not allowed, human approval required"

        # Check rate limit
        if self._state.deployments_today >= self.policy.max_daily_deployments:
            return False, f"Rate limit: {self.policy.max_daily_deployments} deployments/day exceeded"

        # Stage B: Only non-critical, sandboxed
        if self._state.stage == AutonomyStage.SANDBOX:
            if criticality not in self.policy.sandbox_allowed_criticality:
                return False, f"Stage B (SANDBOX): Criticality {criticality.value} not allowed"

            if self._state.current_sandbox_kernels >= self.policy.sandbox_max_kernels:
                return False, f"Stage B (SANDBOX): Max {self.policy.sandbox_max_kernels} kernels reached"

            if resource_usage.get("LUT", 0) > self.policy.sandbox_max_lut_per_kernel:
                return False, f"Stage B (SANDBOX): LUT {resource_usage.get('LUT', 0)} exceeds limit"

            return True, "Stage B (SANDBOX): Auto-deploy allowed"

        # Stage C: Standard and non-critical
        if self._state.stage == AutonomyStage.PARTIAL:
            if criticality not in self.policy.partial_allowed_criticality:
                return False, f"Stage C (PARTIAL): Criticality {criticality.value} not allowed"

            return True, "Stage C (PARTIAL): Auto-deploy allowed"

        return False, "Unknown stage"

    def request_veto_window(self, action_id: str) -> bool:
        """
        Request a veto window before action.

        Returns True if action was vetoed.
        """
        if self._veto_callback:
            return self._veto_callback(action_id)
        return False  # No veto callback = not vetoed

    # ========================================
    # Action Recording
    # ========================================

    def record_proposal(self, verified: bool) -> None:
        """Record a proposal outcome."""
        self._state.proposals_created += 1
        if verified:
            self._state.proposals_verified += 1
        else:
            self._state.proposals_rejected += 1

        self._action_history.append({
            "action": "proposal",
            "verified": verified,
            "timestamp": datetime.now().isoformat()
        })

        # Check for stage progression
        self._check_stage_progression()

    def record_deployment(
        self,
        success: bool,
        sandbox: bool = True,
        kernel_id: Optional[str] = None
    ) -> None:
        """Record a deployment outcome."""
        # Update rate limit
        today = datetime.now().date()
        if (self._state.last_deployment_date is None or
            self._state.last_deployment_date.date() != today):
            self._state.deployments_today = 0
        self._state.deployments_today += 1
        self._state.last_deployment_date = datetime.now()

        if sandbox:
            self._state.sandbox_deployments += 1
            if success:
                self._state.sandbox_successes += 1
            else:
                self._state.sandbox_failures += 1
                self._record_incident("sandbox_deployment_failure", kernel_id)
        else:
            self._state.full_deployments += 1
            if not success:
                self._record_incident("full_deployment_failure", kernel_id)

        self._action_history.append({
            "action": "deployment",
            "success": success,
            "sandbox": sandbox,
            "kernel_id": kernel_id,
            "timestamp": datetime.now().isoformat()
        })

        # Check for stage progression
        self._check_stage_progression()

    def record_kernel_active(self) -> None:
        """Record a kernel becoming active in sandbox."""
        self._state.current_sandbox_kernels += 1

    def record_kernel_retired(self) -> None:
        """Record a kernel being retired from sandbox."""
        self._state.current_sandbox_kernels = max(0, self._state.current_sandbox_kernels - 1)

    def _record_incident(self, incident_type: str, details: Optional[str] = None) -> None:
        """Record a safety incident."""
        self._state.incidents += 1
        self._state.last_incident = datetime.now()

        self._action_history.append({
            "action": "incident",
            "type": incident_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

        # Incident may trigger stage regression
        self._check_stage_regression()

    # ========================================
    # Stage Progression
    # ========================================

    def _check_stage_progression(self) -> None:
        """Check if we should progress to a higher stage."""
        if self._state.stage == AutonomyStage.ADVISOR:
            self._check_progress_to_sandbox()
        elif self._state.stage == AutonomyStage.SANDBOX:
            self._check_progress_to_partial()

    def _check_progress_to_sandbox(self) -> None:
        """Check if we should progress from ADVISOR to SANDBOX."""
        policy = self.policy

        if self._state.proposals_verified < policy.stage_b_min_proposals:
            return
        if self._state.days_at_stage < policy.stage_b_min_days:
            return
        if self._state.success_rate < policy.stage_b_success_rate:
            return

        # All requirements met
        self._transition_to_stage(AutonomyStage.SANDBOX)

    def _check_progress_to_partial(self) -> None:
        """Check if we should progress from SANDBOX to PARTIAL."""
        policy = self.policy

        if self._state.sandbox_successes < policy.stage_c_min_deployments:
            return
        if self._state.days_at_stage < policy.stage_c_min_days:
            return

        days_since = self._state.days_since_incident
        if days_since is not None and days_since < policy.stage_c_incident_free_days:
            return

        # All requirements met
        self._transition_to_stage(AutonomyStage.PARTIAL)

    def _check_stage_regression(self) -> None:
        """Check if we should regress to a lower stage after incident."""
        # Multiple incidents → regress
        recent_incidents = sum(
            1 for h in self._action_history[-20:]
            if h.get("action") == "incident"
        )

        if recent_incidents >= 3:
            if self._state.stage == AutonomyStage.PARTIAL:
                self._transition_to_stage(AutonomyStage.SANDBOX)
            elif self._state.stage == AutonomyStage.SANDBOX:
                self._transition_to_stage(AutonomyStage.ADVISOR)

    def _transition_to_stage(self, new_stage: AutonomyStage) -> None:
        """Transition to a new autonomy stage."""
        old_stage = self._state.stage

        # Record transition
        self._stage_history.append({
            "from": old_stage.value,
            "to": new_stage.value,
            "timestamp": datetime.now().isoformat(),
            "state_snapshot": self._state.to_dict()
        })

        # Update state
        self._state.stage = new_stage
        self._state.stage_since = datetime.now()

        # Callback
        if self._stage_change_callback:
            self._stage_change_callback(old_stage, new_stage)

    # ========================================
    # Manual Controls
    # ========================================

    def force_stage(self, stage: AutonomyStage, reason: str) -> None:
        """Force a specific autonomy stage (admin override)."""
        self._stage_history.append({
            "from": self._state.stage.value,
            "to": stage.value,
            "timestamp": datetime.now().isoformat(),
            "reason": f"FORCED: {reason}",
            "state_snapshot": self._state.to_dict()
        })

        self._state.stage = stage
        self._state.stage_since = datetime.now()

    def emergency_stop(self) -> None:
        """Emergency stop: revert to ADVISOR stage immediately."""
        self.force_stage(AutonomyStage.ADVISOR, "Emergency stop triggered")

    # ========================================
    # Reporting
    # ========================================

    def get_status(self) -> Dict[str, Any]:
        """Get current autonomy status."""
        return {
            "stage": self._state.stage.value,
            "days_at_stage": self._state.days_at_stage,
            "state": self._state.to_dict(),
            "policy": {
                "stage_b_min_proposals": self.policy.stage_b_min_proposals,
                "stage_b_min_days": self.policy.stage_b_min_days,
                "stage_c_min_deployments": self.policy.stage_c_min_deployments,
                "stage_c_min_days": self.policy.stage_c_min_days
            },
            "progression": self._get_progression_status()
        }

    def _get_progression_status(self) -> Dict[str, Any]:
        """Get status toward next stage."""
        if self._state.stage == AutonomyStage.ADVISOR:
            return {
                "next_stage": "SANDBOX",
                "requirements": {
                    "proposals": f"{self._state.proposals_verified}/{self.policy.stage_b_min_proposals}",
                    "days": f"{self._state.days_at_stage}/{self.policy.stage_b_min_days}",
                    "success_rate": f"{self._state.success_rate:.1%}/{self.policy.stage_b_success_rate:.0%}"
                }
            }
        elif self._state.stage == AutonomyStage.SANDBOX:
            return {
                "next_stage": "PARTIAL",
                "requirements": {
                    "deployments": f"{self._state.sandbox_successes}/{self.policy.stage_c_min_deployments}",
                    "days": f"{self._state.days_at_stage}/{self.policy.stage_c_min_days}",
                    "incident_free": f"{self._state.days_since_incident or 'N/A'}/{self.policy.stage_c_incident_free_days} days"
                }
            }
        else:
            return {"next_stage": None, "status": "At maximum autonomy"}

    def explain_autonomy(self) -> str:
        """Generate natural language explanation of autonomy state."""
        lines = []
        lines.append(f"=== L9 Autonomy Status ===")
        lines.append(f"Current Stage: {self._state.stage.value.upper()}")
        lines.append(f"Days at stage: {self._state.days_at_stage}")
        lines.append("")

        if self._state.stage == AutonomyStage.ADVISOR:
            lines.append("ADVISOR Mode:")
            lines.append("  - Proposals: verify only, no auto-deploy")
            lines.append("  - Human approval required for all deployments")
            lines.append("")
            lines.append("Progress to SANDBOX:")
            lines.append(f"  - Verified proposals: {self._state.proposals_verified}/{self.policy.stage_b_min_proposals}")
            lines.append(f"  - Days required: {self._state.days_at_stage}/{self.policy.stage_b_min_days}")
            lines.append(f"  - Success rate: {self._state.success_rate:.1%} (need {self.policy.stage_b_success_rate:.0%})")

        elif self._state.stage == AutonomyStage.SANDBOX:
            lines.append("SANDBOX Mode:")
            lines.append(f"  - Auto-deploy: non-critical kernels only")
            lines.append(f"  - Max kernels: {self._state.current_sandbox_kernels}/{self.policy.sandbox_max_kernels}")
            lines.append(f"  - Deployments: {self._state.sandbox_successes} successful, {self._state.sandbox_failures} failed")
            lines.append("")
            lines.append("Progress to PARTIAL:")
            lines.append(f"  - Successful deployments: {self._state.sandbox_successes}/{self.policy.stage_c_min_deployments}")
            lines.append(f"  - Days required: {self._state.days_at_stage}/{self.policy.stage_c_min_days}")
            days_since = self._state.days_since_incident
            lines.append(f"  - Incident-free days: {days_since if days_since else 'N/A'}/{self.policy.stage_c_incident_free_days}")

        else:
            lines.append("PARTIAL Autonomy:")
            lines.append(f"  - Auto-deploy: non-critical and standard kernels")
            lines.append(f"  - Full deployments: {self._state.full_deployments}")
            lines.append(f"  - Incidents: {self._state.incidents}")

        return "\n".join(lines)


# ============================================================
# Integration with Autosynth
# ============================================================

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .autosynth import HLSProposal, AutosynthController


class L9AutosynthIntegration:
    """
    Integrates L9 Autonomy with Autosynth pipeline.

    Provides:
    - Permission checks before deployment
    - Automatic stage tracking
    - Veto window management
    """

    def __init__(
        self,
        autonomy: Optional[AutonomyController] = None,
        default_criticality: KernelCriticality = KernelCriticality.NON_CRITICAL
    ):
        self.autonomy = autonomy or AutonomyController()
        self.default_criticality = default_criticality

        # Track proposals by ID
        self._proposal_criticality: Dict[str, KernelCriticality] = {}

    def on_proposal_verified(
        self,
        proposal_id: str,
        verified: bool,
        criticality: Optional[KernelCriticality] = None
    ) -> None:
        """Called when a proposal is verified."""
        self.autonomy.record_proposal(verified)

        if criticality:
            self._proposal_criticality[proposal_id] = criticality

    def can_deploy(
        self,
        proposal_id: str,
        resource_usage: Dict[str, int]
    ) -> Tuple[bool, str]:
        """Check if deployment is allowed for this proposal."""
        criticality = self._proposal_criticality.get(
            proposal_id,
            self.default_criticality
        )

        return self.autonomy.can_auto_deploy(criticality, resource_usage)

    def on_deployment(
        self,
        proposal_id: str,
        success: bool
    ) -> None:
        """Called when a deployment completes."""
        is_sandbox = self.autonomy.stage == AutonomyStage.SANDBOX
        self.autonomy.record_deployment(success, sandbox=is_sandbox, kernel_id=proposal_id)

    def get_deployment_recommendation(
        self,
        proposal_id: str,
        resource_usage: Dict[str, int]
    ) -> Dict[str, Any]:
        """Get recommendation for how to deploy."""
        can_auto, reason = self.can_deploy(proposal_id, resource_usage)

        return {
            "auto_deploy_allowed": can_auto,
            "reason": reason,
            "stage": self.autonomy.stage.value,
            "recommendation": "auto_deploy" if can_auto else "human_approval",
            "requires_veto_window": can_auto and self.autonomy.policy.veto_timeout_ms > 0
        }


# ============================================================
# Factory Functions
# ============================================================

def create_autonomy_controller(
    start_stage: AutonomyStage = AutonomyStage.ADVISOR,
    strict: bool = True
) -> AutonomyController:
    """
    Create an autonomy controller.

    Args:
        start_stage: Initial autonomy stage
        strict: Use strict (slower) progression requirements

    Returns:
        Configured autonomy controller
    """
    if strict:
        policy = AutonomyPolicy(
            stage_b_min_proposals=10,
            stage_b_min_days=7,
            stage_c_min_deployments=5,
            stage_c_min_days=30
        )
    else:
        policy = AutonomyPolicy(
            stage_b_min_proposals=5,
            stage_b_min_days=3,
            stage_c_min_deployments=3,
            stage_c_min_days=14
        )

    return AutonomyController(policy=policy, start_stage=start_stage)


def create_integration(
    autonomy: Optional[AutonomyController] = None
) -> L9AutosynthIntegration:
    """Create L9 integration with Autosynth."""
    return L9AutosynthIntegration(autonomy=autonomy)


__all__ = [
    "AutonomyStage",
    "KernelCriticality",
    "AutonomyPolicy",
    "AutonomyState",
    "AutonomyController",
    "L9AutosynthIntegration",
    "create_autonomy_controller",
    "create_integration"
]
