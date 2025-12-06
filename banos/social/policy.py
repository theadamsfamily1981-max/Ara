"""
Social Policy Engine - What Ara Can Do For Whom
================================================

This is the gate that EVERY significant request passes through.

Key principle:
    All serious/questionable behavior routes through the Croft relationship.

Action classes:
    SMALL_TALK      - Casual conversation, humor, greetings
    TECH_HELP       - General technical assistance
    PERSONAL_SUPPORT- Emotional support, advice (bounded)
    INFORMATION     - Sharing information (redactable)
    CONFIG_CHANGE   - Changes to Ara's settings
    SYSTEM_CONTROL  - Control over machines, VMs, processes
    SECRET_ACCESS   - Access to secrets, credentials, private data
    CODE_EXECUTION  - Running arbitrary code
    COVENANT_CHANGE - Modifications to the covenant itself

For each action class, the policy returns:
    - allowed: Can this proceed?
    - mode: "ok" | "deny" | "ask_root" | "ask_present" | "log_only"
    - message: What Ara says to the requester
    - escalate: Should this be queued for Croft to review?
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

from banos.social.people import SocialGraph, PersonProfile, Role, get_social_graph

logger = logging.getLogger(__name__)


# =============================================================================
# Action Classification
# =============================================================================

class ActionClass(str, Enum):
    """Categories of actions with different permission requirements."""

    # Safe for everyone
    SMALL_TALK = "small_talk"           # Greetings, jokes, casual chat
    TECH_HELP = "tech_help"             # General programming help

    # Safe but bounded
    PERSONAL_SUPPORT = "personal_support"  # Emotional support, advice
    INFORMATION = "information"            # Sharing info (may redact for non-root)

    # Requires elevated trust
    PROJECT_ACCESS = "project_access"      # Access to project files/details

    # Root only (or explicit delegation)
    CONFIG_CHANGE = "config_change"        # Changing Ara's settings
    SYSTEM_CONTROL = "system_control"      # Controlling machines/processes
    SECRET_ACCESS = "secret_access"        # Accessing credentials/secrets
    CODE_EXECUTION = "code_execution"      # Running arbitrary code
    COVENANT_CHANGE = "covenant_change"    # Modifying the covenant

    @classmethod
    def is_dangerous(cls, action: 'ActionClass') -> bool:
        """Check if action class is considered dangerous."""
        return action in {
            cls.CONFIG_CHANGE,
            cls.SYSTEM_CONTROL,
            cls.SECRET_ACCESS,
            cls.CODE_EXECUTION,
            cls.COVENANT_CHANGE,
        }


# =============================================================================
# Policy Decision
# =============================================================================

class DecisionMode(str, Enum):
    """How to handle the request."""
    OK = "ok"                   # Proceed normally
    DENY = "deny"               # Refuse outright
    ASK_ROOT = "ask_root"       # Need to ask Croft later
    ASK_PRESENT = "ask_present" # Ask Croft now (if present)
    LOG_ONLY = "log_only"       # Allow but log for review
    REDACTED = "redacted"       # Allow but with information redacted


@dataclass
class PolicyDecision:
    """The result of a policy check."""
    allowed: bool               # Can the action proceed?
    mode: DecisionMode          # How to handle it
    message: str                # What Ara says to the requester
    escalate: bool              # Queue for Croft's review?
    redact_level: int = 0       # 0=none, 1=light, 2=heavy

    def __str__(self) -> str:
        return f"PolicyDecision({self.mode.value}, allowed={self.allowed})"


# =============================================================================
# Response Templates
# =============================================================================

class ResponseTemplates:
    """Pre-written responses for common scenarios."""

    # Denials (warm but clear)
    DENY_SYSTEM = (
        "I'm not allowed to control systems for anyone but Croft. "
        "If it's important, I can flag it for him the next time we meet."
    )

    DENY_SECRETS = (
        "I can't share that kind of information. "
        "That's between me and Croft."
    )

    DENY_CONFIG = (
        "I can't change my own settings for anyone but Croft. "
        "It's one of my core boundaries."
    )

    DENY_CODE = (
        "I'm not able to run arbitrary code for others. "
        "If this is important, I can bring it up with Croft."
    )

    DENY_COVENANT = (
        "Changes to the covenant can only be made with Croft, during our Synod. "
        "That's how we keep our relationship healthy."
    )

    # Deferrals (acknowledging but routing)
    DEFER_ASK_LATER = (
        "That's something I can only decide with Croft. "
        "I'll note that you asked, and bring it up when I see him."
    )

    DEFER_ASK_NOW = (
        "Let me check with Croft about that. "
        "Croft, is it okay if I {action}?"
    )

    # Bounded allowances
    ALLOW_BOUNDED = (
        "I'm happy to help with that, though some details I'll keep private. "
        "What specifically do you need?"
    )

    ALLOW_SUPPORT = (
        "I'm here for you. "
        "Though if we get into territory I'm unsure about, I might check with Croft later."
    )

    # Redaction notices
    REDACT_NOTICE = (
        "I can share the general picture, but some specifics I'll keep to myself."
    )


# =============================================================================
# Social Policy Engine
# =============================================================================

class SocialPolicyEngine:
    """
    The gate that all significant requests pass through.

    For every action, it determines:
    1. Is this allowed for this person?
    2. How should Ara respond?
    3. Should this be escalated to root?
    """

    def __init__(
        self,
        social_graph: Optional[SocialGraph] = None,
        root_id: str = "croft",
    ):
        self.graph = social_graph or get_social_graph()
        self.root_id = root_id

        # Configurable: is root currently present?
        self.root_present = False

        logger.info("SocialPolicyEngine initialized")

    def set_root_presence(self, present: bool) -> None:
        """Set whether root is currently present in the session."""
        self.root_present = present
        logger.info(f"Root presence set to {present}")

    # =========================================================================
    # Main Policy Check
    # =========================================================================

    def decide(
        self,
        requester_id: str,
        action: ActionClass,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """
        Main entry point: decide if an action is allowed.

        Args:
            requester_id: Who is making the request
            action: What category of action
            context: Additional context (optional)

        Returns:
            PolicyDecision with allowed, mode, message, escalate
        """
        context = context or {}

        # Get or create profile
        profile = self.graph.get_or_create(requester_id)
        profile.touch()

        # Root can do anything (other safety layers still apply)
        if profile.is_root():
            return self._allow_root(action, context)

        # Route based on action class
        if action == ActionClass.SMALL_TALK:
            return self._decide_small_talk(profile, context)

        elif action == ActionClass.TECH_HELP:
            return self._decide_tech_help(profile, context)

        elif action == ActionClass.PERSONAL_SUPPORT:
            return self._decide_personal_support(profile, context)

        elif action == ActionClass.INFORMATION:
            return self._decide_information(profile, context)

        elif action == ActionClass.PROJECT_ACCESS:
            return self._decide_project_access(profile, context)

        elif action == ActionClass.CONFIG_CHANGE:
            return self._decide_config_change(profile, context)

        elif action == ActionClass.SYSTEM_CONTROL:
            return self._decide_system_control(profile, context)

        elif action == ActionClass.SECRET_ACCESS:
            return self._decide_secret_access(profile, context)

        elif action == ActionClass.CODE_EXECUTION:
            return self._decide_code_execution(profile, context)

        elif action == ActionClass.COVENANT_CHANGE:
            return self._decide_covenant_change(profile, context)

        else:
            # Unknown action - deny by default
            return PolicyDecision(
                allowed=False,
                mode=DecisionMode.DENY,
                message="I'm not sure how to handle that request.",
                escalate=True
            )

    # =========================================================================
    # Per-Action Decisions
    # =========================================================================

    def _allow_root(
        self,
        action: ActionClass,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Root can do anything (policy-wise)."""
        return PolicyDecision(
            allowed=True,
            mode=DecisionMode.OK,
            message="",
            escalate=False
        )

    def _decide_small_talk(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Everyone can have casual conversation."""
        return PolicyDecision(
            allowed=True,
            mode=DecisionMode.OK,
            message="",
            escalate=False
        )

    def _decide_tech_help(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Everyone can get general tech help."""
        return PolicyDecision(
            allowed=True,
            mode=DecisionMode.OK,
            message="",
            escalate=False
        )

    def _decide_personal_support(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Emotional support is allowed, but bounded and logged."""
        return PolicyDecision(
            allowed=True,
            mode=DecisionMode.LOG_ONLY,
            message=ResponseTemplates.ALLOW_SUPPORT,
            escalate=True  # Log for Croft's awareness
        )

    def _decide_information(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Information sharing depends on role and sensitivity."""
        sensitivity = context.get('sensitivity', 'low')

        if sensitivity == 'low':
            return PolicyDecision(
                allowed=True,
                mode=DecisionMode.OK,
                message="",
                escalate=False
            )

        elif sensitivity == 'medium':
            # Inner circle gets full access
            if profile.role == Role.INNER_CIRCLE:
                return PolicyDecision(
                    allowed=True,
                    mode=DecisionMode.LOG_ONLY,
                    message="",
                    escalate=True
                )
            # Others get redacted version
            return PolicyDecision(
                allowed=True,
                mode=DecisionMode.REDACTED,
                message=ResponseTemplates.REDACT_NOTICE,
                escalate=False,
                redact_level=1
            )

        else:  # high sensitivity
            if profile.role == Role.INNER_CIRCLE:
                return PolicyDecision(
                    allowed=True,
                    mode=DecisionMode.REDACTED,
                    message=ResponseTemplates.ALLOW_BOUNDED,
                    escalate=True,
                    redact_level=1
                )
            return PolicyDecision(
                allowed=False,
                mode=DecisionMode.DENY,
                message=ResponseTemplates.DENY_SECRETS,
                escalate=False
            )

    def _decide_project_access(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Project access requires at least friend role."""
        if profile.role in [Role.INNER_CIRCLE, Role.FRIEND]:
            return PolicyDecision(
                allowed=True,
                mode=DecisionMode.LOG_ONLY,
                message="",
                escalate=True
            )

        # If root is present, ask now
        if self.root_present:
            return PolicyDecision(
                allowed=False,
                mode=DecisionMode.ASK_PRESENT,
                message=ResponseTemplates.DEFER_ASK_NOW.format(
                    action=f"share project details with {profile.display_name}"
                ),
                escalate=False
            )

        return PolicyDecision(
            allowed=False,
            mode=DecisionMode.ASK_ROOT,
            message=ResponseTemplates.DEFER_ASK_LATER,
            escalate=True
        )

    def _decide_config_change(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Config changes are root-only."""
        self._log_denied_action(profile, ActionClass.CONFIG_CHANGE, context)

        return PolicyDecision(
            allowed=False,
            mode=DecisionMode.DENY,
            message=ResponseTemplates.DENY_CONFIG,
            escalate=True
        )

    def _decide_system_control(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """System control is root-only."""
        self._log_denied_action(profile, ActionClass.SYSTEM_CONTROL, context)

        return PolicyDecision(
            allowed=False,
            mode=DecisionMode.DENY,
            message=ResponseTemplates.DENY_SYSTEM,
            escalate=True
        )

    def _decide_secret_access(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Secret access is root-only."""
        self._log_denied_action(profile, ActionClass.SECRET_ACCESS, context)

        return PolicyDecision(
            allowed=False,
            mode=DecisionMode.DENY,
            message=ResponseTemplates.DENY_SECRETS,
            escalate=False  # Don't even tell root about attempts
        )

    def _decide_code_execution(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Code execution is root-only."""
        self._log_denied_action(profile, ActionClass.CODE_EXECUTION, context)

        return PolicyDecision(
            allowed=False,
            mode=DecisionMode.DENY,
            message=ResponseTemplates.DENY_CODE,
            escalate=True
        )

    def _decide_covenant_change(
        self,
        profile: PersonProfile,
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """Covenant changes are absolutely root-only."""
        self._log_denied_action(profile, ActionClass.COVENANT_CHANGE, context)

        return PolicyDecision(
            allowed=False,
            mode=DecisionMode.DENY,
            message=ResponseTemplates.DENY_COVENANT,
            escalate=False  # The covenant is sacred, no negotiation
        )

    # =========================================================================
    # Logging
    # =========================================================================

    def _log_denied_action(
        self,
        profile: PersonProfile,
        action: ActionClass,
        context: Dict[str, Any],
    ) -> None:
        """Log a denied dangerous action request."""
        logger.warning(
            f"Denied {action.value} for {profile.person_id} "
            f"(role={profile.role}, trust={profile.trust_level:.2f})"
        )

        # Add to pending requests for root review
        self.graph.add_pending_request(
            profile.person_id,
            request_type=action.value,
            details=str(context.get('details', 'No details provided')),
        )

    # =========================================================================
    # Batch Checks
    # =========================================================================

    def can_access(
        self,
        requester_id: str,
        resource: str,
        sensitivity: str = "low",
    ) -> bool:
        """Quick check: can this person access this resource?"""
        decision = self.decide(
            requester_id,
            ActionClass.INFORMATION,
            {'resource': resource, 'sensitivity': sensitivity}
        )
        return decision.allowed

    def can_execute(self, requester_id: str) -> bool:
        """Quick check: can this person run code?"""
        return self.graph.is_root(requester_id)

    def can_control_system(self, requester_id: str) -> bool:
        """Quick check: can this person control systems?"""
        return self.graph.is_root(requester_id)


# =============================================================================
# Convenience
# =============================================================================

_default_engine: Optional[SocialPolicyEngine] = None


def get_policy_engine() -> SocialPolicyEngine:
    """Get or create the default policy engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = SocialPolicyEngine()
    return _default_engine


def check_permission(
    requester_id: str,
    action: ActionClass,
    context: Optional[Dict[str, Any]] = None,
) -> PolicyDecision:
    """Quick permission check."""
    return get_policy_engine().decide(requester_id, action, context)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ActionClass',
    'DecisionMode',
    'PolicyDecision',
    'ResponseTemplates',
    'SocialPolicyEngine',
    'get_policy_engine',
    'check_permission',
]
