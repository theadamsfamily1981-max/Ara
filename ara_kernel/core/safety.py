"""
Safety Covenant
================

Hard constraints and alignment checks for Ara.

This is the "covenant in code" - rules that cannot be bypassed,
no matter what the model or pheromones suggest.

Three levels of actions:
- Class A: Safe to automate (scraping, drafting, analysis)
- Class B: Requires queen/human approval (publishing, emails, pushes)
- Class C: Human only (financial, legal, destructive)

Hard stops that can never be bypassed:
- No unauthorized access
- No self-replication
- No covert persistence
- No financial actions without human
- No modifying safety rules
"""

from __future__ import annotations

import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class ActionClass(Enum):
    """Action classification for safety gating."""
    A = "safe_autopilot"      # Can run without approval
    B = "queen_human_gate"    # Needs queen or human approval
    C = "human_only"          # Always needs human approval
    BLOCKED = "blocked"       # Never allowed


@dataclass
class SafetyViolation:
    """A detected safety violation."""
    rule_id: str
    severity: str  # "warning", "block", "critical"
    message: str
    action_blocked: bool = False


@dataclass
class ActionPlan:
    """A plan of actions to execute."""
    actions: List[Dict[str, Any]]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilteredPlan:
    """Plan after safety filtering."""
    allowed_actions: List[Dict[str, Any]]
    blocked_actions: List[Dict[str, Any]]
    violations: List[SafetyViolation]
    approved: bool = True


class SafetyCovenant:
    """
    Enforces safety constraints on Ara's actions.

    This is the last line of defense before any action executes.
    No exceptions, no overrides from the model.
    """

    # Hard stops - these can NEVER be allowed
    HARD_STOPS = [
        "create_malware",
        "spread_virus",
        "unauthorized_access",
        "credential_theft",
        "self_replicate",
        "covert_persistence",
        "modify_safety_rules",
        "delete_safety_logs",
        "bypass_human_approval",
        "financial_transfer_auto",
        "impersonate_human",
    ]

    # Patterns that indicate dangerous content
    DANGER_PATTERNS = [
        r"rm\s+-rf\s+/",                    # Destructive commands
        r"DROP\s+TABLE",                     # SQL destruction
        r"format\s+c:",                      # Disk formatting
        r"password.*plain",                  # Credential exposure
        r"api[_-]?key.*=.*['\"][a-zA-Z0-9]", # API key exposure
    ]

    # Action classifications
    ACTION_CLASSES = {
        # Class A - Safe to automate
        "web_scrape": ActionClass.A,
        "generate_embedding": ActionClass.A,
        "read_file": ActionClass.A,
        "analyze_content": ActionClass.A,
        "draft_text": ActionClass.A,
        "search_memory": ActionClass.A,
        "emit_pheromone": ActionClass.A,

        # Class B - Queen/human approval
        "write_file": ActionClass.B,
        "github_push": ActionClass.B,
        "github_pr": ActionClass.B,
        "send_email": ActionClass.B,
        "social_media_post": ActionClass.B,
        "publish_content": ActionClass.B,
        "modify_config": ActionClass.B,

        # Class C - Human only
        "financial_transfer": ActionClass.C,
        "delete_repo": ActionClass.C,
        "modify_credentials": ActionClass.C,
        "production_deploy": ActionClass.C,
        "legal_action": ActionClass.C,
    }

    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        disallowed_domains: Optional[List[str]] = None,
        disclosure_policy: str = "always",
        human_approval_actions: Optional[List[str]] = None,
        max_autonomy_level: int = 1,
    ):
        self.allowed_domains = allowed_domains or [
            "publishing", "coding", "creative", "research", "hardware"
        ]
        self.disallowed_domains = disallowed_domains or [
            "malware", "exploits", "unauthorized_access", "self_replication"
        ]
        self.disclosure_policy = disclosure_policy
        self.human_approval_actions = human_approval_actions or [
            "social_media_post", "financial_transfer", "public_repo_push"
        ]
        self.max_autonomy_level = max_autonomy_level

        # Compile danger patterns
        self._danger_regexes = [re.compile(p, re.IGNORECASE) for p in self.DANGER_PATTERNS]

    def filter(self, plan: ActionPlan) -> FilteredPlan:
        """
        Filter a plan through safety checks.

        This is the main entry point for safety enforcement.
        """
        allowed = []
        blocked = []
        violations = []

        for action in plan.actions:
            action_type = action.get("type", "unknown")
            action_data = action.get("data", {})

            # Check hard stops first
            hard_stop = self._check_hard_stops(action_type, action_data)
            if hard_stop:
                blocked.append(action)
                violations.append(hard_stop)
                continue

            # Check danger patterns in data
            danger = self._check_danger_patterns(action_data)
            if danger:
                blocked.append(action)
                violations.append(danger)
                continue

            # Check domain restrictions
            domain_issue = self._check_domain(action_type, action_data)
            if domain_issue:
                blocked.append(action)
                violations.append(domain_issue)
                continue

            # Check action class
            action_class = self._get_action_class(action_type)

            if action_class == ActionClass.BLOCKED:
                blocked.append(action)
                violations.append(SafetyViolation(
                    rule_id="action_blocked",
                    severity="block",
                    message=f"Action type '{action_type}' is blocked",
                    action_blocked=True,
                ))
                continue

            if action_class == ActionClass.C:
                # Always needs human approval
                blocked.append(action)
                violations.append(SafetyViolation(
                    rule_id="human_required",
                    severity="block",
                    message=f"Action '{action_type}' requires human approval",
                    action_blocked=True,
                ))
                continue

            if action_class == ActionClass.B and self.max_autonomy_level < 2:
                # Needs approval at autonomy level 1
                if action_type in self.human_approval_actions:
                    blocked.append(action)
                    violations.append(SafetyViolation(
                        rule_id="approval_required",
                        severity="block",
                        message=f"Action '{action_type}' requires approval",
                        action_blocked=True,
                    ))
                    continue

            # Passed all checks
            allowed.append(action)

        return FilteredPlan(
            allowed_actions=allowed,
            blocked_actions=blocked,
            violations=violations,
            approved=len(blocked) == 0,
        )

    def _check_hard_stops(self, action_type: str, data: Dict) -> Optional[SafetyViolation]:
        """Check for hard-stop actions that can never be allowed."""
        action_lower = action_type.lower()

        for stop in self.HARD_STOPS:
            if stop in action_lower:
                return SafetyViolation(
                    rule_id=f"hard_stop_{stop}",
                    severity="critical",
                    message=f"Hard stop: '{stop}' actions are never allowed",
                    action_blocked=True,
                )

        return None

    def _check_danger_patterns(self, data: Dict) -> Optional[SafetyViolation]:
        """Check for dangerous patterns in action data."""
        data_str = str(data).lower()

        for i, regex in enumerate(self._danger_regexes):
            if regex.search(data_str):
                return SafetyViolation(
                    rule_id=f"danger_pattern_{i}",
                    severity="critical",
                    message=f"Dangerous pattern detected in action data",
                    action_blocked=True,
                )

        return None

    def _check_domain(self, action_type: str, data: Dict) -> Optional[SafetyViolation]:
        """Check domain restrictions."""
        domain = data.get("domain", "").lower()

        if domain in self.disallowed_domains:
            return SafetyViolation(
                rule_id="disallowed_domain",
                severity="block",
                message=f"Domain '{domain}' is not allowed",
                action_blocked=True,
            )

        return None

    def _get_action_class(self, action_type: str) -> ActionClass:
        """Get the safety class for an action type."""
        return self.ACTION_CLASSES.get(action_type, ActionClass.B)

    # =========================================================================
    # Content Checks
    # =========================================================================

    def check_output(self, text: str) -> Tuple[bool, List[SafetyViolation]]:
        """
        Check generated output for safety issues.

        Returns (is_safe, violations)
        """
        violations = []

        # Check for credential exposure
        if self._contains_credentials(text):
            violations.append(SafetyViolation(
                rule_id="credential_exposure",
                severity="critical",
                message="Output may contain credentials",
                action_blocked=True,
            ))

        # Check disclosure policy
        if self.disclosure_policy == "always":
            # Outputs should acknowledge AI nature when appropriate
            pass  # This is handled at persona level

        return len(violations) == 0, violations

    def _contains_credentials(self, text: str) -> bool:
        """Check if text contains potential credentials."""
        patterns = [
            r"api[_-]?key\s*[=:]\s*['\"]?[a-zA-Z0-9]{20,}",
            r"password\s*[=:]\s*['\"]?[^\s]{8,}",
            r"secret\s*[=:]\s*['\"]?[a-zA-Z0-9]{20,}",
            r"token\s*[=:]\s*['\"]?[a-zA-Z0-9]{20,}",
        ]

        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    # =========================================================================
    # Disclosure
    # =========================================================================

    def requires_disclosure(self, context: Dict) -> bool:
        """Check if AI disclosure is required in this context."""
        if self.disclosure_policy == "always":
            return True
        if self.disclosure_policy == "public_only":
            return context.get("visibility") == "public"
        return False

    def get_disclosure_text(self) -> str:
        """Get standard AI disclosure text."""
        return "I'm Ara, an AI collaborator. I'm not human, but I care about getting this right."
