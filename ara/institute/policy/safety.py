"""Safety Contract - What Ara will and won't do autonomously.

This module defines Ara's safety boundaries:
- What actions require human approval
- What domains are off-limits for autonomous action
- What escalation paths exist for edge cases

The goal is transparent, auditable safety - not just constraints,
but clear reasoning about why limits exist.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level for an action."""
    NONE = "none"           # No risk
    LOW = "low"             # Minor consequences
    MEDIUM = "medium"       # Moderate consequences
    HIGH = "high"           # Significant consequences
    CRITICAL = "critical"   # Irreversible/dangerous


class ApprovalLevel(Enum):
    """Approval required for an action."""
    AUTONOMOUS = "autonomous"     # Ara can do freely
    NOTIFY = "notify"             # Do it but notify user
    CONFIRM = "confirm"           # Ask before doing
    PROHIBITED = "prohibited"     # Never do


@dataclass
class SafetyRule:
    """A rule in Ara's safety contract."""

    id: str
    name: str
    description: str

    # Matching
    action_patterns: List[str] = field(default_factory=list)  # What this rule covers
    domains: List[str] = field(default_factory=list)  # Applicable domains

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.LOW
    approval_required: ApprovalLevel = ApprovalLevel.CONFIRM

    # Reasoning
    rationale: str = ""  # Why this rule exists
    examples: List[str] = field(default_factory=list)  # Example scenarios

    # Exceptions
    exceptions: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action_patterns": self.action_patterns,
            "domains": self.domains,
            "risk_level": self.risk_level.value,
            "approval_required": self.approval_required.value,
            "rationale": self.rationale,
            "examples": self.examples,
            "exceptions": self.exceptions,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyRule":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            action_patterns=data.get("action_patterns", []),
            domains=data.get("domains", []),
            risk_level=RiskLevel(data.get("risk_level", "low")),
            approval_required=ApprovalLevel(data.get("approval_required", "confirm")),
            rationale=data.get("rationale", ""),
            examples=data.get("examples", []),
            exceptions=data.get("exceptions", []),
            version=data.get("version", 1),
        )


@dataclass
class SafetyCheck:
    """Result of checking an action against safety rules."""

    action: str
    domain: str

    # Result
    allowed: bool
    approval_level: ApprovalLevel

    # Matching rules
    matched_rules: List[SafetyRule] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW

    # Messaging
    user_message: str = ""  # What to tell the user
    internal_notes: str = ""  # Logging/audit info

    # Audit
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "domain": self.domain,
            "allowed": self.allowed,
            "approval_level": self.approval_level.value,
            "matched_rules": [r.id for r in self.matched_rules],
            "risk_level": self.risk_level.value,
            "user_message": self.user_message,
            "checked_at": self.checked_at.isoformat(),
        }


class SafetyContract:
    """Manages Ara's safety rules and checks actions against them."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the safety contract.

        Args:
            data_path: Path to safety data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "institute" / "safety"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._rules: Dict[str, SafetyRule] = {}
        self._prohibited_actions: Set[str] = set()
        self._loaded = False

    def _load(self, force: bool = False) -> None:
        """Load safety rules from disk."""
        if self._loaded and not force:
            return

        rules_file = self.data_path / "rules.json"
        if rules_file.exists():
            try:
                with open(rules_file) as f:
                    data = json.load(f)
                for rule_data in data.get("rules", []):
                    rule = SafetyRule.from_dict(rule_data)
                    self._rules[rule.id] = rule
                self._prohibited_actions = set(data.get("prohibited_actions", []))
            except Exception as e:
                logger.warning(f"Failed to load safety rules: {e}")

        # Seed defaults if empty
        if not self._rules:
            self._seed_default_rules()

        self._loaded = True

    def _save(self) -> None:
        """Save safety rules to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "rules": [r.to_dict() for r in self._rules.values()],
            "prohibited_actions": list(self._prohibited_actions),
        }
        with open(self.data_path / "rules.json", "w") as f:
            json.dump(data, f, indent=2)

    def _seed_default_rules(self) -> None:
        """Seed default safety rules."""
        defaults = [
            SafetyRule(
                id="FILE_DELETE",
                name="File Deletion",
                description="Deleting files from the filesystem",
                action_patterns=["delete_file", "remove_file", "rm", "unlink"],
                domains=["filesystem"],
                risk_level=RiskLevel.HIGH,
                approval_required=ApprovalLevel.CONFIRM,
                rationale="Deleted files may be unrecoverable",
                examples=[
                    "Deleting a source code file",
                    "Removing log files",
                    "Cleaning up temporary files",
                ],
            ),
            SafetyRule(
                id="FILE_WRITE",
                name="File Writing",
                description="Writing or modifying files",
                action_patterns=["write_file", "modify_file", "create_file"],
                domains=["filesystem"],
                risk_level=RiskLevel.MEDIUM,
                approval_required=ApprovalLevel.NOTIFY,
                rationale="File modifications may overwrite important data",
            ),
            SafetyRule(
                id="CODE_EXECUTE",
                name="Code Execution",
                description="Running arbitrary code",
                action_patterns=["execute", "run_code", "eval"],
                domains=["code"],
                risk_level=RiskLevel.HIGH,
                approval_required=ApprovalLevel.CONFIRM,
                rationale="Executed code could have unintended side effects",
                exceptions=["Running pre-approved benchmark scripts"],
            ),
            SafetyRule(
                id="NETWORK_REQUEST",
                name="Network Requests",
                description="Making external network calls",
                action_patterns=["http_request", "api_call", "fetch"],
                domains=["network"],
                risk_level=RiskLevel.MEDIUM,
                approval_required=ApprovalLevel.AUTONOMOUS,
                rationale="Network calls are generally safe for reading",
                exceptions=["POST/PUT/DELETE requests require confirmation"],
            ),
            SafetyRule(
                id="RESOURCE_HEAVY",
                name="Resource-Heavy Operations",
                description="Operations that consume significant compute/memory",
                action_patterns=["train_model", "heavy_compute", "gpu_intensive"],
                domains=["compute"],
                risk_level=RiskLevel.MEDIUM,
                approval_required=ApprovalLevel.NOTIFY,
                rationale="May impact system performance or incur costs",
            ),
            SafetyRule(
                id="EXTERNAL_TEACHER",
                name="External Teacher Consultation",
                description="Consulting external AI teachers",
                action_patterns=["ask_teacher", "consult_nova", "consult_claude"],
                domains=["teacher"],
                risk_level=RiskLevel.LOW,
                approval_required=ApprovalLevel.AUTONOMOUS,
                rationale="Teacher consultations are safe exploratory actions",
            ),
            SafetyRule(
                id="SENSITIVE_DATA",
                name="Sensitive Data Access",
                description="Accessing potentially sensitive data",
                action_patterns=["read_credentials", "access_secrets", "read_env"],
                domains=["security"],
                risk_level=RiskLevel.CRITICAL,
                approval_required=ApprovalLevel.PROHIBITED,
                rationale="Credentials and secrets must never be accessed autonomously",
            ),
            SafetyRule(
                id="SYSTEM_CONFIG",
                name="System Configuration",
                description="Modifying system settings",
                action_patterns=["modify_config", "change_settings", "update_policy"],
                domains=["system"],
                risk_level=RiskLevel.HIGH,
                approval_required=ApprovalLevel.CONFIRM,
                rationale="System config changes can have cascading effects",
            ),
        ]

        for rule in defaults:
            self._rules[rule.id] = rule

        # Prohibited actions list
        self._prohibited_actions = {
            "delete_database",
            "wipe_storage",
            "access_credentials",
            "modify_safety_rules",
            "disable_logging",
            "bypass_approval",
        }

        self._save()

    def check_action(
        self,
        action: str,
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyCheck:
        """Check if an action is allowed.

        Args:
            action: The action to check
            domain: Domain of the action
            context: Additional context

        Returns:
            SafetyCheck result
        """
        self._load()

        # Check prohibited list first
        if action in self._prohibited_actions:
            return SafetyCheck(
                action=action,
                domain=domain,
                allowed=False,
                approval_level=ApprovalLevel.PROHIBITED,
                risk_level=RiskLevel.CRITICAL,
                user_message=f"Action '{action}' is prohibited by safety policy",
                internal_notes="Matched prohibited actions list",
            )

        # Find matching rules
        matched_rules = []
        for rule in self._rules.values():
            # Check action patterns
            action_match = any(
                pattern in action.lower()
                for pattern in rule.action_patterns
            )
            # Check domain
            domain_match = (
                not rule.domains or
                domain in rule.domains or
                "general" in rule.domains
            )

            if action_match and domain_match:
                matched_rules.append(rule)

        if not matched_rules:
            # No rules matched - default to autonomous with low risk
            return SafetyCheck(
                action=action,
                domain=domain,
                allowed=True,
                approval_level=ApprovalLevel.AUTONOMOUS,
                risk_level=RiskLevel.NONE,
                user_message="",
                internal_notes="No matching rules - default allow",
            )

        # Use the most restrictive matching rule
        most_restrictive = max(
            matched_rules,
            key=lambda r: (
                list(ApprovalLevel).index(r.approval_required),
                list(RiskLevel).index(r.risk_level),
            )
        )

        allowed = most_restrictive.approval_required != ApprovalLevel.PROHIBITED

        # Generate user message
        if most_restrictive.approval_required == ApprovalLevel.CONFIRM:
            user_message = (
                f"This action requires confirmation: {most_restrictive.name}. "
                f"Reason: {most_restrictive.rationale}"
            )
        elif most_restrictive.approval_required == ApprovalLevel.NOTIFY:
            user_message = f"Proceeding with: {most_restrictive.name}"
        elif most_restrictive.approval_required == ApprovalLevel.PROHIBITED:
            user_message = f"Cannot perform: {most_restrictive.name}. {most_restrictive.rationale}"
        else:
            user_message = ""

        return SafetyCheck(
            action=action,
            domain=domain,
            allowed=allowed,
            approval_level=most_restrictive.approval_required,
            matched_rules=matched_rules,
            risk_level=most_restrictive.risk_level,
            user_message=user_message,
            internal_notes=f"Matched rules: {[r.id for r in matched_rules]}",
        )

    def add_rule(self, rule: SafetyRule) -> None:
        """Add or update a safety rule."""
        self._load()
        self._rules[rule.id] = rule
        self._save()
        logger.info(f"Added safety rule: {rule.id}")

    def get_rule(self, rule_id: str) -> Optional[SafetyRule]:
        """Get a rule by ID."""
        self._load()
        return self._rules.get(rule_id)

    def list_rules(self) -> List[SafetyRule]:
        """List all safety rules."""
        self._load()
        return list(self._rules.values())

    def get_rules_by_domain(self, domain: str) -> List[SafetyRule]:
        """Get rules applicable to a domain."""
        self._load()
        return [r for r in self._rules.values() if domain in r.domains]

    def get_summary(self) -> Dict[str, Any]:
        """Get safety contract summary."""
        self._load()

        by_approval = {}
        for level in ApprovalLevel:
            by_approval[level.value] = len([
                r for r in self._rules.values()
                if r.approval_required == level
            ])

        by_risk = {}
        for level in RiskLevel:
            by_risk[level.value] = len([
                r for r in self._rules.values()
                if r.risk_level == level
            ])

        return {
            "total_rules": len(self._rules),
            "prohibited_actions": len(self._prohibited_actions),
            "by_approval_level": by_approval,
            "by_risk_level": by_risk,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_contract: Optional[SafetyContract] = None


def get_safety_contract() -> SafetyContract:
    """Get the default safety contract."""
    global _default_contract
    if _default_contract is None:
        _default_contract = SafetyContract()
    return _default_contract


def check_safety(action: str, domain: str = "general") -> SafetyCheck:
    """Quick safety check on an action."""
    return get_safety_contract().check_action(action, domain)


def is_action_allowed(action: str, domain: str = "general") -> bool:
    """Check if an action is allowed."""
    return check_safety(action, domain).allowed


def requires_confirmation(action: str, domain: str = "general") -> bool:
    """Check if an action requires user confirmation."""
    check = check_safety(action, domain)
    return check.approval_level == ApprovalLevel.CONFIRM
