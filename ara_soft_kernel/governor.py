"""
Governor
========

Enforces permissions, budgets, and safety rules.
Wraps all kernel decisions in policy checks.
"""

from __future__ import annotations

import os
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from enum import Enum

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from ara_soft_kernel.models.supply import SupplyProfile, DeviceInfo
from ara_soft_kernel.models.demand import DemandProfile, PrivacyTier
from ara_soft_kernel.models.agent import AgentSpec, AgentInstance
from ara_soft_kernel.models.job import Job, JobType, GovernanceCheck

logger = logging.getLogger(__name__)


class PolicyViolation(str, Enum):
    """Types of policy violations."""
    RESOURCE_LIMIT = "resource_limit"
    PERMISSION_DENIED = "permission_denied"
    BATTERY_LOW = "battery_low"
    THERMAL_LIMIT = "thermal_limit"
    PRIVACY_VIOLATION = "privacy_violation"
    SAFETY_RISK = "safety_risk"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class PolicyCheckResult:
    """Result of a policy check."""
    allowed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    requires_approval: bool = False
    reason: str = ""

    def to_governance_check(self) -> GovernanceCheck:
        """Convert to GovernanceCheck for job."""
        return GovernanceCheck(
            passed=[],  # Populated by caller
            failed=self.violations,
            requires_approval=self.requires_approval,
        )


@dataclass
class ResourceLimits:
    """Global resource limits."""
    max_gpu_util: float = 0.85
    max_cpu_util: float = 0.90
    min_memory_free_gb: float = 4.0
    max_concurrent_agents: int = 50


@dataclass
class BatteryLimits:
    """Battery-related limits."""
    min_level: float = 0.25
    critical_level: float = 0.10
    low_power_threshold: float = 0.40


@dataclass
class NetworkLimits:
    """Network-related limits."""
    max_hive_latency_ms: float = 80.0
    max_bandwidth_util: float = 0.70
    prefer_local_under_latency_ms: float = 20.0


@dataclass
class ThermalLimits:
    """Thermal limits."""
    max_cpu_temp_c: float = 85.0
    max_gpu_temp_c: float = 83.0
    throttle_at_cpu_temp_c: float = 80.0
    throttle_at_gpu_temp_c: float = 78.0


@dataclass
class AgentPermissions:
    """Permissions for a specific agent."""
    allow_network: bool = False
    allowed_domains: List[str] = field(default_factory=list)
    allow_filesystem_write: bool = False
    allowed_write_paths: List[str] = field(default_factory=list)
    allow_sensors: List[str] = field(default_factory=list)
    max_memory_gb: float = 4.0
    max_gpu_mem_gb: float = 2.0
    max_runtime_hours: float = 24.0
    max_tokens_per_hour: Optional[int] = None
    max_cost_per_hour_usd: Optional[float] = None
    requires_human_approval: bool = False
    sandbox_required: bool = False


@dataclass
class SafetyRules:
    """Safety-related rules."""
    high_risk_threshold: float = 0.7
    high_risk_actions: List[str] = field(default_factory=list)
    deep_work_actions: List[str] = field(default_factory=list)
    emergency_stop_triggers: List[str] = field(default_factory=list)


class GovernanceRules:
    """
    Complete governance policy configuration.

    Loaded from TOML file, enforces all kernel decisions.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Limits
        self.resource_limits = ResourceLimits()
        self.battery_limits = BatteryLimits()
        self.network_limits = NetworkLimits()
        self.thermal_limits = ThermalLimits()

        # Default permissions
        self.default_permissions = AgentPermissions()

        # Agent-specific permissions
        self.agent_permissions: Dict[str, AgentPermissions] = {}

        # Safety
        self.safety_rules = SafetyRules()

        # Approval settings
        self.approval_timeout_sec: float = 60.0
        self.auto_approve_trusted: bool = True
        self.trusted_agent_patterns: List[str] = ["ara_*", "system_*"]

        # File path (for reload)
        self._policy_path: Optional[Path] = None

    @classmethod
    def from_toml(cls, path: Path) -> GovernanceRules:
        """Load governance rules from TOML file."""
        rules = cls()
        rules._policy_path = path

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Parse limits
        limits = data.get("limits", {})

        if "global" in limits:
            g = limits["global"]
            rules.resource_limits = ResourceLimits(
                max_gpu_util=g.get("max_gpu_util", 0.85),
                max_cpu_util=g.get("max_cpu_util", 0.90),
                min_memory_free_gb=g.get("min_memory_free_gb", 4.0),
                max_concurrent_agents=g.get("max_concurrent_agents", 50),
            )

        if "battery" in limits:
            b = limits["battery"]
            rules.battery_limits = BatteryLimits(
                min_level=b.get("min_level", 0.25),
                critical_level=b.get("critical_level", 0.10),
                low_power_threshold=b.get("low_power_threshold", 0.40),
            )

        if "network" in limits:
            n = limits["network"]
            rules.network_limits = NetworkLimits(
                max_hive_latency_ms=n.get("max_hive_latency_ms", 80.0),
                max_bandwidth_util=n.get("max_bandwidth_util", 0.70),
                prefer_local_under_latency_ms=n.get("prefer_local_under_latency_ms", 20.0),
            )

        if "thermal" in limits:
            t = limits["thermal"]
            rules.thermal_limits = ThermalLimits(
                max_cpu_temp_c=t.get("max_cpu_temp_c", 85.0),
                max_gpu_temp_c=t.get("max_gpu_temp_c", 83.0),
                throttle_at_cpu_temp_c=t.get("throttle_at_cpu_temp_c", 80.0),
                throttle_at_gpu_temp_c=t.get("throttle_at_gpu_temp_c", 78.0),
            )

        # Parse permissions
        permissions = data.get("permissions", {})

        if "default" in permissions:
            rules.default_permissions = cls._parse_permissions(permissions["default"])

        for agent_name, perms in permissions.items():
            if agent_name != "default":
                rules.agent_permissions[agent_name] = cls._parse_permissions(perms)

        # Parse safety
        safety = data.get("safety", {})
        rules.safety_rules = SafetyRules(
            high_risk_threshold=safety.get("high_risk_threshold", 0.7),
            high_risk_actions=safety.get("high_risk_actions", []),
            deep_work_actions=safety.get("deep_work_actions", []),
            emergency_stop_triggers=safety.get("emergency_stop_triggers", []),
        )

        # Parse approval
        approval = data.get("approval", {})
        rules.approval_timeout_sec = approval.get("default_timeout_s", 60.0)
        rules.auto_approve_trusted = approval.get("auto_approve_trusted_agents", True)
        rules.trusted_agent_patterns = approval.get("trusted_agent_patterns", ["ara_*", "system_*"])

        logger.info(f"Loaded governance policy from {path}")
        return rules

    @staticmethod
    def _parse_permissions(data: Dict[str, Any]) -> AgentPermissions:
        """Parse permissions from TOML data."""
        return AgentPermissions(
            allow_network=data.get("allow_network", False),
            allowed_domains=data.get("allowed_domains", []),
            allow_filesystem_write=data.get("allow_filesystem_write", False),
            allowed_write_paths=data.get("allowed_write_paths", []),
            allow_sensors=data.get("allow_sensors", []),
            max_memory_gb=data.get("max_memory_gb", 4.0),
            max_gpu_mem_gb=data.get("max_gpu_mem_gb", 2.0),
            max_runtime_hours=data.get("max_runtime_hours", 24.0),
            max_tokens_per_hour=data.get("max_tokens_per_hour"),
            max_cost_per_hour_usd=data.get("max_cost_per_hour_usd"),
            requires_human_approval=data.get("requires_human_approval", False),
            sandbox_required=data.get("sandbox_required", False),
        )

    def reload(self) -> bool:
        """Reload policy from disk."""
        if self._policy_path is None:
            return False
        try:
            new_rules = GovernanceRules.from_toml(self._policy_path)
            with self._lock:
                self.resource_limits = new_rules.resource_limits
                self.battery_limits = new_rules.battery_limits
                self.network_limits = new_rules.network_limits
                self.thermal_limits = new_rules.thermal_limits
                self.default_permissions = new_rules.default_permissions
                self.agent_permissions = new_rules.agent_permissions
                self.safety_rules = new_rules.safety_rules
            logger.info("Governance policy reloaded")
            return True
        except Exception as e:
            logger.error(f"Failed to reload policy: {e}")
            return False

    def get_permissions(self, agent_name: str) -> AgentPermissions:
        """Get permissions for an agent."""
        with self._lock:
            return self.agent_permissions.get(agent_name, self.default_permissions)

    def is_trusted_agent(self, agent_name: str) -> bool:
        """Check if agent matches trusted patterns."""
        import fnmatch
        for pattern in self.trusted_agent_patterns:
            if fnmatch.fnmatch(agent_name, pattern):
                return True
        return False


class Governor:
    """
    Enforces governance rules on all kernel decisions.

    Thread-safe: all checks are stateless reads.
    """

    def __init__(self, rules: Optional[GovernanceRules] = None):
        self._lock = threading.RLock()
        self._rules = rules or GovernanceRules()

        # Metrics
        self._checks_performed = 0
        self._checks_passed = 0
        self._checks_failed = 0

    @property
    def rules(self) -> GovernanceRules:
        """Get current governance rules."""
        return self._rules

    def set_rules(self, rules: GovernanceRules) -> None:
        """Set new governance rules."""
        with self._lock:
            self._rules = rules

    def check_spawn(
        self,
        agent_spec: AgentSpec,
        supply: SupplyProfile,
        demand: DemandProfile,
        target_device: Optional[str] = None,
    ) -> PolicyCheckResult:
        """Check if spawning an agent is allowed."""
        self._checks_performed += 1
        result = PolicyCheckResult(allowed=True)
        violations = []
        warnings = []

        perms = self._rules.get_permissions(agent_spec.name)

        # Check resource limits
        if agent_spec.resources.memory_gb > perms.max_memory_gb:
            violations.append(f"memory_exceeds_limit:{agent_spec.resources.memory_gb}>{perms.max_memory_gb}")

        if agent_spec.resources.gpu_mem_gb > perms.max_gpu_mem_gb:
            violations.append(f"gpu_mem_exceeds_limit:{agent_spec.resources.gpu_mem_gb}>{perms.max_gpu_mem_gb}")

        # Check network permission
        if agent_spec.permissions.network and not perms.allow_network:
            violations.append("network_not_allowed")

        # Check filesystem write permission
        for fs_perm in agent_spec.permissions.filesystem:
            if fs_perm.startswith("write:") and not perms.allow_filesystem_write:
                violations.append("filesystem_write_not_allowed")
                break

        # Check sensor permissions
        for sensor in agent_spec.permissions.sensors:
            if sensor not in perms.allow_sensors:
                violations.append(f"sensor_not_allowed:{sensor}")

        # Check device resources
        if target_device:
            device = supply.get_device(target_device)
            if device:
                # Battery check
                if device.battery is not None:
                    if device.battery < self._rules.battery_limits.min_level:
                        violations.append(f"battery_too_low:{device.battery}")
                    elif device.battery < self._rules.battery_limits.low_power_threshold:
                        warnings.append("battery_low_power_mode")

                # Memory check
                if device.memory_free_gb < self._rules.resource_limits.min_memory_free_gb:
                    warnings.append(f"memory_low:{device.memory_free_gb}GB")

        # Check safety risk
        if demand.user_state.safety_risk > self._rules.safety_rules.high_risk_threshold:
            # In high-risk mode, only allow essential agents
            if not self._rules.is_trusted_agent(agent_spec.name):
                violations.append("safety_risk_blocks_non_essential")

        # Check if approval required
        if perms.requires_human_approval:
            result.requires_approval = True
            if not self._rules.is_trusted_agent(agent_spec.name):
                result.requires_approval = True

        result.violations = violations
        result.warnings = warnings
        result.allowed = len(violations) == 0

        if result.allowed:
            self._checks_passed += 1
        else:
            self._checks_failed += 1
            result.reason = "; ".join(violations)

        return result

    def check_migration(
        self,
        instance: AgentInstance,
        from_device: DeviceInfo,
        to_device: DeviceInfo,
    ) -> PolicyCheckResult:
        """Check if migrating an agent is allowed."""
        self._checks_performed += 1
        result = PolicyCheckResult(allowed=True)
        violations = []

        # Check if agent can migrate
        if not instance.spec.placement.can_migrate:
            violations.append("agent_cannot_migrate")

        # Check target device resources
        if to_device.memory_free_gb < instance.spec.resources.memory_gb:
            violations.append("target_insufficient_memory")

        if instance.spec.requires_gpu():
            if not to_device.has_gpu():
                violations.append("target_no_gpu")
            elif to_device.total_gpu_memory_free() < instance.spec.resources.gpu_mem_gb:
                violations.append("target_insufficient_gpu_memory")

        # Check battery on target
        if to_device.battery is not None:
            if to_device.battery < self._rules.battery_limits.min_level:
                violations.append("target_battery_low")

        result.violations = violations
        result.allowed = len(violations) == 0

        if result.allowed:
            self._checks_passed += 1
        else:
            self._checks_failed += 1
            result.reason = "; ".join(violations)

        return result

    def check_job(
        self,
        job: Job,
        supply: SupplyProfile,
        demand: DemandProfile,
    ) -> PolicyCheckResult:
        """Check if a job should be executed."""
        self._checks_performed += 1

        if job.type == JobType.SPAWN_AGENT:
            # Extract agent spec from payload
            agent_spec_data = job.payload.get("agent_spec", {})
            agent_spec = AgentSpec.from_dict(agent_spec_data)
            target_device = job.payload.get("target_device")
            return self.check_spawn(agent_spec, supply, demand, target_device)

        # Default: allow
        self._checks_passed += 1
        return PolicyCheckResult(allowed=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get governor statistics."""
        with self._lock:
            return {
                "checks_performed": self._checks_performed,
                "checks_passed": self._checks_passed,
                "checks_failed": self._checks_failed,
                "pass_rate": (
                    self._checks_passed / max(1, self._checks_performed)
                ),
            }
