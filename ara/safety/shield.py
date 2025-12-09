"""
ARA SHIELD: Security Hardening for Realistic Ara

Security appropriate for a 1-person startup with limited resources.
Not a fantasy corporation. Realistic threats, realistic defenses.

Threat Model:
    1. Script kiddies and automated scanners (HIGH probability)
    2. Opportunistic attackers (MEDIUM probability)
    3. Targeted attacks from individuals (LOW probability)
    4. State actors (NOT in scope - if they want in, they're in)

Defense Layers:
    1. Network (firewall, VPN-only access)
    2. Application (input validation, rate limiting)
    3. Runtime (sandboxing, resource limits)
    4. Data (encryption at rest, memory protection)
    5. Detection (logging, anomaly detection)
    6. Recovery (backups, kill switch, safe mode)

Non-Goals:
    - Perfect security (impossible)
    - Enterprise compliance (HIPAA, SOC2, etc.)
    - Defense against nation-state actors
    - Physical security of hardware

Usage:
    from ara.safety.shield import Shield, run_security_audit

    # Initialize shield
    shield = Shield(config)

    # Check if an operation is allowed
    if shield.authorize("execute_skill", context):
        execute_skill(...)

    # Run security audit
    report = run_security_audit()
"""

from __future__ import annotations

import logging
import time
import os
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Threat Categories
# =============================================================================

class ThreatLevel(str, Enum):
    """Threat level classification."""
    NONE = "none"       # No threat detected
    LOW = "low"         # Minor concern
    MEDIUM = "medium"   # Significant concern
    HIGH = "high"       # Immediate attention needed
    CRITICAL = "critical"  # Emergency response


class ThreatCategory(str, Enum):
    """Categories of threats we defend against."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INJECTION = "injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"


# =============================================================================
# Shield Mode State Machine
# =============================================================================

class ShieldMode(str, Enum):
    """
    Shield operating modes.

    Normal:    Full capabilities
    Sanctuary: Local-only, no cathedral/paid skills
    Debug:     Internal builds only, no obfuscation, no RASP
    """
    NORMAL = "normal"
    SANCTUARY = "sanctuary"
    DEBUG = "debug"


class ShieldTrigger(str, Enum):
    """Events that can trigger mode transitions."""
    DEBUGGER_DETECTED = "debugger_detected"
    HOOK_DETECTED = "hook_detected"
    ROOT_OR_JAILBREAK = "root_or_jailbreak"
    BINARY_TAMPERED = "binary_tampered"
    KEY_COMPROMISED = "key_compromised"
    NETWORK_HOSTILE = "network_hostile"
    USER_REQUEST = "user_request"
    SAFE_RESTART = "safe_restart"


@dataclass
class ShieldState:
    """
    Current state of the Shield.

    Invariants:
    - Debug mode only in internal/dev builds
    - Sanctuary can be entered from Normal but never auto-promoted back
    - Normal is privileged, not default-forever
    """
    mode: ShieldMode = ShieldMode.NORMAL
    last_trigger: Optional[ShieldTrigger] = None
    triggered_at: Optional[float] = None
    sanctuary_reason: str = ""

    # Tracking
    triggers_since_start: List[ShieldTrigger] = field(default_factory=list)
    sanctuary_entries: int = 0

    def enter_sanctuary(self, trigger: ShieldTrigger, reason: str = "") -> None:
        """Transition to Sanctuary mode."""
        if self.mode == ShieldMode.SANCTUARY:
            return  # Already in Sanctuary

        self.mode = ShieldMode.SANCTUARY
        self.last_trigger = trigger
        self.triggered_at = time.time()
        self.sanctuary_reason = reason or f"Triggered by {trigger.value}"
        self.triggers_since_start.append(trigger)
        self.sanctuary_entries += 1

        logger.warning(f"SHIELD: Entering Sanctuary - {self.sanctuary_reason}")

    def exit_sanctuary(self, trigger: ShieldTrigger = ShieldTrigger.SAFE_RESTART) -> bool:
        """
        Attempt to exit Sanctuary mode.

        Only allowed via:
        - App restart
        - Successful safety recheck (fresh boot, no hooks/root)

        Returns True if exit succeeded.
        """
        if self.mode != ShieldMode.SANCTUARY:
            return True

        # Check if safe to exit
        if trigger not in (ShieldTrigger.SAFE_RESTART, ShieldTrigger.USER_REQUEST):
            logger.warning("Cannot exit Sanctuary without safe restart")
            return False

        # Run safety checks before exiting
        if not _run_safety_checks():
            logger.warning("Safety checks failed, staying in Sanctuary")
            return False

        self.mode = ShieldMode.NORMAL
        self.last_trigger = trigger
        self.triggered_at = time.time()
        self.sanctuary_reason = ""

        logger.info("SHIELD: Exited Sanctuary, returning to Normal")
        return True

    def is_sanctuary(self) -> bool:
        """Check if in Sanctuary mode."""
        return self.mode == ShieldMode.SANCTUARY

    def allows_cathedral(self) -> bool:
        """Check if cathedral access is allowed."""
        return self.mode == ShieldMode.NORMAL

    def allows_payments(self) -> bool:
        """Check if payment processing is allowed."""
        return self.mode == ShieldMode.NORMAL

    def max_autonomy_level(self) -> int:
        """Get max autonomy level for current mode."""
        if self.mode == ShieldMode.SANCTUARY:
            return 2  # ASSIST max
        return 4  # EXEC_HIGH

    def get_user_message(self) -> str:
        """Get user-friendly explanation of current state."""
        if self.mode == ShieldMode.NORMAL:
            return "I'm running normally with full capabilities."

        if self.mode == ShieldMode.SANCTUARY:
            return (
                "This device looks modified or unsafe, so I'm running in Sanctuary mode. "
                "I'll keep your memories local and avoid anything risky like payments or lab control. "
                "I'm still here. I still remember you."
            )

        if self.mode == ShieldMode.DEBUG:
            return "Running in debug mode (internal build only)."

        return "Unknown mode."


def _run_safety_checks() -> bool:
    """
    Run safety checks to determine if it's safe to exit Sanctuary.

    Returns True if environment appears safe.
    """
    # Check for debugger
    if _detect_debugger():
        logger.warning("Safety check failed: debugger detected")
        return False

    # Check for root/jailbreak
    if _detect_root_or_jailbreak():
        logger.warning("Safety check failed: root/jailbreak detected")
        return False

    # Check for hooks
    if _detect_hooks():
        logger.warning("Safety check failed: hooks detected")
        return False

    return True


def _detect_debugger() -> bool:
    """Detect if a debugger is attached."""
    try:
        # Linux: check TracerPid in /proc/self/status
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("TracerPid:"):
                    tracer_pid = int(line.split()[1])
                    return tracer_pid != 0
    except Exception:
        pass

    # Check if running under ptrace
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # PTRACE_TRACEME = 0
        result = libc.ptrace(0, 0, 0, 0)
        if result == -1:
            return True  # Already being traced
    except Exception:
        pass

    return False


def _detect_root_or_jailbreak() -> bool:
    """Detect if device is rooted or jailbroken."""
    # Check if running as root
    if os.geteuid() == 0:
        return True

    # Check for common root indicators
    root_indicators = [
        "/system/app/Superuser.apk",  # Android
        "/sbin/su",
        "/system/bin/su",
        "/system/xbin/su",
        "/data/local/xbin/su",
        "/Applications/Cydia.app",  # iOS jailbreak
        "/Library/MobileSubstrate/MobileSubstrate.dylib",
        "/bin/bash",  # iOS normally doesn't have bash
        "/usr/sbin/sshd",  # iOS sshd indicates jailbreak
    ]

    for path in root_indicators:
        if os.path.exists(path):
            return True

    return False


def _detect_hooks() -> bool:
    """Detect if common hooking frameworks are present."""
    # Check for Frida
    frida_indicators = [
        "/data/local/tmp/frida-server",
        "frida-agent",
        "frida-gadget",
    ]

    for indicator in frida_indicators:
        if os.path.exists(indicator):
            return True

    # Check for Xposed
    try:
        with open("/proc/self/maps") as f:
            maps = f.read()
            if "XposedBridge" in maps or "substrate" in maps.lower():
                return True
    except Exception:
        pass

    # Check loaded libraries
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "frida" in line.lower() or "xposed" in line.lower():
                    return True
    except Exception:
        pass

    return False


# =============================================================================
# Security Configuration
# =============================================================================

@dataclass
class ShieldConfig:
    """Security configuration for ARA SHIELD."""
    # Network layer
    vpn_required: bool = True
    allowed_ips: Set[str] = field(default_factory=lambda: {"127.0.0.1", "::1"})
    rate_limit_requests_per_min: int = 60
    rate_limit_burst: int = 10

    # Application layer
    max_input_length: int = 100_000  # 100KB max input
    max_output_length: int = 1_000_000  # 1MB max output
    sanitize_inputs: bool = True
    validate_schemas: bool = True

    # Runtime layer
    max_memory_mb: int = 4096  # 4GB max memory
    max_cpu_percent: float = 90.0  # 90% CPU limit
    max_execution_time_s: float = 300.0  # 5 min timeout
    sandbox_enabled: bool = True

    # Data layer
    encrypt_at_rest: bool = True
    secure_memory_wipe: bool = True
    pii_detection: bool = True

    # Detection layer
    log_all_requests: bool = True
    anomaly_detection: bool = True
    alert_threshold: int = 5  # Alerts after N suspicious events

    # Recovery layer
    auto_backup: bool = True
    backup_interval_hours: int = 24
    kill_switch_enabled: bool = True
    safe_mode_fallback: bool = True


# =============================================================================
# Input Validation
# =============================================================================

class InputValidator:
    """Validate and sanitize user inputs."""

    # Dangerous patterns (injection attempts)
    DANGEROUS_PATTERNS = [
        # Command injection
        r";\s*rm\s+-rf",
        r"&&\s*rm\s+-rf",
        r"\|\s*rm\s+-rf",
        r";\s*shutdown",
        r";\s*reboot",
        r";\s*dd\s+if=",
        r">\s*/dev/sd",

        # SQL injection (shouldn't apply but defensive)
        r"'\s*OR\s+'1'\s*=\s*'1",
        r"'\s*;\s*DROP\s+TABLE",
        r"UNION\s+SELECT",

        # Path traversal
        r"\.\./\.\./\.\.",
        r"/etc/passwd",
        r"/etc/shadow",

        # Shell escapes
        r"\$\(.*\)",
        r"`.*`",
    ]

    def __init__(self, config: ShieldConfig):
        self.config = config
        import re
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]

    def validate(self, input_text: str) -> tuple[bool, str]:
        """
        Validate user input.

        Returns (is_valid, reason).
        """
        # Check length
        if len(input_text) > self.config.max_input_length:
            return False, f"Input too long ({len(input_text)} > {self.config.max_input_length})"

        # Check for dangerous patterns
        for pattern in self._patterns:
            if pattern.search(input_text):
                logger.warning(f"Dangerous pattern detected: {pattern.pattern[:50]}")
                return False, "Potentially dangerous input detected"

        return True, "ok"

    def sanitize(self, input_text: str) -> str:
        """Sanitize input by removing dangerous characters."""
        if not self.config.sanitize_inputs:
            return input_text

        # Remove null bytes
        sanitized = input_text.replace("\x00", "")

        # Truncate if too long
        if len(sanitized) > self.config.max_input_length:
            sanitized = sanitized[:self.config.max_input_length]

        return sanitized


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(
        self,
        requests_per_min: int = 60,
        burst: int = 10,
    ):
        self.rate = requests_per_min / 60.0  # tokens per second
        self.burst = burst
        self._buckets: Dict[str, tuple[float, float]] = {}  # client_id -> (tokens, last_update)

    def allow(self, client_id: str) -> bool:
        """Check if request should be allowed."""
        now = time.time()

        if client_id not in self._buckets:
            self._buckets[client_id] = (self.burst - 1, now)
            return True

        tokens, last_update = self._buckets[client_id]

        # Add tokens for time passed
        elapsed = now - last_update
        tokens = min(self.burst, tokens + elapsed * self.rate)

        if tokens >= 1:
            self._buckets[client_id] = (tokens - 1, now)
            return True

        return False

    def reset(self, client_id: str) -> None:
        """Reset rate limit for a client."""
        self._buckets.pop(client_id, None)


# =============================================================================
# Resource Monitor
# =============================================================================

class ResourceMonitor:
    """Monitor and limit resource usage."""

    def __init__(self, config: ShieldConfig):
        self.config = config
        self._start_times: Dict[str, float] = {}

    def check_memory(self) -> tuple[bool, float]:
        """Check if memory usage is within limits."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb < self.config.max_memory_mb, memory_mb
        except ImportError:
            return True, 0.0

    def check_cpu(self) -> tuple[bool, float]:
        """Check if CPU usage is within limits."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return cpu_percent < self.config.max_cpu_percent, cpu_percent
        except ImportError:
            return True, 0.0

    def start_timer(self, operation_id: str) -> None:
        """Start timing an operation."""
        self._start_times[operation_id] = time.time()

    def check_timeout(self, operation_id: str) -> tuple[bool, float]:
        """Check if operation has exceeded timeout."""
        if operation_id not in self._start_times:
            return True, 0.0

        elapsed = time.time() - self._start_times[operation_id]
        return elapsed < self.config.max_execution_time_s, elapsed

    def end_timer(self, operation_id: str) -> float:
        """End timing and return duration."""
        if operation_id in self._start_times:
            elapsed = time.time() - self._start_times.pop(operation_id)
            return elapsed
        return 0.0


# =============================================================================
# Anomaly Detection
# =============================================================================

@dataclass
class SecurityEvent:
    """A security-relevant event."""
    timestamp: float
    event_type: str
    client_id: str
    details: Dict[str, Any]
    threat_level: ThreatLevel


class AnomalyDetector:
    """Detect anomalous patterns in requests."""

    def __init__(self, config: ShieldConfig):
        self.config = config
        self._events: List[SecurityEvent] = []
        self._client_event_counts: Dict[str, int] = {}

    def record_event(
        self,
        event_type: str,
        client_id: str,
        details: Dict[str, Any],
        threat_level: ThreatLevel = ThreatLevel.LOW,
    ) -> None:
        """Record a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            client_id=client_id,
            details=details,
            threat_level=threat_level,
        )
        self._events.append(event)

        # Track per-client counts
        self._client_event_counts[client_id] = self._client_event_counts.get(client_id, 0) + 1

        # Check threshold
        if self._client_event_counts[client_id] >= self.config.alert_threshold:
            self._trigger_alert(client_id)

        # Cleanup old events (keep last hour)
        cutoff = time.time() - 3600
        self._events = [e for e in self._events if e.timestamp > cutoff]

    def _trigger_alert(self, client_id: str) -> None:
        """Trigger security alert."""
        logger.warning(f"SECURITY ALERT: Client {client_id} exceeded event threshold")

    def get_threat_level(self, client_id: str) -> ThreatLevel:
        """Get current threat level for a client."""
        count = self._client_event_counts.get(client_id, 0)
        if count >= self.config.alert_threshold * 2:
            return ThreatLevel.HIGH
        if count >= self.config.alert_threshold:
            return ThreatLevel.MEDIUM
        if count > 0:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def get_recent_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events."""
        return self._events[-limit:]


# =============================================================================
# Data Protection
# =============================================================================

class DataProtector:
    """Protect sensitive data."""

    # Simple PII patterns (not comprehensive)
    PII_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card (basic)
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    ]

    def __init__(self, config: ShieldConfig):
        self.config = config
        import re
        self._pii_patterns = [re.compile(p) for p in self.PII_PATTERNS]

    def detect_pii(self, text: str) -> List[str]:
        """Detect potential PII in text."""
        if not self.config.pii_detection:
            return []

        findings = []
        for pattern in self._pii_patterns:
            if pattern.search(text):
                findings.append(pattern.pattern)
        return findings

    def redact_pii(self, text: str) -> str:
        """Redact potential PII from text."""
        redacted = text
        for pattern in self._pii_patterns:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted

    @staticmethod
    def secure_hash(data: str) -> str:
        """Create a secure hash of data."""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)


# =============================================================================
# Shield (Main Class)
# =============================================================================

class Shield:
    """
    ARA SHIELD: Main security interface.

    Coordinates all security layers and manages the Shield state machine.

    The Shield:
    1. Monitors for threats (debuggers, hooks, root)
    2. Transitions to Sanctuary when threats detected
    3. Limits capabilities based on current mode
    4. Always fails safe - degraded, not dead

    Key invariant:
    "If someone attacks Ara, she gets smaller and gentler, not harsher or more dangerous."
    """

    def __init__(self, config: Optional[ShieldConfig] = None):
        self.config = config or ShieldConfig()

        # Shield state machine
        self.state = ShieldState()

        # Initialize layers
        self.validator = InputValidator(self.config)
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_requests_per_min,
            self.config.rate_limit_burst,
        )
        self.resource_monitor = ResourceMonitor(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.data_protector = DataProtector(self.config)

        # Run initial RASP check
        self._initial_rasp_check()

    def authorize(
        self,
        operation: str,
        client_id: str,
        input_data: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Authorize an operation.

        Returns (allowed, reason).
        """
        # Check rate limit
        if not self.rate_limiter.allow(client_id):
            self.anomaly_detector.record_event(
                "rate_limit_exceeded",
                client_id,
                {"operation": operation},
                ThreatLevel.MEDIUM,
            )
            return False, "Rate limit exceeded"

        # Validate input
        if input_data:
            valid, reason = self.validator.validate(input_data)
            if not valid:
                self.anomaly_detector.record_event(
                    "invalid_input",
                    client_id,
                    {"operation": operation, "reason": reason},
                    ThreatLevel.MEDIUM,
                )
                return False, reason

        # Check resources
        mem_ok, mem_mb = self.resource_monitor.check_memory()
        if not mem_ok:
            self.anomaly_detector.record_event(
                "resource_exhaustion",
                client_id,
                {"operation": operation, "memory_mb": mem_mb},
                ThreatLevel.HIGH,
            )
            return False, f"Memory limit exceeded ({mem_mb:.0f}MB)"

        # Check client threat level
        threat = self.anomaly_detector.get_threat_level(client_id)
        if threat in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            return False, f"Client blocked due to {threat.value} threat level"

        return True, "ok"

    def sanitize_input(self, input_data: str) -> str:
        """Sanitize user input."""
        return self.validator.sanitize(input_data)

    def protect_output(self, output_data: str) -> str:
        """Protect sensitive data in output."""
        # Redact PII
        protected = self.data_protector.redact_pii(output_data)

        # Truncate if too long
        if len(protected) > self.config.max_output_length:
            protected = protected[:self.config.max_output_length] + "...[truncated]"

        return protected

    def record_suspicious_activity(
        self,
        client_id: str,
        activity: str,
        details: Dict[str, Any],
    ) -> None:
        """Record suspicious activity."""
        self.anomaly_detector.record_event(
            activity,
            client_id,
            details,
            ThreatLevel.MEDIUM,
        )

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        mem_ok, mem_mb = self.resource_monitor.check_memory()
        cpu_ok, cpu_pct = self.resource_monitor.check_cpu()

        return {
            # Shield state
            "mode": self.state.mode.value,
            "is_sanctuary": self.state.is_sanctuary(),
            "allows_cathedral": self.state.allows_cathedral(),
            "allows_payments": self.state.allows_payments(),
            "max_autonomy": self.state.max_autonomy_level(),
            "sanctuary_entries": self.state.sanctuary_entries,
            "last_trigger": self.state.last_trigger.value if self.state.last_trigger else None,

            # Resources
            "memory_ok": mem_ok,
            "memory_mb": mem_mb,
            "cpu_ok": cpu_ok,
            "cpu_percent": cpu_pct,
            "recent_events": len(self.anomaly_detector.get_recent_events()),

            # Config
            "config": {
                "vpn_required": self.config.vpn_required,
                "rate_limit": self.config.rate_limit_requests_per_min,
                "sandbox_enabled": self.config.sandbox_enabled,
                "encrypt_at_rest": self.config.encrypt_at_rest,
            },
        }

    def _initial_rasp_check(self) -> None:
        """Run initial Runtime Application Self-Protection check."""
        # Check for debugger
        if _detect_debugger():
            self.state.enter_sanctuary(
                ShieldTrigger.DEBUGGER_DETECTED,
                "Debugger detected at startup"
            )
            return

        # Check for root/jailbreak
        if _detect_root_or_jailbreak():
            self.state.enter_sanctuary(
                ShieldTrigger.ROOT_OR_JAILBREAK,
                "Device appears to be rooted or jailbroken"
            )
            return

        # Check for hooks
        if _detect_hooks():
            self.state.enter_sanctuary(
                ShieldTrigger.HOOK_DETECTED,
                "Hooking framework detected"
            )
            return

        logger.info("SHIELD: Initial RASP check passed")

    def run_rasp_check(self) -> bool:
        """
        Run a Runtime Application Self-Protection check.

        Returns True if environment is safe.
        If unsafe, automatically enters Sanctuary mode.
        """
        # Check for debugger
        if _detect_debugger():
            self.state.enter_sanctuary(
                ShieldTrigger.DEBUGGER_DETECTED,
                "Debugger detected during runtime"
            )
            return False

        # Check for hooks
        if _detect_hooks():
            self.state.enter_sanctuary(
                ShieldTrigger.HOOK_DETECTED,
                "Hooking framework detected during runtime"
            )
            return False

        return True

    def enter_sanctuary(self, reason: str = "Manual trigger") -> None:
        """Manually enter Sanctuary mode."""
        self.state.enter_sanctuary(ShieldTrigger.USER_REQUEST, reason)

    def try_exit_sanctuary(self) -> bool:
        """Try to exit Sanctuary mode (if safe)."""
        return self.state.exit_sanctuary(ShieldTrigger.SAFE_RESTART)

    def get_user_message(self) -> str:
        """Get user-friendly explanation of current Shield state."""
        return self.state.get_user_message()


# =============================================================================
# Security Audit
# =============================================================================

def run_security_audit() -> Dict[str, Any]:
    """
    Run a security audit of the current environment.

    Returns a report of findings and recommendations.
    """
    report = {
        "timestamp": time.time(),
        "findings": [],
        "recommendations": [],
        "score": 100,  # Start at 100, deduct for issues
    }

    # Check if running as root (bad)
    if os.geteuid() == 0:
        report["findings"].append("Running as root user")
        report["recommendations"].append("Run Ara as a non-root user")
        report["score"] -= 20

    # Check file permissions
    sensitive_paths = [
        Path.home() / ".ara",
        Path("/var/ara"),
    ]
    for path in sensitive_paths:
        if path.exists():
            mode = path.stat().st_mode
            if mode & 0o077:  # World or group readable/writable
                report["findings"].append(f"Loose permissions on {path}")
                report["recommendations"].append(f"Run: chmod 700 {path}")
                report["score"] -= 10

    # Check for kill switch file
    kill_paths = [
        Path("/var/ara/kill_switch"),
        Path.home() / ".ara" / "kill_switch",
    ]
    kill_switch_present = any(p.exists() for p in kill_paths)
    if not kill_switch_present:
        report["findings"].append("Kill switch not configured")
        report["recommendations"].append("Create kill switch file location")
        report["score"] -= 5

    # Check environment variables for secrets
    dangerous_env = ["API_KEY", "SECRET", "PASSWORD", "TOKEN"]
    for key in os.environ:
        if any(d in key.upper() for d in dangerous_env):
            report["findings"].append(f"Sensitive env var: {key}")
            report["recommendations"].append(f"Use secrets manager instead of env var for {key}")
            report["score"] -= 5

    # Check network exposure
    try:
        import socket
        # Check if default ports are exposed
        for port in [8080, 443, 80]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('0.0.0.0', port))
            sock.close()
            if result == 0:
                report["findings"].append(f"Port {port} appears to be listening on all interfaces")
                report["recommendations"].append(f"Bind to 127.0.0.1 or use VPN")
                report["score"] -= 10
    except Exception:
        pass

    # Clamp score
    report["score"] = max(0, report["score"])

    # Overall assessment
    if report["score"] >= 80:
        report["assessment"] = "GOOD - Minor improvements recommended"
    elif report["score"] >= 60:
        report["assessment"] = "FAIR - Several issues should be addressed"
    elif report["score"] >= 40:
        report["assessment"] = "POOR - Significant security improvements needed"
    else:
        report["assessment"] = "CRITICAL - Immediate action required"

    return report


# =============================================================================
# Convenience Functions
# =============================================================================

_default_shield: Optional[Shield] = None


def get_shield() -> Shield:
    """Get the default Shield instance."""
    global _default_shield
    if _default_shield is None:
        _default_shield = Shield()
    return _default_shield


def quick_authorize(operation: str, client_id: str, input_data: Optional[str] = None) -> bool:
    """Quick authorization check."""
    allowed, _ = get_shield().authorize(operation, client_id, input_data)
    return allowed
