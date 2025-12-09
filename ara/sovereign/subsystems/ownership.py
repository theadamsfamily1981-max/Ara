"""
Field Ownership: Who Can Write What

Every field in SovereignState has exactly ONE owner.
This prevents race conditions and makes debugging trivial.

Ownership Matrix:
    BANOS       → hardware.*, safety.kill_switch_engaged
    MindReader  → user.*
    Covenant    → safety.autonomy_level, safety.trust_*
    ChiefOfStaff→ work.*, safety.risk_assessment
    HTC         → soul.axis_*, soul.memory_*
    Teleology   → teleology.*
    Avatar      → avatar.*
    Tracer      → trace.*
    Clock       → time.*

Violations are logged (or rejected in strict mode).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, Optional, Any, List
import re

logger = logging.getLogger(__name__)


class Subsystem(str, Enum):
    """All subsystems that can own state fields."""
    BANOS = "banos"           # Hardware telemetry
    MIND_READER = "mind_reader"  # User state sensing
    COVENANT = "covenant"      # Trust & governance
    CHIEF_OF_STAFF = "chief_of_staff"  # CEO decisions
    HTC = "htc"               # Holographic core
    TELEOLOGY = "teleology"   # Goal management
    AVATAR = "avatar"         # User interface
    TRACER = "tracer"         # Debugging/tracing
    CLOCK = "clock"           # Time management
    SANCTUARY = "sanctuary"   # Comfort shard (special)


# =============================================================================
# Ownership Registry
# =============================================================================

@dataclass
class FieldOwnership:
    """Ownership definition for a field or field pattern."""
    pattern: str          # e.g., "hardware.*" or "safety.kill_switch_engaged"
    owner: Subsystem
    read_only: bool = False  # If True, only owner can read too
    description: str = ""


# The canonical ownership matrix
OWNERSHIP_MATRIX: List[FieldOwnership] = [
    # Time (owned by Clock)
    FieldOwnership("time.*", Subsystem.CLOCK, description="All time-related fields"),

    # Hardware (owned by BANOS)
    FieldOwnership("hardware.*", Subsystem.BANOS, description="Hardware telemetry"),
    FieldOwnership("safety.kill_switch_engaged", Subsystem.BANOS, description="Physical kill switch state"),

    # User state (owned by MindReader)
    FieldOwnership("user.*", Subsystem.MIND_READER, description="User state and cognitive mode"),

    # Soul/HDC (owned by HTC)
    FieldOwnership("soul.axis_*", Subsystem.HTC, description="AxisMundi state"),
    FieldOwnership("soul.memory_*", Subsystem.HTC, description="EternalMemory stats"),
    FieldOwnership("soul.global_coherence", Subsystem.HTC, description="Coherence metric"),
    FieldOwnership("soul.resonance_score", Subsystem.HTC, description="Resonance metric"),
    FieldOwnership("soul.plasticity_mode", Subsystem.HTC, description="Learning mode"),

    # Teleology (owned by Teleology subsystem)
    FieldOwnership("teleology.*", Subsystem.TELEOLOGY, description="Goals and mission"),

    # Work (owned by ChiefOfStaff)
    FieldOwnership("work.*", Subsystem.CHIEF_OF_STAFF, description="Initiatives and skills"),
    FieldOwnership("safety.risk_assessment", Subsystem.CHIEF_OF_STAFF, description="Risk evaluation"),

    # Safety/Trust (owned by Covenant)
    FieldOwnership("safety.autonomy_level", Subsystem.COVENANT, description="Current autonomy"),
    FieldOwnership("safety.trust_*", Subsystem.COVENANT, description="Trust accounting"),
    FieldOwnership("safety.founder_protection_active", Subsystem.COVENANT, description="Protection flag"),
    FieldOwnership("safety.last_human_contact_ts", Subsystem.COVENANT, description="Human contact tracking"),

    # Avatar (owned by Avatar subsystem)
    FieldOwnership("avatar.*", Subsystem.AVATAR, description="UI/conversation state"),

    # Trace (owned by Tracer)
    FieldOwnership("trace.*", Subsystem.TRACER, description="Debug trace"),
]


class OwnershipRegistry:
    """
    Registry that tracks field ownership and validates writes.

    Usage:
        registry = OwnershipRegistry()

        # Check if a subsystem can write a field
        if registry.can_write(Subsystem.BANOS, "hardware.cpu_load"):
            state.hardware.cpu_load = 0.5

        # Or use the guarded write
        registry.write(state, Subsystem.BANOS, "hardware.cpu_load", 0.5)
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, raise on ownership violations. If False, just log.
        """
        self.strict = strict
        self._ownership = OWNERSHIP_MATRIX
        self._cache: Dict[str, Subsystem] = {}

    def get_owner(self, field_path: str) -> Optional[Subsystem]:
        """Get the owner of a field path."""
        # Check cache first
        if field_path in self._cache:
            return self._cache[field_path]

        # Find matching pattern
        for ownership in self._ownership:
            if self._matches_pattern(field_path, ownership.pattern):
                self._cache[field_path] = ownership.owner
                return ownership.owner

        return None

    def _matches_pattern(self, field_path: str, pattern: str) -> bool:
        """Check if a field path matches an ownership pattern."""
        # Convert glob-style pattern to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", r".*")
        return bool(re.match(f"^{regex_pattern}$", field_path))

    def can_write(self, subsystem: Subsystem, field_path: str) -> bool:
        """Check if a subsystem can write to a field."""
        owner = self.get_owner(field_path)
        if owner is None:
            # Unowned field - anyone can write (but log warning)
            logger.warning(f"Unowned field: {field_path}")
            return True
        return owner == subsystem

    def can_read(self, subsystem: Subsystem, field_path: str) -> bool:
        """Check if a subsystem can read a field (most can)."""
        # Find ownership entry
        for ownership in self._ownership:
            if self._matches_pattern(field_path, ownership.pattern):
                if ownership.read_only:
                    return ownership.owner == subsystem
                return True
        return True

    def validate_write(
        self,
        subsystem: Subsystem,
        field_path: str,
    ) -> bool:
        """
        Validate a write operation.

        Returns True if allowed, raises or returns False if not.
        """
        if self.can_write(subsystem, field_path):
            return True

        owner = self.get_owner(field_path)
        msg = f"Ownership violation: {subsystem.value} cannot write {field_path} (owned by {owner.value if owner else 'unknown'})"

        if self.strict:
            raise PermissionError(msg)

        logger.warning(msg)
        return False

    def list_owned_fields(self, subsystem: Subsystem) -> List[str]:
        """List all field patterns owned by a subsystem."""
        return [
            ownership.pattern
            for ownership in self._ownership
            if ownership.owner == subsystem
        ]


# =============================================================================
# Guarded State Writer
# =============================================================================

@dataclass
class WriteRecord:
    """Record of a state write."""
    tick: int
    subsystem: Subsystem
    field_path: str
    old_value: Any
    new_value: Any
    timestamp: float


class GuardedStateWriter:
    """
    State writer with ownership validation and audit trail.

    Usage:
        writer = GuardedStateWriter(state, registry)

        # This will be validated against ownership
        writer.write(Subsystem.BANOS, "hardware.cpu_load", 0.5)

        # Get audit trail
        for record in writer.get_writes():
            print(f"{record.subsystem}: {record.field_path} = {record.new_value}")
    """

    def __init__(
        self,
        state: Any,  # SovereignState
        registry: Optional[OwnershipRegistry] = None,
        record_writes: bool = True,
    ):
        self.state = state
        self.registry = registry or OwnershipRegistry()
        self.record_writes = record_writes
        self._write_log: List[WriteRecord] = []

    def write(
        self,
        subsystem: Subsystem,
        field_path: str,
        value: Any,
    ) -> bool:
        """
        Write a value to a field with ownership validation.

        Args:
            subsystem: The subsystem performing the write
            field_path: Dot-separated path (e.g., "hardware.cpu_load")
            value: The new value

        Returns:
            True if write succeeded, False if ownership violation
        """
        import time

        # Validate ownership
        if not self.registry.validate_write(subsystem, field_path):
            return False

        # Navigate to field and set value
        parts = field_path.split(".")
        obj = self.state
        old_value = None

        for part in parts[:-1]:
            obj = getattr(obj, part)

        final_field = parts[-1]
        old_value = getattr(obj, final_field, None)
        setattr(obj, final_field, value)

        # Record write
        if self.record_writes:
            self._write_log.append(WriteRecord(
                tick=getattr(self.state.time, 'tick', 0),
                subsystem=subsystem,
                field_path=field_path,
                old_value=old_value,
                new_value=value,
                timestamp=time.time(),
            ))

        return True

    def read(self, field_path: str) -> Any:
        """Read a value from a field path."""
        parts = field_path.split(".")
        obj = self.state

        for part in parts:
            obj = getattr(obj, part)

        return obj

    def get_writes(self, subsystem: Optional[Subsystem] = None) -> List[WriteRecord]:
        """Get write records, optionally filtered by subsystem."""
        if subsystem is None:
            return list(self._write_log)
        return [r for r in self._write_log if r.subsystem == subsystem]

    def clear_log(self) -> None:
        """Clear the write log."""
        self._write_log.clear()


# =============================================================================
# Subsystem Base Class
# =============================================================================

class SubsystemBase:
    """
    Base class for all subsystems.

    Provides:
    - Automatic ownership validation
    - Consistent interface
    - State access helpers
    """

    subsystem_id: Subsystem = Subsystem.BANOS  # Override in subclass

    def __init__(self, writer: GuardedStateWriter):
        self.writer = writer

    @property
    def state(self) -> Any:
        """Access the current state."""
        return self.writer.state

    def write(self, field_path: str, value: Any) -> bool:
        """Write a field (with ownership validation)."""
        return self.writer.write(self.subsystem_id, field_path, value)

    def read(self, field_path: str) -> Any:
        """Read a field."""
        return self.writer.read(field_path)

    def owned_fields(self) -> List[str]:
        """Get list of fields this subsystem owns."""
        return self.writer.registry.list_owned_fields(self.subsystem_id)


# =============================================================================
# Global Registry Instance
# =============================================================================

_default_registry: Optional[OwnershipRegistry] = None


def get_ownership_registry(strict: bool = False) -> OwnershipRegistry:
    """Get the default ownership registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = OwnershipRegistry(strict=strict)
    return _default_registry


def reset_ownership_registry() -> None:
    """Reset the default registry (for testing)."""
    global _default_registry
    _default_registry = None
