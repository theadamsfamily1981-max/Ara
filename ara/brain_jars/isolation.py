"""
Brain Jar Isolation Layer

Ensures complete separation between tenant data:
- Each jar gets its own directory tree
- No cross-jar data access
- No access to Founder's private state
- Audit logging of all access attempts
"""

from __future__ import annotations
import os
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Base directory for all brain jars (configurable)
BRAIN_JAR_BASE = Path(os.environ.get("ARA_BRAIN_JAR_BASE", "/var/ara/brain_jars"))

# Founder's private directories (NEVER accessible to jars)
FOUNDER_PROTECTED_PATHS = [
    "/var/ara/founder",
    "/var/ara/cathedral",
    "/var/ara/sovereign",
    "/etc/ara",
    "/home",  # No home directory access
]

# Subdirectories created for each jar
JAR_SUBDIRS = [
    "memory",       # EternalMemory shard
    "conversations",  # Chat logs
    "preferences",  # User preferences
    "state",        # Session state
    "exports",      # Data export staging
    "logs",         # Per-jar audit logs
]


# =============================================================================
# Jar Store
# =============================================================================

@dataclass
class BrainJarStore:
    """
    Isolated storage for a single brain jar tenant.

    All data for this user lives under:
        {BRAIN_JAR_BASE}/{user_id}/

    With subdirectories:
        memory/         - EternalMemory shard
        conversations/  - Chat history
        preferences/    - User preferences
        state/          - Session state
        exports/        - Data export staging
        logs/           - Audit logs
    """
    user_id: str
    base_path: Path = field(default_factory=lambda: BRAIN_JAR_BASE)

    def __post_init__(self):
        self.jar_path = self.base_path / self.user_id
        self._initialized = False

    @property
    def memory_path(self) -> Path:
        return self.jar_path / "memory"

    @property
    def conversations_path(self) -> Path:
        return self.jar_path / "conversations"

    @property
    def preferences_path(self) -> Path:
        return self.jar_path / "preferences"

    @property
    def state_path(self) -> Path:
        return self.jar_path / "state"

    @property
    def exports_path(self) -> Path:
        return self.jar_path / "exports"

    @property
    def logs_path(self) -> Path:
        return self.jar_path / "logs"

    @property
    def policy_path(self) -> Path:
        return self.jar_path / "policy.yaml"

    def initialize(self) -> None:
        """Create the jar directory structure."""
        if self._initialized:
            return

        # Validate user_id (prevent path traversal)
        if not self._is_safe_user_id(self.user_id):
            raise ValueError(f"Invalid user_id: {self.user_id}")

        # Create directories
        for subdir in JAR_SUBDIRS:
            (self.jar_path / subdir).mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions (owner only)
        os.chmod(self.jar_path, 0o700)

        self._initialized = True
        self._audit_log("jar_initialized", {"user_id": self.user_id})
        logger.info(f"Brain jar initialized: {self.user_id}")

    def exists(self) -> bool:
        """Check if this jar exists."""
        return self.jar_path.exists()

    def _is_safe_user_id(self, user_id: str) -> bool:
        """Validate user_id to prevent path traversal attacks."""
        if not user_id:
            return False
        if ".." in user_id or "/" in user_id or "\\" in user_id:
            return False
        if user_id.startswith("."):
            return False
        if len(user_id) > 64:
            return False
        # Only allow alphanumeric, underscore, hyphen
        return all(c.isalnum() or c in "_-" for c in user_id)

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if a path is within this jar's boundary."""
        try:
            resolved = path.resolve()
            jar_resolved = self.jar_path.resolve()

            # Must be within jar directory
            if not str(resolved).startswith(str(jar_resolved)):
                return False

            # Must not be in protected paths
            for protected in FOUNDER_PROTECTED_PATHS:
                if str(resolved).startswith(protected):
                    return False

            return True
        except Exception:
            return False

    def _audit_log(self, event: str, details: Dict[str, Any]) -> None:
        """Write to per-jar audit log."""
        if not self.logs_path.exists():
            self.logs_path.mkdir(parents=True, exist_ok=True)

        log_file = self.logs_path / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": self.user_id,
            "event": event,
            "details": details,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # =========================================================================
    # Safe File Operations
    # =========================================================================

    def read_file(self, relative_path: str) -> Optional[bytes]:
        """Safely read a file from within the jar."""
        full_path = self.jar_path / relative_path
        if not self._is_path_allowed(full_path):
            self._audit_log("access_denied", {"path": relative_path, "op": "read"})
            logger.warning(f"Access denied: {self.user_id} tried to read {relative_path}")
            return None

        if not full_path.exists():
            return None

        self._audit_log("file_read", {"path": relative_path})
        return full_path.read_bytes()

    def write_file(self, relative_path: str, data: bytes) -> bool:
        """Safely write a file within the jar."""
        full_path = self.jar_path / relative_path
        if not self._is_path_allowed(full_path):
            self._audit_log("access_denied", {"path": relative_path, "op": "write"})
            logger.warning(f"Access denied: {self.user_id} tried to write {relative_path}")
            return False

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)
        self._audit_log("file_write", {"path": relative_path, "size": len(data)})
        return True

    def delete_file(self, relative_path: str) -> bool:
        """Safely delete a file within the jar."""
        full_path = self.jar_path / relative_path
        if not self._is_path_allowed(full_path):
            self._audit_log("access_denied", {"path": relative_path, "op": "delete"})
            return False

        if full_path.exists():
            full_path.unlink()
            self._audit_log("file_delete", {"path": relative_path})
            return True
        return False

    def list_files(self, relative_path: str = "") -> List[str]:
        """List files within a jar subdirectory."""
        full_path = self.jar_path / relative_path
        if not self._is_path_allowed(full_path):
            return []

        if not full_path.is_dir():
            return []

        return [f.name for f in full_path.iterdir()]

    # =========================================================================
    # Data Export & Deletion
    # =========================================================================

    def export_all_data(self) -> Path:
        """Export all jar data to a tarball for user download."""
        import tarfile
        from datetime import datetime

        export_name = f"{self.user_id}_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        export_path = self.exports_path / export_name

        with tarfile.open(export_path, "w:gz") as tar:
            for subdir in ["memory", "conversations", "preferences"]:
                subdir_path = self.jar_path / subdir
                if subdir_path.exists():
                    tar.add(subdir_path, arcname=subdir)

        self._audit_log("data_exported", {"export_file": export_name})
        logger.info(f"Data exported for {self.user_id}: {export_path}")
        return export_path

    def nuke_jar(self, confirm: bool = False) -> bool:
        """
        Completely delete this brain jar and all its data.

        IRREVERSIBLE. Requires explicit confirmation.
        """
        if not confirm:
            logger.warning(f"Nuke rejected for {self.user_id}: confirmation required")
            return False

        if not self.jar_path.exists():
            return True

        # Log before deletion
        self._audit_log("jar_nuked", {"user_id": self.user_id, "confirmed": True})

        # Delete everything
        shutil.rmtree(self.jar_path)
        logger.info(f"Brain jar NUKED: {self.user_id}")
        return True

    # =========================================================================
    # Memory Shard Access
    # =========================================================================

    def get_memory_shard_path(self) -> Path:
        """Get path to this jar's EternalMemory shard."""
        return self.memory_path / "eternal_memory.db"

    def get_axis_state_path(self) -> Path:
        """Get path to this jar's AxisMundi state."""
        return self.state_path / "axis_mundi.bin"

    def get_conversation_log_path(self, session_id: str) -> Path:
        """Get path to a conversation log file."""
        safe_session = "".join(c for c in session_id if c.isalnum() or c in "_-")
        return self.conversations_path / f"{safe_session}.jsonl"


# =============================================================================
# Global Jar Store Factory
# =============================================================================

_jar_stores: Dict[str, BrainJarStore] = {}


def get_jar_store(user_id: str, auto_create: bool = False) -> Optional[BrainJarStore]:
    """
    Get the store for a specific brain jar.

    Args:
        user_id: The user identifier
        auto_create: If True, create the jar if it doesn't exist

    Returns:
        BrainJarStore or None if jar doesn't exist and auto_create=False
    """
    if user_id in _jar_stores:
        return _jar_stores[user_id]

    store = BrainJarStore(user_id=user_id)

    if not store.exists():
        if auto_create:
            store.initialize()
        else:
            return None

    _jar_stores[user_id] = store
    return store


def list_all_jars() -> List[str]:
    """List all brain jar user IDs."""
    if not BRAIN_JAR_BASE.exists():
        return []

    return [
        d.name for d in BRAIN_JAR_BASE.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]


def verify_isolation(user_id: str, path: str) -> bool:
    """
    Verify that a path access is allowed for the given user.

    Use this at API boundaries before any file operation.
    """
    store = get_jar_store(user_id)
    if store is None:
        return False

    return store._is_path_allowed(Path(path))
