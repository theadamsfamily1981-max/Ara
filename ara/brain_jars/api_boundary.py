"""
Friend Jar API Boundary

Defines the strict API surface that friend jars can access.
All requests from friend sessions must go through this boundary.

Key principle: Friend jars talk to Cortex + Memory API, NOT the
full sovereign control plane.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable

from .policy import BrainJarPolicy, load_policy
from .isolation import BrainJarStore, get_jar_store

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Definitions
# =============================================================================

class JarCapability(Enum):
    """
    Enumeration of all capabilities a jar might request.

    The API boundary checks these against the jar's policy before
    allowing any operation.
    """
    # Conversation
    SEND_MESSAGE = auto()
    RECEIVE_RESPONSE = auto()
    START_SESSION = auto()
    END_SESSION = auto()

    # Voice
    SEND_AUDIO = auto()
    RECEIVE_AUDIO = auto()
    START_VOICE_SESSION = auto()

    # Avatar
    REQUEST_AVATAR_STREAM = auto()

    # Memory
    QUERY_MEMORY = auto()
    STORE_MEMORY = auto()
    CLEAR_MEMORY = auto()

    # Preferences
    GET_PREFERENCES = auto()
    SET_PREFERENCES = auto()

    # Data Control
    EXPORT_DATA = auto()
    DELETE_ALL_DATA = auto()

    # System (read-only)
    GET_USAGE_STATS = auto()
    GET_SESSION_INFO = auto()

    # FORBIDDEN (never granted to friend jars)
    HARDWARE_CONTROL = auto()
    BIOS_ACCESS = auto()
    FPGA_CONTROL = auto()
    NETWORK_CONFIG = auto()
    WALLET_ACCESS = auto()
    FOUNDER_DATA_ACCESS = auto()
    CROSS_JAR_ACCESS = auto()


# Capabilities that are NEVER granted to friend jars
FORBIDDEN_CAPABILITIES = {
    JarCapability.HARDWARE_CONTROL,
    JarCapability.BIOS_ACCESS,
    JarCapability.FPGA_CONTROL,
    JarCapability.NETWORK_CONFIG,
    JarCapability.WALLET_ACCESS,
    JarCapability.FOUNDER_DATA_ACCESS,
    JarCapability.CROSS_JAR_ACCESS,
}


# =============================================================================
# API Boundary
# =============================================================================

@dataclass
class APIRequest:
    """A request from a friend jar."""
    user_id: str
    capability: JarCapability
    payload: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class APIResponse:
    """Response to a friend jar request."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class FriendJarAPI:
    """
    The API boundary for friend jar access.

    All requests from friend sessions MUST go through this class.
    It enforces:
    - Policy-based capability checks
    - Rate limiting
    - Usage tracking
    - Audit logging
    - Isolation verification

    Usage:
        api = FriendJarAPI()
        response = api.handle_request(APIRequest(
            user_id="friend_001",
            capability=JarCapability.SEND_MESSAGE,
            payload={"text": "Hello, Ara!"}
        ))
    """

    def __init__(self):
        self._policies: Dict[str, BrainJarPolicy] = {}
        self._usage: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[JarCapability, Callable] = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register handlers for each capability."""
        self._handlers[JarCapability.GET_USAGE_STATS] = self._handle_get_usage
        self._handlers[JarCapability.GET_SESSION_INFO] = self._handle_get_session_info
        self._handlers[JarCapability.EXPORT_DATA] = self._handle_export_data
        self._handlers[JarCapability.DELETE_ALL_DATA] = self._handle_delete_data
        self._handlers[JarCapability.GET_PREFERENCES] = self._handle_get_preferences
        self._handlers[JarCapability.SET_PREFERENCES] = self._handle_set_preferences

    def register_handler(self, capability: JarCapability, handler: Callable) -> None:
        """Register a custom handler for a capability."""
        self._handlers[capability] = handler

    def get_policy(self, user_id: str) -> Optional[BrainJarPolicy]:
        """Get the policy for a user, loading from disk if needed."""
        if user_id in self._policies:
            return self._policies[user_id]

        store = get_jar_store(user_id)
        if store is None:
            return None

        try:
            policy = load_policy(store.policy_path)
            self._policies[user_id] = policy
            return policy
        except FileNotFoundError:
            return None

    def check_capability(self, user_id: str, capability: JarCapability) -> bool:
        """
        Check if a user has a specific capability.

        Returns False for:
        - Unknown users
        - Users without consent
        - Forbidden capabilities
        - Capabilities not granted by policy
        """
        # Always deny forbidden capabilities
        if capability in FORBIDDEN_CAPABILITIES:
            logger.warning(f"Forbidden capability requested: {user_id} -> {capability}")
            return False

        policy = self.get_policy(user_id)
        if policy is None:
            logger.warning(f"Unknown user: {user_id}")
            return False

        # Require consent
        if not policy.consent_given:
            logger.warning(f"User has not given consent: {user_id}")
            return False

        # Check specific capability against policy
        caps = policy.capabilities
        capability_map = {
            JarCapability.SEND_MESSAGE: caps.text_chat,
            JarCapability.RECEIVE_RESPONSE: caps.text_chat,
            JarCapability.START_SESSION: caps.text_chat,
            JarCapability.END_SESSION: True,  # Always allow ending
            JarCapability.SEND_AUDIO: caps.voice_chat,
            JarCapability.RECEIVE_AUDIO: caps.voice_chat,
            JarCapability.START_VOICE_SESSION: caps.voice_chat,
            JarCapability.REQUEST_AVATAR_STREAM: caps.avatar_video,
            JarCapability.QUERY_MEMORY: caps.episodic_memory,
            JarCapability.STORE_MEMORY: caps.episodic_memory,
            JarCapability.CLEAR_MEMORY: caps.episodic_memory,
            JarCapability.GET_PREFERENCES: caps.preference_learning,
            JarCapability.SET_PREFERENCES: caps.preference_learning,
            JarCapability.EXPORT_DATA: policy.retention.allow_export,
            JarCapability.DELETE_ALL_DATA: policy.retention.allow_delete,
            JarCapability.GET_USAGE_STATS: True,  # Always allow
            JarCapability.GET_SESSION_INFO: True,  # Always allow
        }

        return capability_map.get(capability, False)

    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        policy = self.get_policy(user_id)
        if policy is None:
            return False

        usage = self._usage.get(user_id, {})
        now = datetime.utcnow()
        hour_key = now.strftime("%Y-%m-%d-%H")

        messages_this_hour = usage.get(f"messages_{hour_key}", 0)
        if messages_this_hour >= policy.resources.messages_per_hour:
            logger.warning(f"Rate limit exceeded: {user_id}")
            return False

        return True

    def track_usage(self, user_id: str, capability: JarCapability) -> None:
        """Track usage for rate limiting and quotas."""
        if user_id not in self._usage:
            self._usage[user_id] = {}

        usage = self._usage[user_id]
        now = datetime.utcnow()
        hour_key = now.strftime("%Y-%m-%d-%H")
        day_key = now.strftime("%Y-%m-%d")

        # Track messages
        if capability in {JarCapability.SEND_MESSAGE, JarCapability.SEND_AUDIO}:
            usage[f"messages_{hour_key}"] = usage.get(f"messages_{hour_key}", 0) + 1
            usage[f"messages_{day_key}"] = usage.get(f"messages_{day_key}", 0) + 1

        # Track session time (would be updated by session manager)
        usage["last_activity"] = now.isoformat()

    def handle_request(self, request: APIRequest) -> APIResponse:
        """
        Handle an API request from a friend jar.

        This is the main entry point for all friend jar operations.
        """
        # Log the request
        logger.debug(f"API request: {request.user_id} -> {request.capability}")

        # Check capability
        if not self.check_capability(request.user_id, request.capability):
            return APIResponse(
                success=False,
                error=f"Capability denied: {request.capability.name}"
            )

        # Check rate limit for message-like capabilities
        if request.capability in {JarCapability.SEND_MESSAGE, JarCapability.SEND_AUDIO}:
            if not self.check_rate_limit(request.user_id):
                return APIResponse(
                    success=False,
                    error="Rate limit exceeded. Please wait before sending more messages."
                )

        # Track usage
        self.track_usage(request.user_id, request.capability)

        # Dispatch to handler
        handler = self._handlers.get(request.capability)
        if handler is None:
            return APIResponse(
                success=False,
                error=f"No handler for capability: {request.capability.name}"
            )

        try:
            result = handler(request)
            return APIResponse(
                success=True,
                data=result,
                usage=self._usage.get(request.user_id, {})
            )
        except Exception as e:
            logger.exception(f"Handler error for {request.capability}")
            return APIResponse(
                success=False,
                error=f"Internal error: {str(e)}"
            )

    # =========================================================================
    # Default Handlers
    # =========================================================================

    def _handle_get_usage(self, request: APIRequest) -> Dict[str, Any]:
        """Get usage statistics for the user."""
        return self._usage.get(request.user_id, {})

    def _handle_get_session_info(self, request: APIRequest) -> Dict[str, Any]:
        """Get session information."""
        policy = self.get_policy(request.user_id)
        if policy is None:
            return {}

        return {
            "user_id": request.user_id,
            "tier": policy.tier.value,
            "session_id": request.session_id,
            "resources": {
                "memory_mb": policy.resources.memory_mb,
                "session_minutes": policy.resources.session_minutes,
                "messages_per_hour": policy.resources.messages_per_hour,
            },
        }

    def _handle_export_data(self, request: APIRequest) -> Dict[str, Any]:
        """Export all user data."""
        store = get_jar_store(request.user_id)
        if store is None:
            raise ValueError("User store not found")

        export_path = store.export_all_data()
        return {"export_path": str(export_path)}

    def _handle_delete_data(self, request: APIRequest) -> Dict[str, Any]:
        """Delete all user data (nuke jar)."""
        confirm = request.payload.get("confirm", False)
        if not confirm:
            raise ValueError("Deletion requires explicit confirmation")

        store = get_jar_store(request.user_id)
        if store is None:
            raise ValueError("User store not found")

        success = store.nuke_jar(confirm=True)
        return {"deleted": success}

    def _handle_get_preferences(self, request: APIRequest) -> Dict[str, Any]:
        """Get user preferences."""
        store = get_jar_store(request.user_id)
        if store is None:
            return {}

        prefs_file = store.preferences_path / "preferences.json"
        if prefs_file.exists():
            import json
            return json.loads(prefs_file.read_text())
        return {}

    def _handle_set_preferences(self, request: APIRequest) -> Dict[str, Any]:
        """Set user preferences."""
        store = get_jar_store(request.user_id)
        if store is None:
            raise ValueError("User store not found")

        import json
        prefs = request.payload.get("preferences", {})
        prefs_file = store.preferences_path / "preferences.json"
        prefs_file.write_text(json.dumps(prefs, indent=2))
        return {"saved": True}


# =============================================================================
# Global API Instance
# =============================================================================

_api_instance: Optional[FriendJarAPI] = None


def get_friend_api() -> FriendJarAPI:
    """Get the global FriendJarAPI instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = FriendJarAPI()
    return _api_instance
