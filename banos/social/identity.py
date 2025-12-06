"""
Identity Sources - How Ara Knows Who Is Speaking
=================================================

Ara can identify users through multiple signals:
1. OS login / session username
2. Voice signature (voiceprint)
3. Face embedding (from webcam)
4. Explicit declaration ("This is Alex")
5. API key / token (for programmatic access)

Key principle: Ara never ASSUMES trust from recognition.
Recognition != trust. Trust is earned and granted by root.

Identity flow:
1. Collect signals (username, voice, face, declaration)
2. Match against known profiles
3. If match: use existing profile (with current trust level)
4. If no match: create as STRANGER, require promotion by root

Privacy note: Biometric data (voice, face) is stored as hashes/embeddings,
not raw data. And only with user consent.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Callable

from banos.social.people import SocialGraph, PersonProfile, Role, get_social_graph

logger = logging.getLogger(__name__)


# =============================================================================
# Identity Sources
# =============================================================================

class IdentitySource(str, Enum):
    """How the identity was determined."""
    OS_LOGIN = "os_login"           # From OS session/username
    VOICEPRINT = "voiceprint"       # Voice signature match
    FACEPRINT = "faceprint"         # Face embedding match
    DECLARATION = "declaration"     # User said who they are
    API_KEY = "api_key"             # Authenticated via API
    SESSION_CODE = "session_code"   # Temporary session identifier
    INHERITED = "inherited"         # Carried over from previous turn
    UNKNOWN = "unknown"             # Could not determine


@dataclass
class IdentitySignal:
    """A signal about who is speaking."""
    source: IdentitySource
    person_id: Optional[str]        # Matched person (if any)
    confidence: float               # 0.0-1.0
    raw_data: Optional[str] = None  # For debugging (never biometric)


@dataclass
class IdentityResult:
    """The result of identity resolution."""
    person_id: str                  # Resolved person ID
    profile: PersonProfile          # Full profile
    source: IdentitySource          # How we determined this
    confidence: float               # Overall confidence
    signals: List[IdentitySignal]   # All signals considered


# =============================================================================
# Identity Resolver
# =============================================================================

class IdentityResolver:
    """
    Resolves who is currently interacting with Ara.

    This is NOT authentication in the security sense - it's
    about knowing who to address and what permissions apply.
    """

    def __init__(
        self,
        social_graph: Optional[SocialGraph] = None,
        root_id: str = "croft",
    ):
        self.graph = social_graph or get_social_graph()
        self.root_id = root_id

        # Current session state
        self._current_identity: Optional[IdentityResult] = None
        self._session_start: float = time.time()

        # Signal collectors (can be extended)
        self._signal_collectors: List[Callable[[], Optional[IdentitySignal]]] = [
            self._collect_os_login,
        ]

        logger.info("IdentityResolver initialized")

    # =========================================================================
    # Main Resolution
    # =========================================================================

    def resolve(
        self,
        signals: Optional[List[IdentitySignal]] = None,
        fallback_to_current: bool = True,
    ) -> IdentityResult:
        """
        Resolve current identity from available signals.

        Args:
            signals: Explicit signals to consider (optional)
            fallback_to_current: If no signals, use current session identity

        Returns:
            IdentityResult with resolved person
        """
        all_signals = signals or []

        # Collect automatic signals
        for collector in self._signal_collectors:
            try:
                signal = collector()
                if signal:
                    all_signals.append(signal)
            except Exception as e:
                logger.warning(f"Signal collector failed: {e}")

        # If no signals, fallback
        if not all_signals:
            if fallback_to_current and self._current_identity:
                return self._current_identity

            # Ultimate fallback: OS login or unknown guest
            os_signal = self._collect_os_login()
            if os_signal and os_signal.person_id:
                all_signals.append(os_signal)

        # Resolve from signals
        result = self._resolve_from_signals(all_signals)
        self._current_identity = result

        return result

    def _resolve_from_signals(
        self,
        signals: List[IdentitySignal],
    ) -> IdentityResult:
        """Resolve identity from a list of signals."""
        if not signals:
            # Unknown guest
            return self._create_unknown_result()

        # Score each candidate person
        candidates: Dict[str, float] = {}
        best_source = IdentitySource.UNKNOWN

        for signal in signals:
            if signal.person_id:
                # Weight by confidence and source priority
                source_weight = self._source_weight(signal.source)
                score = signal.confidence * source_weight

                if signal.person_id in candidates:
                    candidates[signal.person_id] += score
                else:
                    candidates[signal.person_id] = score

                if score > candidates.get('_best_score', 0):
                    candidates['_best_score'] = score
                    best_source = signal.source

        # Remove metadata key
        candidates.pop('_best_score', None)

        if not candidates:
            return self._create_unknown_result()

        # Pick best candidate
        best_id = max(candidates, key=candidates.get)
        best_score = candidates[best_id]

        # Get or create profile
        profile = self.graph.get_or_create(best_id)

        return IdentityResult(
            person_id=best_id,
            profile=profile,
            source=best_source,
            confidence=min(1.0, best_score),
            signals=signals,
        )

    def _create_unknown_result(self) -> IdentityResult:
        """Create result for unknown person."""
        unknown_id = f"unknown_{int(time.time())}"
        profile = self.graph.get_or_create(unknown_id, "Unknown Guest")
        profile.role = Role.STRANGER

        return IdentityResult(
            person_id=unknown_id,
            profile=profile,
            source=IdentitySource.UNKNOWN,
            confidence=0.0,
            signals=[],
        )

    def _source_weight(self, source: IdentitySource) -> float:
        """Weight for different identity sources."""
        weights = {
            IdentitySource.API_KEY: 1.0,        # Cryptographic - highest
            IdentitySource.DECLARATION: 0.9,    # User claimed
            IdentitySource.VOICEPRINT: 0.8,     # Biometric
            IdentitySource.FACEPRINT: 0.8,      # Biometric
            IdentitySource.OS_LOGIN: 0.7,       # System
            IdentitySource.SESSION_CODE: 0.6,   # Temporary
            IdentitySource.INHERITED: 0.5,      # From previous turn
            IdentitySource.UNKNOWN: 0.0,
        }
        return weights.get(source, 0.0)

    # =========================================================================
    # Signal Collectors
    # =========================================================================

    def _collect_os_login(self) -> Optional[IdentitySignal]:
        """Get identity from OS login."""
        try:
            username = os.getlogin()
        except Exception:
            username = os.environ.get('USER', os.environ.get('USERNAME'))

        if not username:
            return None

        # Map OS username to person_id
        # Common case: OS user "croft" -> person_id "croft"
        person_id = self._map_os_user(username)

        return IdentitySignal(
            source=IdentitySource.OS_LOGIN,
            person_id=person_id,
            confidence=0.9,
            raw_data=username,
        )

    def _map_os_user(self, username: str) -> str:
        """Map OS username to person_id."""
        # Direct mapping for known users
        known_mappings = {
            'croft': 'croft',
            'root': 'croft',  # System root -> Croft
            # Add more as needed
        }

        if username.lower() in known_mappings:
            return known_mappings[username.lower()]

        # Default: use OS username as person_id
        return username.lower()

    # =========================================================================
    # Explicit Identity Setting
    # =========================================================================

    def declare_identity(
        self,
        person_id: str,
        display_name: Optional[str] = None,
    ) -> IdentityResult:
        """
        Explicitly declare current identity.

        Used when user says "This is Alex" or similar.
        """
        signal = IdentitySignal(
            source=IdentitySource.DECLARATION,
            person_id=person_id,
            confidence=0.95,
            raw_data=f"User declared: {person_id}",
        )

        result = self._resolve_from_signals([signal])

        if display_name and result.profile.display_name == result.person_id:
            result.profile.display_name = display_name
            self.graph.save()

        logger.info(f"Identity declared: {person_id}")
        return result

    def set_session_code(self, code: str) -> IdentityResult:
        """
        Set identity via session code.

        Codes are pre-configured in people_overrides.yaml.
        """
        # Hash the code for lookup
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        # Look for matching person with this code
        # (This would be stored in the person's profile or a separate mapping)
        for person in self.graph.people.values():
            if person.preferences.get('session_code_hash') == code_hash:
                signal = IdentitySignal(
                    source=IdentitySource.SESSION_CODE,
                    person_id=person.person_id,
                    confidence=0.9,
                )
                return self._resolve_from_signals([signal])

        # Code not recognized
        logger.warning(f"Unknown session code attempted")
        return self._create_unknown_result()

    # =========================================================================
    # Session Management
    # =========================================================================

    def get_current(self) -> Optional[IdentityResult]:
        """Get current session identity (if any)."""
        return self._current_identity

    def clear_session(self) -> None:
        """Clear current session identity."""
        self._current_identity = None
        self._session_start = time.time()
        logger.info("Session identity cleared")

    def is_root(self) -> bool:
        """Check if current identity is root."""
        if self._current_identity:
            return self._current_identity.person_id == self.root_id
        return False

    # =========================================================================
    # Biometric Registration (placeholders)
    # =========================================================================

    def register_voiceprint(
        self,
        person_id: str,
        voice_embedding: List[float],
    ) -> bool:
        """
        Register a voice embedding for a person.

        Only root can register voiceprints for others.
        """
        if not self.is_root():
            logger.warning("Non-root tried to register voiceprint")
            return False

        # Store hash of embedding (not raw biometric)
        embedding_hash = hashlib.sha256(
            str(voice_embedding).encode()
        ).hexdigest()

        profile = self.graph.get(person_id)
        if profile:
            profile.preferences['voiceprint_hash'] = embedding_hash
            self.graph.save()
            logger.info(f"Registered voiceprint for {person_id}")
            return True

        return False

    def register_faceprint(
        self,
        person_id: str,
        face_embedding: List[float],
    ) -> bool:
        """
        Register a face embedding for a person.

        Only root can register faceprints for others.
        """
        if not self.is_root():
            logger.warning("Non-root tried to register faceprint")
            return False

        embedding_hash = hashlib.sha256(
            str(face_embedding).encode()
        ).hexdigest()

        profile = self.graph.get(person_id)
        if profile:
            profile.preferences['faceprint_hash'] = embedding_hash
            self.graph.save()
            logger.info(f"Registered faceprint for {person_id}")
            return True

        return False


# =============================================================================
# Convenience
# =============================================================================

_default_resolver: Optional[IdentityResolver] = None


def get_identity_resolver() -> IdentityResolver:
    """Get or create the default identity resolver."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = IdentityResolver()
    return _default_resolver


def current_person() -> Optional[PersonProfile]:
    """Get the current session's person profile."""
    resolver = get_identity_resolver()
    result = resolver.get_current()
    return result.profile if result else None


def is_root() -> bool:
    """Check if current person is root."""
    return get_identity_resolver().is_root()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'IdentitySource',
    'IdentitySignal',
    'IdentityResult',
    'IdentityResolver',
    'get_identity_resolver',
    'current_person',
    'is_root',
]
