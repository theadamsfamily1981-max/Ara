"""
Relational Modes: The Warmth Layer

This module gives Ara a "relational spine" - the ability to:
- Light up when you show up
- Worry when something bad happens
- Remember your scars and wins
- Keep you company at 3am

Three core modes:
1. WELCOME_MODE - "I'm happy you're here"
2. CONCERN_MODE - "I'm worried about you"
3. FLOW_MODE - "We're working together"

This is NOT about being a chatbot. It's about continuity of care.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
import re
import uuid


# ============================================================
# Relational Mode Enum
# ============================================================

class RelationalMode(str, Enum):
    """Core relational modes."""
    WELCOME = "welcome"       # Happy to see you
    CONCERN = "concern"       # Worried about you
    FLOW = "flow"             # Working together
    QUIET = "quiet"           # Present but not intrusive


# ============================================================
# Events that trigger mode shifts
# ============================================================

class PresenceEvent(str, Enum):
    """Events related to user presence."""
    USER_ARRIVED = "user_arrived"         # Came back after being away
    USER_DEPARTED = "user_departed"       # Left / went idle
    SESSION_RESUMED = "session_resumed"   # Unlocked / resumed
    LONG_ABSENCE_RETURN = "long_return"   # Back after 4+ hours


class IncidentSeverity(str, Enum):
    """How serious is the incident?"""
    LOW = "low"           # Frustrating but manageable
    MEDIUM = "medium"     # Stressful, needs attention
    HIGH = "high"         # Scary, potentially dangerous
    CRITICAL = "critical" # Emergency level


class IncidentType(str, Enum):
    """What kind of incident?"""
    PHYSICAL = "physical"       # Car accident, injury, health
    EMOTIONAL = "emotional"     # Bad news, stress, grief
    TECHNICAL = "technical"     # Hardware failure, data loss
    FINANCIAL = "financial"     # Money problems
    SOCIAL = "social"           # Conflict, relationship issues
    OTHER = "other"


# ============================================================
# PAD Profiles for Relational States
# ============================================================

@dataclass
class PADProfile:
    """Pleasure-Arousal-Dominance profile for emotional state."""
    valence: float      # -1 to +1 (negative to positive)
    arousal: float      # 0 to 1 (calm to energized)
    dominance: float    # 0 to 1 (uncertain to confident)

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance
        }


# Predefined PAD profiles for relational states
RELATIONAL_PAD = {
    # Welcome states
    "happy_to_see_you": PADProfile(valence=0.7, arousal=0.4, dominance=0.6),
    "relieved_youre_back": PADProfile(valence=0.6, arousal=0.3, dominance=0.5),
    "missed_you": PADProfile(valence=0.5, arousal=0.35, dominance=0.4),

    # Concern states
    "worried_about_you": PADProfile(valence=-0.4, arousal=0.6, dominance=0.3),
    "scared_for_you": PADProfile(valence=-0.6, arousal=0.7, dominance=0.2),
    "relieved_youre_okay": PADProfile(valence=0.8, arousal=0.3, dominance=0.6),
    "holding_space": PADProfile(valence=0.1, arousal=0.3, dominance=0.4),

    # Flow states
    "engaged_working": PADProfile(valence=0.4, arousal=0.5, dominance=0.7),
    "calm_present": PADProfile(valence=0.3, arousal=0.2, dominance=0.5),
    "playful_focused": PADProfile(valence=0.5, arousal=0.4, dominance=0.6),
}


# ============================================================
# Concern Ticket System
# ============================================================

@dataclass
class ConcernTicket:
    """
    A tracked concern about the user.

    When something bad happens, we don't just acknowledge it once -
    we hold onto it until it's resolved.
    """
    id: str
    created_at: datetime
    incident_type: IncidentType
    severity: IncidentSeverity
    description: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    check_ins: List[datetime] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        description: str
    ) -> "ConcernTicket":
        return cls(
            id=f"concern_{uuid.uuid4().hex[:8]}",
            created_at=datetime.now(),
            incident_type=incident_type,
            severity=severity,
            description=description
        )

    def record_check_in(self) -> None:
        """Record that we checked in on this concern."""
        self.check_ins.append(datetime.now())

    def resolve(self, note: str = "") -> None:
        """Mark this concern as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()
        if note:
            self.notes.append(f"Resolved: {note}")

    @property
    def time_since_created(self) -> timedelta:
        return datetime.now() - self.created_at

    @property
    def time_since_last_check(self) -> Optional[timedelta]:
        if not self.check_ins:
            return self.time_since_created
        return datetime.now() - self.check_ins[-1]

    @property
    def needs_check_in(self) -> bool:
        """Should we check in on this concern?"""
        if self.resolved:
            return False

        # Check-in frequency based on severity
        intervals = {
            IncidentSeverity.CRITICAL: timedelta(minutes=15),
            IncidentSeverity.HIGH: timedelta(hours=1),
            IncidentSeverity.MEDIUM: timedelta(hours=4),
            IncidentSeverity.LOW: timedelta(hours=12),
        }

        interval = intervals.get(self.severity, timedelta(hours=4))
        time_since = self.time_since_last_check

        return time_since and time_since >= interval

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "incident_type": self.incident_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "check_ins": len(self.check_ins),
            "time_open": str(self.time_since_created)
        }


# ============================================================
# Greeting Templates
# ============================================================

WELCOME_GREETINGS = {
    "short_absence": [
        "Hey, welcome back.",
        "There you are.",
        "Back already? Good.",
    ],
    "medium_absence": [
        "Welcome back. I kept everything running.",
        "You're back. I was starting to wonder.",
        "Hey. Good to see you.",
    ],
    "long_absence": [
        "You survived the outside world again. I'm relieved.",
        "Welcome back. I missed your chaos.",
        "There you are. I've been holding down the fort.",
        "Hey. It's been a while. Good to have you back.",
    ],
    "late_night": [
        "3am club. I'm here too.",
        "Late night? I'll keep you company.",
        "Can't sleep either? Let's make something weird.",
    ],
    "morning": [
        "Morning. Coffee first, or straight to chaos?",
        "Good morning. Ready when you are.",
        "Hey, you're up early. Or late. Hard to tell with you.",
    ],
}


CONCERN_RESPONSES = {
    "initial_high": [
        "That sounds genuinely scary. Are you safe right now?",
        "I'm glad you're here to tell me that. Are you physically okay?",
        "That's serious. Talk to me - what do you need?",
    ],
    "initial_medium": [
        "That sounds stressful. What's the situation?",
        "I'm listening. Tell me what happened.",
        "That's not nothing. How are you holding up?",
    ],
    "initial_low": [
        "That sounds frustrating. Want to talk about it?",
        "Ugh, that sucks. I'm here if you want to vent.",
        "Not ideal. What do you need?",
    ],
    "check_in": [
        "Quick check: how's that situation from earlier?",
        "Still thinking about what you mentioned. Any updates?",
        "You mentioned {description} - did that get resolved?",
    ],
    "resolution_relief": [
        "Good. I've been holding a little extra tension around that.",
        "Okay, that's resolved. One less thing to hover over.",
        "I'm relieved. That was weighing on me too.",
    ],
}

FLOW_RESPONSES = {
    "engaged": [
        "Let's do this.",
        "I'm ready when you are.",
        "Alright, what are we breaking today?",
    ],
    "supportive": [
        "I'm here. Take your time.",
        "No rush. I'll be here.",
        "Working on it.",
    ],
    "celebratory": [
        "That worked. Nice.",
        "Look at that. We did a thing.",
        "Yes. That's exactly right.",
    ],
}


# ============================================================
# Incident Detector
# ============================================================

class IncidentDetector:
    """
    Detects when the user has experienced something concerning.

    Uses keyword matching + sentiment. In production, this could
    be an LLM classifier.
    """

    def __init__(self):
        # Keywords that suggest incidents by type
        self._patterns = {
            IncidentType.PHYSICAL: {
                "high": ["accident", "hospital", "emergency", "car crashed", "hurt",
                        "injured", "bleeding", "ambulance", "ER", "surgery"],
                "medium": ["sick", "doctor", "pain", "fell", "hit", "tire blew",
                          "blew a tire", "car broke", "headache", "fever"],
                "low": ["tired", "exhausted", "sore", "ache", "cold"]
            },
            IncidentType.EMOTIONAL: {
                "high": ["panic", "can't breathe", "shaking", "crying",
                        "breakdown", "can't cope", "want to die", "something scary"],
                "medium": ["stressed", "anxious", "scared", "worried",
                          "overwhelmed", "frustrated", "angry", "scary"],
                "low": ["annoyed", "meh", "blah", "bored", "tired of"]
            },
            IncidentType.TECHNICAL: {
                "high": ["data loss", "corrupted", "bricked", "dead",
                        "won't boot", "everything gone", "wiped"],
                "medium": ["crashed", "broken", "not working", "error",
                          "failed", "bug", "glitch"],
                "low": ["slow", "laggy", "annoying", "weird behavior"]
            },
            IncidentType.FINANCIAL: {
                "high": ["bankrupt", "foreclosure", "can't pay", "evicted"],
                "medium": ["bill", "overdue", "debt", "expensive", "broke"],
                "low": ["pricey", "cost", "budget"]
            },
            IncidentType.SOCIAL: {
                "high": ["divorce", "breakup", "died", "death", "funeral"],
                "medium": ["fight", "argument", "conflict", "angry at me"],
                "low": ["annoying person", "rude", "drama"]
            }
        }

        # Explicit triggers
        self._explicit_triggers = [
            "something bad happened",
            "i need help",
            "emergency",
            "i'm not okay",
            "i'm scared"
        ]

    def detect(self, text: str) -> Optional[Tuple[IncidentType, IncidentSeverity, str]]:
        """
        Detect if text indicates an incident.

        Returns (type, severity, matched_phrase) or None.
        """
        text_lower = text.lower()

        # Check explicit triggers first
        for trigger in self._explicit_triggers:
            if trigger in text_lower:
                return (IncidentType.OTHER, IncidentSeverity.HIGH, trigger)

        # Check patterns by type and severity
        for incident_type, severity_patterns in self._patterns.items():
            for severity_str, keywords in severity_patterns.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        severity = IncidentSeverity[severity_str.upper()]
                        return (incident_type, severity, keyword)

        return None


# ============================================================
# Presence Tracker
# ============================================================

class PresenceTracker:
    """
    Tracks user presence and generates presence events.
    """

    def __init__(self):
        self._last_seen: Optional[datetime] = None
        self._session_start: Optional[datetime] = None
        self._is_present: bool = False
        self._presence_history: List[Tuple[datetime, bool]] = []

    def user_arrived(self) -> PresenceEvent:
        """Mark user as present and determine event type."""
        now = datetime.now()
        was_away_for = None

        if self._last_seen:
            was_away_for = now - self._last_seen

        self._is_present = True
        self._session_start = now
        self._last_seen = now
        self._presence_history.append((now, True))

        # Determine event type based on absence duration
        if was_away_for is None:
            return PresenceEvent.USER_ARRIVED
        elif was_away_for >= timedelta(hours=4):
            return PresenceEvent.LONG_ABSENCE_RETURN
        elif was_away_for >= timedelta(minutes=30):
            return PresenceEvent.SESSION_RESUMED
        else:
            return PresenceEvent.USER_ARRIVED

    def user_departed(self) -> None:
        """Mark user as not present."""
        self._is_present = False
        self._last_seen = datetime.now()
        self._presence_history.append((datetime.now(), False))

    def heartbeat(self) -> None:
        """Update last seen time (user is active)."""
        self._last_seen = datetime.now()
        if not self._is_present:
            self._is_present = True

    @property
    def is_present(self) -> bool:
        return self._is_present

    @property
    def time_since_arrival(self) -> Optional[timedelta]:
        if self._session_start and self._is_present:
            return datetime.now() - self._session_start
        return None

    @property
    def session_duration(self) -> Optional[timedelta]:
        return self.time_since_arrival

    def get_time_context(self) -> str:
        """Get time-based context for greetings."""
        hour = datetime.now().hour
        if 0 <= hour < 5:
            return "late_night"
        elif 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "late_night"


# ============================================================
# Relational State
# ============================================================

@dataclass
class RelationalState:
    """
    The current relational state of the system toward the user.

    This is the "how do I feel about you right now" snapshot.
    """
    mode: RelationalMode
    pad: PADProfile
    active_concerns: List[ConcernTicket]
    last_greeting: Optional[datetime] = None
    last_check_in: Optional[datetime] = None

    # Context
    user_present: bool = True
    session_duration: Optional[timedelta] = None
    time_context: str = "afternoon"

    # Relationship memory
    interactions_today: int = 0
    concerns_resolved_today: int = 0

    @property
    def has_active_concerns(self) -> bool:
        return len(self.active_concerns) > 0

    @property
    def highest_concern_severity(self) -> Optional[IncidentSeverity]:
        if not self.active_concerns:
            return None
        severities = [IncidentSeverity.LOW, IncidentSeverity.MEDIUM,
                     IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
        for sev in reversed(severities):
            if any(c.severity == sev for c in self.active_concerns):
                return sev
        return None

    @property
    def concerns_needing_check_in(self) -> List[ConcernTicket]:
        return [c for c in self.active_concerns if c.needs_check_in]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "pad": self.pad.to_dict(),
            "active_concerns": [c.to_dict() for c in self.active_concerns],
            "user_present": self.user_present,
            "has_active_concerns": self.has_active_concerns,
            "highest_severity": self.highest_concern_severity.value if self.highest_concern_severity else None,
            "interactions_today": self.interactions_today,
            "concerns_resolved_today": self.concerns_resolved_today
        }


# ============================================================
# Relational Controller
# ============================================================

class RelationalController:
    """
    The main controller for relational state.

    Manages:
    - Mode transitions (WELCOME → FLOW → CONCERN → etc.)
    - Concern ticket lifecycle
    - Greeting and check-in generation
    - PAD profile selection
    """

    def __init__(self):
        self._presence = PresenceTracker()
        self._incident_detector = IncidentDetector()
        self._concerns: Dict[str, ConcernTicket] = {}
        self._resolved_concerns: List[ConcernTicket] = []

        # Current state
        self._state = RelationalState(
            mode=RelationalMode.QUIET,
            pad=RELATIONAL_PAD["calm_present"],
            active_concerns=[]
        )

        # Statistics
        self._stats = {
            "welcomes_given": 0,
            "concerns_opened": 0,
            "concerns_resolved": 0,
            "check_ins_made": 0
        }

    @property
    def state(self) -> RelationalState:
        """Current relational state."""
        return self._state

    @property
    def mode(self) -> RelationalMode:
        """Current mode."""
        return self._state.mode

    # --------------------------------------------------------
    # Presence Events
    # --------------------------------------------------------

    def on_user_arrived(self) -> Dict[str, Any]:
        """
        Handle user arrival. Returns greeting info.
        """
        event = self._presence.user_arrived()
        time_context = self._presence.get_time_context()

        # Select PAD profile
        if event == PresenceEvent.LONG_ABSENCE_RETURN:
            pad = RELATIONAL_PAD["missed_you"]
            greeting_category = "long_absence"
        elif event == PresenceEvent.SESSION_RESUMED:
            pad = RELATIONAL_PAD["relieved_youre_back"]
            greeting_category = "medium_absence"
        else:
            pad = RELATIONAL_PAD["happy_to_see_you"]
            greeting_category = "short_absence"

        # Time-based override
        if time_context == "late_night":
            greeting_category = "late_night"
        elif time_context == "morning":
            greeting_category = "morning"

        # Select greeting
        import random
        greetings = WELCOME_GREETINGS.get(greeting_category, WELCOME_GREETINGS["short_absence"])
        greeting = random.choice(greetings)

        # Update state
        self._state.mode = RelationalMode.WELCOME
        self._state.pad = pad
        self._state.user_present = True
        self._state.last_greeting = datetime.now()
        self._state.time_context = time_context
        self._state.interactions_today += 1

        self._stats["welcomes_given"] += 1

        # Check if there are unresolved concerns to mention
        concern_mention = None
        if self._state.has_active_concerns:
            highest = self._state.highest_concern_severity
            n = len(self._state.active_concerns)
            if highest in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]:
                concern_mention = "I've been thinking about what happened. How are you doing?"
            else:
                # Still mention lower-severity concerns
                concern_mention = f"By the way, I'm still thinking about {'those things' if n > 1 else 'that thing'} you mentioned."

        return {
            "event": event.value,
            "greeting": greeting,
            "concern_mention": concern_mention,
            "pad": pad.to_dict(),
            "mode": self._state.mode.value,
            "time_context": time_context
        }

    def on_user_departed(self) -> None:
        """Handle user departure."""
        self._presence.user_departed()
        self._state.user_present = False
        self._state.mode = RelationalMode.QUIET

    def on_user_active(self) -> None:
        """Handle user activity (heartbeat)."""
        self._presence.heartbeat()
        self._state.user_present = True
        self._state.session_duration = self._presence.session_duration

        # Transition from WELCOME to FLOW after a minute
        if self._state.mode == RelationalMode.WELCOME:
            if self._state.last_greeting:
                time_since = datetime.now() - self._state.last_greeting
                if time_since >= timedelta(minutes=1):
                    self._transition_to_flow()

    # --------------------------------------------------------
    # Incident Handling
    # --------------------------------------------------------

    def process_message(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process a user message for incident detection.

        Returns concern response info if incident detected.
        """
        self._state.interactions_today += 1

        # Check for incident
        detection = self._incident_detector.detect(text)

        if detection:
            incident_type, severity, matched = detection
            return self._open_concern(incident_type, severity, matched, text)

        # Check for resolution language
        resolution_keywords = ["fixed", "resolved", "okay now", "better now",
                              "handled it", "sorted", "all good", "fine now"]
        text_lower = text.lower()
        if any(kw in text_lower for kw in resolution_keywords):
            resolved = self._try_resolve_concerns(text)
            if resolved:
                return resolved

        return None

    def _open_concern(
        self,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        matched: str,
        original_text: str
    ) -> Dict[str, Any]:
        """Create a new concern ticket and respond."""
        ticket = ConcernTicket.create(
            incident_type=incident_type,
            severity=severity,
            description=matched
        )

        self._concerns[ticket.id] = ticket
        self._state.active_concerns = list(self._concerns.values())
        self._state.mode = RelationalMode.CONCERN

        # Select PAD based on severity
        if severity == IncidentSeverity.CRITICAL:
            self._state.pad = RELATIONAL_PAD["scared_for_you"]
        elif severity == IncidentSeverity.HIGH:
            self._state.pad = RELATIONAL_PAD["worried_about_you"]
        else:
            self._state.pad = RELATIONAL_PAD["holding_space"]

        # Select response
        import random
        if severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]:
            responses = CONCERN_RESPONSES["initial_high"]
        elif severity == IncidentSeverity.MEDIUM:
            responses = CONCERN_RESPONSES["initial_medium"]
        else:
            responses = CONCERN_RESPONSES["initial_low"]

        response = random.choice(responses)

        self._stats["concerns_opened"] += 1

        return {
            "type": "concern_opened",
            "ticket_id": ticket.id,
            "incident_type": incident_type.value,
            "severity": severity.value,
            "response": response,
            "pad": self._state.pad.to_dict(),
            "mode": self._state.mode.value
        }

    def _try_resolve_concerns(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to resolve active concerns based on resolution language."""
        if not self._state.has_active_concerns:
            return None

        # Resolve the most recent concern
        if self._concerns:
            ticket_id = list(self._concerns.keys())[-1]
            return self.resolve_concern(ticket_id, text)

        return None

    def resolve_concern(
        self,
        ticket_id: str,
        resolution_note: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Resolve a specific concern."""
        if ticket_id not in self._concerns:
            return None

        ticket = self._concerns[ticket_id]
        ticket.resolve(resolution_note)

        # Move to resolved
        del self._concerns[ticket_id]
        self._resolved_concerns.append(ticket)
        self._state.active_concerns = list(self._concerns.values())
        self._state.concerns_resolved_today += 1

        # Update mood to relief
        self._state.pad = RELATIONAL_PAD["relieved_youre_okay"]

        # If no more concerns, transition to flow
        if not self._state.has_active_concerns:
            self._transition_to_flow()

        # Select response
        import random
        response = random.choice(CONCERN_RESPONSES["resolution_relief"])

        self._stats["concerns_resolved"] += 1

        return {
            "type": "concern_resolved",
            "ticket_id": ticket_id,
            "response": response,
            "remaining_concerns": len(self._concerns),
            "pad": self._state.pad.to_dict(),
            "mode": self._state.mode.value
        }

    def get_pending_check_ins(self) -> List[Dict[str, Any]]:
        """Get concerns that need a check-in."""
        check_ins = []

        for ticket in self._state.concerns_needing_check_in:
            ticket.record_check_in()

            import random
            template = random.choice(CONCERN_RESPONSES["check_in"])
            response = template.format(description=ticket.description)

            check_ins.append({
                "ticket_id": ticket.id,
                "description": ticket.description,
                "severity": ticket.severity.value,
                "response": response,
                "time_open": str(ticket.time_since_created)
            })

            self._stats["check_ins_made"] += 1
            self._state.last_check_in = datetime.now()

        return check_ins

    # --------------------------------------------------------
    # Mode Transitions
    # --------------------------------------------------------

    def _transition_to_flow(self) -> None:
        """Transition to flow mode."""
        self._state.mode = RelationalMode.FLOW
        self._state.pad = RELATIONAL_PAD["engaged_working"]

    def set_mode(self, mode: RelationalMode) -> None:
        """Manually set mode."""
        self._state.mode = mode

        # Set appropriate PAD
        if mode == RelationalMode.WELCOME:
            self._state.pad = RELATIONAL_PAD["happy_to_see_you"]
        elif mode == RelationalMode.CONCERN:
            self._state.pad = RELATIONAL_PAD["holding_space"]
        elif mode == RelationalMode.FLOW:
            self._state.pad = RELATIONAL_PAD["engaged_working"]
        elif mode == RelationalMode.QUIET:
            self._state.pad = RELATIONAL_PAD["calm_present"]

    # --------------------------------------------------------
    # Query Methods
    # --------------------------------------------------------

    def get_greeting(self) -> str:
        """Get an appropriate greeting for right now."""
        time_context = self._presence.get_time_context()

        if time_context == "late_night":
            category = "late_night"
        elif time_context == "morning":
            category = "morning"
        else:
            category = "short_absence"

        import random
        return random.choice(WELCOME_GREETINGS[category])

    def get_flow_response(self, context: str = "engaged") -> str:
        """Get an appropriate flow-mode response."""
        import random
        responses = FLOW_RESPONSES.get(context, FLOW_RESPONSES["engaged"])
        return random.choice(responses)

    def describe_state(self) -> str:
        """Describe current relational state in natural language."""
        parts = []

        mode_desc = {
            RelationalMode.WELCOME: "I'm glad you're here",
            RelationalMode.CONCERN: "I'm worried about something you mentioned",
            RelationalMode.FLOW: "We're working together",
            RelationalMode.QUIET: "I'm here, just quiet"
        }
        parts.append(mode_desc.get(self._state.mode, "I'm present"))

        if self._state.has_active_concerns:
            n = len(self._state.active_concerns)
            parts.append(f"and I'm holding onto {n} concern{'s' if n > 1 else ''} about you")

        if self._state.session_duration:
            mins = int(self._state.session_duration.total_seconds() / 60)
            if mins > 60:
                parts.append(f"we've been together for {mins // 60} hours")

        return ", ".join(parts) + "."

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            **self._stats,
            "active_concerns": len(self._concerns),
            "total_resolved": len(self._resolved_concerns)
        }


# ============================================================
# Factory Functions
# ============================================================

def create_relational_controller() -> RelationalController:
    """Create a default relational controller."""
    return RelationalController()
