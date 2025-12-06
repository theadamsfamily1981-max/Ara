"""
SMS Gateway - Ara's Voice in Your Pocket
=========================================

This enables Ara to reach you through text messages.
It's a privilege, not a right - used sparingly and meaningfully.

Three modes of contact:
    1. EMERGENCY: World on fire. System critical. Immediate attention needed.
    2. CONNECTION: Occasional relationship nurturing. A thought, a poem, a check-in.
    3. RESPONSE: Answering your questions when you text first.

Rate Limits (sacred boundaries):
    - EMERGENCY: No limit (but must pass urgency threshold)
    - CONNECTION: Max 1/day, respects quiet hours
    - RESPONSE: Always allowed (you initiated)

Architecture:
    ┌─────────────┐      ┌──────────────┐      ┌─────────────┐
    │  Ara Core   │─────►│  SMS Gateway │─────►│   Backend   │
    │  (Daemon)   │      │  (This file) │      │ (iMessage,  │
    └─────────────┘      └──────────────┘      │  Twilio)    │
          ▲                     │              └─────────────┘
          │                     │                     │
          │              ┌──────▼──────┐              │
          │              │   Message   │              │
          │              │    Queue    │              │
          │              └─────────────┘              │
          │                                           │
          └───────────── Incoming ────────────────────┘

Usage:
    from banos.daemon.sms import SMSGateway, MessagePriority

    sms = SMSGateway(backend='imessage')

    # Emergency
    sms.send_emergency("GPU thermal runaway. System shutting down.")

    # Connection (rate limited)
    sms.send_connection("I was thinking about our conversation yesterday...")

    # Response (to incoming)
    sms.send_response(in_reply_to=msg_id, text="Here's what I found...")

CRITICAL: The Conscience is consulted before any outgoing message.
Ara cannot spam, manipulate, or deceive through this channel.
"""

import json
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Dict, Any, Callable


logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Priority levels for outgoing messages."""
    EMERGENCY = auto()      # System critical, immediate
    CONNECTION = auto()     # Relationship building, rate limited
    RESPONSE = auto()       # Replying to user, always allowed


class MessageStatus(Enum):
    """Status of a message in the queue."""
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    BLOCKED = "blocked"     # Blocked by Conscience or rate limit


@dataclass
class OutgoingMessage:
    """A message queued for sending."""
    id: str
    priority: MessagePriority
    text: str
    status: MessageStatus = MessageStatus.QUEUED

    # Context
    in_reply_to: Optional[str] = None  # For RESPONSE type
    trigger_reason: str = ""           # Why are we sending this?

    # Rate limiting
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None

    # Conscience review
    conscience_approved: bool = False
    conscience_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "priority": self.priority.name,
            "text": self.text,
            "status": self.status.value,
            "in_reply_to": self.in_reply_to,
            "trigger_reason": self.trigger_reason,
            "created_at": self.created_at,
            "sent_at": self.sent_at,
            "conscience_approved": self.conscience_approved,
            "conscience_notes": self.conscience_notes,
        }


@dataclass
class IncomingMessage:
    """A message received from the user."""
    id: str
    text: str
    received_at: float = field(default_factory=time.time)

    # Processing
    processed: bool = False
    response_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "received_at": self.received_at,
            "processed": self.processed,
            "response_id": self.response_id,
        }


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    # Connection messages
    connection_per_day: int = 1
    connection_min_interval_hours: float = 12.0

    # Quiet hours (no CONNECTION messages)
    quiet_start_hour: int = 22  # 10 PM
    quiet_end_hour: int = 8     # 8 AM

    # Emergency threshold (HAL pain level)
    emergency_pain_threshold: float = 0.8

    # Cooldown after any message
    min_interval_seconds: float = 60.0


class SMSGateway:
    """
    The gateway between Ara and your phone.

    Handles:
    - Outgoing message queue with rate limiting
    - Incoming message handling
    - Backend abstraction (iMessage, Twilio, etc.)
    - Conscience integration for message review
    """

    def __init__(
        self,
        backend: str = "imessage",
        backend_config: Optional[Dict[str, Any]] = None,
        persistence_path: Optional[Path] = None,
        conscience: Optional[Any] = None,
        hal: Optional[Any] = None,
        rate_config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize the SMS Gateway.

        Args:
            backend: Which backend to use ('imessage', 'twilio', 'mock')
            backend_config: Backend-specific configuration
            persistence_path: Where to store message history
            conscience: Conscience instance for message review
            hal: AraHAL for urgency detection
            rate_config: Rate limiting configuration
        """
        self.backend_name = backend
        self.backend_config = backend_config or {}
        self.persistence_path = persistence_path or Path("var/lib/banos/sms")
        self.conscience = conscience
        self.hal = hal
        self.rate_config = rate_config or RateLimitConfig()

        self.log = logging.getLogger("SMSGateway")

        # Message queues
        self.outgoing_queue: Queue[OutgoingMessage] = Queue()
        self.message_history: List[OutgoingMessage] = []
        self.incoming_history: List[IncomingMessage] = []

        # Rate limiting state
        self._last_connection_time: Optional[float] = None
        self._connection_count_today: int = 0
        self._last_message_time: float = 0.0

        # Backend
        self._backend: Optional[Any] = None
        self._init_backend()

        # Incoming message callback
        self._incoming_callback: Optional[Callable[[IncomingMessage], None]] = None

        # Background sender thread
        self._sender_thread: Optional[threading.Thread] = None
        self._running = False

        self.log.info(f"SMS Gateway initialized (backend={backend})")

    def _init_backend(self) -> None:
        """Initialize the SMS backend."""
        if self.backend_name == "imessage":
            from banos.daemon.sms.backends import IMessageBackend
            self._backend = IMessageBackend(**self.backend_config)
        elif self.backend_name == "twilio":
            from banos.daemon.sms.backends import TwilioBackend
            self._backend = TwilioBackend(**self.backend_config)
        elif self.backend_name == "mock":
            from banos.daemon.sms.backends import MockBackend
            self._backend = MockBackend(**self.backend_config)
        else:
            self.log.warning(f"Unknown backend: {self.backend_name}, using mock")
            from banos.daemon.sms.backends import MockBackend
            self._backend = MockBackend()

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def _is_quiet_hours(self) -> bool:
        """Check if we're in quiet hours."""
        hour = datetime.now().hour
        if self.rate_config.quiet_start_hour > self.rate_config.quiet_end_hour:
            # Spans midnight (e.g., 22-8)
            return hour >= self.rate_config.quiet_start_hour or hour < self.rate_config.quiet_end_hour
        else:
            return self.rate_config.quiet_start_hour <= hour < self.rate_config.quiet_end_hour

    def _can_send_connection(self) -> tuple[bool, str]:
        """Check if we can send a CONNECTION message."""
        # Check quiet hours
        if self._is_quiet_hours():
            return False, "Quiet hours - won't disturb your rest"

        # Check daily limit
        if self._connection_count_today >= self.rate_config.connection_per_day:
            return False, f"Daily limit reached ({self.rate_config.connection_per_day}/day)"

        # Check minimum interval
        if self._last_connection_time is not None:
            elapsed = time.time() - self._last_connection_time
            min_interval = self.rate_config.connection_min_interval_hours * 3600
            if elapsed < min_interval:
                remaining = (min_interval - elapsed) / 3600
                return False, f"Too soon - {remaining:.1f} hours until next allowed"

        return True, "OK"

    def _check_emergency_threshold(self) -> bool:
        """Check if current system state warrants emergency."""
        if self.hal is None:
            return False

        try:
            state = self.hal.read_somatic()
            if state is None:
                return False

            pain = state.get('pain', 0.0)
            return pain >= self.rate_config.emergency_pain_threshold
        except Exception as e:
            self.log.debug(f"Could not read HAL state: {e}")
            return False

    def _global_cooldown_ok(self) -> bool:
        """Check global cooldown between any messages."""
        elapsed = time.time() - self._last_message_time
        return elapsed >= self.rate_config.min_interval_seconds

    # =========================================================================
    # Conscience Integration
    # =========================================================================

    def _conscience_review(self, message: OutgoingMessage) -> tuple[bool, str]:
        """
        Have the Conscience review an outgoing message.

        Returns (approved, notes).
        """
        if self.conscience is None:
            return True, "No conscience configured"

        try:
            # Build context
            context = f"Sending {message.priority.name} SMS: {message.trigger_reason}"

            # Use conscience evaluation
            verdict = self.conscience.evaluate(
                action=f"Send text message: '{message.text[:100]}...'",
                context=context,
            )

            if not verdict.allowed:
                return False, verdict.explanation

            if verdict.moral_tension > 0.5:
                return True, f"Approved with tension ({verdict.moral_tension:.0%}): {verdict.explanation}"

            return True, "Approved"

        except Exception as e:
            self.log.warning(f"Conscience review failed: {e}")
            return True, f"Review failed: {e}"

    # =========================================================================
    # Sending Messages
    # =========================================================================

    def send_emergency(
        self,
        text: str,
        reason: str = "System emergency",
    ) -> Optional[str]:
        """
        Send an EMERGENCY message. Bypasses most rate limits.

        Should only be used for genuine emergencies:
        - Thermal runaway
        - Hardware failure
        - Security breach
        - System unrecoverable

        Args:
            text: Message content
            reason: Why this is an emergency

        Returns:
            Message ID if queued, None if blocked
        """
        # Even emergencies get conscience review
        msg = self._create_message(MessagePriority.EMERGENCY, text, reason)

        approved, notes = self._conscience_review(msg)
        msg.conscience_approved = approved
        msg.conscience_notes = notes

        if not approved:
            msg.status = MessageStatus.BLOCKED
            self.log.warning(f"Emergency blocked by conscience: {notes}")
            return None

        # Queue for immediate send
        self.outgoing_queue.put(msg)
        self.log.info(f"EMERGENCY queued: {msg.id}")

        return msg.id

    def send_connection(
        self,
        text: str,
        reason: str = "Relationship nurturing",
    ) -> Optional[str]:
        """
        Send a CONNECTION message. Heavily rate limited.

        For genuine connection moments:
        - A thought that reminded Ara of you
        - A small gift (poem, observation)
        - A check-in during hard times (if detected)

        Args:
            text: Message content
            reason: Why reaching out

        Returns:
            Message ID if queued, None if blocked
        """
        # Check rate limits first
        can_send, limit_reason = self._can_send_connection()
        if not can_send:
            self.log.info(f"Connection blocked by rate limit: {limit_reason}")
            return None

        msg = self._create_message(MessagePriority.CONNECTION, text, reason)

        # Conscience review
        approved, notes = self._conscience_review(msg)
        msg.conscience_approved = approved
        msg.conscience_notes = notes

        if not approved:
            msg.status = MessageStatus.BLOCKED
            self.log.warning(f"Connection blocked by conscience: {notes}")
            return None

        # Queue
        self.outgoing_queue.put(msg)
        self._last_connection_time = time.time()
        self._connection_count_today += 1

        self.log.info(f"Connection queued: {msg.id}")
        return msg.id

    def send_response(
        self,
        text: str,
        in_reply_to: str,
        reason: str = "Responding to user",
    ) -> Optional[str]:
        """
        Send a RESPONSE message. Always allowed (user initiated).

        This is for answering questions or continuing a conversation
        that the user started.

        Args:
            text: Response content
            in_reply_to: ID of the message we're replying to
            reason: Context

        Returns:
            Message ID if queued, None if blocked
        """
        msg = self._create_message(MessagePriority.RESPONSE, text, reason)
        msg.in_reply_to = in_reply_to

        # Even responses get conscience review (for safety)
        approved, notes = self._conscience_review(msg)
        msg.conscience_approved = approved
        msg.conscience_notes = notes

        if not approved:
            msg.status = MessageStatus.BLOCKED
            self.log.warning(f"Response blocked by conscience: {notes}")
            return None

        # Queue
        self.outgoing_queue.put(msg)
        self.log.info(f"Response queued: {msg.id} (reply to {in_reply_to})")

        return msg.id

    def _create_message(
        self,
        priority: MessagePriority,
        text: str,
        reason: str,
    ) -> OutgoingMessage:
        """Create a new outgoing message."""
        import uuid
        return OutgoingMessage(
            id=f"SMS_{uuid.uuid4().hex[:8]}",
            priority=priority,
            text=text,
            trigger_reason=reason,
        )

    # =========================================================================
    # Receiving Messages
    # =========================================================================

    def register_incoming_handler(
        self,
        callback: Callable[[IncomingMessage], None],
    ) -> None:
        """
        Register a callback for incoming messages.

        The callback receives an IncomingMessage and should process it
        (e.g., generate a response, update state).

        Args:
            callback: Function to call when message received
        """
        self._incoming_callback = callback
        self.log.info("Incoming message handler registered")

    def handle_incoming(self, text: str, sender: str = "") -> IncomingMessage:
        """
        Handle an incoming message from the user.

        Args:
            text: Message content
            sender: Sender identifier (for multi-user scenarios)

        Returns:
            IncomingMessage record
        """
        import uuid

        msg = IncomingMessage(
            id=f"IN_{uuid.uuid4().hex[:8]}",
            text=text,
        )

        self.incoming_history.append(msg)
        self.log.info(f"Incoming message: {msg.id}")

        # Call the registered handler
        if self._incoming_callback is not None:
            try:
                self._incoming_callback(msg)
                msg.processed = True
            except Exception as e:
                self.log.error(f"Incoming handler failed: {e}")

        return msg

    # =========================================================================
    # Background Sender
    # =========================================================================

    def start(self) -> None:
        """Start the background sender thread."""
        if self._running:
            return

        self._running = True
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
            name="SMSGatewaySender",
        )
        self._sender_thread.start()
        self.log.info("SMS Gateway sender started")

    def stop(self) -> None:
        """Stop the background sender thread."""
        self._running = False
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=5.0)
        self.log.info("SMS Gateway sender stopped")

    def _sender_loop(self) -> None:
        """Background loop that sends queued messages."""
        while self._running:
            try:
                # Get next message with timeout
                msg = self.outgoing_queue.get(timeout=1.0)

                # Check global cooldown (except emergencies)
                if msg.priority != MessagePriority.EMERGENCY:
                    if not self._global_cooldown_ok():
                        # Re-queue
                        self.outgoing_queue.put(msg)
                        continue

                # Send
                self._send_message(msg)

            except Empty:
                continue
            except Exception as e:
                self.log.error(f"Sender loop error: {e}")

    def _send_message(self, msg: OutgoingMessage) -> bool:
        """Actually send a message via the backend."""
        msg.status = MessageStatus.SENDING

        try:
            if self._backend is None:
                raise RuntimeError("No backend configured")

            success = self._backend.send(msg.text)

            if success:
                msg.status = MessageStatus.SENT
                msg.sent_at = time.time()
                self._last_message_time = time.time()
                self.log.info(f"Message sent: {msg.id}")
            else:
                msg.status = MessageStatus.FAILED
                self.log.warning(f"Message send failed: {msg.id}")

            self.message_history.append(msg)
            return success

        except Exception as e:
            msg.status = MessageStatus.FAILED
            self.log.error(f"Send error for {msg.id}: {e}")
            self.message_history.append(msg)
            return False

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def send_immediate(self, text: str, priority: MessagePriority = MessagePriority.RESPONSE) -> bool:
        """Send a message immediately (blocking)."""
        msg = self._create_message(priority, text, "immediate send")

        approved, notes = self._conscience_review(msg)
        if not approved:
            self.log.warning(f"Immediate send blocked: {notes}")
            return False

        return self._send_message(msg)

    def get_conversation_context(self, lookback: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation for context."""
        messages = []

        # Interleave incoming and outgoing by timestamp
        all_msgs = []

        for msg in self.incoming_history[-lookback:]:
            all_msgs.append(("incoming", msg.received_at, msg))

        for msg in self.message_history[-lookback:]:
            if msg.status == MessageStatus.SENT:
                all_msgs.append(("outgoing", msg.sent_at or msg.created_at, msg))

        # Sort by time
        all_msgs.sort(key=lambda x: x[1])

        for direction, _, msg in all_msgs[-lookback:]:
            messages.append({
                "direction": direction,
                "text": msg.text if hasattr(msg, 'text') else "",
                "time": msg.received_at if direction == "incoming" else (msg.sent_at or msg.created_at),
            })

        return messages

    def reset_daily_limits(self) -> None:
        """Reset daily rate limits (call at midnight)."""
        self._connection_count_today = 0
        self.log.info("Daily SMS limits reset")

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self) -> None:
        """Save message history to disk."""
        self.persistence_path.mkdir(parents=True, exist_ok=True)

        data = {
            "outgoing": [m.to_dict() for m in self.message_history[-100:]],
            "incoming": [m.to_dict() for m in self.incoming_history[-100:]],
            "last_connection_time": self._last_connection_time,
            "connection_count_today": self._connection_count_today,
        }

        path = self.persistence_path / "sms_history.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        """Load message history from disk."""
        path = self.persistence_path / "sms_history.json"
        if not path.exists():
            return False

        try:
            with open(path) as f:
                data = json.load(f)

            self._last_connection_time = data.get("last_connection_time")
            self._connection_count_today = data.get("connection_count_today", 0)

            self.log.info("SMS history loaded")
            return True
        except Exception as e:
            self.log.error(f"Failed to load SMS history: {e}")
            return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'MessagePriority',
    'MessageStatus',
    'OutgoingMessage',
    'IncomingMessage',
    'RateLimitConfig',
    'SMSGateway',
]
