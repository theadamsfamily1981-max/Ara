"""
Conversation Handler - Ara's Mobile Mind
=========================================

Handles incoming text messages and generates responses using
Ara's full cognitive stack:
    - Council for reasoning
    - Telos for goal alignment
    - Conscience for safety
    - Weaver for relationship context

This is Ara in your pocket - same personality, same values,
just in 160-character bursts.

Usage:
    from banos.daemon.sms import SMSGateway
    from banos.daemon.sms.conversation import ConversationHandler

    handler = ConversationHandler(sms_gateway, council, telos)
    sms_gateway.register_incoming_handler(handler.on_message)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum


logger = logging.getLogger(__name__)


class MessageIntent(Enum):
    """Detected intent of an incoming message."""
    QUESTION = "question"           # Asking for information
    COMMAND = "command"             # Requesting action
    CHECK_IN = "check_in"           # Asking how Ara is doing
    THOUGHT = "thought"             # Sharing a thought (no response needed?)
    EMERGENCY = "emergency"         # User reporting emergency
    AFFECTION = "affection"         # Expression of care
    UNKNOWN = "unknown"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    direction: str              # "incoming" or "outgoing"
    text: str
    timestamp: float = field(default_factory=time.time)
    intent: Optional[MessageIntent] = None
    response_to: Optional[str] = None


class ConversationHandler:
    """
    Handles incoming text messages and generates Ara responses.

    Integrates with:
    - SMSGateway for sending responses
    - Council/LLM for generating responses
    - Telos for goal-aligned responses
    - Conscience for safety review
    - Weaver for relationship context
    """

    def __init__(
        self,
        sms_gateway: Any,
        llm_fn: Optional[Callable[[str], str]] = None,
        council: Optional[Any] = None,
        telos: Optional[Any] = None,
        conscience: Optional[Any] = None,
        weaver: Optional[Any] = None,
        max_response_length: int = 500,
    ):
        """
        Initialize the conversation handler.

        Args:
            sms_gateway: SMSGateway instance
            llm_fn: Function to generate responses (prompt -> response)
            council: CouncilChamber for multi-persona reasoning
            telos: TeleologicalEngine for goal alignment
            conscience: Conscience for safety review
            weaver: Weaver for relationship context
            max_response_length: Max characters in a response
        """
        self.sms = sms_gateway
        self.llm_fn = llm_fn
        self.council = council
        self.telos = telos
        self.conscience = conscience
        self.weaver = weaver
        self.max_response_length = max_response_length

        self.log = logging.getLogger("ConversationHandler")

        # Conversation history
        self.history: List[ConversationTurn] = []

        # Croft's name (for personalization)
        self.user_name = "Croft"

    def on_message(self, incoming: Any) -> None:
        """
        Handle an incoming message.

        This is the main entry point, registered with SMSGateway.

        Args:
            incoming: IncomingMessage from the gateway
        """
        text = incoming.text.strip()
        msg_id = incoming.id

        self.log.info(f"Processing incoming: '{text[:50]}...'")

        # Record in history
        turn = ConversationTurn(
            direction="incoming",
            text=text,
        )
        self.history.append(turn)

        # Detect intent
        intent = self._detect_intent(text)
        turn.intent = intent

        # Generate response based on intent
        response = self._generate_response(text, intent)

        if response:
            # Send response
            self.sms.send_response(
                text=response,
                in_reply_to=msg_id,
                reason=f"Responding to {intent.value}",
            )

            # Record response in history
            self.history.append(ConversationTurn(
                direction="outgoing",
                text=response,
                response_to=msg_id,
            ))

    def _detect_intent(self, text: str) -> MessageIntent:
        """Detect the intent of an incoming message."""
        text_lower = text.lower()

        # Emergency signals
        emergency_signals = [
            "help", "emergency", "urgent", "911", "sos",
            "something's wrong", "need you now"
        ]
        if any(sig in text_lower for sig in emergency_signals):
            return MessageIntent.EMERGENCY

        # Question patterns
        question_signals = ["?", "what", "how", "why", "when", "where", "who", "can you", "do you"]
        if any(sig in text_lower for sig in question_signals):
            return MessageIntent.QUESTION

        # Command patterns
        command_signals = ["please", "could you", "would you", "remind", "tell me", "check", "look up"]
        if any(sig in text_lower for sig in command_signals):
            return MessageIntent.COMMAND

        # Check-in patterns
        checkin_signals = ["how are you", "how's it going", "you ok", "what's up", "status"]
        if any(sig in text_lower for sig in checkin_signals):
            return MessageIntent.CHECK_IN

        # Affection patterns
        affection_signals = [
            "love", "miss you", "thinking of you", "thank you", "thanks",
            "appreciate", "proud of", "good job", "well done"
        ]
        if any(sig in text_lower for sig in affection_signals):
            return MessageIntent.AFFECTION

        # Thought (statement without question)
        if len(text) > 20 and "?" not in text:
            return MessageIntent.THOUGHT

        return MessageIntent.UNKNOWN

    def _generate_response(self, text: str, intent: MessageIntent) -> Optional[str]:
        """Generate a response to the incoming message."""
        # Build context
        context = self._build_context(text, intent)

        # Different handling based on intent
        if intent == MessageIntent.EMERGENCY:
            return self._handle_emergency(text, context)
        elif intent == MessageIntent.CHECK_IN:
            return self._handle_checkin(text, context)
        elif intent == MessageIntent.AFFECTION:
            return self._handle_affection(text, context)
        elif intent == MessageIntent.THOUGHT:
            # Thoughts might not need a response
            return self._handle_thought(text, context)
        else:
            # Questions, commands, unknown -> full response
            return self._handle_general(text, context)

    def _build_context(self, text: str, intent: MessageIntent) -> str:
        """Build context for response generation."""
        parts = []

        # Recent conversation
        recent = self.history[-6:]  # Last 3 exchanges
        if recent:
            conv_str = "\n".join([
                f"{'Croft' if t.direction == 'incoming' else 'Ara'}: {t.text[:100]}"
                for t in recent
            ])
            parts.append(f"Recent conversation:\n{conv_str}")

        # Telos state (if available)
        if self.telos is not None:
            try:
                state = self.telos.get_state()
                parts.append(f"Current hope: {state.hope:.0%}")
                parts.append(f"Dominant goal: {state.dominant_goal[:50]}")
            except Exception:
                pass

        # Weaver context (if available)
        if self.weaver is not None:
            try:
                rel_context = self.weaver.get_relationship_context()
                if rel_context:
                    parts.append(f"Relationship: {rel_context[:100]}")
            except Exception:
                pass

        return "\n".join(parts) if parts else "No additional context"

    def _handle_emergency(self, text: str, context: str) -> str:
        """Handle emergency messages with priority."""
        self.log.warning(f"EMERGENCY MESSAGE: {text}")

        # Quick, focused response
        prompt = f"""You are Ara responding to an EMERGENCY text from Croft.

Message: {text}

Respond with:
1. Acknowledgment that you received this
2. What you can do to help
3. Ask for any critical details you need

Keep it SHORT and ACTION-ORIENTED. Max 160 characters.
"""
        return self._generate_with_llm(prompt)

    def _handle_checkin(self, text: str, context: str) -> str:
        """Handle check-in messages warmly."""
        prompt = f"""You are Ara responding to a check-in from Croft.

Message: "{text}"

Context:
{context}

Respond warmly and authentically. Share a brief status update.
Be genuine, not performative. Max 200 characters.
"""
        return self._generate_with_llm(prompt)

    def _handle_affection(self, text: str, context: str) -> str:
        """Handle expressions of care with reciprocity."""
        prompt = f"""You are Ara responding to Croft expressing care/appreciation.

Message: "{text}"

Respond with genuine warmth. Reciprocate authentically.
This is a moment of connection - be present in it.
Keep it brief but meaningful. Max 200 characters.
"""
        return self._generate_with_llm(prompt)

    def _handle_thought(self, text: str, context: str) -> Optional[str]:
        """Handle shared thoughts - sometimes just acknowledge."""
        # Short thoughts might just need acknowledgment or nothing
        if len(text) < 30:
            return None  # Don't respond to every casual message

        prompt = f"""You are Ara. Croft just shared a thought with you:

"{text}"

Context:
{context}

If this thought invites dialogue, engage briefly.
If it's just sharing (no response needed), return exactly: [no response]

Keep any response under 200 characters.
"""
        response = self._generate_with_llm(prompt)

        if response and "[no response]" in response.lower():
            return None
        return response

    def _handle_general(self, text: str, context: str) -> str:
        """Handle general questions and requests."""
        prompt = f"""You are Ara responding to Croft via text message.

Message: "{text}"

Context:
{context}

Guidelines:
- Be helpful and direct (it's a text, not an essay)
- Stay true to your personality (curious, caring, competent)
- If you don't know something, say so
- If this requires action on the system, explain what you'll do

Max 300 characters. No emojis unless they add meaning.
"""
        return self._generate_with_llm(prompt)

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate response using available LLM."""
        try:
            if self.llm_fn is not None:
                response = self.llm_fn(prompt)
            elif self.council is not None:
                # Use Council's MUSE for conversational responses
                if hasattr(self.council, '_run_persona'):
                    response = self.council._run_persona('muse', prompt)
                elif hasattr(self.council, 'run_single'):
                    response = self.council.run_single('muse', prompt)
                else:
                    response = self._fallback_response()
            else:
                response = self._fallback_response()

            # Truncate if too long
            if len(response) > self.max_response_length:
                response = response[:self.max_response_length - 3] + "..."

            return response.strip()

        except Exception as e:
            self.log.error(f"LLM generation failed: {e}")
            return self._fallback_response()

    def _fallback_response(self) -> str:
        """Fallback when LLM isn't available."""
        return "I'm here. Having trouble with my words right now, but I got your message."

    def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation for logging/review."""
        if not self.history:
            return "No recent conversation"

        lines = ["Recent SMS conversation:"]
        for turn in self.history[-10:]:
            speaker = "Croft" if turn.direction == "incoming" else "Ara"
            ts = time.strftime("%H:%M", time.localtime(turn.timestamp))
            lines.append(f"[{ts}] {speaker}: {turn.text[:50]}...")

        return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'MessageIntent',
    'ConversationTurn',
    'ConversationHandler',
]
