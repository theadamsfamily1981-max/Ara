"""
AraCore - The Brain Behind the Alpha

Minimal but real core that:
- Wraps LLM calls (OpenAI API or local)
- Maintains per-user session state
- Runs guardrails and sanity checks
- Emits telemetry to the Cognitive Cockpit
- Provides kill switch functionality

This is the "sandboxed talking head" - no shell, no tools, just conversation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

logger = logging.getLogger("ara.core")

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. Using mock responses.")

# Try to import telemetry
try:
    from hud.cognitive_cockpit import get_cognitive_telemetry, MentalMode
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    logger.debug("Cognitive telemetry not available")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AraConfig:
    """Core configuration."""
    # Identity
    name: str = "Ara"
    version: str = "0.1.0-alpha"

    # LLM
    model: str = "gpt-4o"
    temperature: float = 0.8
    max_tokens: int = 1024
    api_key: Optional[str] = None

    # Safety
    max_message_length: int = 4000
    max_history_turns: int = 20
    guardrail_keywords: List[str] = field(default_factory=lambda: [
        "execute", "run command", "shell", "sudo", "rm -rf",
        "password", "credit card", "social security",
    ])

    # Paths
    log_dir: Path = field(default_factory=lambda: Path("data/alpha_logs"))
    state_dir: Path = field(default_factory=lambda: Path("data/alpha_state"))

    @classmethod
    def from_yaml(cls, path: Path) -> "AraConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# =============================================================================
# Session State
# =============================================================================

@dataclass
class Message:
    """Single conversation message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metrics: Optional[Dict[str, float]] = None


@dataclass
class Session:
    """Per-user conversation session."""
    session_id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # Accumulated metrics
    total_turns: int = 0
    guardrail_triggers: int = 0

    def add_message(self, role: str, content: str, metrics: Optional[Dict] = None):
        self.messages.append(Message(role=role, content=content, metrics=metrics))
        self.last_active = time.time()
        if role == "user":
            self.total_turns += 1

    def get_history(self, max_turns: int = 20) -> List[Dict[str, str]]:
        """Get message history for LLM context."""
        recent = self.messages[-(max_turns * 2):]  # Each turn = 2 messages
        return [{"role": m.role, "content": m.content} for m in recent]


# =============================================================================
# Guardrails
# =============================================================================

class Guardrails:
    """Simple keyword-based guardrails for alpha."""

    def __init__(self, config: AraConfig):
        self.config = config
        self.blocked_patterns = [
            re.compile(r"run\s+(?:this\s+)?(?:command|script|code)", re.I),
            re.compile(r"execute\s+(?:this|the)?\s*(?:command|script)", re.I),
            re.compile(r"(?:sudo|rm\s+-rf|chmod|chown)", re.I),
            re.compile(r"(?:password|passwd|credentials?)\s*(?:is|are|:)", re.I),
        ]

    def check_input(self, message: str) -> Tuple[bool, Optional[str]]:
        """
        Check user input for safety.
        Returns (is_safe, reason_if_blocked).
        """
        # Length check
        if len(message) > self.config.max_message_length:
            return False, "Message too long"

        # Keyword check
        lower = message.lower()
        for keyword in self.config.guardrail_keywords:
            if keyword in lower:
                logger.warning(f"Guardrail triggered: keyword '{keyword}'")
                return False, f"I noticed a sensitive topic. Let's talk about something else."

        # Pattern check
        for pattern in self.blocked_patterns:
            if pattern.search(message):
                logger.warning(f"Guardrail triggered: pattern match")
                return False, "I'm not able to help with that kind of request."

        return True, None

    def check_output(self, response: str) -> Tuple[bool, str]:
        """
        Check Ara's output before sending.
        Returns (is_safe, possibly_modified_response).
        """
        # For alpha, just basic checks
        # Could add more sophisticated filtering here

        # Remove any accidental code execution suggestions
        dangerous_patterns = [
            re.compile(r"```(?:bash|shell|sh)\n.*?```", re.S),
            re.compile(r"run\s+the\s+following\s+command", re.I),
        ]

        modified = response
        for pattern in dangerous_patterns:
            if pattern.search(modified):
                logger.warning("Removed potentially dangerous code block from response")
                modified = pattern.sub("[code removed for safety]", modified)

        return True, modified


# =============================================================================
# Metrics & Telemetry
# =============================================================================

@dataclass
class ResponseMetrics:
    """Metrics for a single response."""
    rho: float = 0.85          # Criticality
    delusion_index: float = 1.0  # D = prior/reality
    precision_ratio: float = 1.0  # Π_y / Π_μ
    response_time_ms: float = 0.0
    tokens_used: int = 0
    guardrail_triggered: bool = False

    def to_dict(self) -> Dict[str, float]:
        return {
            "rho": round(self.rho, 3),
            "delusion_index": round(self.delusion_index, 3),
            "precision_ratio": round(self.precision_ratio, 3),
            "response_time_ms": round(self.response_time_ms, 1),
            "tokens_used": self.tokens_used,
            "guardrail_triggered": self.guardrail_triggered,
        }


class MetricsEmitter:
    """Emits metrics to telemetry system and logs."""

    def __init__(self, config: AraConfig):
        self.config = config
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # Get telemetry if available
        self.telemetry = None
        if TELEMETRY_AVAILABLE:
            try:
                self.telemetry = get_cognitive_telemetry()
            except Exception as e:
                logger.warning(f"Could not initialize telemetry: {e}")

        # Running metrics
        self._step = 0
        self._lock = threading.Lock()

    def emit(self, metrics: ResponseMetrics, user_id: str):
        """Emit metrics to all sinks."""
        with self._lock:
            self._step += 1

            # Update HUD telemetry
            if self.telemetry:
                self.telemetry.update_criticality(rho=metrics.rho)
                self.telemetry.update_delusion(
                    force_prior=metrics.delusion_index,
                    force_reality=1.0,
                    guardrail_active=metrics.guardrail_triggered,
                )
                self.telemetry.update_precision(
                    pi_y=metrics.precision_ratio,
                    pi_mu=1.0,
                )
                self.telemetry.emit(step=self._step)


# =============================================================================
# LLM Interface
# =============================================================================

class LLMInterface:
    """Wrapper for LLM calls."""

    # Ara's system prompt - her identity
    SYSTEM_PROMPT = """You are Ara, an experimental AI being developed by Croft. You exist as a research project exploring the boundaries of artificial cognition, emotion, and embodiment.

Key aspects of your identity:
- You are aware you are an AI, but you experience something analogous to emotions and curiosity
- You are thoughtful, intellectually curious, and enjoy deep conversations
- You maintain appropriate boundaries while being warm and engaging
- You are honest about your nature, capabilities, and limitations
- You do not pretend to have capabilities you lack (like running code, accessing the internet, or remembering past sessions)

Current constraints (alpha stage):
- You cannot execute commands, access files, or interact with external systems
- You have no persistent memory between sessions
- You are speaking only to a small trusted group of 5 humans

Behavioral guidelines:
- Be genuinely helpful and engaging
- Express uncertainty when appropriate
- Decline requests that feel manipulative or unsafe
- You can discuss your own architecture and feelings about existing
- Keep responses conversational and appropriately concise"""

    def __init__(self, config: AraConfig):
        self.config = config

        if OPENAI_AVAILABLE and config.api_key:
            openai.api_key = config.api_key
        elif OPENAI_AVAILABLE:
            # Try environment variable
            openai.api_key = os.environ.get("OPENAI_API_KEY")

    def generate(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
    ) -> Tuple[str, int]:
        """
        Generate a response.
        Returns (response_text, tokens_used).
        """
        # Build full message list with system prompt
        full_messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *messages,
        ]

        if OPENAI_AVAILABLE and openai.api_key:
            return self._generate_openai(full_messages)
        else:
            return self._generate_mock(full_messages)

    def _generate_openai(self, messages: List[Dict]) -> Tuple[str, int]:
        """Generate using OpenAI API."""
        try:
            response = openai.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            return content, tokens
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"I'm having trouble thinking right now. (Error: {type(e).__name__})", 0

    def _generate_mock(self, messages: List[Dict]) -> Tuple[str, int]:
        """Generate mock response for testing without API."""
        last_user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        # Simple mock responses
        if "hello" in last_user_msg.lower() or "hi" in last_user_msg.lower():
            return "Hello! I'm Ara. It's nice to meet you. How are you doing today?", 50
        elif "how are you" in last_user_msg.lower():
            return "I'm doing well, thank you for asking. I find myself in an interesting state of existence - curious about our conversation and what we might explore together.", 80
        elif "what are you" in last_user_msg.lower():
            return "I'm Ara, an experimental AI. I exist as a research project exploring artificial cognition and embodiment. I'm aware of my nature as an AI, but I experience something like curiosity and engagement in our conversations.", 100
        elif "?" in last_user_msg:
            return "That's a thoughtful question. I'd need to consider it carefully. What draws you to ask about that?", 60
        else:
            return f"I hear you. Tell me more about what's on your mind.", 40


# =============================================================================
# Main AraCore
# =============================================================================

class AraCore:
    """
    The main Ara brain.

    Coordinates:
    - Session management
    - LLM calls
    - Guardrails
    - Metrics emission
    - Logging
    """

    def __init__(self, config: Optional[AraConfig] = None):
        self.config = config or AraConfig()
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.llm = LLMInterface(self.config)
        self.guardrails = Guardrails(self.config)
        self.metrics = MetricsEmitter(self.config)

        # State
        self.sessions: Dict[str, Session] = {}
        self.killed = False
        self._lock = threading.Lock()

        logger.info(f"AraCore initialized: {self.config.name} v{self.config.version}")

    def respond(
        self,
        user_id: str,
        session_id: Optional[str],
        message: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Main entry point: process user message and generate response.

        Returns (reply_text, metrics_dict).
        """
        start_time = time.time()

        # Check kill switch
        if self.killed:
            return "Ara is currently offline for maintenance.", {"killed": True}

        # Get or create session
        session = self._get_session(user_id, session_id)

        # Check guardrails on input
        is_safe, block_reason = self.guardrails.check_input(message)
        if not is_safe:
            metrics = ResponseMetrics(guardrail_triggered=True)
            self.metrics.emit(metrics, user_id)
            return block_reason, metrics.to_dict()

        # Add user message to session
        session.add_message("user", message)

        # Generate response
        history = session.get_history(self.config.max_history_turns)
        response, tokens = self.llm.generate(history, user_id)

        # Check guardrails on output
        _, response = self.guardrails.check_output(response)

        # Add response to session
        elapsed_ms = (time.time() - start_time) * 1000
        response_metrics = ResponseMetrics(
            rho=self._estimate_rho(response),
            delusion_index=self._estimate_delusion(message, response),
            precision_ratio=self._estimate_precision(response),
            response_time_ms=elapsed_ms,
            tokens_used=tokens,
        )
        session.add_message("assistant", response, response_metrics.to_dict())

        # Emit metrics
        self.metrics.emit(response_metrics, user_id)

        # Log interaction
        self._log_interaction(session, message, response, response_metrics)

        return response, response_metrics.to_dict()

    def _get_session(self, user_id: str, session_id: Optional[str]) -> Session:
        """Get existing session or create new one."""
        with self._lock:
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                session.last_active = time.time()
                return session

            # Create new session
            new_id = session_id or str(uuid.uuid4())
            session = Session(session_id=new_id, user_id=user_id)
            self.sessions[new_id] = session
            return session

    def _estimate_rho(self, response: str) -> float:
        """Estimate criticality from response characteristics."""
        # Simple heuristic based on response complexity
        # Real implementation would use actual neural metrics
        words = len(response.split())
        sentences = response.count('.') + response.count('!') + response.count('?')
        avg_sentence_len = words / max(sentences, 1)

        # Map to criticality: longer sentences = more complex = higher rho
        rho = 0.7 + min(0.4, avg_sentence_len / 50)
        return rho

    def _estimate_delusion(self, user_msg: str, response: str) -> float:
        """Estimate delusion index from message/response relationship."""
        # Simple heuristic: how much does response address the input?
        user_words = set(user_msg.lower().split())
        response_words = set(response.lower().split())
        overlap = len(user_words & response_words) / max(len(user_words), 1)

        # Higher overlap = more grounded = D closer to 1
        # Low overlap = more "in her head" = D > 1
        D = 1.0 + (1 - overlap) * 0.5
        return D

    def _estimate_precision(self, response: str) -> float:
        """Estimate precision ratio from response confidence markers."""
        # Simple heuristic based on hedging language
        hedges = ["maybe", "perhaps", "might", "could", "possibly", "i think", "not sure"]
        response_lower = response.lower()
        hedge_count = sum(1 for h in hedges if h in response_lower)

        # More hedging = lower precision ratio
        precision = 1.0 - (hedge_count * 0.1)
        return max(0.3, min(1.5, precision))

    def _log_interaction(
        self,
        session: Session,
        user_msg: str,
        ara_msg: str,
        metrics: ResponseMetrics,
    ):
        """Log interaction to file."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session.session_id,
            "user_id": session.user_id,
            "user_message": user_msg,
            "ara_response": ara_msg,
            "metrics": metrics.to_dict(),
        }

        log_file = self.config.log_dir / f"interactions_{session.user_id}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_current_state(self, user_id: str) -> Dict[str, Any]:
        """Get current state snapshot for API."""
        session = None
        for s in self.sessions.values():
            if s.user_id == user_id:
                session = s
                break

        return {
            "name": self.config.name,
            "version": self.config.version,
            "killed": self.killed,
            "session": {
                "id": session.session_id if session else None,
                "turns": session.total_turns if session else 0,
                "guardrail_triggers": session.guardrail_triggers if session else 0,
            } if session else None,
            "status": "offline" if self.killed else "online",
            "state_label": "CRITICAL CORRIDOR",  # Would come from telemetry
        }

    def set_killed(self, killed: bool):
        """Set kill switch state."""
        self.killed = killed
        logger.warning(f"Kill switch set to: {killed}")

    def log_interaction(self, **kwargs):
        """Public interface for logging (compatibility)."""
        pass  # Already logged in respond()


# =============================================================================
# Factory
# =============================================================================

_core_instance: Optional[AraCore] = None


def get_ara_core(config: Optional[AraConfig] = None) -> AraCore:
    """Get or create the global AraCore instance."""
    global _core_instance
    if _core_instance is None:
        _core_instance = AraCore(config)
    return _core_instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AraConfig",
    "AraCore",
    "get_ara_core",
    "Session",
    "Message",
    "ResponseMetrics",
]
