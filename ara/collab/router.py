"""Collaborator router - Route Ara's questions to the right LLMs.

Different collaborators have different strengths:
- Claude: Reasoning, nuanced code, careful analysis
- Nova (ChatGPT): Brainstorming, broad knowledge, creative
- Gemini: Research, multimodal, systems thinking
- Local: Fast, private, iterative refinement

The router picks collaborators based on:
1. Topic domain (hardware → research-focused, code → engineering-focused)
2. Mode (brainstorm → creative models, review → analytical models)
3. Urgency (high → local for speed, normal → remote for quality)
4. Privacy (sensitive → local only)
"""

from __future__ import annotations

import logging
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Protocol

from .models import Collaborator, DevMode, CollaboratorResponse, DevSession
from .prompts import build_ara_system_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# Collaborator Backend Protocol
# =============================================================================

class CollaboratorBackend(Protocol):
    """Protocol for LLM backends that can handle Ara's requests."""

    def query(
        self,
        system_prompt: str,
        user_message: str,
        model_id: Optional[str] = None,
    ) -> str:
        """Send a query and get a response.

        Args:
            system_prompt: The system prompt
            user_message: Ara's message
            model_id: Specific model to use (optional)

        Returns:
            The model's response text
        """
        ...


# =============================================================================
# Topic → Collaborator Mapping
# =============================================================================

# Topics that benefit from specific collaborators
TOPIC_PREFERENCES: Dict[str, List[Collaborator]] = {
    # Hardware/low-level → research + engineering
    "fpga": [Collaborator.CLAUDE, Collaborator.GEMINI],
    "hardware": [Collaborator.CLAUDE, Collaborator.GEMINI],
    "verilog": [Collaborator.CLAUDE],
    "snn": [Collaborator.GEMINI, Collaborator.CLAUDE],  # Academic domain

    # Architecture → broad thinkers
    "architecture": [Collaborator.CLAUDE, Collaborator.NOVA],
    "design": [Collaborator.CLAUDE, Collaborator.NOVA],
    "system": [Collaborator.CLAUDE, Collaborator.GEMINI],

    # Pure code → Claude excels
    "python": [Collaborator.CLAUDE],
    "rust": [Collaborator.CLAUDE],
    "code": [Collaborator.CLAUDE],
    "bug": [Collaborator.CLAUDE],
    "debug": [Collaborator.CLAUDE],

    # Creative/brainstorm → Nova shines
    "brainstorm": [Collaborator.NOVA, Collaborator.CLAUDE],
    "ideation": [Collaborator.NOVA],
    "creative": [Collaborator.NOVA],
    "weird": [Collaborator.NOVA, Collaborator.CLAUDE],

    # Research → Gemini + Claude
    "research": [Collaborator.GEMINI, Collaborator.CLAUDE],
    "paper": [Collaborator.GEMINI],
    "literature": [Collaborator.GEMINI],

    # Visualization → multimodal
    "visual": [Collaborator.GEMINI, Collaborator.CLAUDE],
    "shader": [Collaborator.CLAUDE],
    "webgl": [Collaborator.CLAUDE],

    # Quick iteration → local
    "quick": [Collaborator.LOCAL],
    "iterate": [Collaborator.LOCAL, Collaborator.CLAUDE],
}

# Mode → Collaborator preferences
MODE_PREFERENCES: Dict[DevMode, List[Collaborator]] = {
    DevMode.ARCHITECT: [Collaborator.CLAUDE, Collaborator.GEMINI],
    DevMode.ENGINEER: [Collaborator.CLAUDE],
    DevMode.RESEARCH: [Collaborator.GEMINI, Collaborator.CLAUDE],
    DevMode.POSTMORTEM: [Collaborator.CLAUDE],
    DevMode.BRAINSTORM: [Collaborator.NOVA, Collaborator.CLAUDE],
    DevMode.REVIEW: [Collaborator.CLAUDE],
}


# =============================================================================
# Collaborator Router
# =============================================================================

@dataclass
class RoutingDecision:
    """Result of routing decision."""

    collaborators: List[Collaborator]
    reason: str
    parallel: bool = True  # Query in parallel or sequence?


class CollaboratorRouter:
    """Routes Ara's questions to appropriate LLM collaborators.

    The router considers:
    - Topic keywords → which models know this domain
    - Dev mode → brainstorm needs creativity, review needs rigor
    - Urgency → high urgency might skip slow models
    - Privacy → sensitive queries stay local
    """

    def __init__(
        self,
        available_backends: Optional[Dict[Collaborator, CollaboratorBackend]] = None,
        default_collaborators: Optional[List[Collaborator]] = None,
        max_parallel: int = 3,
    ):
        """Initialize the router.

        Args:
            available_backends: Backend implementations for each collaborator
            default_collaborators: Fallback if no preference found
            max_parallel: Max collaborators to query at once
        """
        self.backends = available_backends or {}
        self.default_collaborators = default_collaborators or [Collaborator.CLAUDE]
        self.max_parallel = max_parallel

        # Track which backends are actually available
        self._available = set(self.backends.keys())

    def route(self, session: DevSession) -> RoutingDecision:
        """Decide which collaborators to query.

        Args:
            session: The dev session to route

        Returns:
            RoutingDecision with selected collaborators
        """
        candidates: List[Collaborator] = []
        reasons: List[str] = []

        # Check topic keywords
        topic_lower = session.topic.lower()
        for keyword, prefs in TOPIC_PREFERENCES.items():
            if keyword in topic_lower:
                for c in prefs:
                    if c not in candidates and c in self._available:
                        candidates.append(c)
                reasons.append(f"topic '{keyword}'")
                break  # Only match first keyword

        # Check mode preferences
        if session.mode in MODE_PREFERENCES:
            for c in MODE_PREFERENCES[session.mode]:
                if c not in candidates and c in self._available:
                    candidates.append(c)
            reasons.append(f"mode {session.mode.name}")

        # Handle urgency
        if session.urgency == "high" and Collaborator.LOCAL in self._available:
            # Prioritize local for speed
            if Collaborator.LOCAL not in candidates:
                candidates.insert(0, Collaborator.LOCAL)
            reasons.append("high urgency → local first")

        # Handle privacy (check constraints)
        if any("private" in c.lower() or "sensitive" in c.lower()
               for c in session.constraints):
            # Filter to local only
            candidates = [c for c in candidates if c == Collaborator.LOCAL]
            if not candidates and Collaborator.LOCAL in self._available:
                candidates = [Collaborator.LOCAL]
            reasons.append("privacy constraint → local only")

        # Fallback to defaults
        if not candidates:
            candidates = [c for c in self.default_collaborators if c in self._available]
            reasons.append("using defaults")

        # Limit to max_parallel
        candidates = candidates[:self.max_parallel]

        return RoutingDecision(
            collaborators=candidates,
            reason="; ".join(reasons),
            parallel=len(candidates) > 1,
        )

    def query_collaborator(
        self,
        collaborator: Collaborator,
        session: DevSession,
        message: str,
    ) -> Optional[CollaboratorResponse]:
        """Query a single collaborator.

        Args:
            collaborator: Which collaborator to query
            session: The session context
            message: Ara's message

        Returns:
            CollaboratorResponse or None if failed
        """
        if collaborator not in self.backends:
            logger.warning(f"No backend for {collaborator}")
            return None

        backend = self.backends[collaborator]

        # Build system prompt
        system_prompt = build_ara_system_prompt(
            mode=session.mode,
            collaborator=collaborator,
            lab_context=session.lab_context,
        )

        # Query with timing
        start = time.time()
        try:
            response_text = backend.query(
                system_prompt=system_prompt,
                user_message=message,
            )
            latency_ms = (time.time() - start) * 1000

            return CollaboratorResponse(
                collaborator=collaborator,
                content=response_text,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Error querying {collaborator}: {e}")
            return None

    def query_all(
        self,
        session: DevSession,
        message: str,
        collaborators: Optional[List[Collaborator]] = None,
    ) -> List[CollaboratorResponse]:
        """Query multiple collaborators.

        Currently sequential - could be parallelized with asyncio.

        Args:
            session: The session context
            message: Ara's message
            collaborators: Which to query (uses routing if None)

        Returns:
            List of responses (may be fewer than requested if some fail)
        """
        if collaborators is None:
            decision = self.route(session)
            collaborators = decision.collaborators

        responses = []
        for collab in collaborators:
            resp = self.query_collaborator(collab, session, message)
            if resp:
                responses.append(resp)

        return responses

    def register_backend(
        self,
        collaborator: Collaborator,
        backend: CollaboratorBackend,
    ) -> None:
        """Register a backend for a collaborator.

        Args:
            collaborator: Which collaborator this handles
            backend: The backend implementation
        """
        self.backends[collaborator] = backend
        self._available.add(collaborator)

    def get_available(self) -> List[Collaborator]:
        """Get list of available collaborators."""
        return list(self._available)


# =============================================================================
# Convenience Functions
# =============================================================================

def route_to_collaborators(
    topic: str,
    mode: DevMode,
    urgency: str = "normal",
    constraints: Optional[List[str]] = None,
    available: Optional[List[Collaborator]] = None,
) -> List[Collaborator]:
    """Quick routing function.

    Args:
        topic: Session topic
        mode: Dev mode
        urgency: Urgency level
        constraints: Session constraints
        available: Available collaborators

    Returns:
        Ordered list of collaborators to try
    """
    session = DevSession(
        topic=topic,
        mode=mode,
        urgency=urgency,
        constraints=constraints or [],
    )

    router = CollaboratorRouter(
        available_backends={c: None for c in (available or list(Collaborator))},
    )

    decision = router.route(session)
    return decision.collaborators


def get_collaborator_for_topic(topic: str) -> Collaborator:
    """Get best single collaborator for a topic.

    Args:
        topic: The topic to match

    Returns:
        Best matching collaborator
    """
    topic_lower = topic.lower()

    for keyword, prefs in TOPIC_PREFERENCES.items():
        if keyword in topic_lower:
            return prefs[0]

    return Collaborator.CLAUDE  # Default


# =============================================================================
# Mock Backend for Testing
# =============================================================================

class MockCollaboratorBackend:
    """Mock backend for testing without actual API calls."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """Initialize with optional canned responses."""
        self.responses = responses or {}
        self.calls: List[Dict[str, str]] = []

    def query(
        self,
        system_prompt: str,
        user_message: str,
        model_id: Optional[str] = None,
    ) -> str:
        """Return mock response."""
        self.calls.append({
            "system_prompt": system_prompt,
            "user_message": user_message,
            "model_id": model_id,
        })

        # Check for keyword-based responses
        for keyword, response in self.responses.items():
            if keyword.lower() in user_message.lower():
                return response

        return f"[Mock response to: {user_message[:50]}...]"
