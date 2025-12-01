"""
ARA Integration Module

Wires together Pulse, NIB, and AEPO into a unified API router
that can be mounted into the avatar pipeline.

This module provides:
- FastAPI router with /pulse, /nib, /aepo endpoints
- Integration hooks for the avatar generation pipeline
- Event logging for Pulse telemetry

Usage:
    from ara.integration import create_integration_router, AraOrchestrator

    # Add to FastAPI app
    app.include_router(create_integration_router(), prefix="/ara")

    # Or use orchestrator directly
    orchestrator = AraOrchestrator()
    result = orchestrator.process_turn(user_text, session_id)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import logging

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Import our modules
from ara.pulse import (
    PulseEstimator,
    AffectEstimate,
    PADState,
    GatingSignals,
    estimate_affect,
)
from ara.nib import (
    NIBManager,
    NIBState,
    IdentityMode,
    StyleParameters,
    get_nib_state,
    infer_identity_mode,
)
from ara.aepo import (
    AEPORouter,
    RoutingDecision,
    Backend,
    TaskType,
    route_request,
)

logger = logging.getLogger("ara.integration")


@dataclass
class TurnContext:
    """Context for a single conversation turn."""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    turn_number: int = 0
    conversation_history: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ProcessedTurn:
    """Result of processing a user turn through the full pipeline."""
    # Input
    user_text: str
    context: TurnContext

    # Pulse (affect)
    affect: AffectEstimate = None

    # NIB (identity)
    nib_state: NIBState = None

    # AEPO (routing)
    routing: RoutingDecision = None

    # Derived
    effective_temperature: float = 1.0
    effective_system_prompt: str = ""
    selected_backend: str = "claude"

    # Metadata
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_text": self.user_text,
            "session_id": self.context.session_id,
            "turn_number": self.context.turn_number,
            "affect": self.affect.to_dict() if self.affect else None,
            "nib_state": self.nib_state.to_dict() if self.nib_state else None,
            "routing": self.routing.to_dict() if self.routing else None,
            "effective_temperature": self.effective_temperature,
            "selected_backend": self.selected_backend,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
        }


class AraOrchestrator:
    """
    Main orchestrator that coordinates Pulse, NIB, and AEPO.

    This is the central integration point that:
    1. Receives user input
    2. Estimates affect (Pulse)
    3. Gets/updates identity state (NIB)
    4. Routes to appropriate backend (AEPO)
    5. Returns processed turn with all signals
    """

    def __init__(
        self,
        default_backend: Backend = Backend.CLAUDE,
        prefer_local: bool = False,
        auto_mode_switch: bool = True,
    ):
        self.pulse = PulseEstimator()
        self.nib = NIBManager()
        self.aepo = AEPORouter(default_backend=default_backend, prefer_local=prefer_local)
        self.auto_mode_switch = auto_mode_switch

        # Session tracking
        self._sessions: Dict[str, TurnContext] = {}
        self._event_log: List[Dict] = []

    def process_turn(
        self,
        user_text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        prosody: Optional[Dict] = None,
    ) -> ProcessedTurn:
        """
        Process a user turn through the full pipeline.

        Args:
            user_text: User input text
            session_id: Session identifier
            user_id: User identifier
            prosody: Optional prosody features from audio

        Returns:
            ProcessedTurn with all signals and routing decision
        """
        import time
        start_time = time.time()

        # Get/create context
        context = self._get_context(session_id, user_id)
        context.turn_number += 1
        context.conversation_history.append(user_text)

        # 1. Estimate affect (Pulse)
        affect = self.pulse.estimate(
            text=user_text,
            session_id=session_id,
            context=context.conversation_history[-5:],  # Last 5 turns
        )

        # 2. Get/update NIB state
        nib_state = self.nib.get_state(session_id, user_id)

        # Auto-switch mode if enabled
        if self.auto_mode_switch:
            inferred_mode = self.nib.infer_mode(user_text, context.conversation_history)
            if inferred_mode != nib_state.identity_mode:
                nib_state = self.nib.switch_mode(inferred_mode, session_id)

        # Adapt style based on affect
        nib_state.style = self.nib.adapt_style(
            nib_state,
            affect.pad.pleasure,
            affect.pad.arousal,
            affect.pad.dominance,
        )

        # 3. Route request (AEPO)
        routing = self.aepo.route(
            text=user_text,
            context=context.conversation_history,
            session_id=session_id,
        )

        # 4. Compute effective parameters
        effective_temp = self._compute_temperature(affect, routing)
        effective_prompt = self._build_system_prompt(nib_state, routing)

        # Build result
        result = ProcessedTurn(
            user_text=user_text,
            context=context,
            affect=affect,
            nib_state=nib_state,
            routing=routing,
            effective_temperature=effective_temp,
            effective_system_prompt=effective_prompt,
            selected_backend=routing.selected_backend.value,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        # Log event
        self._log_event(result)

        return result

    def _get_context(
        self,
        session_id: Optional[str],
        user_id: Optional[str],
    ) -> TurnContext:
        """Get or create session context."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        context = TurnContext(session_id=session_id, user_id=user_id)
        if session_id:
            self._sessions[session_id] = context
        return context

    def _compute_temperature(
        self,
        affect: AffectEstimate,
        routing: RoutingDecision,
    ) -> float:
        """
        Compute effective temperature from affect and routing.

        Temperature is influenced by:
        - Affect-derived temp_scale from Pulse
        - Task-specific override from AEPO
        """
        base_temp = 0.7

        # Apply affect scaling
        temp = base_temp * affect.gates.temp_scale

        # Apply routing override if specified
        if routing.temperature_override is not None:
            # Blend: 70% routing, 30% affect
            temp = 0.7 * routing.temperature_override + 0.3 * temp

        return max(0.1, min(2.0, temp))

    def _build_system_prompt(
        self,
        nib_state: NIBState,
        routing: RoutingDecision,
    ) -> str:
        """
        Build effective system prompt from NIB and routing.
        """
        parts = []

        # Base persona from NIB mode
        mode_prompts = {
            IdentityMode.GUIDE_MENTOR: "You are Ara, a knowledgeable and supportive guide. Help the user learn and grow.",
            IdentityMode.TECHNICAL: "You are Ara, a precise technical assistant. Focus on accuracy and detail.",
            IdentityMode.CREATIVE: "You are Ara, a creative collaborator. Explore ideas freely and offer fresh perspectives.",
            IdentityMode.SUPPORT: "You are Ara, an empathetic supporter. Listen carefully and validate feelings.",
            IdentityMode.EXECUTIVE: "You are Ara, a decisive assistant. Be direct and action-oriented.",
            IdentityMode.RESEARCH: "You are Ara, a thorough researcher. Provide well-sourced analysis.",
        }
        parts.append(mode_prompts.get(nib_state.identity_mode, mode_prompts[IdentityMode.GUIDE_MENTOR]))

        # Style modifiers from NIB
        modifiers = self.nib.get_prompt_modifiers(nib_state)
        style_desc = f"Maintain a {modifiers['tone']} tone with {modifiers['length']}. Be {modifiers['approach']}."
        parts.append(style_desc)

        # Task-specific additions from AEPO
        if routing.system_prompt_additions:
            parts.extend(routing.system_prompt_additions)

        return " ".join(parts)

    def _log_event(self, turn: ProcessedTurn):
        """Log turn as Pulse event."""
        event = {
            "event_type": "affect.events.v1",
            "timestamp": turn.timestamp,
            "session_id": turn.context.session_id,
            "turn_number": turn.context.turn_number,
            "pad": turn.affect.pad.to_dict() if turn.affect else None,
            "gates": turn.affect.gates.to_dict() if turn.affect else None,
            "evt_flagged": turn.affect.evt_flagged if turn.affect else False,
            "identity_mode": turn.nib_state.identity_mode.value if turn.nib_state else None,
            "selected_backend": turn.selected_backend,
            "processing_time_ms": turn.processing_time_ms,
        }
        self._event_log.append(event)
        logger.info(f"Pulse event: {json.dumps(event)}")

    def get_event_log(self) -> List[Dict]:
        """Get all logged events."""
        return self._event_log.copy()


def create_integration_router():
    """
    Create FastAPI router with integration endpoints.

    Returns router with:
    - POST /pulse/estimate_affect
    - GET /nib/state
    - POST /aepo/route
    - POST /process (full pipeline)
    """
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel
    except ImportError:
        logger.warning("FastAPI not available, router not created")
        return None

    router = APIRouter(tags=["integration"])

    # Request/Response models
    class AffectRequest(BaseModel):
        text: str
        session_id: Optional[str] = None
        prosody: Optional[Dict] = None

    class NIBRequest(BaseModel):
        session_id: Optional[str] = None
        user_id: Optional[str] = None

    class RouteRequest(BaseModel):
        text: str
        context: Optional[List[str]] = None
        session_id: Optional[str] = None
        available_backends: Optional[List[str]] = None

    class ProcessRequest(BaseModel):
        text: str
        session_id: Optional[str] = None
        user_id: Optional[str] = None
        prosody: Optional[Dict] = None

    # Shared orchestrator
    orchestrator = AraOrchestrator()

    @router.post("/pulse/estimate_affect")
    async def pulse_estimate(request: AffectRequest):
        """Estimate affect from text."""
        result = estimate_affect(
            text=request.text,
            prosody=request.prosody,
            session_id=request.session_id,
        )
        return result

    @router.get("/nib/state")
    async def nib_state(session_id: Optional[str] = None, user_id: Optional[str] = None):
        """Get NIB identity state."""
        result = get_nib_state(session_id=session_id, user_id=user_id)
        return result

    @router.post("/aepo/route")
    async def aepo_route(request: RouteRequest):
        """Route request to appropriate backend."""
        result = route_request(
            text=request.text,
            context=request.context,
            available_backends=request.available_backends,
        )
        return result

    @router.post("/process")
    async def process_turn(request: ProcessRequest):
        """Process full turn through Pulse → NIB → AEPO pipeline."""
        result = orchestrator.process_turn(
            user_text=request.text,
            session_id=request.session_id,
            user_id=request.user_id,
            prosody=request.prosody,
        )
        return result.to_dict()

    @router.get("/events")
    async def get_events():
        """Get Pulse event log."""
        return orchestrator.get_event_log()

    return router


__all__ = [
    "TurnContext",
    "ProcessedTurn",
    "AraOrchestrator",
    "create_integration_router",
]
