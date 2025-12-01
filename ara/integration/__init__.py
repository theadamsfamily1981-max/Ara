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
from ara.metacontrol import (
    L3MetacontrolService,
    WorkspaceMode,
    PADState as MetacontrolPADState,
    ControlModulation,
    WORKSPACE_PAD_MAPPINGS,
    get_metacontrol_service,
    set_workspace_mode,
    compute_pad_gating,
    get_metacontrol_status,
)

# Import telemetry (optional)
try:
    from ara.telemetry import (
        PulseTelemetry,
        get_telemetry,
        record_turn_metrics,
        create_telemetry_router,
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    get_telemetry = None
    record_turn_metrics = None

# Import TP-RL (Topological Meta-Plasticity)
try:
    from ara.tprl import (
        TPRLAgent,
        MaskEnvironment,
        evolve_mask,
        get_tprl_agent,
    )
    TPRL_AVAILABLE = True
except ImportError:
    TPRL_AVAILABLE = False
    evolve_mask = None

# Import Interoception Core
try:
    from ara.interoception import (
        InteroceptionCore,
        InteroceptivePAD,
        get_interoception_core,
        process_sensory_input,
    )
    INTEROCEPTION_AVAILABLE = True
except ImportError:
    INTEROCEPTION_AVAILABLE = False
    get_interoception_core = None

# Import CXL Control Plane
try:
    from ara.cxl_control import (
        ControlPlane,
        get_control_plane,
        fast_control,
    )
    CXL_AVAILABLE = True
except ImportError:
    CXL_AVAILABLE = False
    get_control_plane = None

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

    # L3 Metacontrol
    metacontrol: Optional[ControlModulation] = None
    workspace_mode: str = "default"

    # Derived
    effective_temperature: float = 1.0
    effective_memory_p: float = 1.0
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
            "metacontrol": self.metacontrol.to_dict() if self.metacontrol else None,
            "workspace_mode": self.workspace_mode,
            "effective_temperature": self.effective_temperature,
            "effective_memory_p": self.effective_memory_p,
            "selected_backend": self.selected_backend,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
        }


class AraOrchestrator:
    """
    Main orchestrator that coordinates the full brain-body stack.

    This is the central integration point that:
    1. Receives user input + sensory data
    2. Estimates affect (Pulse) OR generates from Interoception
    3. Gets/updates identity state (NIB)
    4. Routes to appropriate backend (AEPO)
    5. Applies L3 Metacontrol (with CXL fast path)
    6. Evolves topology (TP-RL) periodically
    7. Returns processed turn with all signals

    Revolutionary features:
    - TP-RL: Topological Meta-Plasticity via RL
    - Interoception: True internal PAD from SNN membrane dynamics
    - CXL Control: Sub-200us latency control plane
    """

    def __init__(
        self,
        default_backend: Backend = Backend.CLAUDE,
        prefer_local: bool = False,
        auto_mode_switch: bool = True,
        default_workspace_mode: WorkspaceMode = WorkspaceMode.DEFAULT,
        enable_interoception: bool = True,
        enable_cxl_control: bool = True,
        enable_tprl: bool = True,
        tprl_evolve_interval: int = 50,  # Evolve mask every N turns
    ):
        # Core components
        self.pulse = PulseEstimator()
        self.nib = NIBManager()
        self.aepo = AEPORouter(default_backend=default_backend, prefer_local=prefer_local)
        self.metacontrol = get_metacontrol_service()
        self.auto_mode_switch = auto_mode_switch

        # Set initial workspace mode
        self.metacontrol.set_workspace_mode(default_workspace_mode)

        # Revolutionary components
        self.enable_interoception = enable_interoception and INTEROCEPTION_AVAILABLE
        self.enable_cxl_control = enable_cxl_control and CXL_AVAILABLE
        self.enable_tprl = enable_tprl and TPRL_AVAILABLE
        self.tprl_evolve_interval = tprl_evolve_interval

        # Initialize revolutionary components
        if self.enable_interoception:
            self.interoception = get_interoception_core()
            logger.info("✅ Interoception Core enabled")

        if self.enable_cxl_control:
            self.cxl_plane = get_control_plane()
            logger.info("✅ CXL Control Plane enabled")

        if self.enable_tprl:
            self.tprl_agent = get_tprl_agent()
            logger.info("✅ TP-RL Agent enabled")

        # Session tracking
        self._sessions: Dict[str, TurnContext] = {}
        self._event_log: List[Dict] = []
        self._turn_count = 0

    def process_turn(
        self,
        user_text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        prosody: Optional[Dict] = None,
        audio_features: Optional[Dict] = None,
        text_embedding: Optional[List[float]] = None,
    ) -> ProcessedTurn:
        """
        Process a user turn through the full brain-body pipeline.

        Args:
            user_text: User input text
            session_id: Session identifier
            user_id: User identifier
            prosody: Optional prosody features from audio (legacy)
            audio_features: Prosody features for interoception
            text_embedding: Text embedding for interoception

        Returns:
            ProcessedTurn with all signals and routing decision
        """
        import time
        start_time = time.time()
        self._turn_count += 1

        # Get/create context
        context = self._get_context(session_id, user_id)
        context.turn_number += 1
        context.conversation_history.append(user_text)

        # 1. Estimate affect - use Interoception if available, else Pulse
        if self.enable_interoception and (audio_features or text_embedding):
            # Revolutionary: True internal PAD from SNN membrane dynamics
            intero_pad = self.interoception.process_sensory(
                audio_features=audio_features or prosody,
                text_embedding=text_embedding,
            )
            # Convert InteroceptivePAD to standard AffectEstimate format
            affect = self.pulse.estimate(
                text=user_text,
                session_id=session_id,
                context=context.conversation_history[-5:],
            )
            # Override PAD with interoceptive values
            affect.pad.pleasure = intero_pad.valence
            affect.pad.arousal = intero_pad.arousal * 2 - 1  # [0,1] → [-1,1]
            affect.pad.dominance = intero_pad.dominance * 2 - 1
            affect.confidence = intero_pad.confidence
        else:
            # Standard Pulse estimation
            affect = self.pulse.estimate(
                text=user_text,
                session_id=session_id,
                context=context.conversation_history[-5:],
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

        # 4. Compute L3 Metacontrol - use CXL fast path if available
        metacontrol_pad = MetacontrolPADState(
            valence=affect.pad.pleasure,
            arousal=max(0.0, min(1.0, (affect.pad.arousal + 1.0) / 2.0)),
            dominance=max(0.0, min(1.0, (affect.pad.dominance + 1.0) / 2.0)),
            confidence=affect.confidence,
        )

        if self.enable_cxl_control:
            # Revolutionary: Sub-200us latency control via CXL plane
            cxl_result = self.cxl_plane.fast_control_step(
                valence=metacontrol_pad.valence,
                arousal=metacontrol_pad.arousal,
                dominance=metacontrol_pad.dominance,
            )
            # Use CXL results
            metacontrol_modulation = self.metacontrol.compute_modulation(metacontrol_pad)
            # Override with CXL fast path values
            metacontrol_modulation.temperature_multiplier = cxl_result["temperature_mult"]
            metacontrol_modulation.memory_write_multiplier = cxl_result["memory_mult"]
            metacontrol_modulation.attention_gain = cxl_result["attention_gain"]
        else:
            # Standard metacontrol
            metacontrol_modulation = self.metacontrol.compute_modulation(metacontrol_pad)

        # 5. TP-RL: Evolve mask periodically
        if self.enable_tprl and self._turn_count % self.tprl_evolve_interval == 0:
            # Get jerk from metacontrol history
            jerk = getattr(metacontrol_modulation, 'effective_weight', 0.1)
            tprl_result = evolve_mask(
                jerk_signal=jerk,
                confidence=affect.confidence,
            )
            logger.info(f"TP-RL evolved mask: density={tprl_result['config']['density']:.4f}")

        # 6. Compute effective parameters (with metacontrol)
        effective_temp = self._compute_temperature(affect, routing, metacontrol_modulation)
        effective_mem_p = metacontrol_modulation.memory_write_multiplier
        effective_prompt = self._build_system_prompt(nib_state, routing)

        # Build result
        result = ProcessedTurn(
            user_text=user_text,
            context=context,
            affect=affect,
            nib_state=nib_state,
            routing=routing,
            metacontrol=metacontrol_modulation,
            workspace_mode=self.metacontrol.get_current_mode().value,
            effective_temperature=effective_temp,
            effective_memory_p=effective_mem_p,
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
        metacontrol: Optional[ControlModulation] = None,
    ) -> float:
        """
        Compute effective temperature from affect, routing, and L3 metacontrol.

        Temperature is influenced by (in order of priority):
        1. L3 Metacontrol temperature_multiplier (from workspace mode/PAD)
        2. Task-specific override from AEPO
        3. Affect-derived temp_scale from Pulse (fallback)

        Control law: final_temp = base × metacontrol_mult × routing_blend
        """
        base_temp = 0.7

        # Start with affect scaling
        temp = base_temp * affect.gates.temp_scale

        # Apply L3 metacontrol if available (highest priority)
        if metacontrol is not None:
            # Metacontrol provides a multiplier based on PAD/workspace mode
            # Blend: 60% metacontrol, 40% affect-based
            temp = base_temp * (0.6 * metacontrol.temperature_multiplier + 0.4 * affect.gates.temp_scale)

        # Apply routing override if specified (task-specific)
        if routing.temperature_override is not None:
            # Blend: 50% routing, 50% current (metacontrol+affect)
            temp = 0.5 * routing.temperature_override + 0.5 * temp

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
        """Log turn as Pulse event with L3 metacontrol telemetry."""
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
            # L3 Metacontrol telemetry
            "metacontrol": {
                "workspace_mode": turn.workspace_mode,
                "temperature_multiplier": turn.metacontrol.temperature_multiplier if turn.metacontrol else 1.0,
                "memory_write_multiplier": turn.metacontrol.memory_write_multiplier if turn.metacontrol else 1.0,
                "attention_gain": turn.metacontrol.attention_gain if turn.metacontrol else 1.0,
                "effective_weight": turn.metacontrol.effective_weight if turn.metacontrol else 1.0,
            },
            "effective_temperature": turn.effective_temperature,
            "effective_memory_p": turn.effective_memory_p,
        }
        self._event_log.append(event)
        logger.info(f"Pulse event: {json.dumps(event)}")

        # Record to Prometheus telemetry if available
        if TELEMETRY_AVAILABLE and record_turn_metrics is not None:
            record_turn_metrics(turn.to_dict())

    def get_event_log(self) -> List[Dict]:
        """Get all logged events."""
        return self._event_log.copy()


def create_integration_router():
    """
    Create FastAPI router with integration endpoints.

    Returns router with:
    - POST /pulse/estimate_affect - Estimate affect from text
    - GET /nib/state - Get NIB identity state
    - POST /aepo/route - Route request to backend
    - POST /process - Full pipeline (Pulse→NIB→AEPO→Metacontrol)
    - GET /events - Get Pulse event log

    L3 Metacontrol endpoints:
    - POST /control/pad_gating - Compute modulation from raw PAD
    - POST /control/mode - Set workspace mode (work/relax/creative/support)
    - GET /control/status - Get current metacontrol status
    - GET /control/modes - Get available modes with PAD mappings
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

    # L3 Metacontrol models
    class PADGatingRequest(BaseModel):
        valence: float  # [-1, 1] pleasure/displeasure
        arousal: float  # [0, 1] activation level
        dominance: float = 0.5  # [0, 1] control/agency
        confidence: float = 1.0  # [0, 1] confidence in estimate

    class WorkspaceModeRequest(BaseModel):
        mode: str  # work, relax, creative, support, default

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

    # L3 Metacontrol endpoints
    @router.post("/control/pad_gating")
    async def pad_gating(request: PADGatingRequest):
        """
        Compute L3 metacontrol modulation from raw PAD values.

        Returns temperature and memory multipliers based on the control law:
        - Arousal → Temperature (0.8-1.3): Higher arousal = more exploratory
        - Valence → Memory P (0.7-1.2): Lower valence = more conservative writes
        """
        result = compute_pad_gating(
            valence=request.valence,
            arousal=request.arousal,
            dominance=request.dominance,
            confidence=request.confidence,
        )
        return result

    @router.post("/control/mode")
    async def set_mode(request: WorkspaceModeRequest):
        """
        Set workspace mode for L3 metacontrol.

        Modes map to PAD states:
        - work: High valence (0.8), high arousal (0.7) → focused, energetic
        - relax: Low valence (-0.2), low arousal (0.3) → calm, conservative
        - creative: High arousal (0.8), neutral valence (0.3) → exploratory
        - support: High valence (0.6), low arousal (0.4) → warm, stable
        - default: Neutral baseline
        """
        result = set_workspace_mode(request.mode)
        # Also update orchestrator's metacontrol
        try:
            ws_mode = WorkspaceMode(request.mode.lower())
            orchestrator.metacontrol.set_workspace_mode(ws_mode)
        except ValueError:
            pass
        return result

    @router.get("/control/status")
    async def control_status():
        """Get current L3 metacontrol status."""
        return get_metacontrol_status()

    @router.get("/control/modes")
    async def available_modes():
        """Get available workspace modes with their PAD mappings."""
        return {
            mode.value: {
                "valence": pad.valence,
                "arousal": pad.arousal,
                "dominance": pad.dominance,
                "confidence": pad.confidence,
            }
            for mode, pad in WORKSPACE_PAD_MAPPINGS.items()
        }

    return router


__all__ = [
    # Core classes
    "TurnContext",
    "ProcessedTurn",
    "AraOrchestrator",
    "create_integration_router",
    # Re-exported from submodules
    "PADState",
    "GatingSignals",
    "AffectEstimate",
    "estimate_affect",
    "NIBState",
    "IdentityMode",
    "get_nib_state",
    "RoutingDecision",
    "Backend",
    "route_request",
    # L3 Metacontrol
    "WorkspaceMode",
    "ControlModulation",
    "WORKSPACE_PAD_MAPPINGS",
    "set_workspace_mode",
    "compute_pad_gating",
    "get_metacontrol_status",
]
