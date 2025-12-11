#!/usr/bin/env python3
# api/routers/narrative_router.py
"""
Narrative API Router: REST/WebSocket endpoints for Ara's self-narration.

Provides:
- Current narrative state (all audience types)
- Phase transition history
- Real-time narrative streaming via WebSocket
- Metrics update endpoint for external integrators
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ara.narrative.interface import (
    NarrativeEngine,
    NarrativeStreamer,
    SystemMetrics,
    LifecyclePhase,
    PHASE_PROFILES,
)
from ara.narrative.dojo_hook import NarrativeGovernanceAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/narrative", tags=["narrative"])


# ============================================================================
# Pydantic Models
# ============================================================================

class MetricsInput(BaseModel):
    """Input schema for metrics update."""
    throughput_agents_per_sec: float = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    cpu_utilization_pct: float = Field(default=0, ge=0, le=100)
    planning_horizon_steps: int = Field(default=0, ge=0)
    futures_explored_count: int = Field(default=0, ge=0)
    safety_prevented_pct: float = Field(default=0, ge=0, le=100)
    prediction_accuracy_pct: float = Field(default=0, ge=0, le=100)
    covenant_violations: int = Field(default=0, ge=0)
    entropy_cost_bits: float = Field(default=0, ge=0)
    reversibility_score: float = Field(default=0, ge=0, le=1)
    total_decisions: int = Field(default=0, ge=0)
    hours_since_deployment: float = Field(default=0, ge=0)


class NarrativeResponse(BaseModel):
    """Response schema for narrative endpoints."""
    timestamp: float
    datetime: str
    phase: str
    efficiency: Dict[str, float]
    narrative: str
    audience: str


class PhaseTransition(BaseModel):
    """Schema for phase transition events."""
    from_phase: str = Field(alias="from")
    to_phase: str = Field(alias="to")
    timestamp: float
    total_decisions: int


class NarrativeStateResponse(BaseModel):
    """Full state response with all audiences."""
    operator: NarrativeResponse
    public: NarrativeResponse
    technical: NarrativeResponse
    mythic: NarrativeResponse
    phase_transitions: List[Dict[str, Any]]


class PhaseProfileResponse(BaseModel):
    """Response for phase profile lookup."""
    name: str
    efficiency_target: float
    risk_tolerance: float
    narrative_tone: str
    learning_priority: str


# ============================================================================
# Global State
# ============================================================================

# Singleton engine and streamer
_engine: Optional[NarrativeEngine] = None
_streamer: Optional[NarrativeStreamer] = None
_governance: Optional[NarrativeGovernanceAdapter] = None
_current_reports: Dict[str, Dict[str, Any]] = {}
_ws_connections: List[WebSocket] = []


def get_engine() -> NarrativeEngine:
    """Get or create the narrative engine singleton."""
    global _engine
    if _engine is None:
        _engine = NarrativeEngine()
    return _engine


def get_streamer() -> NarrativeStreamer:
    """Get or create the narrative streamer singleton."""
    global _streamer, _engine
    if _streamer is None:
        _streamer = NarrativeStreamer(get_engine())
    return _streamer


def get_governance() -> NarrativeGovernanceAdapter:
    """Get or create the governance adapter singleton."""
    global _governance
    if _governance is None:
        _governance = NarrativeGovernanceAdapter(get_engine())
    return _governance


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class NarrativeConnectionManager:
    """Manage WebSocket connections for narrative streaming."""

    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.append(websocket)
        logger.info(
            "Narrative WebSocket connected. Total: %d",
            len(self.connections)
        )

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)
        logger.info(
            "Narrative WebSocket disconnected. Total: %d",
            len(self.connections)
        )

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast to all connected clients."""
        disconnected = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning("Failed to send to WebSocket: %s", e)
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)


ws_manager = NarrativeConnectionManager()


# ============================================================================
# REST Endpoints
# ============================================================================

@router.get("/current", response_model=NarrativeStateResponse)
async def get_current_narrative():
    """
    Get current narrative state for all audience types.

    Returns the most recent narrative reports including:
    - Operator view (concise, actionable)
    - Public view (narrative-focused)
    - Technical view (detailed diagnostics)
    - Mythic view (archetypal framing)
    """
    global _current_reports

    if not _current_reports:
        # Generate initial report with default metrics
        engine = get_engine()
        metrics = SystemMetrics(
            timestamp=time.time(),
            total_decisions=0,
            hours_since_deployment=0,
            current_phase=LifecyclePhase.EMBRYO
        )
        efficiency = engine.compute_operational_efficiency(metrics)
        _current_reports = {
            'operator': engine.generate_narrative(metrics, efficiency, "operator"),
            'public': engine.generate_narrative(metrics, efficiency, "public"),
            'technical': engine.generate_narrative(metrics, efficiency, "technical"),
            'mythic': engine.generate_narrative(metrics, efficiency, "mythic"),
        }

    return NarrativeStateResponse(
        operator=_current_reports['operator'],
        public=_current_reports['public'],
        technical=_current_reports['technical'],
        mythic=_current_reports['mythic'],
        phase_transitions=get_engine().phase_transitions
    )


@router.get("/audience/{audience}", response_model=NarrativeResponse)
async def get_narrative_by_audience(audience: str):
    """
    Get narrative for a specific audience type.

    Args:
        audience: One of "operator", "public", "technical", "mythic"
    """
    valid_audiences = ["operator", "public", "technical", "mythic"]
    if audience not in valid_audiences:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audience. Must be one of: {valid_audiences}"
        )

    global _current_reports
    if not _current_reports or audience not in _current_reports:
        # Generate fresh
        engine = get_engine()
        metrics = SystemMetrics(timestamp=time.time())
        efficiency = engine.compute_operational_efficiency(metrics)
        return engine.generate_narrative(metrics, efficiency, audience)

    return _current_reports[audience]


@router.get("/phases")
async def get_phase_definitions():
    """Get all lifecycle phase definitions and thresholds."""
    return {
        phase.value: {
            "name": profile.name,
            "efficiency_target": profile.efficiency_target,
            "risk_tolerance": profile.risk_tolerance,
            "narrative_tone": profile.narrative_tone,
            "learning_priority": profile.learning_priority
        }
        for phase, profile in PHASE_PROFILES.items()
    }


@router.get("/phases/{phase}", response_model=PhaseProfileResponse)
async def get_phase_profile(phase: str):
    """Get profile for a specific lifecycle phase."""
    try:
        lifecycle_phase = LifecyclePhase(phase)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown phase: {phase}. Valid phases: {[p.value for p in LifecyclePhase]}"
        )

    profile = PHASE_PROFILES[lifecycle_phase]
    return PhaseProfileResponse(
        name=profile.name,
        efficiency_target=profile.efficiency_target,
        risk_tolerance=profile.risk_tolerance,
        narrative_tone=profile.narrative_tone,
        learning_priority=profile.learning_priority
    )


@router.get("/transitions")
async def get_phase_transitions():
    """Get history of phase transitions."""
    return {
        "transitions": get_engine().phase_transitions,
        "total": len(get_engine().phase_transitions)
    }


@router.post("/update")
async def update_metrics(metrics_input: MetricsInput):
    """
    Update system metrics and regenerate narratives.

    This endpoint allows external systems to push metrics updates,
    triggering narrative regeneration and WebSocket broadcast.
    """
    global _current_reports

    engine = get_engine()

    # Convert to SystemMetrics
    metrics = SystemMetrics(
        timestamp=time.time(),
        throughput_agents_per_sec=metrics_input.throughput_agents_per_sec,
        latency_ms=metrics_input.latency_ms,
        cpu_utilization_pct=metrics_input.cpu_utilization_pct,
        planning_horizon_steps=metrics_input.planning_horizon_steps,
        futures_explored_count=metrics_input.futures_explored_count,
        safety_prevented_pct=metrics_input.safety_prevented_pct,
        prediction_accuracy_pct=metrics_input.prediction_accuracy_pct,
        covenant_violations=metrics_input.covenant_violations,
        entropy_cost_bits=metrics_input.entropy_cost_bits,
        reversibility_score=metrics_input.reversibility_score,
        total_decisions=metrics_input.total_decisions,
        hours_since_deployment=metrics_input.hours_since_deployment,
        current_phase=LifecyclePhase.EMBRYO  # Will be recalculated
    )

    # Compute efficiency and determine phase
    efficiency = engine.compute_operational_efficiency(metrics)
    metrics.current_phase = engine.determine_phase(
        metrics.total_decisions,
        efficiency['overall_efficiency']
    )

    # Generate all narratives
    _current_reports = {
        'operator': engine.generate_narrative(metrics, efficiency, "operator"),
        'public': engine.generate_narrative(metrics, efficiency, "public"),
        'technical': engine.generate_narrative(metrics, efficiency, "technical"),
        'mythic': engine.generate_narrative(metrics, efficiency, "mythic"),
    }

    # Broadcast to WebSocket clients
    await ws_manager.broadcast({
        'type': 'narrative_update',
        'data': _current_reports
    })

    return {
        "status": "success",
        "phase": metrics.current_phase.value,
        "efficiency": efficiency
    }


@router.get("/governance/summary")
async def get_governance_summary():
    """Get narrative governance summary for MEIS integration."""
    governance = get_governance()
    return governance.get_narrative_summary()


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@router.websocket("/ws")
async def narrative_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time narrative streaming.

    Clients receive:
    - narrative_update: New narrative generated
    - phase_transition: Phase changed
    - ping/pong: Keepalive

    Clients can send:
    - ping: Request pong response
    - request_current: Request current narrative state
    """
    await ws_manager.connect(websocket)

    try:
        # Send current state on connect
        if _current_reports:
            await websocket.send_json({
                'type': 'initial_state',
                'data': _current_reports
            })

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "request_current":
                    if _current_reports:
                        await websocket.send_json({
                            'type': 'current_state',
                            'data': _current_reports
                        })

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text("ping")

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        ws_manager.disconnect(websocket)


# ============================================================================
# Dashboard HTML (optional inline for simplicity)
# ============================================================================

@router.get("/dashboard", response_class=HTMLResponse)
async def narrative_dashboard():
    """
    Serve the narrative dashboard HTML page.

    This provides a standalone view without requiring template files.
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ara Narrative Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff00;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #00ff00;
            text-shadow: 0 0 10px #00ff00;
            border-bottom: 2px solid #00ff00;
            padding-bottom: 15px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .panel {
            background: rgba(26, 26, 46, 0.9);
            border: 1px solid #00ff00;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.1);
        }
        .panel h2 {
            color: #00ffff;
            border-bottom: 1px solid #00ff00;
            padding-bottom: 8px;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        .gauge-container { margin: 12px 0; }
        .gauge-label { margin-bottom: 5px; font-size: 0.9em; }
        .gauge {
            width: 100%;
            height: 24px;
            background: #0a0a0a;
            border: 1px solid #00ff00;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        .gauge-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff0000 0%, #ffff00 50%, #00ff00 100%);
            transition: width 0.5s ease;
            border-radius: 12px 0 0 12px;
        }
        .gauge-value {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            font-size: 0.85em;
            text-shadow: 0 0 3px #000;
        }
        .narrative-text {
            white-space: pre-wrap;
            font-size: 0.85em;
            line-height: 1.5;
            background: #0f0f0f;
            padding: 12px;
            border-radius: 4px;
            max-height: 350px;
            overflow-y: auto;
            border: 1px solid #333;
        }
        .phase-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9em;
        }
        .phase-embryo { background: #666; color: #fff; }
        .phase-infant { background: #ff6b6b; color: #000; }
        .phase-adolescent { background: #ffd93d; color: #000; }
        .phase-adult { background: #6bcf7f; color: #000; }
        .phase-sage { background: #4d96ff; color: #fff; }
        .phase-crisis { background: #ff0000; color: #fff; animation: blink 1s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.5; } }
        .status { margin: 10px 0; padding: 8px; border-radius: 4px; }
        .status-connected { background: rgba(0, 255, 0, 0.2); border: 1px solid #00ff00; }
        .status-disconnected { background: rgba(255, 0, 0, 0.2); border: 1px solid #ff0000; }
        .timeline { margin-top: 10px; }
        .timeline-item {
            padding: 8px;
            background: #0f0f0f;
            border-left: 3px solid #00ff00;
            margin-bottom: 8px;
            border-radius: 0 4px 4px 0;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ARA NARRATIVE DASHBOARD</h1>

        <div id="connection-status" class="status status-disconnected">
            Connecting to narrative stream...
        </div>

        <div class="grid">
            <!-- Operator View -->
            <div class="panel">
                <h2>OPERATOR COCKPIT</h2>
                <div>
                    <strong>Phase:</strong>
                    <span id="current-phase" class="phase-badge phase-embryo">EMBRYO</span>
                </div>

                <div class="gauge-container">
                    <div class="gauge-label">Overall Efficiency</div>
                    <div class="gauge">
                        <div class="gauge-fill" id="gauge-overall" style="width: 0%"></div>
                        <div class="gauge-value" id="value-overall">0%</div>
                    </div>
                </div>

                <div class="gauge-container">
                    <div class="gauge-label">Throughput</div>
                    <div class="gauge">
                        <div class="gauge-fill" id="gauge-throughput" style="width: 0%"></div>
                        <div class="gauge-value" id="value-throughput">0%</div>
                    </div>
                </div>

                <div class="gauge-container">
                    <div class="gauge-label">Cognitive</div>
                    <div class="gauge">
                        <div class="gauge-fill" id="gauge-cognitive" style="width: 0%"></div>
                        <div class="gauge-value" id="value-cognitive">0%</div>
                    </div>
                </div>

                <div id="operator-narrative" class="narrative-text">
                    Waiting for data...
                </div>
            </div>

            <!-- Mythic View -->
            <div class="panel">
                <h2>MYTHIC CHRONICLE</h2>
                <div id="mythic-narrative" class="narrative-text">
                    Waiting for data...
                </div>
            </div>

            <!-- Public View -->
            <div class="panel">
                <h2>PUBLIC VOICE</h2>
                <div id="public-narrative" class="narrative-text">
                    Waiting for data...
                </div>
            </div>

            <!-- Timeline -->
            <div class="panel">
                <h2>LIFECYCLE TIMELINE</h2>
                <div id="phase-timeline" class="timeline">
                    <div class="timeline-item">Waiting for phase transitions...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 10;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/api/narrative/ws`);

            ws.onopen = function() {
                document.getElementById('connection-status').className = 'status status-connected';
                document.getElementById('connection-status').textContent = 'Connected to narrative stream';
                reconnectAttempts = 0;
            };

            ws.onclose = function() {
                document.getElementById('connection-status').className = 'status status-disconnected';
                document.getElementById('connection-status').textContent = 'Disconnected. Reconnecting...';

                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connect, 2000 * reconnectAttempts);
                }
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };

            ws.onmessage = function(event) {
                if (event.data === 'ping') {
                    ws.send('pong');
                    return;
                }

                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'initial_state' || msg.type === 'current_state' || msg.type === 'narrative_update') {
                        updateDashboard(msg.data);
                    }
                } catch (e) {
                    console.error('Failed to parse message:', e);
                }
            };
        }

        function updateDashboard(reports) {
            if (!reports || !reports.operator) return;

            // Update phase
            const phase = reports.operator.phase || 'unknown';
            const phaseBadge = document.getElementById('current-phase');
            phaseBadge.textContent = phase.toUpperCase();
            phaseBadge.className = `phase-badge phase-${phase}`;

            // Update gauges
            const eff = reports.operator.efficiency || {};
            updateGauge('overall', eff.overall_efficiency || 0);
            updateGauge('throughput', eff.throughput_pct || 0);
            updateGauge('cognitive', eff.cognitive_pct || 0);

            // Update narratives
            if (reports.operator.narrative) {
                document.getElementById('operator-narrative').textContent = reports.operator.narrative;
            }
            if (reports.mythic && reports.mythic.narrative) {
                document.getElementById('mythic-narrative').textContent = reports.mythic.narrative;
            }
            if (reports.public && reports.public.narrative) {
                document.getElementById('public-narrative').textContent = reports.public.narrative;
            }
        }

        function updateGauge(id, value) {
            const fill = document.getElementById(`gauge-${id}`);
            const valueEl = document.getElementById(`value-${id}`);
            if (fill && valueEl) {
                fill.style.width = `${Math.min(100, value)}%`;
                valueEl.textContent = `${value.toFixed(1)}%`;
            }
        }

        // Initial connection
        connect();

        // Fetch initial state via REST
        fetch('/api/narrative/current')
            .then(response => response.json())
            .then(data => updateDashboard(data))
            .catch(err => console.error('Failed to fetch initial state:', err));

        // Periodic phase history update
        setInterval(function() {
            fetch('/api/narrative/transitions')
                .then(response => response.json())
                .then(data => {
                    const timeline = document.getElementById('phase-timeline');
                    if (data.transitions && data.transitions.length > 0) {
                        timeline.innerHTML = data.transitions.map(t => `
                            <div class="timeline-item">
                                <strong>${new Date(t.timestamp * 1000).toLocaleString()}</strong>:
                                ${t.from} &rarr;
                                <span class="phase-badge phase-${t.to}">${t.to.toUpperCase()}</span>
                            </div>
                        `).join('');
                    }
                })
                .catch(err => console.error('Failed to fetch transitions:', err));
        }, 10000);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
