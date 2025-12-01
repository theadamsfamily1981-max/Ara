"""
Ara Avatar System API Router

Endpoints for voice command processing, system state queries,
and bidirectional WebSocket communication with Ara.

Endpoints:
    POST /api/ara/command  - Process voice command
    GET  /api/ara/status   - Get current system state
    GET  /api/ara/history  - Get command history
    POST /api/ara/avatar   - Update avatar config
    POST /api/ara/mode     - Set workspace mode
    WS   /ws/ara           - Bidirectional event stream
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
import asyncio
import logging

from ..models.schemas import (
    AraCommand,
    AraCommandResponse,
    AraAvatarConfig,
    AraSystemState,
    AraEvent,
    AraStatusReport,
)
from ..services.ara_service import AraService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ara", tags=["Ara Avatar System"])

# Security
security = HTTPBearer(auto_error=False)
API_TOKEN = "tfan-secure-token-change-me"  # TODO: Use env var


def verify_token_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional token verification (some endpoints are public)."""
    if credentials and credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials


# Service instance
ara_service = AraService()

# WebSocket connection manager for Ara
class AraConnectionManager:
    """Manages WebSocket connections to Ara avatar system."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"[Ara] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"[Ara] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, event: dict):
        """Broadcast event to all connected Ara clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(event)
            except Exception as e:
                logger.error(f"[Ara] Error sending to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)

    async def send_to(self, websocket: WebSocket, event: dict):
        """Send event to specific client."""
        try:
            await websocket.send_json(event)
        except Exception as e:
            logger.error(f"[Ara] Error sending to client: {e}")


ara_manager = AraConnectionManager()


# ============================================================================
# REST Endpoints
# ============================================================================

@router.post("/command", response_model=AraCommandResponse)
async def process_voice_command(command: AraCommand):
    """
    Process a voice command from Ara.

    Ara sends natural language commands, this endpoint:
    1. Parses the command
    2. Maps to system action
    3. Returns response text for Ara to speak

    Example commands:
    - "show me GPU stats"
    - "switch to work mode"
    - "engage topology visualization"
    - "start training"
    """
    result = ara_service.process_command(command.command)

    # Broadcast command event to WebSocket clients
    await ara_manager.broadcast({
        'type': 'command_processed',
        'data': {
            'command': command.command,
            'action': result['action'],
            'success': result['success']
        }
    })

    return AraCommandResponse(**result)


@router.get("/status", response_model=AraSystemState)
async def get_system_state():
    """
    Get current system state for Ara.

    Returns all relevant state information that Ara needs to:
    - Know what's currently happening
    - Adjust her personality
    - Make informed responses
    """
    state = ara_service.get_system_state()
    personality = ara_service.get_personality_config()

    return AraSystemState(
        workspace_mode=state.get('workspace_mode', 'work'),
        current_view=state.get('current_view', 'dashboard'),
        training_active=state.get('training_active', False),
        topology_visible=state.get('topology_visible', False),
        fullscreen=state.get('fullscreen', False),
        cockpit_active=state.get('cockpit_active', False),
        mode=state.get('mode', 'work'),
        avatar=AraAvatarConfig(**ara_service.avatar_config),
        personality=personality
    )


@router.get("/status/report", response_model=AraStatusReport)
async def get_status_report():
    """
    Get a spoken status report for Ara.

    Generates natural language text appropriate for Ara's current
    personality mode (professional for work, casual for relax).
    """
    # Get metrics for report
    from ..services.metrics_service import MetricsService
    metrics_service = MetricsService()
    metrics = metrics_service.get_current_metrics()

    report_text = ara_service.generate_status_report(metrics.dict())
    state = ara_service.get_system_state()
    personality = ara_service.get_personality_config()

    return AraStatusReport(
        report_text=report_text,
        mode=ara_service.mode.value,
        system_state=AraSystemState(
            workspace_mode=state.get('workspace_mode', 'work'),
            current_view=state.get('current_view', 'dashboard'),
            training_active=state.get('training_active', False),
            topology_visible=state.get('topology_visible', False),
            fullscreen=state.get('fullscreen', False),
            cockpit_active=state.get('cockpit_active', False),
            mode=state.get('mode', 'work'),
            avatar=AraAvatarConfig(**ara_service.avatar_config),
            personality=personality
        )
    )


@router.get("/history")
async def get_command_history(limit: int = 50):
    """Get recent command history."""
    return {
        'commands': ara_service.get_command_history(limit),
        'total': len(ara_service.command_history)
    }


@router.post("/avatar")
async def update_avatar_config(config: AraAvatarConfig):
    """
    Update Ara's avatar appearance configuration.

    Can be called from cockpit HUD or main dashboard.
    """
    ara_service.set_avatar_config(
        profile=config.profile,
        style=config.style,
        mood=config.mood
    )

    # Broadcast avatar update
    await ara_manager.broadcast({
        'type': 'avatar_updated',
        'data': config.dict()
    })

    return {'status': 'success', 'avatar': config.dict()}


@router.post("/mode/{mode}")
async def set_workspace_mode(mode: str):
    """
    Set workspace mode (work or relax).

    Changes Ara's personality and the system theme.
    """
    if mode not in ['work', 'relax']:
        raise HTTPException(status_code=400, detail="Mode must be 'work' or 'relax'")

    success = ara_service.set_workspace_mode(mode)

    if success:
        personality = ara_service.get_personality_config()

        # Broadcast mode change
        await ara_manager.broadcast({
            'type': 'mode_changed',
            'data': {
                'mode': mode,
                'personality': personality
            }
        })

        return {
            'status': 'success',
            'mode': mode,
            'personality': personality
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to set workspace mode")


@router.get("/personality")
async def get_personality_config():
    """Get Ara's current personality configuration."""
    return {
        'mode': ara_service.mode.value,
        'personality': ara_service.get_personality_config()
    }


@router.get("/commands")
async def get_available_commands():
    """Get list of available voice commands."""
    return {
        'commands': list(ara_service.command_mappings.keys()),
        'categories': {
            'navigation': [k for k in ara_service.command_mappings if 'show' in k],
            'topology': [k for k in ara_service.command_mappings if 'topology' in k],
            'control': [k for k in ara_service.command_mappings if any(x in k for x in ['start', 'stop', 'mode'])],
            'avatar': [k for k in ara_service.command_mappings if 'avatar' in k],
        }
    }


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@router.websocket("/ws")
async def ara_websocket(websocket: WebSocket):
    """
    Bidirectional WebSocket for Ara communication.

    Ara can:
    - Send commands and receive responses
    - Receive system events in real-time
    - Subscribe to specific event types

    Message format (client → server):
    {
        "type": "command" | "subscribe" | "ping",
        "data": {...}
    }

    Message format (server → client):
    {
        "type": "response" | "event" | "pong",
        "data": {...}
    }
    """
    await ara_manager.connect(websocket)

    # Send initial state
    state = ara_service.get_system_state()
    await ara_manager.send_to(websocket, {
        'type': 'initial_state',
        'data': state
    })

    try:
        while True:
            # Receive message from Ara
            message = await websocket.receive_json()
            msg_type = message.get('type', '')
            data = message.get('data', {})

            if msg_type == 'command':
                # Process voice command
                command_text = data.get('command', '')
                result = ara_service.process_command(command_text)

                await ara_manager.send_to(websocket, {
                    'type': 'command_response',
                    'data': result
                })

                # Broadcast to other clients
                await ara_manager.broadcast({
                    'type': 'command_processed',
                    'data': {
                        'command': command_text,
                        'action': result['action'],
                        'success': result['success']
                    }
                })

            elif msg_type == 'state_update':
                # Ara updating system state
                for key, value in data.items():
                    ara_service.update_system_state(key, value)

                await ara_manager.broadcast({
                    'type': 'state_updated',
                    'data': data
                })

            elif msg_type == 'ping':
                await ara_manager.send_to(websocket, {
                    'type': 'pong',
                    'data': {}
                })

            elif msg_type == 'get_state':
                state = ara_service.get_system_state()
                await ara_manager.send_to(websocket, {
                    'type': 'state',
                    'data': state
                })

    except WebSocketDisconnect:
        ara_manager.disconnect(websocket)
        logger.info("[Ara] WebSocket disconnected")

    except Exception as e:
        logger.error(f"[Ara] WebSocket error: {e}")
        ara_manager.disconnect(websocket)


# Helper function to broadcast events from other parts of the app
async def broadcast_ara_event(event_type: str, data: dict):
    """Broadcast an event to all connected Ara clients."""
    await ara_manager.broadcast({
        'type': event_type,
        'data': data
    })
