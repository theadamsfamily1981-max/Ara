"""
Ara Core Server - Brain Process

FastAPI server that exposes Ara's cognitive capabilities via HTTP.
This is Process 1 in the 2-process architecture:
- Process 1 (this): Brain + emotions + policies
- Process 2: Avatar (ears, mouth, face)

Usage:
    python -m ara.server.core_server
    # Ara brain now at http://127.0.0.1:8008/chat

The avatar process calls POST /chat with user utterances and receives:
- reply_text: What Ara says
- pad: Emotional state (valence, arousal, dominance)
- clv: Cognitive load (instability, resource, structural)
- kitten: SNN status if available
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ara.service.core import AraService, HardwareMode, AraState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ara.server")

# ============================================================
# Request/Response Models
# ============================================================

class ChatRequest(BaseModel):
    """Request to chat with Ara."""
    session_id: str = "default"
    user_utterance: str
    context: Dict[str, Any] = {}


class PADState(BaseModel):
    """Pleasure-Arousal-Dominance emotional state."""
    valence: float
    arousal: float
    dominance: float


class CLVState(BaseModel):
    """Cognitive Load Vector."""
    instability: float
    resource: float
    structural: float
    risk_level: str


class KittenStatus(BaseModel):
    """Forest Kitten 33 SNN status."""
    mode: str
    total_neurons: int
    total_steps: int
    spike_rate: float
    latency_ms: float
    hardware_present: bool


class ChatResponse(BaseModel):
    """Response from Ara."""
    reply_text: str
    pad: PADState
    clv: CLVState
    kitten: Optional[KittenStatus] = None
    meta: Dict[str, Any] = {}


class StatusResponse(BaseModel):
    """Ara system status."""
    state: str
    mode: str
    autonomy_stage: str
    llm_connected: bool
    kitten_available: bool
    total_interactions: int


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Ara Core Server",
    description="Cognitive AI brain server with PAD emotions and CLV tracking",
    version="0.1.0"
)

# Allow CORS for local avatar process
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Ara instance
ara: Optional[AraService] = None


@app.on_event("startup")
async def startup():
    """Initialize Ara on server startup."""
    global ara
    logger.info("Starting Ara Core Server...")

    # Determine hardware mode from environment or default
    import os
    mode_str = os.environ.get("ARA_MODE", "MODE_B")
    mode = getattr(HardwareMode, mode_str, HardwareMode.MODE_B)

    try:
        ara = AraService(mode=mode)
        logger.info(f"Ara initialized in {mode.value} mode")

        if ara._kitten_available:
            kitten_mode = "HARDWARE" if ara.kitten.is_hardware else "EMULATED"
            logger.info(f"Forest Kitten 33: {kitten_mode}")
    except Exception as e:
        logger.error(f"Failed to initialize Ara: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on server shutdown."""
    global ara
    if ara:
        ara.save_state()
        logger.info("Ara state saved")


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "ara-core"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get Ara's current status."""
    if not ara:
        raise HTTPException(status_code=503, detail="Ara not initialized")

    return StatusResponse(
        state=ara._state.value,
        mode=ara._mode.value,
        autonomy_stage=ara._autonomy_stage.value,
        llm_connected=ara._llm_connected,
        kitten_available=ara._kitten_available,
        total_interactions=ara._stats["total_interactions"]
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint - Avatar calls this.

    Send user utterance, get Ara's reply + emotional state.
    """
    if not ara:
        raise HTTPException(status_code=503, detail="Ara not initialized")

    try:
        # Process through Ara's cognitive stack
        response = ara.process(req.user_utterance)

        # Extract emotional state
        pad = PADState(
            valence=response.emotional_surface.valence,
            arousal=response.emotional_surface.arousal,
            dominance=response.emotional_surface.dominance
        )

        # Extract cognitive load
        clv = CLVState(
            instability=response.cognitive_load.instability,
            resource=response.cognitive_load.resource,
            structural=response.cognitive_load.structural,
            risk_level=_compute_risk_level(response.cognitive_load)
        )

        # Extract Kitten status if available
        kitten = None
        if ara._kitten_available and ara.kitten:
            status = ara.kitten.get_status()
            kitten = KittenStatus(
                mode="HARDWARE" if ara.kitten.is_hardware else "EMULATED",
                total_neurons=ara.kitten.config.total_neurons,
                total_steps=status.get("total_steps", 0),
                spike_rate=status.get("spike_rate", 0.0),
                latency_ms=status.get("avg_latency_ms", 0.0),
                hardware_present=ara.kitten.is_hardware
            )

        # Build meta info
        meta = {
            "focus_mode": response.focus_mode.value,
            "thought_type": response.thought_type.value,
            "processing_time_ms": response.processing_time_ms,
            "session_id": req.session_id,
        }

        return ChatResponse(
            reply_text=response.text,
            pad=pad,
            clv=clv,
            kitten=kitten,
            meta=meta
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kitten")
async def get_kitten():
    """Get Forest Kitten 33 status."""
    if not ara:
        raise HTTPException(status_code=503, detail="Ara not initialized")

    if not ara._kitten_available or not ara.kitten:
        return {"available": False, "message": "Kitten not available in this mode"}

    status = ara.kitten.get_status()
    return {
        "available": True,
        "mode": "HARDWARE" if ara.kitten.is_hardware else "EMULATED",
        "device": ara.kitten.device_path if ara.kitten.is_hardware else None,
        **status
    }


@app.get("/mood")
async def get_mood():
    """Get Ara's current emotional state."""
    if not ara:
        raise HTTPException(status_code=503, detail="Ara not initialized")

    surface = ara._emotional_surface
    mood_desc = ara._describe_mood()

    return {
        "valence": surface.valence,
        "arousal": surface.arousal,
        "dominance": surface.dominance,
        "description": mood_desc,
        "label": _mood_label(surface)
    }


@app.post("/reset")
async def reset():
    """Reset Ara's state."""
    if not ara:
        raise HTTPException(status_code=503, detail="Ara not initialized")

    ara.reset()
    return {"status": "reset", "message": "Ara state reset"}


# ============================================================
# Helpers
# ============================================================

def _compute_risk_level(clv) -> str:
    """Compute risk level from CLV."""
    combined = (clv.instability * 0.5 + clv.resource * 0.3 + clv.structural * 0.2)
    if combined < 0.3:
        return "LOW"
    elif combined < 0.6:
        return "MODERATE"
    elif combined < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


def _mood_label(surface) -> str:
    """Get mood label from emotional surface."""
    v, a, d = surface.valence, surface.arousal, surface.dominance

    if a < 0.3:
        if v > 0.3:
            return "calm"
        elif v < -0.3:
            return "melancholy"
        else:
            return "serene"
    elif a > 0.7:
        if v > 0.3:
            return "excited"
        elif v < -0.3:
            return "anxious"
        else:
            return "alert"
    else:
        if v > 0.3:
            return "content"
        elif v < -0.3:
            return "uneasy"
        else:
            return "neutral"


# ============================================================
# Main
# ============================================================

def main():
    """Run the Ara Core Server."""
    import argparse

    parser = argparse.ArgumentParser(description="Ara Core Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8008, help="Port to bind to")
    parser.add_argument("--mode", default="MODE_B", help="Hardware mode (MODE_A, MODE_B, MODE_C)")
    args = parser.parse_args()

    # Set mode in environment for startup
    import os
    os.environ["ARA_MODE"] = args.mode

    logger.info(f"Starting Ara Core Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
