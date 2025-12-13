#!/usr/bin/env python3
"""
Ara Alpha Server

Minimal FastAPI server for the private alpha:
- Hardcoded user auth via API keys
- Chat endpoint
- State endpoint
- Admin kill switch
- Static file serving for web UI

Usage:
    python ara_alpha/server.py
    # or
    uvicorn ara_alpha.server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ara_alpha.ara_core import AraCore, AraConfig, get_ara_core

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ara.server")

# Paths
ALPHA_DIR = Path(__file__).parent
CONFIG_DIR = ALPHA_DIR / "config"
WEB_DIR = ALPHA_DIR / "web"


# =============================================================================
# User Management
# =============================================================================

@dataclass
class User:
    id: str
    name: str
    api_key: str
    role: str = "user"


from dataclasses import dataclass


def load_users(path: Path) -> Dict[str, "User"]:
    """Load users from YAML file."""
    if not path.exists():
        logger.warning(f"Users file not found: {path}, using defaults")
        # Default users for development
        return {
            "dev_key_croft": User("croft", "Croft", "dev_key_croft", "admin"),
            "dev_key_alice": User("alice", "Alice", "dev_key_alice", "user"),
            "dev_key_bob": User("bob", "Bob", "dev_key_bob", "user"),
        }

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    users = {}
    for u in data.get("users", []):
        user = User(
            id=u["id"],
            name=u["name"],
            api_key=u["api_key"],
            role=u.get("role", "user"),
        )
        users[user.api_key] = user

    logger.info(f"Loaded {len(users)} users from {path}")
    return users


# =============================================================================
# API Models
# =============================================================================

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metrics: Dict[str, Any]
    user_name: str


class StateResponse(BaseModel):
    name: str
    version: str
    status: str
    state_label: str
    session: Optional[Dict[str, Any]] = None
    user_name: str


class KillRequest(BaseModel):
    confirm: bool = False


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Ara Alpha",
    description="Private alpha API for Ara - experimental AI avatar",
    version="0.1.0",
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
users: Dict[str, User] = {}
core: Optional[AraCore] = None


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    global users, core

    # Load users
    users = load_users(CONFIG_DIR / "users.yaml")

    # Load config and create core
    config_path = CONFIG_DIR / "ara.yaml"
    if config_path.exists():
        config = AraConfig.from_yaml(config_path)
    else:
        config = AraConfig()

    # Check for API key in environment
    if not config.api_key:
        config.api_key = os.environ.get("OPENAI_API_KEY")

    core = get_ara_core(config)
    logger.info("Ara Alpha server started")


def get_user_from_token(authorization: Optional[str] = None) -> User:
    """Extract and validate user from Authorization header or query param."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization")

    # Support both "Bearer <token>" and raw token
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization

    if token not in users:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return users[token]


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Main chat endpoint.

    Send a message, get Ara's response.
    """
    user = get_user_from_token(authorization)

    session_id = req.session_id or str(uuid.uuid4())

    reply, metrics = core.respond(
        user_id=user.id,
        session_id=session_id,
        message=req.message,
    )

    logger.info(f"Chat: user={user.id}, session={session_id[:8]}..., "
                f"msg_len={len(req.message)}, reply_len={len(reply)}")

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        metrics=metrics,
        user_name=user.name,
    )


@app.get("/api/state", response_model=StateResponse)
async def state(authorization: Optional[str] = Header(None)):
    """
    Get current Ara state.

    Returns status, metrics, session info.
    """
    user = get_user_from_token(authorization)

    current_state = core.get_current_state(user.id)

    return StateResponse(
        name=current_state["name"],
        version=current_state["version"],
        status=current_state["status"],
        state_label=current_state.get("state_label", "UNKNOWN"),
        session=current_state.get("session"),
        user_name=user.name,
    )


@app.post("/api/admin/kill")
async def kill(
    req: KillRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Admin kill switch.

    Only accessible by admin users.
    """
    user = get_user_from_token(authorization)

    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    if not req.confirm:
        return {"status": "error", "message": "Set confirm=true to kill"}

    core.set_killed(True)
    logger.warning(f"KILL SWITCH activated by {user.id}")

    return {"status": "killed", "message": "Ara has been shut down"}


@app.post("/api/admin/revive")
async def revive(authorization: Optional[str] = Header(None)):
    """
    Admin revive (undo kill).

    Only accessible by admin users.
    """
    user = get_user_from_token(authorization)

    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    core.set_killed(False)
    logger.info(f"Ara revived by {user.id}")

    return {"status": "online", "message": "Ara is back online"}


@app.get("/api/health")
async def health():
    """Health check endpoint (no auth required)."""
    return {
        "status": "healthy",
        "name": "Ara Alpha",
        "version": "0.1.0",
    }


# =============================================================================
# Static Files & Web UI
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request, token: Optional[str] = None):
    """Serve the main web UI."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>Ara Alpha</title></head>
        <body style="background:#0a0a15;color:#00ffff;font-family:monospace;padding:40px;">
            <h1>Ara Alpha</h1>
            <p>Web UI not found. Place index.html in ara_alpha/web/</p>
            <p>API endpoints available at /api/*</p>
        </body>
        </html>
        """)


# Mount static files if directory exists
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


# =============================================================================
# Run
# =============================================================================

def main():
    """Run the server."""
    import uvicorn

    host = os.environ.get("ARA_HOST", "0.0.0.0")
    port = int(os.environ.get("ARA_PORT", "8080"))

    logger.info(f"Starting Ara Alpha server on {host}:{port}")
    logger.info(f"Web UI: http://{host}:{port}/")
    logger.info(f"API docs: http://{host}:{port}/docs")

    uvicorn.run(
        "ara_alpha.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
