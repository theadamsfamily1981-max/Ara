"""
T-FAN REST API

FastAPI server for T-FAN with live weight tuning, training control,
and real-time metrics streaming.

Features:
- Live Pareto weight tuning via REST
- Training start/stop control
- Real-time metrics via WebSocket
- Config management
- Secure token authentication

Usage:
    uvicorn api.main:app --reload
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from pathlib import Path
import json
import asyncio
from typing import Dict, List, Optional
import logging

from .models.schemas import (
    MetricsResponse,
    ParetoWeights,
    TrainingRequest,
    ConfigResponse,
    ParetoFrontResponse,
)
from .services.metrics_service import MetricsService
from .services.training_service import TrainingService
from .services.pareto_service import ParetoService
from .routers.ara_router import router as ara_router
from .routers.narrative_router import router as narrative_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
API_TOKEN = "tfan-secure-token-change-me"  # TODO: Use env var


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials


# Services
metrics_service = MetricsService()
training_service = TrainingService()
pareto_service = ParetoService()


# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 T-FAN API starting up...")
    asyncio.create_task(broadcast_metrics())
    yield
    # Shutdown
    logger.info("🛑 T-FAN API shutting down...")


# Create app
app = FastAPI(
    title="T-FAN API",
    description="REST API for T-FAN neural network training and optimization",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# Include routers
app.include_router(ara_router)
app.include_router(narrative_router)


# Background task for broadcasting metrics
async def broadcast_metrics():
    """Periodically broadcast metrics to all WebSocket clients."""
    while True:
        try:
            metrics = metrics_service.get_current_metrics()
            await manager.broadcast({
                "type": "metrics",
                "data": metrics.dict()
            })
        except Exception as e:
            logger.error(f"Error broadcasting metrics: {e}")

        await asyncio.sleep(2)  # Update every 2 seconds


# ============================================================================
# Web Dashboard
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request):
    """Serve main web dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ============================================================================
# Metrics Endpoints
# ============================================================================

@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get current training metrics."""
    return metrics_service.get_current_metrics()


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================================
# Pareto Optimization Endpoints
# ============================================================================

@app.get("/api/pareto/weights")
async def get_pareto_weights():
    """Get current Pareto optimization weights."""
    return pareto_service.get_weights()


@app.post("/api/pareto/weights")
async def update_pareto_weights(
    weights: ParetoWeights,
    token: str = Depends(verify_token)
):
    """
    Update Pareto optimization decision weights.

    This allows live tuning of the multi-objective optimization
    to prioritize different objectives (accuracy, latency, etc.).
    """
    try:
        pareto_service.update_weights(weights)

        # Broadcast update to WebSocket clients
        await manager.broadcast({
            "type": "weights_updated",
            "data": weights.dict()
        })

        return {"status": "success", "weights": weights.dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/pareto/front", response_model=ParetoFrontResponse)
async def get_pareto_front():
    """Get current Pareto front."""
    return pareto_service.get_front()


@app.post("/api/pareto/run")
async def run_pareto_optimization(
    n_iterations: int = 100,
    n_initial: int = 10,
    token: str = Depends(verify_token)
):
    """Run Pareto optimization in background."""
    try:
        task_id = pareto_service.run_optimization_async(n_iterations, n_initial)
        return {"status": "started", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pareto/status/{task_id}")
async def get_pareto_status(task_id: str):
    """Get status of Pareto optimization task."""
    status = pareto_service.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


# ============================================================================
# Training Control Endpoints
# ============================================================================

@app.post("/api/training/start")
async def start_training(
    request: TrainingRequest,
    token: str = Depends(verify_token)
):
    """Start training session."""
    try:
        result = training_service.start_training(request)

        await manager.broadcast({
            "type": "training_started",
            "data": {"config": request.config_path}
        })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/stop")
async def stop_training(token: str = Depends(verify_token)):
    """Stop current training session."""
    try:
        result = training_service.stop_training()

        await manager.broadcast({
            "type": "training_stopped"
        })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/status")
async def get_training_status():
    """Get training status."""
    return training_service.get_status()


@app.get("/api/training/logs")
async def get_training_logs(lines: int = 100):
    """Get recent training logs."""
    return training_service.get_logs(lines)


# ============================================================================
# Configuration Endpoints
# ============================================================================

@app.get("/api/configs")
async def list_configs():
    """List available configurations."""
    configs_dir = Path("configs")
    configs = []

    for config_file in configs_dir.rglob("*.yaml"):
        if config_file.is_file():
            configs.append({
                "path": str(config_file.relative_to(configs_dir.parent)),
                "name": config_file.stem
            })

    return {"configs": configs}


@app.get("/api/configs/{config_path:path}")
async def get_config(config_path: str):
    """Get configuration details."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise HTTPException(status_code=404, detail="Config not found")

    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        return {"path": config_path, "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/configs/select")
async def select_config(
    config_path: str,
    token: str = Depends(verify_token)
):
    """Select active configuration."""
    # Update configs/auto/best.yaml symlink or copy
    try:
        source = Path(config_path)
        target = Path("configs/auto/best.yaml")

        if not source.exists():
            raise HTTPException(status_code=404, detail="Config not found")

        target.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy(source, target)

        await manager.broadcast({
            "type": "config_selected",
            "data": {"config": config_path}
        })

        return {"status": "success", "config": config_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "training_active": training_service.is_training(),
        "api_version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
