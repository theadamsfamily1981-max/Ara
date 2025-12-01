"""
Pydantic models for T-FAN API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class MetricsResponse(BaseModel):
    """Current training metrics."""
    training_active: bool = False
    step: int = 0
    accuracy: float = 0.0
    latency_ms: float = 0.0
    hypervolume: float = 0.0
    epr_cv: float = 0.0
    topo_gap: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ParetoWeights(BaseModel):
    """Pareto optimization decision weights."""
    neg_accuracy: float = Field(10.0, description="Weight for accuracy (higher = maximize)")
    latency: float = Field(1.0, description="Weight for latency (higher = minimize)")
    epr_cv: float = Field(2.0, description="Weight for EPR CV (higher = minimize)")
    topo_gap: float = Field(1.0, description="Weight for topology gap (higher = minimize)")
    energy: float = Field(0.5, description="Weight for energy (higher = minimize)")


class TrainingRequest(BaseModel):
    """Training start request."""
    config_path: str = Field("configs/auto/best.yaml", description="Path to training config")
    max_steps: Optional[int] = Field(None, description="Max training steps")
    logdir: str = Field("runs/api_training", description="Log directory")


class TrainingStatus(BaseModel):
    """Training status response."""
    active: bool
    step: int = 0
    config: Optional[str] = None
    started_at: Optional[str] = None


class ConfigResponse(BaseModel):
    """Configuration response."""
    path: str
    name: str
    config: Dict


class ParetoConfig(BaseModel):
    """Pareto configuration point."""
    n_heads: int
    d_model: int
    n_layers: int
    keep_ratio: float
    alpha: float
    lr: float
    objectives: List[float]


class ParetoFrontResponse(BaseModel):
    """Pareto front response."""
    n_pareto_points: int
    hypervolume: float
    configurations: List[ParetoConfig]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Ara Avatar System Schemas
# ============================================================================

class AraCommand(BaseModel):
    """Voice command from Ara."""
    command: str = Field(..., description="Natural language voice command")
    context: Optional[Dict] = Field(None, description="Additional context (e.g., current view)")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat())


class AraCommandResponse(BaseModel):
    """Response to Ara command."""
    success: bool
    action: str
    params: Dict = {}
    response: str = Field(..., description="Text for Ara to speak")
    original_command: str


class AraAvatarConfig(BaseModel):
    """Avatar appearance configuration."""
    profile: str = Field("Default", description="Avatar profile name")
    style: str = Field("Realistic", description="Visual style")
    mood: str = Field("Neutral", description="Current mood/expression")


class AraSystemState(BaseModel):
    """Current system state for Ara."""
    workspace_mode: str = "work"
    current_view: str = "dashboard"
    training_active: bool = False
    topology_visible: bool = False
    fullscreen: bool = False
    cockpit_active: bool = False
    mode: str = "work"
    avatar: AraAvatarConfig
    personality: Dict = {}
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AraEvent(BaseModel):
    """Event pushed to Ara via WebSocket."""
    type: str = Field(..., description="Event type (e.g., 'training_started', 'metrics_update')")
    data: Dict = {}
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AraStatusReport(BaseModel):
    """Status report for Ara to speak."""
    report_text: str = Field(..., description="Natural language status report")
    mode: str = Field(..., description="Personality mode used")
    system_state: AraSystemState
