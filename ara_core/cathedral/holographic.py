"""
Cathedral 4D Holographic Control Plane
======================================

Hypervector-of-hypervectors representation for spatiotemporal state.

Architecture:
    Event → HyperVector (16kD)
    Window → Bundle of event HVs
    History → HV-of-HVs (holographic)
    Projection → 3D latent for human viz

Control objectives:
    1. Maximize T_s (topology stability)
    2. Minimize cost during low-yield phases
    3. Maximize A_g (antifragility gain)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import math


# =============================================================================
# 4D EVENT SCHEMA
# =============================================================================

class ResourceType(str, Enum):
    """Resource types in the cathedral."""
    GPU = "gpu"
    CPU = "cpu"
    FPGA = "fpga"
    NETWORK = "network"
    MEMORY = "memory"
    POWER = "power"
    AGENT = "agent"


@dataclass
class CathedralEvent:
    """A single event in the cathedral - the atomic unit for embedding."""
    timestamp: datetime
    event_type: str  # "job_start", "job_end", "stress", "morph", etc.

    # Spatial coordinates (where in the cathedral graph)
    module_id: str        # Which module/agent
    resource_type: ResourceType
    layer: int           # Swarm layer (0-3)

    # Metrics at this event
    T_s: float           # Topology stability
    H_s: float           # Homeostasis
    sigma: float         # Current stress level
    cost: float          # Resource cost
    latency_ms: float

    # Context
    job_id: Optional[str] = None
    pattern_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """Convert to raw feature vector for embedding."""
        # Time features (cyclical encoding)
        hour = self.timestamp.hour
        day_of_week = self.timestamp.weekday()

        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin = math.sin(2 * math.pi * day_of_week / 7)
        dow_cos = math.cos(2 * math.pi * day_of_week / 7)

        # Resource type one-hot
        resource_onehot = [1.0 if r.value == self.resource_type.value else 0.0
                          for r in ResourceType]

        # Combine all features
        features = [
            # Time (4D cyclical)
            hour_sin, hour_cos, dow_sin, dow_cos,
            # Spatial
            float(self.layer) / 3.0,  # Normalized layer
            # Metrics
            self.T_s, self.H_s, self.sigma,
            math.log1p(self.cost),  # Log-scale cost
            math.log1p(self.latency_ms),  # Log-scale latency
        ] + resource_onehot

        return np.array(features, dtype=np.float32)


# =============================================================================
# HYPERVECTOR ENGINE
# =============================================================================

class HyperVectorEngine:
    """Engine for hyperdimensional computing operations."""

    def __init__(self, dim: int = 16384, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # Base vectors for binding
        self._base_cache: Dict[str, np.ndarray] = {}

    def random_hv(self) -> np.ndarray:
        """Generate a random bipolar hypervector."""
        return self.rng.choice([-1, 1], size=self.dim).astype(np.float32)

    def get_base(self, name: str) -> np.ndarray:
        """Get or create a base vector for a concept."""
        if name not in self._base_cache:
            # Deterministic from name
            seed = hash(name) % (2**32)
            rng = np.random.default_rng(seed)
            self._base_cache[name] = rng.choice([-1, 1], size=self.dim).astype(np.float32)
        return self._base_cache[name]

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two hypervectors (element-wise XOR for bipolar = multiply)."""
        return a * b

    def bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple hypervectors (majority vote)."""
        if len(hvs) == 0:
            return self.random_hv()
        if len(hvs) == 1:
            return hvs[0].copy()

        summed = np.sum(hvs, axis=0)
        # Bipolar threshold
        result = np.sign(summed)
        # Handle ties randomly
        ties = (result == 0)
        result[ties] = self.rng.choice([-1, 1], size=np.sum(ties))
        return result.astype(np.float32)

    def permute(self, hv: np.ndarray, shifts: int = 1) -> np.ndarray:
        """Permute hypervector (circular shift for sequence encoding)."""
        return np.roll(hv, shifts)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between hypervectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def encode_scalar(self, value: float, min_val: float, max_val: float,
                      levels: int = 100) -> np.ndarray:
        """Encode a scalar value as a hypervector using level encoding."""
        # Normalize to [0, 1]
        norm = (value - min_val) / (max_val - min_val + 1e-10)
        norm = max(0.0, min(1.0, norm))

        # Get level index
        level_idx = int(norm * (levels - 1))

        # Generate level HV deterministically
        base = self.get_base(f"level_{level_idx}")
        return base

    def encode_event(self, event: CathedralEvent) -> np.ndarray:
        """Encode a cathedral event as a hypervector."""
        features = event.to_feature_vector()

        # Encode each feature and bind with position
        hvs = []
        for i, val in enumerate(features):
            # Get base for this feature position
            pos_hv = self.get_base(f"feature_pos_{i}")
            # Encode value
            val_hv = self.encode_scalar(val, -1.0, 1.0)
            # Bind position with value
            hvs.append(self.bind(pos_hv, val_hv))

        # Bundle all bound pairs
        return self.bundle(hvs)


# =============================================================================
# 4D HOLOGRAPHIC STATE
# =============================================================================

@dataclass
class TimeWindow:
    """A time window containing multiple events."""
    start: datetime
    end: datetime
    events: List[CathedralEvent] = field(default_factory=list)
    hv: Optional[np.ndarray] = None  # Bundled hypervector

    # Aggregated metrics
    avg_T_s: float = 0.0
    avg_H_s: float = 0.0
    total_cost: float = 0.0
    event_count: int = 0


class HolographicState:
    """4D holographic representation of cathedral state over time."""

    def __init__(self, engine: HyperVectorEngine, window_minutes: int = 15):
        self.engine = engine
        self.window_minutes = window_minutes
        self.windows: List[TimeWindow] = []
        self.history_hv: Optional[np.ndarray] = None  # HV-of-HVs

    def add_event(self, event: CathedralEvent):
        """Add an event to the current window."""
        # Find or create window
        window = self._get_or_create_window(event.timestamp)
        window.events.append(event)
        window.event_count += 1

        # Update aggregates
        window.avg_T_s = (window.avg_T_s * (window.event_count - 1) + event.T_s) / window.event_count
        window.avg_H_s = (window.avg_H_s * (window.event_count - 1) + event.H_s) / window.event_count
        window.total_cost += event.cost

    def _get_or_create_window(self, ts: datetime) -> TimeWindow:
        """Get or create a time window for a timestamp."""
        # Quantize to window boundary
        window_start = ts.replace(
            minute=(ts.minute // self.window_minutes) * self.window_minutes,
            second=0, microsecond=0
        )

        # Look for existing window
        for w in self.windows:
            if w.start == window_start:
                return w

        # Create new window
        from datetime import timedelta
        window = TimeWindow(
            start=window_start,
            end=window_start + timedelta(minutes=self.window_minutes)
        )
        self.windows.append(window)
        return window

    def compute_window_hvs(self):
        """Compute hypervector for each window."""
        for window in self.windows:
            if window.events:
                # Encode each event
                event_hvs = [self.engine.encode_event(e) for e in window.events]

                # Sequence encoding: permute by position in window
                sequenced_hvs = []
                for i, hv in enumerate(event_hvs):
                    sequenced_hvs.append(self.engine.permute(hv, i))

                # Bundle into window HV
                window.hv = self.engine.bundle(sequenced_hvs)

    def compute_history_hv(self) -> np.ndarray:
        """Compute hypervector-of-hypervectors for full history."""
        # Ensure windows have HVs
        self.compute_window_hvs()

        # Get windows with HVs
        window_hvs = [w.hv for w in self.windows if w.hv is not None]

        if not window_hvs:
            self.history_hv = self.engine.random_hv()
            return self.history_hv

        # Sequence encode windows by time order
        self.windows.sort(key=lambda w: w.start)
        sequenced = []
        for i, w in enumerate(self.windows):
            if w.hv is not None:
                sequenced.append(self.engine.permute(w.hv, i))

        # Bundle into history HV
        self.history_hv = self.engine.bundle(sequenced)
        return self.history_hv

    def similarity_to(self, other: 'HolographicState') -> float:
        """Compare two holographic states."""
        if self.history_hv is None:
            self.compute_history_hv()
        if other.history_hv is None:
            other.compute_history_hv()

        return self.engine.similarity(self.history_hv, other.history_hv)


# =============================================================================
# QUANTUM-STYLE CONTROL PLANE
# =============================================================================

@dataclass
class ControlField:
    """A control field (small nudge) applied to the system."""
    name: str
    target_modules: List[str]  # Which modules to affect
    sigma_delta: float = 0.0   # Change to stress level
    routing_bias: float = 0.0  # Bias toward certain agents
    power_mode: str = "normal"  # "low", "normal", "high"


@dataclass
class ControlObjective:
    """Objective for the control plane to optimize."""
    name: str
    weight: float
    target: str  # "maximize" or "minimize"
    metric: str  # "T_s", "cost", "A_g", "latency"

    def evaluate(self, state: HolographicState) -> float:
        """Evaluate this objective on a state."""
        if not state.windows:
            return 0.0

        # Get latest window
        latest = max(state.windows, key=lambda w: w.start)

        if self.metric == "T_s":
            value = latest.avg_T_s
        elif self.metric == "H_s":
            value = latest.avg_H_s
        elif self.metric == "cost":
            value = latest.total_cost
        elif self.metric == "A_g":
            # Antifragility gain = T_s improvement under stress
            # Simplified: T_s * (1 + variance in T_s)
            t_s_values = [w.avg_T_s for w in state.windows if w.event_count > 0]
            if len(t_s_values) > 1:
                variance = np.var(t_s_values)
                value = latest.avg_T_s * (1 + variance)
            else:
                value = latest.avg_T_s
        else:
            value = 0.0

        # Apply direction
        if self.target == "minimize":
            value = -value

        return value * self.weight


class QuantumControlPlane:
    """Quantum-style control plane for cathedral optimization."""

    def __init__(self, engine: HyperVectorEngine):
        self.engine = engine
        self.objectives: List[ControlObjective] = []
        self.active_fields: List[ControlField] = []

        # Default objectives
        self._init_default_objectives()

    def _init_default_objectives(self):
        """Initialize default control objectives."""
        self.objectives = [
            # Primary: topology stability
            ControlObjective("topology", weight=1.0, target="maximize", metric="T_s"),
            # Secondary: cost efficiency
            ControlObjective("efficiency", weight=0.5, target="minimize", metric="cost"),
            # Tertiary: antifragility
            ControlObjective("antifragility", weight=0.3, target="maximize", metric="A_g"),
        ]

    def evaluate_state(self, state: HolographicState) -> Dict[str, float]:
        """Evaluate all objectives on a state."""
        results = {}
        total = 0.0

        for obj in self.objectives:
            score = obj.evaluate(state)
            results[obj.name] = score
            total += score

        results["total"] = total
        return results

    def propose_field(self, state: HolographicState) -> ControlField:
        """Propose a control field to improve the state."""
        # Get current metrics
        if not state.windows:
            return ControlField(name="default", target_modules=["*"])

        latest = max(state.windows, key=lambda w: w.start)

        # Decision logic based on current state
        if latest.avg_T_s < 0.95:
            # Topology unstable - reduce stress
            return ControlField(
                name="stabilize",
                target_modules=["*"],
                sigma_delta=-0.02,
                power_mode="normal",
            )
        elif latest.total_cost > 1000:
            # High cost - enter power-save
            return ControlField(
                name="conserve",
                target_modules=["*"],
                sigma_delta=0.0,
                power_mode="low",
            )
        else:
            # Stable and efficient - try to gain antifragility
            return ControlField(
                name="strengthen",
                target_modules=["*"],
                sigma_delta=0.01,  # Small stress increase
                power_mode="normal",
            )

    def apply_field(self, field: ControlField):
        """Apply a control field."""
        self.active_fields.append(field)
        # In real impl, this would modify scheduler/stress params


# =============================================================================
# 3D PROJECTION FOR VISUALIZATION
# =============================================================================

class HologramProjector:
    """Project holographic state to 3D for human visualization."""

    def __init__(self, engine: HyperVectorEngine):
        self.engine = engine
        # Projection matrix (random projection to 3D)
        self.projection = np.random.randn(3, engine.dim).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)

    def project_hv(self, hv: np.ndarray) -> Tuple[float, float, float]:
        """Project a hypervector to 3D coordinates."""
        coords = self.projection @ hv
        return tuple(coords)

    def project_state(self, state: HolographicState) -> List[Dict[str, Any]]:
        """Project a holographic state to 3D point cloud."""
        points = []

        for window in state.windows:
            if window.hv is not None:
                x, y, z = self.project_hv(window.hv)
                points.append({
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "T_s": window.avg_T_s,
                    "H_s": window.avg_H_s,
                    "cost": window.total_cost,
                    "time": window.start.isoformat(),
                    "events": window.event_count,
                })

        return points

    def render_ascii(self, state: HolographicState, width: int = 60, height: int = 20) -> str:
        """Render a simple ASCII projection."""
        points = self.project_state(state)
        if not points:
            return "No data"

        # Get bounds
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]

        if max(xs) == min(xs):
            xs = [0, 1]
        if max(ys) == min(ys):
            ys = [0, 1]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Plot points
        for p in points:
            px = int((p["x"] - x_min) / (x_max - x_min + 1e-10) * (width - 1))
            py = int((p["y"] - y_min) / (y_max - y_min + 1e-10) * (height - 1))
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))

            # Color by T_s
            if p["T_s"] > 0.99:
                char = '█'
            elif p["T_s"] > 0.95:
                char = '▓'
            elif p["T_s"] > 0.90:
                char = '▒'
            else:
                char = '░'

            grid[height - 1 - py][px] = char

        # Render
        lines = ['┌' + '─' * width + '┐']
        for row in grid:
            lines.append('│' + ''.join(row) + '│')
        lines.append('└' + '─' * width + '┘')
        lines.append(f"  T_s: █>0.99 ▓>0.95 ▒>0.90 ░<0.90  Points: {len(points)}")

        return '\n'.join(lines)


# =============================================================================
# CONVENIENCE / SINGLETON
# =============================================================================

_engine: Optional[HyperVectorEngine] = None
_control_plane: Optional[QuantumControlPlane] = None


def get_engine() -> HyperVectorEngine:
    global _engine
    if _engine is None:
        _engine = HyperVectorEngine()
    return _engine


def get_control_plane() -> QuantumControlPlane:
    global _control_plane
    if _control_plane is None:
        _control_plane = QuantumControlPlane(get_engine())
    return _control_plane


def create_holographic_state() -> HolographicState:
    return HolographicState(get_engine())
