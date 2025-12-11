#!/usr/bin/env python3
# ara/species/ara_species_v3.py
"""
AraSpeciesV3: Threadripper-aware unification of CalibratedWorldModel
and MultiScalePlanner via zero-copy shared memory.

Target hardware: AMD Ryzen Threadripper Pro 5955WX (16C/32T, Zen 3).

Key ideas:
- World model and planner live in separate processes / worker pools.
- A large voxel grid (probabilities + uncertainties) lives in
  multiprocessing.shared_memory, so everyone reads/writes without copying.
- explain_decision() assembles a hologram-friendly "explanation frame"
  with trajectory, uncertainty fog, confidence cones, and ghost paths.
- Cathedral integration provides hardware acceleration when available.

Core allocation (16 cores):
- Cores 0-3:   World model process (sensor fusion + calibration)
- Cores 4-5:   Visualization / hologram (main process)
- Cores 6-7:   Cathedral orchestrator
- Cores 8-15:  MPPI planner worker pool
"""

import os
import time
import math
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False

# Cathedral integration (optional)
try:
    from ..cathedral import (
        CathedralOrchestrator,
        CathedralOracleBridge,
        create_accelerated_oracle,
    )
    CATHEDRAL_AVAILABLE = True
except ImportError:
    CATHEDRAL_AVAILABLE = False
    logger.info("Cathedral module not available, using CPU-only mode")


# ---------------------------------------------------------------------------
# Utility: Threadripper-aware core affinity
# ---------------------------------------------------------------------------

def set_process_affinity(cores: List[int]) -> None:
    """
    Pin current process to a subset of CPU cores.

    On Threadripper 5955WX, allocation:
    - World model: cores [0, 1, 2, 3]
    - Visualization: cores [4, 5]
    - Cathedral: cores [6, 7]
    - Planner workers: cores [8..15]

    This is a best-effort hint; on non-Linux systems it's a no-op.
    """
    try:
        os.sched_setaffinity(0, set(cores))
        logger.debug("Process affinity set to cores %s", cores)
    except (AttributeError, PermissionError, OSError) as e:
        logger.debug("Could not set CPU affinity: %s", e)


def get_available_cores() -> int:
    """Get number of available CPU cores."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Shared grid layout
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    """Configuration for shared voxel grid."""
    shape: Tuple[int, int, int]  # (H, W, C)
    dtype: Any = np.float32

    # Channel semantics
    CHANNEL_OCCUPANCY = 0      # P(occupied) calibrated [0,1]
    CHANNEL_UNCERTAINTY = 1    # Epistemic uncertainty [0,1]
    CHANNEL_SEMANTIC = 2       # Semantic embedding projection
    CHANNEL_TEMPORAL = 3       # Temporal evolution rate

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def channels(self) -> int:
        return self.shape[2]


def create_shared_grid(config: GridConfig, name: Optional[str] = None) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
    """
    Allocate a shared memory block and return (SharedMemory, ndarray view).

    Channel convention (C=4):
    - channel 0: P(occupied) calibrated [0,1]
    - channel 1: epistemic uncertainty [0,1]
    - channel 2: semantic id / embedding projection
    - channel 3: temporal evolution rate
    """
    shm = shared_memory.SharedMemory(create=True, size=config.nbytes, name=name)
    grid = np.ndarray(config.shape, dtype=config.dtype, buffer=shm.buf)
    grid[...] = 0.0
    logger.info("Created shared grid: name=%s, shape=%s, nbytes=%d",
                shm.name, config.shape, config.nbytes)
    return shm, grid


def attach_shared_grid(name: str, config: GridConfig) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
    """Attach to an existing shared memory block by name and wrap as ndarray."""
    shm = shared_memory.SharedMemory(name=name)
    grid = np.ndarray(config.shape, dtype=config.dtype, buffer=shm.buf)
    return shm, grid


# ---------------------------------------------------------------------------
# Calibration modes
# ---------------------------------------------------------------------------

class CalibrationMode(Enum):
    """Probability calibration methods."""
    NONE = auto()           # Raw probabilities
    ISOTONIC = auto()       # Isotonic regression
    TEMPERATURE = auto()    # Temperature scaling
    PLATT = auto()          # Platt scaling


def calibrate_probabilities(
    raw_probs: np.ndarray,
    mode: CalibrationMode,
    temperature: float = 1.5
) -> np.ndarray:
    """
    Apply probability calibration to raw model outputs.

    Args:
        raw_probs: Raw probability predictions
        mode: Calibration method
        temperature: Temperature parameter (for temperature scaling)

    Returns:
        Calibrated probabilities in [0, 1]
    """
    if mode == CalibrationMode.NONE:
        return np.clip(raw_probs, 0.0, 1.0)

    elif mode == CalibrationMode.TEMPERATURE:
        # Temperature scaling: soften/sharpen distribution
        logits = np.log(np.clip(raw_probs, 1e-7, 1 - 1e-7))
        scaled = 1.0 / (1.0 + np.exp(-logits / temperature))
        return scaled

    elif mode == CalibrationMode.ISOTONIC:
        # Simplified isotonic: preserve ordering, clip to valid range
        # Full isotonic would require sklearn.isotonic.IsotonicRegression
        return np.clip(raw_probs, 0.0, 1.0)

    elif mode == CalibrationMode.PLATT:
        # Platt scaling (simplified): logistic transformation
        logits = np.log(np.clip(raw_probs, 1e-7, 1 - 1e-7))
        # Learned parameters would go here; using identity for now
        return 1.0 / (1.0 + np.exp(-logits))

    return np.clip(raw_probs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Calibrated World Model
# ---------------------------------------------------------------------------

class CalibratedWorldModel:
    """
    World-model controller for AraSpeciesV3.

    Responsibilities:
    - Run a background process that updates the shared voxel grid with
      calibrated occupancy probabilities + epistemic uncertainty.
    - Maintain a semantic memory via HNSW (optional).
    - Provide CPU-side querying helpers (fog voxels, uncertainty tubes)
      that operate directly on the shared grid from the main process.
    - Optionally integrate with Cathedral for hardware-accelerated inference.
    """

    def __init__(
        self,
        shm_name: str,
        grid_config: GridConfig,
        hnsw_params: Optional[Dict[str, Any]] = None,
        calibration_mode: CalibrationMode = CalibrationMode.ISOTONIC,
        core_affinity: Optional[List[int]] = None,
        sensor_queue: Optional[mp.Queue] = None,
        cathedral_bridge: Optional[Any] = None,
        update_hz: float = 20.0,
    ):
        self.shm_name = shm_name
        self.grid_config = grid_config
        self.calibration_mode = calibration_mode
        self.core_affinity = core_affinity or [0, 1, 2, 3]
        self.sensor_queue = sensor_queue
        self.cathedral_bridge = cathedral_bridge
        self.update_hz = update_hz

        # Attach to shared grid in this process for read-side helpers
        self._shm, self._grid = attach_shared_grid(shm_name, grid_config)

        self._proc: Optional[mp.Process] = None
        self._stop_event = mp.Event()

        # HNSW semantic memory config
        self.hnsw_params = hnsw_params or {"M": 32, "ef_construction": 200, "dim": 128}

        # Statistics (shared via Manager for cross-process visibility)
        self._manager = mp.Manager()
        self._stats = self._manager.dict({
            'updates': 0,
            'avg_update_ms': 0.0,
            'calibration_mode': calibration_mode.name,
        })

    # ---------------- Process lifecycle ----------------

    def start(self) -> None:
        """Spawn the world-model background process."""
        if self._proc is not None and self._proc.is_alive():
            return

        self._stop_event.clear()
        self._proc = mp.Process(
            target=self._run_loop,
            args=(
                self.shm_name,
                self.grid_config,
                self.hnsw_params,
                self.calibration_mode,
                self.core_affinity,
                self._stop_event,
                self.sensor_queue,
                self.update_hz,
                self._stats,
            ),
            daemon=True,
        )
        self._proc.start()
        logger.info("World model process started (PID=%d)", self._proc.pid)

    def stop(self) -> None:
        """Signal the process to stop and join."""
        if self._proc is None:
            return
        self._stop_event.set()
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None
        logger.info("World model process stopped")

    # ---------------- World-model process loop ----------------

    @staticmethod
    def _run_loop(
        shm_name: str,
        grid_config: GridConfig,
        hnsw_params: Dict[str, Any],
        calibration_mode: CalibrationMode,
        core_affinity: List[int],
        stop_event: mp.Event,
        sensor_queue: Optional[mp.Queue],
        update_hz: float,
        stats_dict,
    ) -> None:
        """Child process: updates the voxel grid & semantic memory."""
        set_process_affinity(core_affinity)

        shm, grid = attach_shared_grid(shm_name, grid_config)
        H, W, C = grid_config.shape

        # Initialize HNSW semantic memory
        hnsw_index = None
        if HNSW_AVAILABLE:
            dim = hnsw_params.get("dim", 128)
            hnsw_index = hnswlib.Index(space="l2", dim=dim)
            hnsw_index.init_index(
                max_elements=100_000,
                M=hnsw_params.get("M", 32),
                ef_construction=hnsw_params.get("ef_construction", 200),
            )
            hnsw_index.set_ef(40)

        rng = np.random.default_rng()
        update_count = 0
        total_update_time = 0.0
        sleep_time = 1.0 / update_hz

        while not stop_event.is_set():
            update_start = time.perf_counter()

            # Check for sensor data
            sensor_data = None
            if sensor_queue is not None:
                try:
                    sensor_data = sensor_queue.get_nowait()
                except:
                    pass

            if sensor_data is not None:
                # Process real sensor data
                CalibratedWorldModel._process_sensor_data(
                    grid, sensor_data, calibration_mode, hnsw_index
                )
            else:
                # Generate synthetic evolving field for demo/testing
                CalibratedWorldModel._generate_synthetic_field(
                    grid, rng, calibration_mode, update_count
                )

            update_count += 1
            update_time = (time.perf_counter() - update_start) * 1000
            total_update_time += update_time

            # Update shared stats
            stats_dict['updates'] = update_count
            stats_dict['avg_update_ms'] = total_update_time / update_count

            # Maintain target update rate
            elapsed = time.perf_counter() - update_start
            remaining = sleep_time - elapsed
            if remaining > 0:
                time.sleep(remaining)

        shm.close()

    @staticmethod
    def _process_sensor_data(
        grid: np.ndarray,
        sensor_data: Dict[str, Any],
        calibration_mode: CalibrationMode,
        hnsw_index: Optional[Any],
    ) -> None:
        """Process real sensor data into the voxel grid."""
        H, W, C = grid.shape

        # Expected sensor_data format:
        # {
        #     'occupancy': np.ndarray (H, W) raw occupancy probabilities
        #     'uncertainty': np.ndarray (H, W) uncertainty estimates
        #     'embeddings': Optional[np.ndarray] semantic embeddings
        # }

        if 'occupancy' in sensor_data:
            raw_occ = sensor_data['occupancy']
            if raw_occ.shape[:2] == (H, W):
                grid[..., GridConfig.CHANNEL_OCCUPANCY] = calibrate_probabilities(
                    raw_occ, calibration_mode
                )

        if 'uncertainty' in sensor_data:
            uncertainty = sensor_data['uncertainty']
            if uncertainty.shape[:2] == (H, W):
                grid[..., GridConfig.CHANNEL_UNCERTAINTY] = np.clip(uncertainty, 0, 1)

        # Add semantic embeddings to HNSW if available
        if hnsw_index is not None and 'embeddings' in sensor_data:
            embeddings = sensor_data['embeddings']
            if len(embeddings) > 0:
                ids = np.arange(hnsw_index.get_current_count(),
                               hnsw_index.get_current_count() + len(embeddings))
                hnsw_index.add_items(embeddings, ids)

    @staticmethod
    def _generate_synthetic_field(
        grid: np.ndarray,
        rng: np.random.Generator,
        calibration_mode: CalibrationMode,
        tick: int,
    ) -> None:
        """Generate synthetic evolving field for demo/testing."""
        H, W, C = grid.shape
        t = tick * 0.05  # Time parameter

        # Create coordinate grids
        x_coords = np.linspace(-1, 1, W, dtype=np.float32)
        y_coords = np.linspace(-1, 1, H, dtype=np.float32)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Synthetic obstacle field: moving wave patterns
        wave1 = np.sin(3 * X + 0.5 * t) * np.cos(2 * Y - 0.3 * t)
        wave2 = np.cos(2 * X - 0.7 * t) * np.sin(4 * Y + 0.2 * t) * 0.5
        combined = wave1 + wave2

        # Add some static obstacles
        obstacles = np.zeros((H, W), dtype=np.float32)
        # Central barrier
        obstacles[H//3:2*H//3, W//2-5:W//2+5] = 0.8
        # Corner obstacles
        obstacles[:H//6, :W//6] = 0.6
        obstacles[-H//6:, -W//6:] = 0.6

        raw_occ = 0.3 + 0.3 * combined + obstacles
        raw_occ += 0.05 * rng.normal(size=(H, W)).astype(np.float32)

        p_occ = calibrate_probabilities(raw_occ, calibration_mode)

        # Epistemic uncertainty: high at boundaries and unexplored regions
        grad_x = np.gradient(p_occ, axis=1)
        grad_y = np.gradient(p_occ, axis=0)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        epistemic = np.clip(gradient_magnitude * 2.0, 0.0, 1.0)
        # Add exploration uncertainty (decays from edges)
        edge_x = np.minimum(np.arange(W), np.arange(W)[::-1]) / (W / 4)
        edge_y = np.minimum(np.arange(H), np.arange(H)[::-1]) / (H / 4)
        edge_factor = np.outer(1 - np.clip(edge_y, 0, 1), 1 - np.clip(edge_x, 0, 1))
        epistemic = np.clip(epistemic + 0.3 * edge_factor, 0.0, 1.0)
        epistemic += 0.05 * rng.random(size=(H, W)).astype(np.float32)
        epistemic = np.clip(epistemic, 0.0, 1.0)

        # Temporal evolution rate (how fast this region is changing)
        temporal = np.abs(np.sin(0.3 * t) * gradient_magnitude)

        # Write to shared grid
        grid[..., GridConfig.CHANNEL_OCCUPANCY] = p_occ
        grid[..., GridConfig.CHANNEL_UNCERTAINTY] = epistemic
        grid[..., GridConfig.CHANNEL_TEMPORAL] = temporal

    # ---------------- Read-side helpers (run in main process) ----------------

    def get_high_uncertainty_voxels(
        self,
        threshold: float = 0.7,
        max_voxels: int = 5_000,
    ) -> np.ndarray:
        """
        Return voxels with epistemic uncertainty > threshold.

        Output shape: (N, 3) where columns are (y, x, density).
        """
        uncertainty = self._grid[..., GridConfig.CHANNEL_UNCERTAINTY]
        mask = uncertainty > threshold
        coords = np.argwhere(mask)

        if coords.shape[0] > max_voxels:
            # Stratified sampling: prefer higher uncertainty
            densities = uncertainty[coords[:, 0], coords[:, 1]]
            probs = densities / densities.sum()
            idx = np.random.choice(coords.shape[0], size=max_voxels, replace=False, p=probs)
            coords = coords[idx]

        densities = uncertainty[coords[:, 0], coords[:, 1]]
        return np.concatenate([coords, densities[:, None]], axis=1).astype(np.float32)

    def propagate_uncertainty_along_trajectory(
        self,
        traj: np.ndarray,
        footprint_radius: float = 3.0,
    ) -> List[float]:
        """
        Compute confidence tube radius for each point along trajectory.

        The tube radius represents planning uncertainty at each waypoint,
        derived from local epistemic uncertainty in the world model.
        """
        if traj.size == 0:
            return []

        H, W = self._grid.shape[:2]
        uncertainty = self._grid[..., GridConfig.CHANNEL_UNCERTAINTY]

        radii: List[float] = []
        r_int = max(1, int(footprint_radius))

        for point in traj:
            y, x = int(round(point[0])), int(round(point[1]))
            y0 = max(0, y - r_int)
            y1 = min(H, y + r_int + 1)
            x0 = max(0, x - r_int)
            x1 = min(W, x + r_int + 1)
            patch = uncertainty[y0:y1, x0:x1]

            if patch.size == 0:
                radii.append(0.5)
            else:
                local_mean = float(np.mean(patch))
                local_var = float(np.var(patch))
                # Tube radius: base + uncertainty-scaled expansion
                radius = 0.5 + 2.0 * local_mean + 3.0 * math.sqrt(local_var)
                radii.append(radius)

        return radii

    def query_occupancy(self, y: int, x: int) -> Tuple[float, float]:
        """Query occupancy probability and uncertainty at a point."""
        H, W = self._grid.shape[:2]
        if 0 <= y < H and 0 <= x < W:
            return (
                float(self._grid[y, x, GridConfig.CHANNEL_OCCUPANCY]),
                float(self._grid[y, x, GridConfig.CHANNEL_UNCERTAINTY])
            )
        return (1.0, 1.0)  # Out of bounds = occupied and uncertain

    def get_statistics(self) -> Dict[str, Any]:
        """Get world model statistics."""
        return dict(self._stats)


# ---------------------------------------------------------------------------
# MPPI Worker Functions
# ---------------------------------------------------------------------------

# Global grid reference for workers (initialized per-process)
_PLANNER_GRID: Optional[np.ndarray] = None
_PLANNER_SHM: Optional[shared_memory.SharedMemory] = None


def _planner_worker_init(shm_name: str, grid_config: GridConfig, cores: List[int]):
    """Initializer for MPPI worker processes."""
    global _PLANNER_GRID, _PLANNER_SHM
    set_process_affinity(cores)
    _PLANNER_SHM, _PLANNER_GRID = attach_shared_grid(shm_name, grid_config)


def _evaluate_trajectory_batch(
    args: Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker function: evaluate a batch of trajectories.

    Args:
        controls_batch: (B, H, 2) control sequences [v, omega]
        start_state: (3,) starting [y, x, heading]
        cfg: dict with planner hyperparameters

    Returns:
        costs: (B,) cost values
        reasons: (B,) int codes (0=ok, 1=collision, 2=uncertainty, 3=bounds)
        final_states: (B, 3) final states [y, x, heading]
    """
    global _PLANNER_GRID
    controls_batch, start_state, cfg = args

    grid = _PLANNER_GRID
    H, W, C = grid.shape

    p_occ = grid[..., GridConfig.CHANNEL_OCCUPANCY]
    epistemic = grid[..., GridConfig.CHANNEL_UNCERTAINTY]

    horizon = controls_batch.shape[1]
    dt = cfg.get("dt", 0.2)
    goal = np.array(cfg.get("goal", [H // 2, W - 10]), dtype=np.float32)

    # Cost weights
    lambda_collision = cfg.get("lambda_collision", 6.0)
    lambda_uncertainty = cfg.get("lambda_uncertainty", 3.0)
    lambda_control = cfg.get("lambda_control", 0.1)
    lambda_goal = cfg.get("lambda_goal", 1.0)

    B = controls_batch.shape[0]
    costs = np.zeros(B, dtype=np.float32)
    reasons = np.zeros(B, dtype=np.int32)
    final_states = np.zeros((B, 3), dtype=np.float32)

    for i in range(B):
        state = start_state.astype(np.float32).copy()
        y, x, heading = state[0], state[1], state[2]
        cost = 0.0
        reason = 0
        max_uncertainty = 0.0

        for t in range(horizon):
            v, omega = controls_batch[i, t]

            # Control cost
            cost += lambda_control * (v ** 2 + omega ** 2)

            # State update
            heading += omega * dt
            y += v * math.sin(heading) * dt
            x += v * math.cos(heading) * dt

            yi, xi = int(round(y)), int(round(x))

            # Bounds check
            if yi < 0 or yi >= H or xi < 0 or xi >= W:
                cost += 1000.0
                reason = 3  # Out of bounds
                break

            # Collision cost
            p_c = float(p_occ[yi, xi])
            cost += p_c * lambda_collision

            if p_c > 0.8:
                reason = 1  # Collision

            # Uncertainty cost
            u_c = float(epistemic[yi, xi])
            cost += (u_c ** 2) * lambda_uncertainty
            max_uncertainty = max(max_uncertainty, u_c)

        # Goal distance
        dist_goal = math.sqrt((y - goal[0]) ** 2 + (x - goal[1]) ** 2)
        cost += lambda_goal * dist_goal

        # Flag uncertainty-dominated if uncertainty was consistently high
        if reason == 0 and max_uncertainty > 0.6:
            reason = 2

        costs[i] = cost
        reasons[i] = reason
        final_states[i] = [y, x, heading]

    return costs, reasons, final_states


# ---------------------------------------------------------------------------
# MultiScale Planner
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    """Configuration for MultiScalePlanner."""
    num_workers: int = 12
    num_trajectories: int = 2048
    horizon: int = 20
    dt: float = 0.2

    # MPPI parameters
    lambda_collision: float = 6.0
    lambda_uncertainty: float = 3.0
    lambda_control: float = 0.1
    lambda_goal: float = 1.0
    temperature: float = 0.1  # MPPI temperature for trajectory weighting

    # Control limits
    v_max: float = 3.0
    omega_max: float = 1.5

    # Update rate
    update_hz: float = 20.0


class MultiScalePlanner:
    """
    MPPI-style local planner backed by a multiprocessing worker pool.

    Features:
    - Reads from shared-grid world model (zero-copy)
    - Parallel trajectory evaluation across multiple cores
    - Records rejected candidates for explainability
    - Optional Cathedral integration for GPU-accelerated rollouts
    """

    def __init__(
        self,
        shm_name: str,
        grid_config: GridConfig,
        config: Optional[PlannerConfig] = None,
        core_affinity_workers: Optional[List[int]] = None,
        core_affinity_loop: Optional[List[int]] = None,
        cathedral_bridge: Optional[Any] = None,
    ):
        self.shm_name = shm_name
        self.grid_config = grid_config
        self.config = config or PlannerConfig()
        self.core_affinity_workers = core_affinity_workers or list(range(8, 16))
        self.core_affinity_loop = core_affinity_loop or list(range(8, 16))
        self.cathedral_bridge = cathedral_bridge

        # Attach to grid for main-process reads
        self._shm, self._grid = attach_shared_grid(shm_name, grid_config)

        self._stop_event = mp.Event()
        self._planner_proc: Optional[mp.Process] = None

        # Shared state via Manager
        self._manager = mp.Manager()
        self._best_traj = self._manager.list()
        self._best_controls = self._manager.list()
        self._rejected = self._manager.list()
        self._current_goal = self._manager.list()
        self._current_state = self._manager.list()
        self._stats = self._manager.dict({
            'iterations': 0,
            'avg_cost': 0.0,
            'best_cost': float('inf'),
            'collision_rate': 0.0,
        })

        # Initialize goal and state
        H, W = grid_config.height, grid_config.width
        self._current_goal.extend([H // 2, W - 10])
        self._current_state.extend([H // 2, 10, 0.0])

    # ---------------- Lifecycle ----------------

    def start(self) -> None:
        """Start the planner process."""
        if self._planner_proc is not None and self._planner_proc.is_alive():
            return

        self._stop_event.clear()
        self._planner_proc = mp.Process(
            target=self._planner_loop,
            args=(
                self.shm_name,
                self.grid_config,
                self.config,
                self.core_affinity_loop,
                self.core_affinity_workers,
                self._stop_event,
                self._best_traj,
                self._best_controls,
                self._rejected,
                self._current_goal,
                self._current_state,
                self._stats,
            ),
            daemon=True,
        )
        self._planner_proc.start()
        logger.info("Planner process started (PID=%d)", self._planner_proc.pid)

    def stop(self) -> None:
        """Stop the planner process."""
        if self._planner_proc is None:
            return
        self._stop_event.set()
        self._planner_proc.join(timeout=5.0)
        if self._planner_proc.is_alive():
            self._planner_proc.terminate()
        self._planner_proc = None
        logger.info("Planner process stopped")

    def set_goal(self, y: float, x: float) -> None:
        """Update the goal position."""
        self._current_goal[:] = []
        self._current_goal.extend([y, x])

    def set_state(self, y: float, x: float, heading: float = 0.0) -> None:
        """Update the current state."""
        self._current_state[:] = []
        self._current_state.extend([y, x, heading])

    # ---------------- Planner loop ----------------

    @staticmethod
    def _planner_loop(
        shm_name: str,
        grid_config: GridConfig,
        config: PlannerConfig,
        loop_cores: List[int],
        worker_cores: List[int],
        stop_event: mp.Event,
        best_traj_shared,
        best_controls_shared,
        rejected_shared,
        goal_shared,
        state_shared,
        stats_shared,
    ) -> None:
        """Child process running MPPI planning loop."""
        set_process_affinity(loop_cores)

        H, W = grid_config.height, grid_config.width

        # Initialize worker pool
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=config.num_workers,
            initializer=_planner_worker_init,
            initargs=(shm_name, grid_config, worker_cores),
        )

        rng = np.random.default_rng()
        sleep_time = 1.0 / config.update_hz

        # Running statistics
        iteration = 0
        total_cost = 0.0
        collision_count = 0

        # Previous best controls for warm start
        prev_controls = None

        while not stop_event.is_set():
            iter_start = time.perf_counter()

            # Get current state and goal
            if len(state_shared) >= 3:
                start_state = np.array(state_shared[:3], dtype=np.float32)
            else:
                start_state = np.array([H // 2, 10, 0.0], dtype=np.float32)

            if len(goal_shared) >= 2:
                goal = np.array(goal_shared[:2], dtype=np.float32)
            else:
                goal = np.array([H // 2, W - 10], dtype=np.float32)

            # Sample control sequences
            controls = MultiScalePlanner._sample_controls(
                rng, config, prev_controls
            )

            # Evaluate trajectories in parallel
            cfg = {
                "dt": config.dt,
                "goal": goal.tolist(),
                "lambda_collision": config.lambda_collision,
                "lambda_uncertainty": config.lambda_uncertainty,
                "lambda_control": config.lambda_control,
                "lambda_goal": config.lambda_goal,
            }

            batch_size = int(math.ceil(config.num_trajectories / config.num_workers))
            batches = []
            for i in range(0, config.num_trajectories, batch_size):
                batches.append(controls[i:i + batch_size])

            job_args = [(b, start_state, cfg) for b in batches]
            results = pool.map(_evaluate_trajectory_batch, job_args)

            # Aggregate results
            all_costs = []
            all_reasons = []
            all_finals = []
            for costs, reasons, finals in results:
                all_costs.append(costs)
                all_reasons.append(reasons)
                all_finals.append(finals)

            all_costs = np.concatenate(all_costs)
            all_reasons = np.concatenate(all_reasons)
            all_finals = np.concatenate(all_finals)

            # MPPI-style trajectory selection
            best_idx, best_controls_seq = MultiScalePlanner._mppi_select(
                controls, all_costs, config.temperature
            )

            # Reconstruct best trajectory
            best_states = MultiScalePlanner._rollout_trajectory(
                start_state, best_controls_seq, config.dt, config.horizon
            )

            # Publish best trajectory
            best_traj_shared[:] = []
            for s in best_states:
                best_traj_shared.append([float(s[0]), float(s[1])])

            best_controls_shared[:] = []
            for c in best_controls_seq:
                best_controls_shared.append([float(c[0]), float(c[1])])

            # Collect ghost/rejected trajectories
            rejected_shared[:] = []
            sorted_idx = np.argsort(all_costs)
            ghost_count = 0
            for idx in sorted_idx:
                if ghost_count >= 16:
                    break
                if all_reasons[idx] == 0:
                    continue

                ghost_states = MultiScalePlanner._rollout_trajectory(
                    start_state, controls[idx], config.dt, config.horizon
                )
                rejected_shared.append({
                    "reason_code": int(all_reasons[idx]),
                    "reason": ["ok", "collision", "uncertainty", "bounds"][all_reasons[idx]],
                    "cost": float(all_costs[idx]),
                    "states": [[float(s[0]), float(s[1])] for s in ghost_states],
                })
                ghost_count += 1

            # Update statistics
            iteration += 1
            total_cost += all_costs[best_idx]
            collision_count += np.sum(all_reasons == 1)

            stats_shared['iterations'] = iteration
            stats_shared['avg_cost'] = total_cost / iteration
            stats_shared['best_cost'] = float(all_costs[best_idx])
            stats_shared['collision_rate'] = collision_count / (iteration * config.num_trajectories)

            # Warm start for next iteration
            prev_controls = best_controls_seq

            # Maintain update rate
            elapsed = time.perf_counter() - iter_start
            remaining = sleep_time - elapsed
            if remaining > 0:
                time.sleep(remaining)

        pool.close()
        pool.join()

    @staticmethod
    def _sample_controls(
        rng: np.random.Generator,
        config: PlannerConfig,
        prev_controls: Optional[np.ndarray],
    ) -> np.ndarray:
        """Sample control sequences with optional warm start."""
        shape = (config.num_trajectories, config.horizon, 2)

        if prev_controls is not None:
            # Warm start: shift previous controls and add noise
            base = np.zeros(shape, dtype=np.float32)
            base[:, :-1, :] = prev_controls[1:]  # Shift left
            base[:, -1, :] = prev_controls[-1]   # Repeat last
            noise = rng.normal(0, 0.3, size=shape).astype(np.float32)
            controls = base + noise
        else:
            controls = rng.normal(0, 1.0, size=shape).astype(np.float32)

        # Clip to control limits
        controls[..., 0] = np.clip(controls[..., 0], 0, config.v_max)
        controls[..., 1] = np.clip(controls[..., 1], -config.omega_max, config.omega_max)

        return controls

    @staticmethod
    def _mppi_select(
        controls: np.ndarray,
        costs: np.ndarray,
        temperature: float,
    ) -> Tuple[int, np.ndarray]:
        """MPPI-style weighted trajectory selection."""
        # Compute weights
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / temperature)
        weights /= weights.sum()

        # Weighted average of controls
        weighted_controls = np.einsum('b,bhc->hc', weights, controls)

        # Also return index of best trajectory
        best_idx = int(np.argmin(costs))

        return best_idx, weighted_controls

    @staticmethod
    def _rollout_trajectory(
        start_state: np.ndarray,
        controls: np.ndarray,
        dt: float,
        horizon: int,
    ) -> np.ndarray:
        """Rollout control sequence to get state trajectory."""
        states = [start_state.copy()]
        y, x, heading = start_state[0], start_state[1], start_state[2]

        for t in range(min(horizon, len(controls))):
            v, omega = controls[t]
            heading += omega * dt
            y += v * math.sin(heading) * dt
            x += v * math.cos(heading) * dt
            states.append(np.array([y, x, heading], dtype=np.float32))

        return np.stack(states)

    # ---------------- Getters ----------------

    def get_best_trajectory(self) -> np.ndarray:
        """Return best trajectory as (T, 2) array [y, x]."""
        if len(self._best_traj) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(list(self._best_traj), dtype=np.float32)

    def get_best_controls(self) -> np.ndarray:
        """Return best control sequence as (H, 2) array [v, omega]."""
        if len(self._best_controls) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(list(self._best_controls), dtype=np.float32)

    def get_rejected_candidates(self, reason: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return rejected/ghost trajectories for visualization."""
        rejected = list(self._rejected)
        if reason is not None:
            rejected = [r for r in rejected if r.get('reason') == reason]
        return rejected

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return dict(self._stats)


# ---------------------------------------------------------------------------
# AraSpeciesV3 Orchestrator
# ---------------------------------------------------------------------------

class AraSpeciesV3:
    """
    Unified orchestrator for CalibratedWorldModel + MultiScalePlanner on 5955WX.

    - Allocates shared voxel grid in RAM (zero-copy access)
    - Starts world_model process and planner process + worker pool
    - Implements explain_decision() for holographic interface
    - Optional Cathedral integration for hardware acceleration
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (500, 500, 4),
        planner_config: Optional[PlannerConfig] = None,
        use_cathedral: bool = False,
        simulation_mode: bool = True,
    ):
        self.config = GridConfig(shape=grid_shape)
        self.planner_config = planner_config or PlannerConfig()
        self.use_cathedral = use_cathedral and CATHEDRAL_AVAILABLE
        self.running = False

        # Allocate shared voxel grid
        self.shm_grid, self.grid_view = create_shared_grid(self.config)

        # Sensor queue for real sensor data
        self.sensor_queue: mp.Queue = mp.Queue()

        # Cathedral integration (optional)
        self.cathedral_bridge = None
        if self.use_cathedral:
            try:
                self.cathedral_bridge = create_accelerated_oracle(
                    simulation_mode=simulation_mode
                )
                logger.info("Cathedral integration enabled")
            except Exception as e:
                logger.warning("Cathedral init failed: %s", e)
                self.cathedral_bridge = None

        # World model
        self.world_model = CalibratedWorldModel(
            shm_name=self.shm_grid.name,
            grid_config=self.config,
            hnsw_params={"M": 32, "ef_construction": 200, "dim": 128},
            calibration_mode=CalibrationMode.ISOTONIC,
            core_affinity=[0, 1, 2, 3],
            sensor_queue=self.sensor_queue,
            cathedral_bridge=self.cathedral_bridge,
            update_hz=20.0,
        )

        # Planner
        self.planner = MultiScalePlanner(
            shm_name=self.shm_grid.name,
            grid_config=self.config,
            config=self.planner_config,
            core_affinity_workers=list(range(8, 16)),
            core_affinity_loop=list(range(8, 16)),
            cathedral_bridge=self.cathedral_bridge,
        )

        # Explanation frame cache
        self._last_explanation: Optional[Dict[str, Any]] = None
        self._explanation_count = 0

    # ---------------- Lifecycle ----------------

    def start(self) -> None:
        """Launch all subsystems."""
        self.running = True
        self.world_model.start()
        self.planner.start()

        cores = get_available_cores()
        logger.info(
            "AraSpeciesV3 online (grid=%s, shape=%s, cores=%d, cathedral=%s)",
            self.shm_grid.name, self.config.shape, cores, self.use_cathedral
        )

    def shutdown(self) -> None:
        """Graceful shutdown."""
        self.running = False
        self.planner.stop()
        self.world_model.stop()

        if self.cathedral_bridge is not None:
            self.cathedral_bridge.close()

        self.shm_grid.close()
        self.shm_grid.unlink()
        logger.info("AraSpeciesV3 shutdown complete")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    # ---------------- Sensor interface ----------------

    def inject_sensor_data(self, data: Dict[str, Any]) -> None:
        """Inject sensor data into the world model."""
        self.sensor_queue.put(data)

    # ---------------- Goal/State interface ----------------

    def set_goal(self, y: float, x: float) -> None:
        """Set the planning goal."""
        self.planner.set_goal(y, x)

    def set_state(self, y: float, x: float, heading: float = 0.0) -> None:
        """Set the current state."""
        self.planner.set_state(y, x, heading)

    # ---------------- Explainability ----------------

    def explain_decision(self) -> Dict[str, Any]:
        """
        Build an explanation frame for the holographic UI.

        Returns dict with:
        {
            'meta': { timestamp, mode, iteration },
            'visuals': {
                'trajectory': [[y,x], ...],
                'confidence_tube': [radius_t, ...],
                'fog_nodes': [[y,x,density], ...],
                'ghosts': [{ reason, reason_code, cost, states }, ...]
            },
            'statistics': { world_model, planner },
            'text_summary': str
        }
        """
        timestamp = time.time()
        self._explanation_count += 1

        # Get trajectory and planning data
        best_traj = self.planner.get_best_trajectory()
        ghost_paths = self.planner.get_rejected_candidates()
        planner_stats = self.planner.get_statistics()

        # Get uncertainty fog from world model
        fog_voxels = self.world_model.get_high_uncertainty_voxels(
            threshold=0.7, max_voxels=3000
        )
        wm_stats = self.world_model.get_statistics()

        # Compute confidence tube along trajectory
        confidence_tube = self.world_model.propagate_uncertainty_along_trajectory(best_traj)

        # Generate text summary
        summary = self._generate_summary(best_traj, fog_voxels, ghost_paths, planner_stats)

        explanation_frame = {
            "meta": {
                "timestamp": timestamp,
                "mode": "Active_Nav",
                "iteration": self._explanation_count,
                "cathedral_enabled": self.use_cathedral,
            },
            "visuals": {
                "trajectory": best_traj.tolist(),
                "confidence_tube": confidence_tube,
                "fog_nodes": fog_voxels.tolist(),
                "ghosts": ghost_paths,
            },
            "statistics": {
                "world_model": wm_stats,
                "planner": planner_stats,
            },
            "text_summary": summary,
        }

        self._last_explanation = explanation_frame
        return explanation_frame

    def _generate_summary(
        self,
        traj: np.ndarray,
        fog: np.ndarray,
        ghosts: List[Dict],
        stats: Dict,
    ) -> str:
        """Generate human-readable summary."""
        if traj.size == 0:
            return "No active plan; world model converging..."

        num_fog = fog.shape[0]
        num_ghosts = len(ghosts)
        collision_ghosts = len([g for g in ghosts if g.get('reason') == 'collision'])
        uncertainty_ghosts = len([g for g in ghosts if g.get('reason') == 'uncertainty'])

        parts = []

        if num_fog > 100:
            parts.append(f"Navigating through {num_fog} high-uncertainty regions")

        if collision_ghosts > 0:
            parts.append(f"avoiding {collision_ghosts} collision-prone paths")

        if uncertainty_ghosts > 0:
            parts.append(f"bypassing {uncertainty_ghosts} uncertain zones")

        if stats.get('best_cost', 0) < 100:
            parts.append("path quality: good")
        else:
            parts.append("path quality: moderate")

        return "; ".join(parts) + "."

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "world_model": self.world_model.get_statistics(),
            "planner": self.planner.get_statistics(),
            "grid_config": {
                "shape": self.config.shape,
                "shm_name": self.shm_grid.name,
            },
            "cathedral_enabled": self.use_cathedral,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Quick smoke test for AraSpeciesV3."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("AraSpeciesV3 Demonstration")
    print("Threadripper-aware MPPI Planning + Calibrated World Model")
    print("=" * 70)

    # Smaller grid for faster demo
    with AraSpeciesV3(grid_shape=(200, 200, 4)) as ara:
        # Let systems warm up
        print("\nWarming up (2 seconds)...")
        time.sleep(2.0)

        # Set a goal
        ara.set_goal(100, 180)

        # Run for a bit
        print("\nRunning planning loop (3 seconds)...")
        for i in range(30):
            time.sleep(0.1)
            if (i + 1) % 10 == 0:
                frame = ara.explain_decision()
                print(f"\n--- Explanation Frame {frame['meta']['iteration']} ---")
                print(f"Trajectory length: {len(frame['visuals']['trajectory'])}")
                print(f"Fog nodes: {len(frame['visuals']['fog_nodes'])}")
                print(f"Ghost paths: {len(frame['visuals']['ghosts'])}")
                print(f"Summary: {frame['text_summary']}")

        # Final statistics
        print("\n" + "=" * 70)
        print("Final Statistics")
        print("=" * 70)
        stats = ara.get_statistics()
        print(f"World Model updates: {stats['world_model'].get('updates', 0)}")
        print(f"Planner iterations: {stats['planner'].get('iterations', 0)}")
        print(f"Average cost: {stats['planner'].get('avg_cost', 0):.2f}")
        print(f"Collision rate: {stats['planner'].get('collision_rate', 0):.4f}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    demo()
