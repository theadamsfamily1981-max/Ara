#!/usr/bin/env python3
"""
TWE Kitten Fabric Interface - SNN/FPGA Real-Time Control
=========================================================

The Kitten Fabric is the neuromorphic control layer that handles:
- Real-time sensor processing (kHz rates)
- λ estimation from sensor avalanches
- Π-weighted precision control
- Actuator command generation

GUTC Integration:
    λ_fabric: Global gain on SNN controllers
    λ_tissue: Estimated from sensor field avalanches
    Π_sensory: Gain on viability error signals
    Π_prior: Gain on blueprint conformance

This module provides the Python interface to the FPGA/SNN fabric.
In production, this wraps actual hardware; here we provide simulation.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum, auto
import numpy as np

from .blueprint import PrintLayer, OrganBlueprint


# =============================================================================
# Sensor State
# =============================================================================

@dataclass
class SensorTile:
    """
    A single tile in the sensor grid.

    Each tile aggregates local sensor readings into a vector.
    """
    tile_id: str
    position: Tuple[int, int]       # Grid position (i, j)

    # Chemical sensors
    oxygen_mmhg: float = 150.0      # Partial pressure O2
    ph: float = 7.4
    co2_mmhg: float = 40.0
    lactate_mm: float = 1.0

    # Electrical/structural
    impedance_ohm: float = 100.0    # Tissue impedance
    connectivity: float = 1.0       # 0-1, structural integrity

    # Flow
    flow_rate_ul_min: float = 0.0
    pressure_mmhg: float = 0.0

    # Temperature
    temperature_c: float = 37.0

    # Event detection (for avalanche analysis)
    event_count: int = 0
    last_event_time: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for SNN input."""
        return np.array([
            self.oxygen_mmhg / 150.0,        # Normalized
            (self.ph - 7.0) / 0.8,           # Centered, scaled
            self.co2_mmhg / 50.0,
            self.lactate_mm / 5.0,
            self.impedance_ohm / 200.0,
            self.connectivity,
            self.flow_rate_ul_min / 100.0,
            self.temperature_c / 40.0,
        ], dtype=np.float32)

    def viability_score(self) -> float:
        """Quick viability heuristic (0-1, higher is better)."""
        # Oxygen: want 80-150 mmHg
        o2_score = 1.0 - abs(self.oxygen_mmhg - 100) / 100.0

        # pH: want 7.35-7.45
        ph_score = 1.0 - abs(self.ph - 7.4) / 0.4

        # Lactate: want < 2 mM
        lac_score = max(0, 1.0 - self.lactate_mm / 4.0)

        # Connectivity: want high
        conn_score = self.connectivity

        return max(0, min(1, (o2_score + ph_score + lac_score + conn_score) / 4))


@dataclass
class SensorGrid:
    """
    Complete sensor grid state.

    The print bed is tiled into an (nx x ny) grid of sensor tiles.
    """
    shape: Tuple[int, int]          # (nx, ny) tiles
    tiles: Dict[Tuple[int, int], SensorTile] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Initialize tiles if empty
        if not self.tiles:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.tiles[(i, j)] = SensorTile(
                        tile_id=f"tile_{i}_{j}",
                        position=(i, j),
                    )

    def get_tile(self, i: int, j: int) -> Optional[SensorTile]:
        return self.tiles.get((i, j))

    def to_tensor(self) -> np.ndarray:
        """Convert grid to 3D tensor (nx, ny, n_features)."""
        nx, ny = self.shape
        n_features = 8  # From SensorTile.to_vector()

        tensor = np.zeros((nx, ny, n_features), dtype=np.float32)
        for (i, j), tile in self.tiles.items():
            tensor[i, j, :] = tile.to_vector()

        return tensor

    def mean_viability(self) -> float:
        """Average viability across all tiles."""
        if not self.tiles:
            return 0.0
        return np.mean([t.viability_score() for t in self.tiles.values()])

    def viability_map(self) -> np.ndarray:
        """2D viability heatmap."""
        nx, ny = self.shape
        vmap = np.zeros((nx, ny), dtype=np.float32)
        for (i, j), tile in self.tiles.items():
            vmap[i, j] = tile.viability_score()
        return vmap


# =============================================================================
# Avalanche Analysis (λ Estimation)
# =============================================================================

@dataclass
class Avalanche:
    """A detected avalanche (cascade) in sensor events."""
    avalanche_id: int
    start_time: float
    duration_ms: float
    size: int                       # Number of tiles involved
    tiles_involved: List[Tuple[int, int]] = field(default_factory=list)
    peak_amplitude: float = 0.0


class AvalancheDetector:
    """
    Detects avalanches in sensor event streams.

    Used to estimate the branching ratio λ of the tissue/system.
    """

    def __init__(self, time_bin_ms: float = 10.0, spatial_threshold: int = 2):
        self.time_bin_ms = time_bin_ms
        self.spatial_threshold = spatial_threshold

        self.event_buffer: List[Tuple[float, int, int]] = []  # (time, i, j)
        self.avalanches: List[Avalanche] = []
        self.avalanche_counter = 0

    def add_event(self, timestamp: float, tile_i: int, tile_j: int):
        """Record a sensor event."""
        self.event_buffer.append((timestamp, tile_i, tile_j))

        # Keep buffer bounded
        cutoff = timestamp - 1.0  # 1 second history
        self.event_buffer = [(t, i, j) for t, i, j in self.event_buffer if t > cutoff]

    def detect_avalanches(self, current_time: float) -> List[Avalanche]:
        """
        Detect avalanches in the recent event buffer.

        Returns newly detected avalanches.
        """
        # Simple binning approach
        bin_width = self.time_bin_ms / 1000.0
        lookback = 0.5  # 500ms

        # Bin events
        bins: Dict[int, List[Tuple[int, int]]] = {}
        for t, i, j in self.event_buffer:
            if t > current_time - lookback:
                bin_idx = int((current_time - t) / bin_width)
                if bin_idx not in bins:
                    bins[bin_idx] = []
                bins[bin_idx].append((i, j))

        new_avalanches = []

        # Find contiguous bins with activity
        sorted_bins = sorted(bins.keys())
        if not sorted_bins:
            return []

        current_avalanche_tiles = set()
        start_bin = sorted_bins[0]

        for bin_idx in sorted_bins:
            if bin_idx - start_bin > 2:  # Gap in activity
                # Close current avalanche
                if len(current_avalanche_tiles) >= self.spatial_threshold:
                    self.avalanche_counter += 1
                    av = Avalanche(
                        avalanche_id=self.avalanche_counter,
                        start_time=current_time - (bin_idx * bin_width),
                        duration_ms=(bin_idx - start_bin) * self.time_bin_ms,
                        size=len(current_avalanche_tiles),
                        tiles_involved=list(current_avalanche_tiles),
                    )
                    new_avalanches.append(av)
                    self.avalanches.append(av)

                # Reset
                current_avalanche_tiles = set()
                start_bin = bin_idx

            current_avalanche_tiles.update(bins[bin_idx])

        return new_avalanches

    def estimate_branching_ratio(self, window_s: float = 5.0) -> float:
        """
        Estimate λ (branching ratio) from recent avalanches.

        λ ≈ 1 means critical (healthy).
        λ < 1 means subcritical (under-responsive).
        λ > 1 means supercritical (unstable).
        """
        cutoff = time.time() - window_s
        recent = [av for av in self.avalanches if av.start_time > cutoff]

        if len(recent) < 3:
            return 1.0  # Insufficient data, assume critical

        # Simple estimate: ratio of consecutive avalanche sizes
        sizes = [av.size for av in recent]
        ratios = []
        for i in range(1, len(sizes)):
            if sizes[i-1] > 0:
                ratios.append(sizes[i] / sizes[i-1])

        if not ratios:
            return 1.0

        # Geometric mean of ratios
        log_ratios = [math.log(r) for r in ratios if r > 0]
        if not log_ratios:
            return 1.0

        return math.exp(np.mean(log_ratios))


# =============================================================================
# Precision Control (Π Fields)
# =============================================================================

@dataclass
class PrecisionState:
    """
    Current precision (Π) weights for control.

    GUTC mapping:
        Π_sensory: Gain on viability errors (what sensors say)
        Π_prior: Gain on blueprint conformance (what the plan says)
    """
    pi_sensory: float = 1.0         # Viability precision
    pi_prior: float = 1.0           # Blueprint precision

    # Per-modality precision
    pi_oxygen: float = 1.0
    pi_ph: float = 1.0
    pi_temperature: float = 1.0
    pi_structure: float = 1.0

    # Adaptive bounds
    pi_min: float = 0.1
    pi_max: float = 5.0

    def clamp(self):
        """Clamp all Π values to bounds."""
        self.pi_sensory = max(self.pi_min, min(self.pi_max, self.pi_sensory))
        self.pi_prior = max(self.pi_min, min(self.pi_max, self.pi_prior))
        self.pi_oxygen = max(self.pi_min, min(self.pi_max, self.pi_oxygen))
        self.pi_ph = max(self.pi_min, min(self.pi_max, self.pi_ph))
        self.pi_temperature = max(self.pi_min, min(self.pi_max, self.pi_temperature))
        self.pi_structure = max(self.pi_min, min(self.pi_max, self.pi_structure))


# =============================================================================
# Actuator Commands
# =============================================================================

@dataclass
class ActuatorCommand:
    """
    Command vector for TWE actuators.

    Generated by the Kitten Fabric control loop.
    """
    # Nozzle control
    scaffold_flow_rate: float = 0.0     # μL/s
    cell_flow_rate: float = 0.0         # μL/s
    sacrificial_flow_rate: float = 0.0  # μL/s

    # Motion
    stage_velocity_mm_s: float = 1.0
    stage_position: Tuple[float, float, float] = (0, 0, 0)

    # Environmental
    bed_temperature_c: float = 37.0
    chamber_humidity: float = 0.95
    media_flow_ul_min: float = 0.0

    # Field shaping
    acoustic_amplitude: float = 0.0     # 0-1
    acoustic_frequency_hz: float = 1e6
    magnetic_field_mt: float = 0.0      # milliTesla

    # Emergency
    pause_requested: bool = False
    abort_requested: bool = False


# =============================================================================
# Kitten Fabric Controller
# =============================================================================

class KittenFabric:
    """
    Kitten Fabric - Neuromorphic real-time controller.

    In production: wraps FPGA/SNN hardware.
    Here: simulation for development/testing.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (10, 10),
        control_rate_hz: float = 100.0,
        verbose: bool = True,
    ):
        self.grid_shape = grid_shape
        self.control_rate_hz = control_rate_hz
        self.verbose = verbose

        # State
        self.sensor_grid = SensorGrid(grid_shape)
        self.precision = PrecisionState()
        self.avalanche_detector = AvalancheDetector()

        # Current estimates
        self.lambda_hat: float = 1.0
        self.lambda_history: List[Tuple[float, float]] = []

        # Control state
        self.connected = False
        self.running = False

        # Callbacks for hardware integration
        self.read_sensors_fn: Optional[Callable[[], SensorGrid]] = None
        self.apply_commands_fn: Optional[Callable[[ActuatorCommand], None]] = None

    def connect(self) -> bool:
        """Connect to Kitten Fabric hardware/simulation."""
        self.connected = True
        if self.verbose:
            print(f"[KittenFabric] Connected (grid={self.grid_shape}, "
                  f"rate={self.control_rate_hz}Hz)")
        return True

    def disconnect(self):
        """Disconnect from hardware."""
        self.connected = False
        self.running = False
        if self.verbose:
            print("[KittenFabric] Disconnected")

    # ------------------------------------------------------------------
    # Sensor Interface
    # ------------------------------------------------------------------

    def read_sensors(self) -> SensorGrid:
        """Read current sensor state."""
        if self.read_sensors_fn:
            self.sensor_grid = self.read_sensors_fn()
        else:
            # Simulation: add some noise to current state
            self._simulate_sensor_dynamics()

        self.sensor_grid.timestamp = time.time()
        return self.sensor_grid

    def _simulate_sensor_dynamics(self):
        """Simulate sensor evolution (for testing without hardware)."""
        for tile in self.sensor_grid.tiles.values():
            # Random walk on key parameters
            tile.oxygen_mmhg += np.random.normal(0, 2)
            tile.oxygen_mmhg = max(50, min(200, tile.oxygen_mmhg))

            tile.ph += np.random.normal(0, 0.02)
            tile.ph = max(6.8, min(7.8, tile.ph))

            tile.lactate_mm += np.random.normal(0, 0.1)
            tile.lactate_mm = max(0, min(10, tile.lactate_mm))

            tile.temperature_c += np.random.normal(0, 0.1)
            tile.temperature_c = max(30, min(42, tile.temperature_c))

            # Random events for avalanche detection
            if np.random.random() < 0.05:
                tile.event_count += 1
                tile.last_event_time = time.time()
                self.avalanche_detector.add_event(
                    time.time(), tile.position[0], tile.position[1]
                )

    # ------------------------------------------------------------------
    # λ Estimation
    # ------------------------------------------------------------------

    def estimate_lambda(self, sensor_state: SensorGrid) -> float:
        """
        Estimate branching ratio λ from sensor avalanches.

        This is the core GUTC metric: λ ≈ 1 is critical (healthy).
        """
        # Detect avalanches from recent events
        self.avalanche_detector.detect_avalanches(time.time())

        # Estimate λ
        self.lambda_hat = self.avalanche_detector.estimate_branching_ratio()

        # Record history
        self.lambda_history.append((time.time(), self.lambda_hat))
        if len(self.lambda_history) > 1000:
            self.lambda_history = self.lambda_history[-1000:]

        return self.lambda_hat

    # ------------------------------------------------------------------
    # Π Adaptation
    # ------------------------------------------------------------------

    def update_precision(
        self,
        sensor_state: SensorGrid,
        blueprint: OrganBlueprint,
        lambda_hat: float,
    ) -> Tuple[float, float]:
        """
        Update precision weights based on current state.

        Returns (pi_sensory, pi_prior).
        """
        # Base adaptation: balance sensory and prior based on λ
        if lambda_hat < 0.8:
            # Subcritical: tissue under-responsive, trust sensors more
            self.precision.pi_sensory *= 1.05
            self.precision.pi_prior *= 0.95
        elif lambda_hat > 1.2:
            # Supercritical: unstable, trust blueprint more (stabilize)
            self.precision.pi_sensory *= 0.95
            self.precision.pi_prior *= 1.05
        else:
            # Near-critical: gradually balance
            target_ratio = 1.0
            current_ratio = self.precision.pi_sensory / max(self.precision.pi_prior, 0.01)
            if current_ratio > target_ratio:
                self.precision.pi_sensory *= 0.99
            else:
                self.precision.pi_prior *= 0.99

        # Viability-based adaptation
        mean_viab = sensor_state.mean_viability()
        if mean_viab < 0.5:
            # Low viability: increase sensory gain
            self.precision.pi_sensory *= 1.1
            self.precision.pi_oxygen *= 1.2
        elif mean_viab > 0.9:
            # High viability: can trust blueprint more
            self.precision.pi_prior *= 1.02

        # Clamp to bounds
        self.precision.clamp()

        return self.precision.pi_sensory, self.precision.pi_prior

    # ------------------------------------------------------------------
    # Control Step
    # ------------------------------------------------------------------

    def control_step(
        self,
        sensor_state: SensorGrid,
        layer: PrintLayer,
        lambda_hat: float,
        pi_sensory: float,
        pi_prior: float,
    ) -> ActuatorCommand:
        """
        Compute actuator commands for current state.

        This is the core closed-loop control function.
        """
        cmd = ActuatorCommand()

        # Base flow rates from layer
        cmd.scaffold_flow_rate = 1.0 * layer.print_speed_modifier
        cmd.cell_flow_rate = 0.5 * layer.print_speed_modifier

        # Viability-based adjustments
        mean_viab = sensor_state.mean_viability()
        viability_map = sensor_state.viability_map()

        # Find problem regions
        low_viab_tiles = np.sum(viability_map < 0.5)
        total_tiles = np.prod(viability_map.shape)
        problem_fraction = low_viab_tiles / max(total_tiles, 1)

        if problem_fraction > 0.2:
            # Too many problem areas: slow down, increase media flow
            cmd.stage_velocity_mm_s = 0.5
            cmd.media_flow_ul_min = 200.0

            if self.verbose:
                print(f"  [FABRIC] Viability issue: {problem_fraction:.1%} tiles low, "
                      f"slowing print")

        elif problem_fraction > 0.05:
            # Some problem areas: moderate adjustment
            cmd.stage_velocity_mm_s = 0.8
            cmd.media_flow_ul_min = 100.0

        else:
            # Good viability: normal operation
            cmd.stage_velocity_mm_s = 1.0
            cmd.media_flow_ul_min = 50.0

        # λ-based adjustments
        if lambda_hat < 0.7:
            # Subcritical: increase acoustic stimulation
            cmd.acoustic_amplitude = 0.5
            cmd.acoustic_frequency_hz = 1.5e6
        elif lambda_hat > 1.3:
            # Supercritical: reduce stimulation, stabilize
            cmd.acoustic_amplitude = 0.1
            cmd.stage_velocity_mm_s *= 0.8

        # Temperature control (always maintain 37°C)
        mean_temp = np.mean([t.temperature_c for t in sensor_state.tiles.values()])
        if mean_temp < 36.5:
            cmd.bed_temperature_c = 38.0
        elif mean_temp > 37.5:
            cmd.bed_temperature_c = 36.5
        else:
            cmd.bed_temperature_c = 37.0

        # Emergency checks
        if mean_viab < 0.3:
            cmd.pause_requested = True
            if self.verbose:
                print("  [FABRIC] PAUSE REQUESTED: Viability critical")

        return cmd

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current fabric status."""
        return {
            "connected": self.connected,
            "running": self.running,
            "lambda_hat": self.lambda_hat,
            "pi_sensory": self.precision.pi_sensory,
            "pi_prior": self.precision.pi_prior,
            "mean_viability": self.sensor_grid.mean_viability(),
            "n_avalanches": len(self.avalanche_detector.avalanches),
            "grid_shape": self.grid_shape,
        }

    def status_string(self) -> str:
        """Get formatted status string."""
        status = self.get_status()
        lambda_status = "🟢" if 0.8 <= status["lambda_hat"] <= 1.2 else "🟡" if 0.6 <= status["lambda_hat"] <= 1.4 else "🔴"

        return (
            f"[KittenFabric] {lambda_status}\n"
            f"  λ̂ = {status['lambda_hat']:.3f}\n"
            f"  Π_sensory = {status['pi_sensory']:.2f}, Π_prior = {status['pi_prior']:.2f}\n"
            f"  Viability = {status['mean_viability']:.2f}\n"
            f"  Avalanches detected: {status['n_avalanches']}"
        )
