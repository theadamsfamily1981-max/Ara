#!/usr/bin/env python3
"""
TWE Hardware Abstraction Layer
==============================

Abstraction for TWE physical hardware:
- Multi-modal printhead (scaffold, cell, sacrificial nozzles)
- Motion stage (3-axis + rotation)
- Sensor bed (chemical, electrical, optical)
- Environmental control (temperature, humidity, gas)
- Field shaping (acoustic, magnetic)

This provides a hardware-agnostic interface so the same control
code works with simulated or real hardware.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import numpy as np

from .blueprint import OrganBlueprint, PrintLayer
from .fabric import SensorGrid, SensorTile, ActuatorCommand


# =============================================================================
# Hardware State
# =============================================================================

class PrinterState(Enum):
    """Overall printer state machine."""
    IDLE = auto()
    INITIALIZING = auto()
    HOMING = auto()
    READY = auto()
    PRINTING = auto()
    PAUSED = auto()
    ERROR = auto()
    MAINTENANCE = auto()
    SHUTDOWN = auto()


@dataclass
class NozzleState:
    """State of a single nozzle."""
    nozzle_id: str
    nozzle_type: str              # scaffold, cell, sacrificial
    is_primed: bool = False
    flow_rate_ul_s: float = 0.0
    temperature_c: float = 25.0
    pressure_kpa: float = 0.0
    material_remaining_ml: float = 100.0

    # Calibration
    offset_um: Tuple[float, float, float] = (0, 0, 0)
    flow_coefficient: float = 1.0


@dataclass
class StageState:
    """State of the motion stage."""
    position_mm: Tuple[float, float, float] = (0, 0, 0)
    velocity_mm_s: Tuple[float, float, float] = (0, 0, 0)
    is_homed: bool = False

    # Limits
    bounds_min_mm: Tuple[float, float, float] = (0, 0, 0)
    bounds_max_mm: Tuple[float, float, float] = (200, 200, 100)

    # Motion parameters
    max_velocity_mm_s: float = 50.0
    max_acceleration_mm_s2: float = 100.0


@dataclass
class ChamberState:
    """Environmental chamber state."""
    temperature_c: float = 37.0
    humidity_percent: float = 95.0
    co2_percent: float = 5.0
    o2_percent: float = 20.0

    # Media perfusion
    media_flow_ul_min: float = 0.0
    media_temperature_c: float = 37.0

    # Field shaping
    acoustic_amplitude: float = 0.0
    acoustic_frequency_hz: float = 1e6
    magnetic_field_mt: float = 0.0
    magnetic_direction: Tuple[float, float, float] = (0, 0, 1)


# =============================================================================
# Hardware Interface (Abstract)
# =============================================================================

class TWEHardware(ABC):
    """
    Abstract base class for TWE hardware interface.

    Implementations:
    - SimulatedHardware: For testing without physical hardware
    - RealHardware: Wraps actual printer (future)
    """

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize hardware, return True on success."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Safely shutdown hardware."""
        pass

    @abstractmethod
    def get_state(self) -> PrinterState:
        """Get current printer state."""
        pass

    @abstractmethod
    def home(self) -> bool:
        """Home all axes, return True on success."""
        pass

    # --- Chamber Control ---

    @abstractmethod
    def prepare_chamber(self, blueprint: OrganBlueprint) -> bool:
        """Prepare chamber for print (temperature, humidity, etc.)."""
        pass

    @abstractmethod
    def get_chamber_state(self) -> ChamberState:
        """Get current chamber state."""
        pass

    @abstractmethod
    def set_chamber_temperature(self, temp_c: float) -> None:
        """Set chamber temperature."""
        pass

    # --- Motion Control ---

    @abstractmethod
    def get_stage_state(self) -> StageState:
        """Get current stage state."""
        pass

    @abstractmethod
    def move_to(self, x: float, y: float, z: float,
                velocity_mm_s: Optional[float] = None) -> bool:
        """Move stage to position, return True when complete."""
        pass

    # --- Nozzle Control ---

    @abstractmethod
    def prime_nozzle(self, nozzle_id: str) -> bool:
        """Prime a nozzle, return True on success."""
        pass

    @abstractmethod
    def set_nozzle_flow(self, nozzle_id: str, flow_ul_s: float) -> None:
        """Set nozzle flow rate."""
        pass

    @abstractmethod
    def get_nozzle_state(self, nozzle_id: str) -> Optional[NozzleState]:
        """Get state of a nozzle."""
        pass

    # --- Sensor Interface ---

    @abstractmethod
    def read_sensors(self) -> SensorGrid:
        """Read all sensors, return grid state."""
        pass

    # --- Command Application ---

    @abstractmethod
    def apply_commands(self, cmd: ActuatorCommand) -> None:
        """Apply actuator commands from Kitten Fabric."""
        pass

    # --- Print Operations ---

    @abstractmethod
    def pause_print(self) -> None:
        """Pause current print."""
        pass

    @abstractmethod
    def resume_print(self) -> None:
        """Resume paused print."""
        pass

    @abstractmethod
    def abort_print(self) -> None:
        """Abort print and safe hardware."""
        pass

    @abstractmethod
    def finalize_print(self) -> None:
        """Finalize print (retract, cool down, etc.)."""
        pass


# =============================================================================
# Simulated Hardware
# =============================================================================

class SimulatedHardware(TWEHardware):
    """
    Simulated TWE hardware for testing.

    Provides realistic behavior without physical hardware.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (10, 10),
        verbose: bool = True,
    ):
        self.grid_shape = grid_shape
        self.verbose = verbose

        # State
        self._state = PrinterState.IDLE
        self._stage = StageState()
        self._chamber = ChamberState()
        self._sensor_grid = SensorGrid(grid_shape)

        # Nozzles
        self._nozzles: Dict[str, NozzleState] = {
            "scaffold": NozzleState("scaffold", "scaffold"),
            "cell_1": NozzleState("cell_1", "cell"),
            "cell_2": NozzleState("cell_2", "cell"),
            "sacrificial": NozzleState("sacrificial", "sacrificial"),
        }

        # Print state
        self._current_blueprint: Optional[OrganBlueprint] = None
        self._current_layer: int = 0
        self._print_start_time: float = 0.0

    def initialize(self) -> bool:
        self._state = PrinterState.INITIALIZING
        if self.verbose:
            print("[HW] Initializing simulated TWE hardware...")
            print(f"     Grid: {self.grid_shape}, Nozzles: {list(self._nozzles.keys())}")

        # Simulate initialization delay
        time.sleep(0.1)

        self._state = PrinterState.READY
        if self.verbose:
            print("[HW] Initialization complete")
        return True

    def shutdown(self) -> None:
        if self.verbose:
            print("[HW] Shutting down...")
        self._state = PrinterState.SHUTDOWN

    def get_state(self) -> PrinterState:
        return self._state

    def home(self) -> bool:
        self._state = PrinterState.HOMING
        if self.verbose:
            print("[HW] Homing all axes...")

        # Simulate homing
        time.sleep(0.1)
        self._stage.position_mm = (0, 0, 0)
        self._stage.is_homed = True

        self._state = PrinterState.READY
        if self.verbose:
            print("[HW] Homing complete")
        return True

    # --- Chamber Control ---

    def prepare_chamber(self, blueprint: OrganBlueprint) -> bool:
        if self.verbose:
            print(f"[HW] Preparing chamber for {blueprint.organ_type}...")

        self._chamber.temperature_c = 37.0
        self._chamber.humidity_percent = 95.0
        self._chamber.co2_percent = 5.0

        self._current_blueprint = blueprint

        if self.verbose:
            print(f"     T={self._chamber.temperature_c}°C, "
                  f"RH={self._chamber.humidity_percent}%, "
                  f"CO2={self._chamber.co2_percent}%")
        return True

    def get_chamber_state(self) -> ChamberState:
        return self._chamber

    def set_chamber_temperature(self, temp_c: float) -> None:
        self._chamber.temperature_c = temp_c

    # --- Motion Control ---

    def get_stage_state(self) -> StageState:
        return self._stage

    def move_to(self, x: float, y: float, z: float,
                velocity_mm_s: Optional[float] = None) -> bool:
        # Clamp to bounds
        x = max(self._stage.bounds_min_mm[0], min(self._stage.bounds_max_mm[0], x))
        y = max(self._stage.bounds_min_mm[1], min(self._stage.bounds_max_mm[1], y))
        z = max(self._stage.bounds_min_mm[2], min(self._stage.bounds_max_mm[2], z))

        self._stage.position_mm = (x, y, z)
        return True

    # --- Nozzle Control ---

    def prime_nozzle(self, nozzle_id: str) -> bool:
        if nozzle_id not in self._nozzles:
            return False

        if self.verbose:
            print(f"[HW] Priming nozzle {nozzle_id}...")

        self._nozzles[nozzle_id].is_primed = True
        return True

    def set_nozzle_flow(self, nozzle_id: str, flow_ul_s: float) -> None:
        if nozzle_id in self._nozzles:
            self._nozzles[nozzle_id].flow_rate_ul_s = flow_ul_s

    def get_nozzle_state(self, nozzle_id: str) -> Optional[NozzleState]:
        return self._nozzles.get(nozzle_id)

    # --- Sensor Interface ---

    def read_sensors(self) -> SensorGrid:
        self._simulate_sensors()
        self._sensor_grid.timestamp = time.time()
        return self._sensor_grid

    def _simulate_sensors(self):
        """Simulate realistic sensor evolution."""
        for tile in self._sensor_grid.tiles.values():
            # Oxygen: affected by perfusion and consumption
            perfusion_effect = self._chamber.media_flow_ul_min / 100.0
            consumption = 0.1 * np.random.random()

            tile.oxygen_mmhg += (perfusion_effect - consumption) * 5
            tile.oxygen_mmhg = max(50, min(180, tile.oxygen_mmhg))
            tile.oxygen_mmhg += np.random.normal(0, 1)

            # pH: slow drift with random walk
            tile.ph += np.random.normal(0, 0.01)
            tile.ph = max(6.9, min(7.6, tile.ph))

            # Lactate: accumulates, cleared by perfusion
            tile.lactate_mm += 0.05 - perfusion_effect * 0.03
            tile.lactate_mm = max(0, min(8, tile.lactate_mm))
            tile.lactate_mm += np.random.normal(0, 0.05)

            # Temperature: tracks chamber with lag
            tile.temperature_c += (self._chamber.temperature_c - tile.temperature_c) * 0.1
            tile.temperature_c += np.random.normal(0, 0.05)

            # Connectivity: affected by printing state
            if self._state == PrinterState.PRINTING:
                tile.connectivity = min(1.0, tile.connectivity + 0.01)
            tile.connectivity += np.random.normal(0, 0.005)
            tile.connectivity = max(0, min(1, tile.connectivity))

    # --- Command Application ---

    def apply_commands(self, cmd: ActuatorCommand) -> None:
        # Apply flow rates
        if "scaffold" in self._nozzles:
            self._nozzles["scaffold"].flow_rate_ul_s = cmd.scaffold_flow_rate
        if "cell_1" in self._nozzles:
            self._nozzles["cell_1"].flow_rate_ul_s = cmd.cell_flow_rate
        if "sacrificial" in self._nozzles:
            self._nozzles["sacrificial"].flow_rate_ul_s = cmd.sacrificial_flow_rate

        # Apply chamber controls
        self._chamber.temperature_c = cmd.bed_temperature_c
        self._chamber.humidity_percent = cmd.chamber_humidity * 100
        self._chamber.media_flow_ul_min = cmd.media_flow_ul_min

        # Field shaping
        self._chamber.acoustic_amplitude = cmd.acoustic_amplitude
        self._chamber.acoustic_frequency_hz = cmd.acoustic_frequency_hz
        self._chamber.magnetic_field_mt = cmd.magnetic_field_mt

        # Emergency handling
        if cmd.pause_requested:
            self.pause_print()
        if cmd.abort_requested:
            self.abort_print()

    # --- Print Operations ---

    def pause_print(self) -> None:
        if self._state == PrinterState.PRINTING:
            self._state = PrinterState.PAUSED
            # Stop all flows
            for nozzle in self._nozzles.values():
                nozzle.flow_rate_ul_s = 0.0
            if self.verbose:
                print("[HW] Print PAUSED")

    def resume_print(self) -> None:
        if self._state == PrinterState.PAUSED:
            self._state = PrinterState.PRINTING
            if self.verbose:
                print("[HW] Print RESUMED")

    def abort_print(self) -> None:
        self._state = PrinterState.ERROR
        for nozzle in self._nozzles.values():
            nozzle.flow_rate_ul_s = 0.0
        if self.verbose:
            print("[HW] Print ABORTED")

    def finalize_print(self) -> None:
        # Retract, cool down
        for nozzle in self._nozzles.values():
            nozzle.flow_rate_ul_s = 0.0

        self._stage.position_mm = (0, 0, self._stage.bounds_max_mm[2])
        self._state = PrinterState.IDLE

        if self.verbose:
            print("[HW] Print FINALIZED")

    def start_print(self) -> None:
        """Start print operation."""
        self._state = PrinterState.PRINTING
        self._print_start_time = time.time()
        self._current_layer = 0

        if self.verbose:
            print("[HW] Print STARTED")

    # --- Status ---

    def get_status(self) -> Dict[str, Any]:
        """Get complete hardware status."""
        return {
            "state": self._state.name,
            "stage_position": self._stage.position_mm,
            "stage_homed": self._stage.is_homed,
            "chamber": {
                "temperature_c": self._chamber.temperature_c,
                "humidity_percent": self._chamber.humidity_percent,
                "media_flow_ul_min": self._chamber.media_flow_ul_min,
            },
            "nozzles": {
                nid: {"primed": n.is_primed, "flow": n.flow_rate_ul_s}
                for nid, n in self._nozzles.items()
            },
            "current_layer": self._current_layer,
            "mean_viability": self._sensor_grid.mean_viability(),
        }

    def status_string(self) -> str:
        """Get formatted status string."""
        status = self.get_status()
        state_emoji = {
            "IDLE": "⚪",
            "READY": "🟢",
            "PRINTING": "🔵",
            "PAUSED": "🟡",
            "ERROR": "🔴",
        }.get(status["state"], "⚪")

        return (
            f"[TWE Hardware] {state_emoji} {status['state']}\n"
            f"  Stage: {status['stage_position']}\n"
            f"  Chamber: {status['chamber']['temperature_c']:.1f}°C, "
            f"media={status['chamber']['media_flow_ul_min']:.0f}μL/min\n"
            f"  Viability: {status['mean_viability']:.2f}"
        )
