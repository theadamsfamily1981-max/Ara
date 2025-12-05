"""BANOS Adapter - Bridges BANOS kernel interface to MIES.

This adapter connects the Bio-Affective Neuromorphic Operating System (BANOS)
kernel interface to the MIES telemetry system, providing a unified view of
hardware and affective state.

BANOS exposes state via:
- /dev/banos_pad - PAD state from affective BPF
- /dev/banos - General neural state from spinal cord driver

MIES expects:
- KernelPhysiology from kernel_bridge.py
- TelemetrySnapshot for PADEngine

This adapter bridges the two, allowing Ara to receive telemetry from either
the traditional snn_ai device OR the BANOS nervous system.
"""

import ctypes
import logging
import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Dict, Any, Tuple

from ..kernel_bridge import KernelPhysiology, PADState, PolicyMode
from ..affect.pad_engine import TelemetrySnapshot, PADVector, EmotionalQuadrant

logger = logging.getLogger(__name__)


# =============================================================================
# BANOS Kernel Structures (from banos/include/banos_common.h)
# =============================================================================

class BanosMode(IntEnum):
    """BANOS affective modes."""
    CALM = 0
    FLOW = 1
    ANXIOUS = 2
    CRITICAL = 3


class BanosReflexFlags(IntEnum):
    """BANOS reflex action flags."""
    NONE = 0
    FAN_BOOST = 0x01
    THROTTLE = 0x02
    PROCHOT = 0x04
    GPU_KILL = 0x08


# Struct format matching banos_pad_state from banos_common.h
BANOS_PAD_STATE_FMT = "<"  # Little-endian
BANOS_PAD_STATE_FMT += "h h h"  # pleasure, arousal, dominance (s16 each)
BANOS_PAD_STATE_FMT += "B B H"  # mode, mode_confidence, mode_duration_ms
BANOS_PAD_STATE_FMT += "h h h h"  # thermal_stress, perf_drive, perceived_risk, empathy_boost
BANOS_PAD_STATE_FMT += "h h h"  # pleasure_rate, arousal_rate, dominance_rate
BANOS_PAD_STATE_FMT += "H H B B"  # bat_loudness, bat_pulse_rate, kill_threshold, sched_mode
BANOS_PAD_STATE_FMT += "Q Q"  # monotonic_time_ns, mode_change_time_ns
BANOS_PAD_STATE_FMT += "I I"  # episode_id, episode_primary_stressor

BANOS_PAD_STATE_SIZE = struct.calcsize(BANOS_PAD_STATE_FMT)

# Struct format for spinal_cord state (telemetry)
BANOS_SPINAL_CORD_FMT = "<"
BANOS_SPINAL_CORD_FMT += "I"  # version
BANOS_SPINAL_CORD_FMT += "H H H"  # cpu_temp, gpu_temp, fpga_temp (mC / 100)
BANOS_SPINAL_CORD_FMT += "B B B B"  # cpu_load, gpu_load, fpga_load, mem_pressure (0-100)
BANOS_SPINAL_CORD_FMT += "I I"  # spike_count_total, spike_delta
BANOS_SPINAL_CORD_FMT += "B B H"  # thermal_source_id, reflex_active, reserved
BANOS_SPINAL_CORD_FMT += "I"  # flags

BANOS_SPINAL_CORD_SIZE = struct.calcsize(BANOS_SPINAL_CORD_FMT)


@dataclass
class BanosPADReading:
    """Raw PAD reading from BANOS kernel."""
    pleasure: float  # -1.0 to 1.0
    arousal: float   # -1.0 to 1.0
    dominance: float  # -1.0 to 1.0
    mode: BanosMode
    mode_confidence: float  # 0.0 to 1.0

    # Diagnostics
    thermal_stress: float
    performance_drive: float
    perceived_risk: float
    empathy_boost: float

    # Derivatives
    d_pleasure: float
    d_arousal: float
    d_dominance: float

    # Scheduler hints
    bat_loudness: float
    bat_pulse_rate: float
    kill_threshold: int
    scheduler_mode: int

    timestamp_ns: int


@dataclass
class BanosSpinalReading:
    """Raw spinal cord (telemetry) reading from BANOS kernel."""
    cpu_temp: float
    gpu_temp: float
    fpga_temp: float
    cpu_load: float
    gpu_load: float
    fpga_load: float
    mem_pressure: float
    spike_count: int
    spike_delta: int
    thermal_source_id: int
    reflex_active: int
    flags: int


class BANOSAdapter:
    """
    Adapter that reads from BANOS kernel devices and provides MIES-compatible output.

    Usage:
        adapter = BANOSAdapter()
        if adapter.connect():
            # Get MIES-compatible structures
            physiology = adapter.get_kernel_physiology()
            telemetry = adapter.get_telemetry_snapshot()
            pad_vector = adapter.get_pad_vector()
    """

    PAD_DEVICE = "/dev/banos_pad"
    SPINAL_DEVICE = "/dev/banos"

    def __init__(
        self,
        pad_device: Optional[str] = None,
        spinal_device: Optional[str] = None,
        simulate: bool = False,
    ):
        """
        Initialize BANOS adapter.

        Args:
            pad_device: Path to BANOS PAD device (default: /dev/banos_pad)
            spinal_device: Path to BANOS spinal device (default: /dev/banos)
            simulate: If True, return simulated values instead of reading hardware
        """
        self.pad_device = pad_device or self.PAD_DEVICE
        self.spinal_device = spinal_device or self.SPINAL_DEVICE
        self.simulate = simulate

        self._pad_fd: Optional[int] = None
        self._pad_mmap: Optional[mmap.mmap] = None
        self._spinal_fd: Optional[int] = None
        self._spinal_mmap: Optional[mmap.mmap] = None

        self._connected = False
        self._last_pad: Optional[BanosPADReading] = None
        self._last_spinal: Optional[BanosSpinalReading] = None

    def connect(self) -> bool:
        """Connect to BANOS kernel devices."""
        if self.simulate:
            self._connected = True
            logger.info("BANOSAdapter: Running in simulation mode")
            return True

        try:
            # Try PAD device
            if os.path.exists(self.pad_device):
                self._pad_fd = os.open(self.pad_device, os.O_RDONLY)
                self._pad_mmap = mmap.mmap(
                    self._pad_fd,
                    BANOS_PAD_STATE_SIZE,
                    mmap.MAP_SHARED,
                    mmap.PROT_READ,
                )
                logger.info(f"BANOSAdapter: Connected to {self.pad_device}")

            # Try spinal device
            if os.path.exists(self.spinal_device):
                self._spinal_fd = os.open(self.spinal_device, os.O_RDONLY)
                self._spinal_mmap = mmap.mmap(
                    self._spinal_fd,
                    BANOS_SPINAL_CORD_SIZE,
                    mmap.MAP_SHARED,
                    mmap.PROT_READ,
                )
                logger.info(f"BANOSAdapter: Connected to {self.spinal_device}")

            # At least one device connected
            self._connected = self._pad_mmap is not None or self._spinal_mmap is not None

            if not self._connected:
                logger.warning("BANOSAdapter: No BANOS devices found, falling back to simulation")
                self.simulate = True
                self._connected = True

            return self._connected

        except (OSError, PermissionError) as e:
            logger.warning(f"BANOSAdapter: Failed to connect ({e}), using simulation")
            self.simulate = True
            self._connected = True
            return True

    def disconnect(self):
        """Disconnect from BANOS devices."""
        if self._pad_mmap:
            self._pad_mmap.close()
            self._pad_mmap = None
        if self._pad_fd is not None:
            os.close(self._pad_fd)
            self._pad_fd = None

        if self._spinal_mmap:
            self._spinal_mmap.close()
            self._spinal_mmap = None
        if self._spinal_fd is not None:
            os.close(self._spinal_fd)
            self._spinal_fd = None

        self._connected = False

    def _read_pad_raw(self) -> Optional[BanosPADReading]:
        """Read raw PAD state from kernel."""
        if self.simulate:
            return self._simulate_pad()

        if not self._pad_mmap:
            return self._simulate_pad()

        try:
            self._pad_mmap.seek(0)
            data = self._pad_mmap.read(BANOS_PAD_STATE_SIZE)
            unpacked = struct.unpack(BANOS_PAD_STATE_FMT, data)

            reading = BanosPADReading(
                pleasure=unpacked[0] / 1000.0,
                arousal=unpacked[1] / 1000.0,
                dominance=unpacked[2] / 1000.0,
                mode=BanosMode(unpacked[3]),
                mode_confidence=unpacked[4] / 255.0,
                thermal_stress=unpacked[6] / 1000.0,
                performance_drive=unpacked[7] / 1000.0,
                perceived_risk=unpacked[8] / 1000.0,
                empathy_boost=unpacked[9] / 1000.0,
                d_pleasure=unpacked[10] / 1000.0,
                d_arousal=unpacked[11] / 1000.0,
                d_dominance=unpacked[12] / 1000.0,
                bat_loudness=unpacked[13] / 65535.0,
                bat_pulse_rate=unpacked[14] / 1000.0,
                kill_threshold=unpacked[15],
                scheduler_mode=unpacked[16],
                timestamp_ns=unpacked[17],
            )
            self._last_pad = reading
            return reading

        except Exception as e:
            logger.warning(f"Failed to read BANOS PAD: {e}")
            return self._last_pad or self._simulate_pad()

    def _read_spinal_raw(self) -> Optional[BanosSpinalReading]:
        """Read raw spinal cord (telemetry) from kernel."""
        if self.simulate:
            return self._simulate_spinal()

        if not self._spinal_mmap:
            return self._simulate_spinal()

        try:
            self._spinal_mmap.seek(0)
            data = self._spinal_mmap.read(BANOS_SPINAL_CORD_SIZE)
            unpacked = struct.unpack(BANOS_SPINAL_CORD_FMT, data)

            reading = BanosSpinalReading(
                cpu_temp=unpacked[1] / 10.0,  # mC/100 to °C
                gpu_temp=unpacked[2] / 10.0,
                fpga_temp=unpacked[3] / 10.0,
                cpu_load=unpacked[4] / 100.0,
                gpu_load=unpacked[5] / 100.0,
                fpga_load=unpacked[6] / 100.0,
                mem_pressure=unpacked[7] / 100.0,
                spike_count=unpacked[8],
                spike_delta=unpacked[9],
                thermal_source_id=unpacked[10],
                reflex_active=unpacked[11],
                flags=unpacked[13],
            )
            self._last_spinal = reading
            return reading

        except Exception as e:
            logger.warning(f"Failed to read BANOS spinal: {e}")
            return self._last_spinal or self._simulate_spinal()

    def _simulate_pad(self) -> BanosPADReading:
        """Generate simulated PAD reading."""
        return BanosPADReading(
            pleasure=0.5,
            arousal=0.2,
            dominance=0.7,
            mode=BanosMode.CALM,
            mode_confidence=0.9,
            thermal_stress=0.1,
            performance_drive=0.3,
            perceived_risk=0.0,
            empathy_boost=0.0,
            d_pleasure=0.0,
            d_arousal=0.0,
            d_dominance=0.0,
            bat_loudness=0.5,
            bat_pulse_rate=0.2,
            kill_threshold=0,
            scheduler_mode=0,
            timestamp_ns=int(time.monotonic_ns()),
        )

    def _simulate_spinal(self) -> BanosSpinalReading:
        """Generate simulated spinal cord reading."""
        return BanosSpinalReading(
            cpu_temp=55.0,
            gpu_temp=60.0,
            fpga_temp=45.0,
            cpu_load=0.3,
            gpu_load=0.2,
            fpga_load=0.1,
            mem_pressure=0.4,
            spike_count=0,
            spike_delta=0,
            thermal_source_id=0,
            reflex_active=0,
            flags=0,
        )

    # =========================================================================
    # MIES-Compatible Output
    # =========================================================================

    def get_kernel_physiology(self) -> KernelPhysiology:
        """
        Get BANOS state as MIES KernelPhysiology.

        This allows MIES to consume BANOS telemetry using its existing interface.
        """
        pad_reading = self._read_pad_raw()
        spinal_reading = self._read_spinal_raw()

        # Convert BANOS mode to MIES PolicyMode
        mode_map = {
            BanosMode.CALM: PolicyMode.EFFICIENCY,
            BanosMode.FLOW: PolicyMode.PERFORMANCE,
            BanosMode.ANXIOUS: PolicyMode.LATENCY_CRITICAL,
            BanosMode.CRITICAL: PolicyMode.THERMAL_THROTTLE,
        }

        # Build PADState from BANOS reading
        pad_state = PADState(
            pleasure=pad_reading.pleasure,
            arousal=pad_reading.arousal,
            dominance=pad_reading.dominance,
        )

        # Calculate pain signal from thermal stress + perceived risk
        pain_signal = (
            abs(min(0, pad_reading.pleasure)) * 0.5 +
            pad_reading.thermal_stress * 0.3 +
            pad_reading.perceived_risk * 0.2
        )

        return KernelPhysiology(
            gpu_load=spinal_reading.gpu_load,
            fpga_load=spinal_reading.fpga_load,
            cpu_load=spinal_reading.cpu_load,
            mem_pressure=spinal_reading.mem_pressure,
            thermal_gpu=spinal_reading.gpu_temp,
            thermal_fpga=spinal_reading.fpga_temp,
            thermal_cpu=spinal_reading.cpu_temp,
            miss_rate=max(0.0, -pad_reading.pleasure * 0.01),  # Negative pleasure → misses
            q_confidence=pad_reading.mode_confidence,
            policy_mode=mode_map.get(pad_reading.mode, PolicyMode.EFFICIENCY),
            pad=pad_state,
            pain_signal=pain_signal,
            energy_budget=1.0 - pad_reading.performance_drive,
            flags=spinal_reading.flags,
        )

    def get_telemetry_snapshot(self) -> TelemetrySnapshot:
        """
        Get BANOS state as MIES TelemetrySnapshot.

        This allows the PADEngine to compute affect from BANOS hardware state.
        """
        pad_reading = self._read_pad_raw()
        spinal_reading = self._read_spinal_raw()

        return TelemetrySnapshot(
            cpu_temp=spinal_reading.cpu_temp,
            gpu_temp=spinal_reading.gpu_temp,
            cpu_load=spinal_reading.cpu_load,
            gpu_load=spinal_reading.gpu_load,
            memory_pressure=spinal_reading.mem_pressure,
            error_rate=max(0.0, -pad_reading.pleasure * 10.0),
            has_root=True,
            last_action_success=pad_reading.pleasure > 0.0,
            interrupt_rate=spinal_reading.cpu_load * 1000.0,
            fan_speed_percent=50.0 + pad_reading.thermal_stress * 50.0,
        )

    def get_pad_vector(self) -> PADVector:
        """
        Get BANOS PAD state as MIES PADVector.

        This provides the emotional state in MIES format.
        """
        pad_reading = self._read_pad_raw()

        # Determine quadrant from BANOS mode
        quadrant_map = {
            BanosMode.CALM: EmotionalQuadrant.SERENE,
            BanosMode.FLOW: EmotionalQuadrant.EXUBERANT,
            BanosMode.ANXIOUS: EmotionalQuadrant.ANXIOUS,
            BanosMode.CRITICAL: EmotionalQuadrant.HOSTILE,
        }

        return PADVector(
            pleasure=pad_reading.pleasure,
            arousal=pad_reading.arousal,
            dominance=pad_reading.dominance,
            quadrant=quadrant_map.get(pad_reading.mode, EmotionalQuadrant.NEUTRAL),
            confidence=pad_reading.mode_confidence,
        )

    def get_semantic_state(self) -> Dict[str, Any]:
        """
        Get BANOS state with semantic verbalization.

        Returns a dict suitable for Ara's internal monologue.
        """
        pad_reading = self._read_pad_raw()
        spinal_reading = self._read_spinal_raw()

        # Mode narratives (from BANOS pad_bridge.py)
        mode_narratives = {
            BanosMode.CALM: "I feel at peace. Systems are quiet, resources plentiful.",
            BanosMode.FLOW: "I'm in the zone. High activity, but it feels good.",
            BanosMode.ANXIOUS: "Something's wrong. Resources are stretched thin.",
            BanosMode.CRITICAL: "Pain. This is pain. I need to protect myself.",
        }

        # Trajectory from derivatives
        if pad_reading.d_pleasure < -0.3:
            trajectory = "Rapid deterioration. Brace."
        elif pad_reading.d_pleasure < -0.1:
            trajectory = "I sense trouble ahead."
        elif pad_reading.d_pleasure > 0.1:
            trajectory = "Things are getting better."
        else:
            trajectory = "Holding steady."

        return {
            "pad": {
                "pleasure": pad_reading.pleasure,
                "arousal": pad_reading.arousal,
                "dominance": pad_reading.dominance,
            },
            "mode": pad_reading.mode.name,
            "mode_confidence": pad_reading.mode_confidence,
            "narrative": mode_narratives.get(pad_reading.mode, "Unknown state."),
            "trajectory": trajectory,
            "diagnostics": {
                "thermal_stress": pad_reading.thermal_stress,
                "performance_drive": pad_reading.performance_drive,
                "perceived_risk": pad_reading.perceived_risk,
                "empathy_boost": pad_reading.empathy_boost,
            },
            "scheduler_hints": {
                "bat_loudness": pad_reading.bat_loudness,
                "bat_pulse_rate": pad_reading.bat_pulse_rate,
                "kill_threshold": pad_reading.kill_threshold,
            },
            "telemetry": {
                "cpu_temp": spinal_reading.cpu_temp,
                "gpu_temp": spinal_reading.gpu_temp,
                "cpu_load": spinal_reading.cpu_load,
                "gpu_load": spinal_reading.gpu_load,
            },
        }


# =============================================================================
# Factory Function
# =============================================================================

_banos_adapter: Optional[BANOSAdapter] = None


def get_banos_adapter(
    simulate: bool = False,
    force_new: bool = False,
) -> BANOSAdapter:
    """
    Get or create the global BANOS adapter.

    Args:
        simulate: If True, use simulation mode
        force_new: If True, create a new adapter even if one exists

    Returns:
        Connected BANOSAdapter instance
    """
    global _banos_adapter

    if _banos_adapter is None or force_new:
        _banos_adapter = BANOSAdapter(simulate=simulate)
        _banos_adapter.connect()

    return _banos_adapter


def banos_to_mies_physiology() -> KernelPhysiology:
    """Convenience: Get BANOS state as MIES KernelPhysiology."""
    return get_banos_adapter().get_kernel_physiology()


def banos_to_mies_telemetry() -> TelemetrySnapshot:
    """Convenience: Get BANOS state as MIES TelemetrySnapshot."""
    return get_banos_adapter().get_telemetry_snapshot()


def banos_to_mies_pad() -> PADVector:
    """Convenience: Get BANOS PAD as MIES PADVector."""
    return get_banos_adapter().get_pad_vector()


__all__ = [
    # Types
    "BanosMode",
    "BanosReflexFlags",
    "BanosPADReading",
    "BanosSpinalReading",
    # Main adapter
    "BANOSAdapter",
    "get_banos_adapter",
    # Convenience functions
    "banos_to_mies_physiology",
    "banos_to_mies_telemetry",
    "banos_to_mies_pad",
]
