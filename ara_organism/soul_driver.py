# ara_organism/soul_driver.py
"""
FPGASoulDriver - Optimized 5 kHz Hardware Interface
====================================================

Tightened hardware loop for real FPGA performance:
- Map once, reuse always with memoryview
- Pre-packed buffers for CSR writes
- struct.unpack_from for fast status reads
- Zero Python loops in critical path

This driver is called from a dedicated OS thread at 5 kHz,
NOT from asyncio. The soul heartbeat must not fight the
event loop scheduler.

Register Map (Stratix-10 AraHD Accelerator):
    0x0000: CTRL_STATUS      - Control/status register
    0x0004: SOUL_MODE        - Soul operating mode
    0x0008: HV_INJ_CTRL      - HV injection control
    0x000C: HV_INJ_DATA      - HV injection data start
    0x0100: METRICS_BASE     - Metrics read-back base
    0x0104: RESONANCE        - Current resonance score
    0x0108: FATIGUE          - Current fatigue level
    0x010C: TEMPERATURE      - FPGA junction temperature
    0x0110: TICK_COUNT       - Hardware tick counter

Usage:
    driver = FPGASoulDriver("/dev/ara_fpga0")
    driver.connect()

    # In dedicated 5 kHz thread:
    while running:
        metrics = driver.tick(mode, hv_injection)
        state_manager.update_soul(metrics)
        time.sleep(0.0002)  # 200 µs
"""

from __future__ import annotations

import mmap
import struct
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

log = logging.getLogger("Ara.SoulDriver")


# =============================================================================
# Register Definitions
# =============================================================================

class SoulRegisters:
    """FPGA register addresses for soul accelerator."""

    # Control registers
    CTRL_STATUS     = 0x0000
    SOUL_MODE       = 0x0004
    HV_INJ_CTRL     = 0x0008
    HV_INJ_DATA     = 0x000C

    # Metrics registers (read-only)
    METRICS_BASE    = 0x0100
    RESONANCE       = 0x0104
    FATIGUE         = 0x0108
    TEMPERATURE     = 0x010C
    TICK_COUNT      = 0x0110

    # HV injection buffer
    HV_BUFFER_SIZE  = 128  # 128 bytes = 64 x 16-bit HV components


class SoulMode:
    """Soul operating modes."""
    IDLE        = 0
    STANDBY     = 1
    ACTIVE      = 2
    RESONANCE   = 3
    RECOVERY    = 4


class StatusBits:
    """Status register bit definitions."""
    READY       = 0x01
    BUSY        = 0x02
    ERROR       = 0x04
    THERMAL_OK  = 0x08
    HV_VALID    = 0x10


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SoulMetrics:
    """Metrics from a single soul tick."""
    tick: int
    resonance: float          # [0, 1] current resonance score
    fatigue: float            # [0, 1] fatigue level
    temperature_c: float      # FPGA junction temperature
    status: int               # Raw status register
    latency_us: float         # Tick latency in microseconds

    @property
    def is_ready(self) -> bool:
        return bool(self.status & StatusBits.READY)

    @property
    def is_thermal_ok(self) -> bool:
        return bool(self.status & StatusBits.THERMAL_OK)

    @property
    def has_error(self) -> bool:
        return bool(self.status & StatusBits.ERROR)


# =============================================================================
# FPGA Soul Driver
# =============================================================================

class FPGASoulDriver:
    """
    Optimized FPGA driver for 5 kHz soul loop.

    Key optimizations:
        1. Map once, reuse always - single mmap for lifetime
        2. Pre-bound memoryviews for each register to avoid slicing
        3. struct.unpack_from on pre-bound view (no allocation)
        4. Pre-packed HV buffer - just memcpy, no Python loops

    The 200 µs tick budget breaks down as:
        - 10-20 µs: Python overhead (this driver)
        - 150+ µs:  FPGA computation
        - 20-30 µs: Status read-back
    """

    # mmap size: 4 KB covers all registers
    MMAP_SIZE = 4096

    def __init__(
        self,
        device_path: str = "/dev/ara_fpga0",
        mock_mode: bool = False,
    ):
        """
        Initialize the soul driver.

        Args:
            device_path: Path to FPGA device file
            mock_mode: If True, simulate hardware
        """
        self.device_path = device_path
        self.mock_mode = mock_mode

        # mmap and memoryviews (initialized on connect)
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None

        # Pre-bound memoryviews for fast access
        self._status_view: Optional[memoryview] = None
        self._mode_view: Optional[memoryview] = None
        self._resonance_view: Optional[memoryview] = None
        self._fatigue_view: Optional[memoryview] = None
        self._temp_view: Optional[memoryview] = None
        self._tick_view: Optional[memoryview] = None
        self._hv_ctrl_view: Optional[memoryview] = None
        self._hv_data_view: Optional[memoryview] = None

        # Pre-packed HV buffer (reused each tick)
        self._hv_buffer = bytearray(SoulRegisters.HV_BUFFER_SIZE)

        # State
        self._connected = False
        self._tick_count = 0

        # Mock state
        self._mock_resonance = 0.5
        self._mock_fatigue = 0.0

    # =========================================================================
    # Connection
    # =========================================================================

    def connect(self) -> bool:
        """
        Connect to FPGA and set up memory mappings.

        Returns:
            True if connected successfully
        """
        if self._connected:
            return True

        if self.mock_mode:
            log.info("SoulDriver: Running in mock mode")
            self._connected = True
            return True

        try:
            # Open device file
            import os
            self._fd = os.open(self.device_path, os.O_RDWR | os.O_SYNC)

            # Create mmap
            self._mm = mmap.mmap(
                self._fd,
                self.MMAP_SIZE,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )

            # Create memoryviews for fast register access
            self._bind_views()

            self._connected = True
            log.info("SoulDriver: Connected to %s", self.device_path)

            # Read initial status
            status = self._read_status()
            log.info("SoulDriver: Initial status=0x%04x", status)

            return True

        except FileNotFoundError:
            log.warning("SoulDriver: Device not found, falling back to mock mode")
            self.mock_mode = True
            self._connected = True
            return True

        except Exception as e:
            log.error("SoulDriver: Connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from FPGA."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None

        if self._fd is not None:
            import os
            os.close(self._fd)
            self._fd = None

        self._connected = False
        log.info("SoulDriver: Disconnected")

    def _bind_views(self) -> None:
        """Bind memoryviews to register locations."""
        mm = self._mm

        # Control registers (read/write)
        self._status_view = memoryview(mm)[
            SoulRegisters.CTRL_STATUS:SoulRegisters.CTRL_STATUS + 4
        ]
        self._mode_view = memoryview(mm)[
            SoulRegisters.SOUL_MODE:SoulRegisters.SOUL_MODE + 4
        ]
        self._hv_ctrl_view = memoryview(mm)[
            SoulRegisters.HV_INJ_CTRL:SoulRegisters.HV_INJ_CTRL + 4
        ]
        self._hv_data_view = memoryview(mm)[
            SoulRegisters.HV_INJ_DATA:SoulRegisters.HV_INJ_DATA + SoulRegisters.HV_BUFFER_SIZE
        ]

        # Metrics registers (read-only)
        self._resonance_view = memoryview(mm)[
            SoulRegisters.RESONANCE:SoulRegisters.RESONANCE + 4
        ]
        self._fatigue_view = memoryview(mm)[
            SoulRegisters.FATIGUE:SoulRegisters.FATIGUE + 4
        ]
        self._temp_view = memoryview(mm)[
            SoulRegisters.TEMPERATURE:SoulRegisters.TEMPERATURE + 4
        ]
        self._tick_view = memoryview(mm)[
            SoulRegisters.TICK_COUNT:SoulRegisters.TICK_COUNT + 4
        ]

    # =========================================================================
    # Fast Register Access
    # =========================================================================

    def _read_status(self) -> int:
        """Read status register (fast path)."""
        if self.mock_mode:
            return StatusBits.READY | StatusBits.THERMAL_OK

        # struct.unpack_from on pre-bound view - no allocation
        return struct.unpack_from('<I', self._status_view, 0)[0]

    def _read_resonance(self) -> float:
        """Read resonance register (Q16.16 fixed-point)."""
        if self.mock_mode:
            return self._mock_resonance

        raw = struct.unpack_from('<I', self._resonance_view, 0)[0]
        return raw / 65536.0

    def _read_fatigue(self) -> float:
        """Read fatigue register (Q16.16 fixed-point)."""
        if self.mock_mode:
            return self._mock_fatigue

        raw = struct.unpack_from('<I', self._fatigue_view, 0)[0]
        return raw / 65536.0

    def _read_temperature(self) -> float:
        """Read temperature register (Q8.8 fixed-point)."""
        if self.mock_mode:
            return 45.0

        raw = struct.unpack_from('<H', self._temp_view, 0)[0]
        return raw / 256.0

    def _read_tick_count(self) -> int:
        """Read hardware tick counter."""
        if self.mock_mode:
            return self._tick_count

        return struct.unpack_from('<I', self._tick_view, 0)[0]

    def _write_mode(self, mode: int) -> None:
        """Write soul mode register."""
        if self.mock_mode:
            return

        struct.pack_into('<I', self._mode_view, 0, mode)

    def _write_hv(self, hv_data: bytes) -> None:
        """
        Write HV injection data (fast bulk write).

        Args:
            hv_data: Pre-packed HV data (up to 128 bytes)
        """
        if self.mock_mode:
            return

        # Direct memcpy into mmap - no Python loop
        size = min(len(hv_data), SoulRegisters.HV_BUFFER_SIZE)
        self._hv_data_view[:size] = hv_data[:size]

        # Trigger HV injection
        struct.pack_into('<I', self._hv_ctrl_view, 0, 0x01)

    # =========================================================================
    # Main Tick Function
    # =========================================================================

    def tick(
        self,
        mode: int = SoulMode.ACTIVE,
        hv_injection: Optional[bytes] = None,
    ) -> SoulMetrics:
        """
        Execute one soul tick.

        This is the hot path called at 5 kHz from a dedicated OS thread.
        Must complete in <50 µs Python overhead.

        Args:
            mode: Soul operating mode
            hv_injection: Optional HV data to inject

        Returns:
            SoulMetrics from this tick
        """
        tick_start = time.perf_counter()

        # Write mode
        self._write_mode(mode)

        # Inject HV if provided
        if hv_injection is not None:
            self._write_hv(hv_injection)

        # Read metrics (fast path)
        status = self._read_status()
        resonance = self._read_resonance()
        fatigue = self._read_fatigue()
        temperature = self._read_temperature()
        hw_tick = self._read_tick_count()

        # Update mock state
        if self.mock_mode:
            self._tick_count += 1
            # Simulate resonance decay and fatigue accumulation
            self._mock_resonance = max(0.0, min(1.0, self._mock_resonance + 0.001 * (0.5 - self._mock_resonance)))
            self._mock_fatigue = min(1.0, self._mock_fatigue + 0.0001)

        tick_end = time.perf_counter()
        latency_us = (tick_end - tick_start) * 1e6

        return SoulMetrics(
            tick=hw_tick if not self.mock_mode else self._tick_count,
            resonance=resonance,
            fatigue=fatigue,
            temperature_c=temperature,
            status=status,
            latency_us=latency_us,
        )

    def reset_fatigue(self) -> None:
        """Reset fatigue counter (recovery mode)."""
        if self.mock_mode:
            self._mock_fatigue = 0.0
        else:
            # Write reset command
            struct.pack_into('<I', self._status_view, 0, 0x100)  # Reset bit

    def get_stats(self) -> Dict[str, Any]:
        """Get driver statistics."""
        return {
            "connected": self._connected,
            "mock_mode": self.mock_mode,
            "device_path": self.device_path,
            "tick_count": self._tick_count if self.mock_mode else self._read_tick_count(),
        }


# =============================================================================
# Convenience
# =============================================================================

_default_driver: Optional[FPGASoulDriver] = None


def get_soul_driver(mock_mode: bool = True) -> FPGASoulDriver:
    """Get the default soul driver."""
    global _default_driver
    if _default_driver is None:
        _default_driver = FPGASoulDriver(mock_mode=mock_mode)
        _default_driver.connect()
    return _default_driver


__all__ = [
    'SoulRegisters',
    'SoulMode',
    'StatusBits',
    'SoulMetrics',
    'FPGASoulDriver',
    'get_soul_driver',
]
