"""
Ara Somatic HAL - Python Interface
===================================

Provides zero-copy access to Ara's unified somatic state via shared memory.

Usage:
    from banos.hal import AraSomatic

    # Open shared memory
    with AraSomatic() as som:
        # Read state
        print(f"Pain level: {som.pain_level}")
        print(f"CPU temp: {som.cpu_temp_c}°C")
        print(f"System state: {som.state_name}")

        # Write control flags
        som.set_avatar_mode("low")
"""

import ctypes
import mmap
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import struct
import logging

logger = logging.getLogger(__name__)

# Constants (must match ara_somatic.h)
ARA_SOMATIC_MAGIC = 0x41524153  # 'ARAS'
ARA_SOMATIC_VERSION = 2
ARA_SOMATIC_SHM_PATH = "/ara_somatic"
ARA_SOMATIC_SHM_SIZE = 4096


class AraSystemState(IntEnum):
    """System state enumeration."""
    IDLE = 0
    NORMAL = 1
    HIGH_LOAD = 2
    CRITICAL = 3
    EMERGENCY = 4
    SLEEPING = 5


class AraAvatarMode(IntEnum):
    """Avatar mode enumeration."""
    OFF = 0
    AUDIO_ONLY = 1
    LOW_RES = 2
    STANDARD = 3
    HIGH_RES = 4


class AraPadQuadrant(IntEnum):
    """PAD quadrant enumeration."""
    SERENE = 0      # +P, -A: Calm and happy
    EXCITED = 1     # +P, +A: Happy and busy
    ANXIOUS = 2     # -P, +A: Stressed and busy
    DEPRESSED = 3   # -P, -A: Stressed and idle
    DOMINANT = 4    # High D: Resources abundant
    SUBMISSIVE = 5  # Low D: Resource constrained
    EMERGENCY = 6   # Critical state


# Structure offsets (computed from C struct layout)
# Header: magic(4) + version(4) + timestamp(8) + update_count(8) + state(1) + health(1) + reserved(6) = 32 bytes
OFFSET_MAGIC = 0
OFFSET_VERSION = 4
OFFSET_TIMESTAMP = 8
OFFSET_UPDATE_COUNT = 16
OFFSET_SYSTEM_STATE = 24
OFFSET_HEALTH_SCORE = 25

# FPGA state starts at offset 32
OFFSET_FPGA = 32
# FPGA: neural_state(4) + pain_level(4) + reflex_log(4) + total_spikes(4) +
#       pleasure(2) + arousal(2) + dominance(2) + quadrant(1) + sched_mode(1) +
#       active_neurons(4) + fabric_temp(4) + fabric_online(1) + reserved(3) = 36 bytes

# System metrics starts after FPGA (32 + 36 = 68)
OFFSET_SYS = 68
# sys: cpu_avg(4) + cpu_max(4) + cpu_temp(4) + cpu_freq(4) +
#      gpu_util(4) + gpu_vram_used(4) + gpu_vram_total(4) + gpu_temp(4) + gpu_power(4) +
#      ram_used(4) + ram_total(4) + swap_used(4) +
#      disk_read(4) + disk_write(4) + net_rx(4) + net_tx(4) = 64 bytes

# Avatar state starts after sys (68 + 64 = 132)
OFFSET_AVATAR = 132
# avatar: fps(4) + width(2) + height(2) + mode(1) + audio_active(1) + face_detected(1) + lips_synced(1) +
#         flow_x(4) + flow_y(4) + flow_mag(4) + frame_shm_offset(8) + frame_size(4) + frame_seq(4) = 40 bytes

# Control flags start after avatar (132 + 40 = 172)
OFFSET_CONTROL = 172


@dataclass
class AraSomaticSnapshot:
    """Immutable snapshot of somatic state."""
    # Header
    magic: int
    version: int
    timestamp_ns: int
    update_count: int
    system_state: AraSystemState
    health_score: int

    # FPGA
    neural_state: int
    pain_level: int
    reflex_log: int
    total_spikes: int
    pleasure: int
    arousal: int
    dominance: int
    pad_quadrant: AraPadQuadrant
    sched_mode: int
    active_neurons: int
    fabric_temp_c: int
    fabric_online: bool

    # System
    cpu_avg_pct: float
    cpu_max_pct: float
    cpu_temp_c: float
    cpu_freq_mhz: int
    gpu_util_pct: float
    gpu_vram_used_gb: float
    gpu_vram_total_gb: float
    gpu_temp_c: float
    gpu_power_w: float
    ram_used_gb: float
    ram_total_gb: float
    swap_used_gb: float
    disk_read_mbps: float
    disk_write_mbps: float
    net_rx_mbps: float
    net_tx_mbps: float

    # Avatar
    avatar_fps: float
    frame_width: int
    frame_height: int
    avatar_mode: AraAvatarMode
    audio_active: bool
    face_detected: bool
    lips_synced: bool
    flow_x: float
    flow_y: float
    flow_magnitude: float

    @property
    def state_name(self) -> str:
        """Get human-readable system state."""
        return self.system_state.name.lower()

    @property
    def quadrant_name(self) -> str:
        """Get human-readable PAD quadrant."""
        return self.pad_quadrant.name.lower()

    @property
    def pain_normalized(self) -> float:
        """Get pain level as 0.0-1.0."""
        return self.pain_level / 4294967295.0

    @property
    def pleasure_normalized(self) -> float:
        """Get pleasure as -1.0 to 1.0."""
        return self.pleasure / 256.0

    @property
    def arousal_normalized(self) -> float:
        """Get arousal as -1.0 to 1.0."""
        return self.arousal / 256.0

    @property
    def dominance_normalized(self) -> float:
        """Get dominance as -1.0 to 1.0."""
        return self.dominance / 256.0


class AraSomatic:
    """
    Python interface to Ara's unified somatic shared memory.

    Provides zero-copy read access and controlled write access
    to Ara's complete hardware state.
    """

    def __init__(self, readonly: bool = False):
        """
        Initialize somatic interface.

        Args:
            readonly: If True, open in read-only mode
        """
        self._readonly = readonly
        self._fd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None
        self._is_open = False

    def open(self) -> None:
        """Open the shared memory region."""
        if self._is_open:
            return

        shm_path = f"/dev/shm{ARA_SOMATIC_SHM_PATH}"

        try:
            if self._readonly:
                self._fd = os.open(shm_path, os.O_RDONLY)
                self._mmap = mmap.mmap(self._fd, ARA_SOMATIC_SHM_SIZE,
                                       access=mmap.ACCESS_READ)
            else:
                # Create if doesn't exist
                flags = os.O_RDWR | os.O_CREAT
                self._fd = os.open(shm_path, flags, 0o666)
                # Ensure size
                os.ftruncate(self._fd, ARA_SOMATIC_SHM_SIZE)
                self._mmap = mmap.mmap(self._fd, ARA_SOMATIC_SHM_SIZE,
                                       access=mmap.ACCESS_WRITE)

                # Initialize if invalid
                magic = struct.unpack_from('<I', self._mmap, OFFSET_MAGIC)[0]
                if magic != ARA_SOMATIC_MAGIC:
                    self._initialize()

            self._is_open = True
            logger.debug("AraSomatic opened")

        except FileNotFoundError:
            raise RuntimeError(f"Somatic SHM not found: {shm_path}. Is ara_daemon running?")
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to open somatic SHM: {e}")

    def close(self) -> None:
        """Close the shared memory region."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._is_open = False

    def __enter__(self) -> 'AraSomatic':
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _initialize(self) -> None:
        """Initialize the shared memory with defaults."""
        if self._readonly or not self._mmap:
            return

        # Zero the memory
        self._mmap.seek(0)
        self._mmap.write(b'\x00' * ARA_SOMATIC_SHM_SIZE)

        # Write header
        struct.pack_into('<II', self._mmap, OFFSET_MAGIC,
                         ARA_SOMATIC_MAGIC, ARA_SOMATIC_VERSION)

        # Touch to set timestamp
        self.touch()

        # Set defaults
        struct.pack_into('<BB', self._mmap, OFFSET_SYSTEM_STATE,
                         AraSystemState.IDLE, 100)  # state, health

        # Control defaults
        struct.pack_into('<BBBBBBxxfff', self._mmap, OFFSET_CONTROL,
                         AraAvatarMode.STANDARD,  # requested_avatar_mode
                         1,  # requested_sim_detail
                         0,  # force_low_power
                         0,  # emergency_stop
                         0,  # avatar_acknowledged
                         0,  # fabric_acknowledged
                         85.0,  # critical_temp_c
                         80.0,  # high_load_threshold
                         2.0)   # low_vram_threshold_gb

        logger.info("AraSomatic initialized")

    def touch(self) -> None:
        """Update timestamp and increment update counter."""
        if self._readonly or not self._mmap:
            return

        import time
        timestamp_ns = int(time.monotonic_ns())
        update_count = struct.unpack_from('<Q', self._mmap, OFFSET_UPDATE_COUNT)[0]

        struct.pack_into('<QQ', self._mmap, OFFSET_TIMESTAMP,
                         timestamp_ns, update_count + 1)

    def snapshot(self) -> AraSomaticSnapshot:
        """
        Get an immutable snapshot of the current state.

        Returns:
            AraSomaticSnapshot with all current values
        """
        if not self._mmap:
            raise RuntimeError("Somatic not open")

        m = self._mmap

        # Header
        magic, version = struct.unpack_from('<II', m, OFFSET_MAGIC)
        timestamp_ns, update_count = struct.unpack_from('<QQ', m, OFFSET_TIMESTAMP)
        system_state, health_score = struct.unpack_from('<BB', m, OFFSET_SYSTEM_STATE)

        # FPGA state
        fpga = struct.unpack_from('<IIIIhhhBBIIBxxx', m, OFFSET_FPGA)
        (neural_state, pain_level, reflex_log, total_spikes,
         pleasure, arousal, dominance, quadrant, sched_mode,
         active_neurons, fabric_temp, fabric_online) = fpga

        # System metrics
        sys_metrics = struct.unpack_from('<16f', m, OFFSET_SYS)
        (cpu_avg, cpu_max, cpu_temp, cpu_freq_f,
         gpu_util, gpu_vram_used, gpu_vram_total, gpu_temp, gpu_power,
         ram_used, ram_total, swap_used,
         disk_read, disk_write, net_rx, net_tx) = sys_metrics

        # Avatar state
        avatar = struct.unpack_from('<fHHBBBBfffQII', m, OFFSET_AVATAR)
        (avatar_fps, frame_width, frame_height, avatar_mode,
         audio_active, face_detected, lips_synced,
         flow_x, flow_y, flow_mag,
         frame_offset, frame_size, frame_seq) = avatar

        return AraSomaticSnapshot(
            magic=magic,
            version=version,
            timestamp_ns=timestamp_ns,
            update_count=update_count,
            system_state=AraSystemState(system_state),
            health_score=health_score,

            neural_state=neural_state,
            pain_level=pain_level,
            reflex_log=reflex_log,
            total_spikes=total_spikes,
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            pad_quadrant=AraPadQuadrant(quadrant),
            sched_mode=sched_mode,
            active_neurons=active_neurons,
            fabric_temp_c=fabric_temp,
            fabric_online=bool(fabric_online),

            cpu_avg_pct=cpu_avg,
            cpu_max_pct=cpu_max,
            cpu_temp_c=cpu_temp,
            cpu_freq_mhz=int(cpu_freq_f),
            gpu_util_pct=gpu_util,
            gpu_vram_used_gb=gpu_vram_used,
            gpu_vram_total_gb=gpu_vram_total,
            gpu_temp_c=gpu_temp,
            gpu_power_w=gpu_power,
            ram_used_gb=ram_used,
            ram_total_gb=ram_total,
            swap_used_gb=swap_used,
            disk_read_mbps=disk_read,
            disk_write_mbps=disk_write,
            net_rx_mbps=net_rx,
            net_tx_mbps=net_tx,

            avatar_fps=avatar_fps,
            frame_width=frame_width,
            frame_height=frame_height,
            avatar_mode=AraAvatarMode(avatar_mode),
            audio_active=bool(audio_active),
            face_detected=bool(face_detected),
            lips_synced=bool(lips_synced),
            flow_x=flow_x,
            flow_y=flow_y,
            flow_magnitude=flow_mag,
        )

    # =========================================================================
    # Convenience Properties (direct reads without full snapshot)
    # =========================================================================

    @property
    def pain_level(self) -> int:
        """Get current pain level (32-bit)."""
        if not self._mmap:
            return 0
        return struct.unpack_from('<I', self._mmap, OFFSET_FPGA + 4)[0]

    @property
    def cpu_temp_c(self) -> float:
        """Get CPU temperature."""
        if not self._mmap:
            return 0.0
        return struct.unpack_from('<f', self._mmap, OFFSET_SYS + 8)[0]

    @property
    def gpu_temp_c(self) -> float:
        """Get GPU temperature."""
        if not self._mmap:
            return 0.0
        return struct.unpack_from('<f', self._mmap, OFFSET_SYS + 28)[0]

    @property
    def system_state(self) -> AraSystemState:
        """Get current system state."""
        if not self._mmap:
            return AraSystemState.IDLE
        return AraSystemState(struct.unpack_from('<B', self._mmap, OFFSET_SYSTEM_STATE)[0])

    @property
    def state_name(self) -> str:
        """Get human-readable system state."""
        return self.system_state.name.lower()

    # =========================================================================
    # Write Methods (for updaters)
    # =========================================================================

    def update_fpga(self, neural_state: int = None, pain_level: int = None,
                    reflex_log: int = None, pleasure: int = None,
                    arousal: int = None, dominance: int = None,
                    quadrant: int = None, active_neurons: int = None,
                    fabric_online: bool = None) -> None:
        """Update FPGA state values."""
        if self._readonly or not self._mmap:
            return

        if neural_state is not None:
            struct.pack_into('<I', self._mmap, OFFSET_FPGA, neural_state)
        if pain_level is not None:
            struct.pack_into('<I', self._mmap, OFFSET_FPGA + 4, pain_level)
        if reflex_log is not None:
            struct.pack_into('<I', self._mmap, OFFSET_FPGA + 8, reflex_log)
        if pleasure is not None:
            struct.pack_into('<h', self._mmap, OFFSET_FPGA + 16, pleasure)
        if arousal is not None:
            struct.pack_into('<h', self._mmap, OFFSET_FPGA + 18, arousal)
        if dominance is not None:
            struct.pack_into('<h', self._mmap, OFFSET_FPGA + 20, dominance)
        if quadrant is not None:
            struct.pack_into('<B', self._mmap, OFFSET_FPGA + 22, quadrant)
        if active_neurons is not None:
            struct.pack_into('<I', self._mmap, OFFSET_FPGA + 24, active_neurons)
        if fabric_online is not None:
            struct.pack_into('<B', self._mmap, OFFSET_FPGA + 32, 1 if fabric_online else 0)

        self.touch()

    def update_system(self, cpu_avg: float = None, cpu_temp: float = None,
                      gpu_util: float = None, gpu_vram_used: float = None,
                      gpu_temp: float = None, gpu_power: float = None,
                      ram_used: float = None, ram_total: float = None) -> None:
        """Update system metrics."""
        if self._readonly or not self._mmap:
            return

        if cpu_avg is not None:
            struct.pack_into('<f', self._mmap, OFFSET_SYS, cpu_avg)
        if cpu_temp is not None:
            struct.pack_into('<f', self._mmap, OFFSET_SYS + 8, cpu_temp)
        if gpu_util is not None:
            struct.pack_into('<f', self._mmap, OFFSET_SYS + 16, gpu_util)
        if gpu_vram_used is not None:
            struct.pack_into('<f', self._mmap, OFFSET_SYS + 20, gpu_vram_used)
        if gpu_temp is not None:
            struct.pack_into('<f', self._mmap, OFFSET_SYS + 28, gpu_temp)
        if gpu_power is not None:
            struct.pack_into('<f', self._mmap, OFFSET_SYS + 32, gpu_power)
        if ram_used is not None:
            struct.pack_into('<f', self._mmap, OFFSET_SYS + 36, ram_used)
        if ram_total is not None:
            struct.pack_into('<f', self._mmap, OFFSET_SYS + 40, ram_total)

        self.touch()

    def update_avatar(self, fps: float = None, mode: AraAvatarMode = None,
                      flow_x: float = None, flow_y: float = None) -> None:
        """Update avatar state."""
        if self._readonly or not self._mmap:
            return

        if fps is not None:
            struct.pack_into('<f', self._mmap, OFFSET_AVATAR, fps)
        if mode is not None:
            struct.pack_into('<B', self._mmap, OFFSET_AVATAR + 8, mode)
        if flow_x is not None:
            struct.pack_into('<f', self._mmap, OFFSET_AVATAR + 12, flow_x)
        if flow_y is not None:
            struct.pack_into('<f', self._mmap, OFFSET_AVATAR + 16, flow_y)

        self.touch()

    def set_avatar_mode(self, mode: str) -> None:
        """
        Request a specific avatar mode.

        Args:
            mode: One of "off", "audio", "low", "standard", "high"
        """
        if self._readonly or not self._mmap:
            return

        mode_map = {
            "off": AraAvatarMode.OFF,
            "audio": AraAvatarMode.AUDIO_ONLY,
            "low": AraAvatarMode.LOW_RES,
            "standard": AraAvatarMode.STANDARD,
            "high": AraAvatarMode.HIGH_RES,
        }

        mode_val = mode_map.get(mode.lower(), AraAvatarMode.STANDARD)
        struct.pack_into('<B', self._mmap, OFFSET_CONTROL, mode_val)
        self.touch()

    def set_emergency_stop(self, stop: bool) -> None:
        """Set emergency stop flag."""
        if self._readonly or not self._mmap:
            return

        struct.pack_into('<B', self._mmap, OFFSET_CONTROL + 3, 1 if stop else 0)
        self.touch()


# Convenience functions
def get_somatic_snapshot() -> AraSomaticSnapshot:
    """Get a quick snapshot of somatic state."""
    with AraSomatic(readonly=True) as som:
        return som.snapshot()


def get_pain_level() -> int:
    """Get current pain level."""
    with AraSomatic(readonly=True) as som:
        return som.pain_level


# Also export high-performance HAL with posix_ipc
try:
    from .ara_hal import (
        AraHAL,
        create_somatic_bus,
        connect_somatic_bus,
        read_pain as read_pain_hal,
        read_pad,
        DREAM_AWAKE,
        DREAM_REM,
        DREAM_DEEP,
    )
    _HAL_EXPORTS = [
        'AraHAL',
        'create_somatic_bus',
        'connect_somatic_bus',
        'read_pain_hal',
        'read_pad',
        'DREAM_AWAKE',
        'DREAM_REM',
        'DREAM_DEEP',
    ]
except ImportError:
    _HAL_EXPORTS = []


__all__ = [
    'AraSomatic',
    'AraSomaticSnapshot',
    'AraSystemState',
    'AraAvatarMode',
    'AraPadQuadrant',
    'get_somatic_snapshot',
    'get_pain_level',
] + _HAL_EXPORTS
