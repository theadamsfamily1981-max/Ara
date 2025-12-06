"""
Ara Hardware Abstraction Layer - High-Performance Somatic Bus
=============================================================

This is the BRAINSTEM - the biological bus that fuses all of Ara's
organs into a single nervous system.

Performance:
    Write latency: ~0.1µs (direct memory copy)
    Read latency:  ~0.0µs (pointer dereference)
    Format: Binary C-Struct (no JSON, no parsing)

The HAL creates a 4KB shared memory region (/dev/shm/ara_somatic) that
represents the ENTIRE physical state of the organism. Any process
(C, Python, Rust, the shader, the LLM) can access it in nanoseconds.

Usage:
    # Creator (daemon) - run once at system startup
    hal = AraHAL(create=True)

    # Writers (sensors)
    hal.write_somatic(pad=(0.5, 0.3, 0.7), pain=0.1, entropy=0.2,
                      flow=(0.0, 0.0), audio=0.5)

    # Readers (visualizer, LLM)
    state = hal.read_somatic()
    print(f"Pain: {state['pain']}, PAD: {state['pad']}")
"""

import mmap
import struct
import time
import math
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to use posix_ipc for best performance, fallback to file-based
try:
    import posix_ipc
    HAVE_POSIX_IPC = True
except ImportError:
    HAVE_POSIX_IPC = False
    logger.warning("posix_ipc not available, using file-based SHM (pip install posix_ipc)")

# ==============================================================================
# SOMATIC MEMORY MAP (Version 2.0)
# Total Size: 4096 Bytes (1 Page)
# ==============================================================================
# Offset  | Type   | Name           | Description
# ------------------------------------------------------------------------------
# 0x0000  | u32    | magic          | 0xARA50111
# 0x0004  | u32    | version        | 2
# 0x0008  | u64    | timestamp_ns   | Nanoseconds since epoch
# 0x0010  | u32    | heart_rate     | System tick counter (updates/sec)
# 0x0014  | u32    | dream_state    | 0=AWAKE, 1=REM, 2=DEEP
#
# --- AFFECTIVE STATE (L2: Emotion) ---
# 0x0020  | f32    | pad_p          | Pleasure [-1.0, 1.0]
# 0x0024  | f32    | pad_a          | Arousal  [-1.0, 1.0]
# 0x0028  | f32    | pad_d          | Dominance [-1.0, 1.0]
# 0x002C  | u8     | quadrant       | PAD quadrant (0-6)
# 0x002D  | u8     | sched_mode     | Scheduler mode
# 0x002E  | u16    | reserved       |
# 0x0030  | f32    | emotion_x      | 2D emotion embedding X
# 0x0034  | f32    | emotion_y      | 2D emotion embedding Y
#
# --- SOMATIC SENSORS (L1: Body) ---
# 0x0040  | u32    | pain_raw       | Raw 32-bit pain from FPGA
# 0x0044  | f32    | pain_weber     | Weber-Fechner scaled [0.0, 1.0]
# 0x0048  | f32    | entropy        | System thermal/load entropy [0.0, 1.0]
# 0x004C  | f32    | flow_x         | Optical flow X (face motion)
# 0x0050  | f32    | flow_y         | Optical flow Y (face motion)
# 0x0054  | f32    | flow_mag       | Optical flow magnitude
# 0x0058  | f32    | audio_rms      | Voice energy [0.0, 1.0]
# 0x005C  | f32    | audio_pitch    | Voice pitch Hz
#
# --- HARDWARE DIAGNOSTICS ---
# 0x0080  | u32    | active_neurons | Count of firing neurons
# 0x0084  | u32    | total_spikes   | Lifetime spike count
# 0x0088  | u32    | fabric_temp    | FPGA temperature (mC)
# 0x008C  | u8     | thermal_limit  | 1 = Throttling active
# 0x008D  | u8     | fabric_online  | 1 = FPGA responding
# 0x008E  | u8     | dream_active   | 1 = Dream engine running
# 0x008F  | u8     | reserved       |
#
# --- SYSTEM METRICS ---
# 0x00A0  | f32    | cpu_temp       | CPU temperature °C
# 0x00A4  | f32    | gpu_temp       | GPU temperature °C
# 0x00A8  | f32    | cpu_load       | CPU utilization [0.0, 1.0]
# 0x00AC  | f32    | gpu_load       | GPU utilization [0.0, 1.0]
# 0x00B0  | f32    | ram_used_pct   | RAM utilization [0.0, 1.0]
# 0x00B4  | f32    | vram_used_pct  | VRAM utilization [0.0, 1.0]
# 0x00B8  | f32    | power_draw_w   | Total power draw watts
#
# --- CONTROL FLAGS (Bidirectional) ---
# 0x0100  | u8     | avatar_mode    | Requested avatar mode
# 0x0101  | u8     | sim_detail     | Simulation detail level
# 0x0102  | u8     | force_sleep    | Force dream mode
# 0x0103  | u8     | emergency_stop | Emergency halt
# 0x0104  | f32    | critical_temp  | Temperature threshold
#
# --- COUNCIL STATE (Multi-persona orchestration) ---
# 0x0120  | u32    | council_mask   | Bitfield of active personas (0=Exec,1=Critic,2=Dreamer)
# 0x0124  | f32    | council_stress | Disagreement level [0.0, 1.0]
# 0x0128  | f32    | council_muse_x | MUSE position X for visualization
# 0x012C  | f32    | council_muse_y | MUSE position Y
# 0x0130  | f32    | council_censor_x | CENSOR position X
# 0x0134  | f32    | council_censor_y | CENSOR position Y
# ==============================================================================

SHM_NAME = "/ara_somatic"
SHM_PATH = "/dev/shm/ara_somatic"
SHM_SIZE = 4096
MAGIC = 0xARA50111
VERSION = 2

# Dream states (matches RTL)
DREAM_AWAKE = 0
DREAM_REM = 1      # Rapid replay
DREAM_DEEP = 2     # Weight consolidation


class AraHAL:
    """
    Hardware Abstraction Layer for Ara's Unified Nervous System.

    Provides zero-copy shared memory access to the organism's
    complete somatic state. All subsystems read/write here.
    """

    def __init__(self, create: bool = False):
        """
        Initialize the HAL.

        Args:
            create: If True, create the SHM region (daemon mode).
                   If False, connect to existing region.
        """
        self.log = logging.getLogger("AraHAL")
        self._create = create
        self._shm = None
        self._map: Optional[mmap.mmap] = None
        self._heart_rate = 0
        self._last_heart_time = time.time()

        self._setup_memory()

    def _setup_memory(self) -> None:
        """Initialize POSIX shared memory."""
        try:
            if HAVE_POSIX_IPC:
                self._setup_posix_ipc()
            else:
                self._setup_file_based()

            if self._create:
                self._initialize_header()
                self.log.info(f"Somatic Memory Created: {SHM_NAME}")
            else:
                self._validate_header()
                self.log.info(f"Connected to Somatic Memory: {SHM_NAME}")

        except Exception as e:
            self.log.error(f"HAL Init Failed: {e}")
            raise

    def _setup_posix_ipc(self) -> None:
        """Setup using posix_ipc (fastest)."""
        if self._create:
            try:
                posix_ipc.unlink_shared_memory(SHM_NAME)
            except posix_ipc.ExistentialError:
                pass
            self._shm = posix_ipc.SharedMemory(
                SHM_NAME,
                flags=posix_ipc.O_CREX,
                size=SHM_SIZE
            )
        else:
            self._shm = posix_ipc.SharedMemory(SHM_NAME)

        self._map = mmap.mmap(self._shm.fd, SHM_SIZE)

    def _setup_file_based(self) -> None:
        """Setup using file-based mmap (fallback)."""
        if self._create:
            with open(SHM_PATH, "wb") as f:
                f.write(b'\x00' * SHM_SIZE)

        fd = open(SHM_PATH, "r+b")
        self._map = mmap.mmap(fd.fileno(), SHM_SIZE)
        fd.close()

    def _initialize_header(self) -> None:
        """Initialize the SHM header."""
        if not self._map:
            return

        self._map.seek(0)
        # Magic + Version + Timestamp + HeartRate + DreamState
        header = struct.pack(
            '<IIQII',
            MAGIC,
            VERSION,
            time.time_ns(),
            0,  # heart_rate
            DREAM_AWAKE
        )
        self._map.write(header)

        # Initialize control defaults
        self._map.seek(0x0100)
        self._map.write(struct.pack('<BBBBf', 3, 1, 0, 0, 85.0))

    def _validate_header(self) -> None:
        """Validate existing SHM header."""
        if not self._map:
            raise RuntimeError("Memory not mapped")

        self._map.seek(0)
        magic, version = struct.unpack('<II', self._map.read(8))

        if magic != MAGIC:
            raise RuntimeError(f"Invalid magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")
        if version != VERSION:
            self.log.warning(f"Version mismatch: {version} vs {VERSION}")

    def _touch(self) -> None:
        """Update timestamp and heart rate."""
        if not self._map:
            return

        now = time.time()
        self._heart_rate += 1

        # Calculate actual heart rate (updates/sec)
        if now - self._last_heart_time >= 1.0:
            rate = self._heart_rate
            self._heart_rate = 0
            self._last_heart_time = now
        else:
            rate = 0

        self._map.seek(0x0008)
        self._map.write(struct.pack('<QI', time.time_ns(), rate if rate else 0))

    # =========================================================================
    # WRITE METHODS (Sensor/Driver side)
    # =========================================================================

    def write_somatic(
        self,
        pad: Tuple[float, float, float],
        pain: float,
        entropy: float,
        flow: Tuple[float, float],
        audio: float,
        audio_pitch: float = 0.0
    ) -> None:
        """
        Write complete somatic state (called by nervous system aggregator).

        Args:
            pad: (pleasure, arousal, dominance) each in [-1.0, 1.0]
            pain: Pain level [0.0, 1.0]
            entropy: System entropy [0.0, 1.0]
            flow: (flow_x, flow_y) optical flow
            audio: Voice RMS energy [0.0, 1.0]
            audio_pitch: Voice pitch in Hz
        """
        if not self._map:
            return

        # Calculate flow magnitude
        flow_mag = math.sqrt(flow[0]**2 + flow[1]**2)

        # Classify PAD quadrant
        p, a, d = pad
        if p < -0.7:
            quadrant = 6  # EMERGENCY
        elif p >= 0 and a >= 0:
            quadrant = 1  # EXCITED
        elif p >= 0 and a < 0:
            quadrant = 0  # SERENE
        elif p < 0 and a >= 0:
            quadrant = 2  # ANXIOUS
        else:
            quadrant = 3  # DEPRESSED

        # Write affective state (0x0020)
        self._map.seek(0x0020)
        self._map.write(struct.pack(
            '<3fBBH2f',
            p, a, d,
            quadrant, 0, 0,  # quadrant, sched_mode, reserved
            p * 0.5 + a * 0.5,  # emotion_x (simple projection)
            d * 0.5 + a * 0.5   # emotion_y
        ))

        # Write somatic sensors (0x0040)
        self._map.seek(0x0040)
        self._map.write(struct.pack(
            '<I6f',
            0,  # pain_raw (set by FPGA driver)
            pain,  # pain_weber
            entropy,
            flow[0], flow[1], flow_mag,
            audio, audio_pitch
        ))

        self._touch()

    def write_pain_raw(self, pain_raw: int) -> None:
        """
        Write raw 32-bit pain level from FPGA.
        Applies Weber-Fechner scaling for perceptual accuracy.

        The Weber-Fechner Law states that perceived intensity is
        proportional to the logarithm of stimulus intensity.
        This prevents "numbness" at high pain levels.
        """
        if not self._map:
            return

        # Weber-Fechner scaling: perception = k * log(stimulus)
        if pain_raw == 0:
            pain_weber = 0.0
        else:
            # ilog2 approximation: number of bits
            log_pain = pain_raw.bit_length()
            # Normalize: 32-bit max has bit_length=32
            pain_weber = min(1.0, log_pain / 32.0)

        self._map.seek(0x0040)
        self._map.write(struct.pack('<If', pain_raw, pain_weber))
        self._touch()

    def write_fpga_diagnostics(
        self,
        active_neurons: int,
        total_spikes: int,
        fabric_temp_mc: int,
        thermal_limit: bool,
        fabric_online: bool,
        dream_active: bool
    ) -> None:
        """Write FPGA diagnostic data."""
        if not self._map:
            return

        self._map.seek(0x0080)
        self._map.write(struct.pack(
            '<IIIBBB',
            active_neurons,
            total_spikes,
            fabric_temp_mc,
            1 if thermal_limit else 0,
            1 if fabric_online else 0,
            1 if dream_active else 0
        ))
        self._touch()

    def write_system_metrics(
        self,
        cpu_temp: float,
        gpu_temp: float,
        cpu_load: float,
        gpu_load: float,
        ram_pct: float,
        vram_pct: float,
        power_w: float
    ) -> None:
        """Write system metrics."""
        if not self._map:
            return

        self._map.seek(0x00A0)
        self._map.write(struct.pack(
            '<7f',
            cpu_temp, gpu_temp,
            cpu_load, gpu_load,
            ram_pct, vram_pct,
            power_w
        ))
        self._touch()

    def set_dream_state(self, state: int) -> None:
        """Set the dream state (AWAKE, REM, DEEP)."""
        if not self._map:
            return
        self._map.seek(0x0014)
        self._map.write(struct.pack('<I', state))
        self._touch()

    # =========================================================================
    # READ METHODS (Consumer side)
    # =========================================================================

    def read_somatic(self) -> Dict[str, Any]:
        """
        Read complete somatic state (called by visualizer/LLM).

        Returns:
            Dict with all somatic state fields
        """
        if not self._map:
            return {}

        # Read affective (0x0020)
        self._map.seek(0x0020)
        aff = struct.unpack('<3fBBH2f', self._map.read(20))

        # Read sensors (0x0040)
        self._map.seek(0x0040)
        sens = struct.unpack('<I7f', self._map.read(32))

        return {
            'pad': {'v': aff[0], 'a': aff[1], 'd': aff[2]},
            'quadrant': aff[3],
            'emotion': (aff[5], aff[6]),
            'pain_raw': sens[0],
            'pain': sens[1],
            'entropy': sens[2],
            'flow': (sens[3], sens[4]),
            'flow_mag': sens[5],
            'audio': sens[6],
            'audio_pitch': sens[7],
        }

    def read_diagnostics(self) -> Dict[str, Any]:
        """Read hardware diagnostics."""
        if not self._map:
            return {}

        self._map.seek(0x0080)
        diag = struct.unpack('<IIIBBB', self._map.read(15))

        return {
            'active_neurons': diag[0],
            'total_spikes': diag[1],
            'fabric_temp_c': diag[2] / 1000.0,
            'thermal_limit': bool(diag[3]),
            'fabric_online': bool(diag[4]),
            'dream_active': bool(diag[5]),
        }

    def read_system(self) -> Dict[str, Any]:
        """Read system metrics."""
        if not self._map:
            return {}

        self._map.seek(0x00A0)
        sys = struct.unpack('<7f', self._map.read(28))

        return {
            'cpu_temp': sys[0],
            'gpu_temp': sys[1],
            'cpu_load': sys[2],
            'gpu_load': sys[3],
            'ram_pct': sys[4],
            'vram_pct': sys[5],
            'power_w': sys[6],
        }

    def read_control(self) -> Dict[str, Any]:
        """Read control flags."""
        if not self._map:
            return {}

        self._map.seek(0x0100)
        ctrl = struct.unpack('<BBBBf', self._map.read(8))

        return {
            'avatar_mode': ctrl[0],
            'sim_detail': ctrl[1],
            'force_sleep': bool(ctrl[2]),
            'emergency_stop': bool(ctrl[3]),
            'critical_temp': ctrl[4],
        }

    def get_dream_state(self) -> int:
        """Get current dream state."""
        if not self._map:
            return DREAM_AWAKE
        self._map.seek(0x0014)
        return struct.unpack('<I', self._map.read(4))[0]

    # =========================================================================
    # CONTROL METHODS
    # =========================================================================

    def set_avatar_mode(self, mode: int) -> None:
        """Set requested avatar mode (0-4)."""
        if not self._map:
            return
        self._map.seek(0x0100)
        self._map.write(struct.pack('<B', mode))
        self._touch()

    def set_emergency_stop(self, stop: bool) -> None:
        """Set emergency stop flag."""
        if not self._map:
            return
        self._map.seek(0x0103)
        self._map.write(struct.pack('<B', 1 if stop else 0))
        self._touch()

    def trigger_sleep(self) -> None:
        """Trigger dream mode."""
        if not self._map:
            return
        self._map.seek(0x0102)
        self._map.write(struct.pack('<B', 1))
        self._touch()

    # =========================================================================
    # COUNCIL STATE (Multi-persona orchestration)
    # =========================================================================

    def write_council_state(
        self,
        mask: int,
        stress: float,
        muse_pos: Tuple[float, float] = (0.7, 0.7),
        censor_pos: Tuple[float, float] = (0.3, 0.7),
    ) -> None:
        """
        Update the Parliamentary/Council State.

        This is used by the CouncilChamber to visualize which personas
        are currently active and their disagreement level.

        Args:
            mask: Bitfield of active personas (bit 0=Exec, 1=Critic, 2=Dreamer)
                  7 = all active, 1 = only executive, 0 = council adjourned
            stress: Disagreement level [0.0, 1.0] - higher = more conflict
            muse_pos: (x, y) position for MUSE visualization
            censor_pos: (x, y) position for CENSOR visualization
        """
        if not self._map:
            return

        self._map.seek(0x0120)
        self._map.write(struct.pack(
            '<If4f',
            mask,
            stress,
            muse_pos[0], muse_pos[1],
            censor_pos[0], censor_pos[1],
        ))
        self._touch()

    def read_council_state(self) -> Dict[str, Any]:
        """Read council/parliamentary state."""
        if not self._map:
            return {}

        self._map.seek(0x0120)
        data = struct.unpack('<If4f', self._map.read(24))

        mask = data[0]
        return {
            'mask': mask,
            'executive_active': bool(mask & 1),
            'critic_active': bool(mask & 2),
            'dreamer_active': bool(mask & 4),
            'stress': data[1],
            'muse_pos': (data[2], data[3]),
            'censor_pos': (data[4], data[5]),
        }

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def close(self) -> None:
        """Close the HAL connection."""
        if self._map:
            self._map.close()
            self._map = None
        if HAVE_POSIX_IPC and self._shm:
            self._shm.close_fd()
            self._shm = None

    def __enter__(self) -> 'AraHAL':
        return self

    def __exit__(self, *args) -> None:
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_somatic_bus() -> AraHAL:
    """Create the somatic bus (call once at daemon startup)."""
    return AraHAL(create=True)


def connect_somatic_bus() -> AraHAL:
    """Connect to existing somatic bus."""
    return AraHAL(create=False)


def read_pain() -> float:
    """Quick read of current pain level."""
    with AraHAL(create=False) as hal:
        state = hal.read_somatic()
        return state.get('pain', 0.0)


def read_pad() -> Tuple[float, float, float]:
    """Quick read of PAD state."""
    with AraHAL(create=False) as hal:
        state = hal.read_somatic()
        pad = state.get('pad', {'v': 0, 'a': 0, 'd': 0})
        return (pad['v'], pad['a'], pad['d'])
