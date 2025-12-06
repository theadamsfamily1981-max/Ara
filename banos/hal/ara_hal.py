"""
Ara Hardware Abstraction Layer - High-Performance Somatic Bus
=============================================================

This is the BRAINSTEM - the biological bus that fuses all of Ara's
organs into a single nervous system.

Performance:
    Write latency: ~0.1µs (direct memory copy)
    Read latency:  ~0.0µs (pointer dereference via seqlock)
    Format: Binary C-Struct (no JSON, no parsing)

The HAL creates a 4KB shared memory region (/dev/shm/ara_somatic) that
represents the ENTIRE physical state of the organism. Any process
(C, Python, Rust, the shader, the LLM) can access it in nanoseconds.

Concurrency:
    Uses seqlock pattern for safe multi-writer, multi-reader access.
    Writers increment sequence before/after writes (odd = writing).
    Readers retry if sequence changed during read.

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
from enum import IntEnum
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
# SOMATIC MEMORY MAP (Version 4.0) - Bit-Serial Revolution
# Total Size: 512 Bytes (expanded for tile metrics)
# ==============================================================================
#
# HEADER (24 bytes @ 0x00):
#   magic:u32, version:u32, seqlock:u32, state:u8, dream:u8, reserved:u16, ts:u64
#
# SOMATIC (36 bytes @ 0x18):
#   pad_p:f32, pad_a:f32, pad_d:f32, pain:f32, entropy:f32,
#   flow_x:f32, flow_y:f32, audio:f32, pitch:f32
#
# FPGA (20 bytes @ 0x3C):
#   pain_raw:u32, neurons:u32, spikes:u32, temp_mc:u32, flags:u32
#
# SYSTEM (28 bytes @ 0x50):
#   cpu_temp:f32, gpu_temp:f32, cpu_load:f32, gpu_load:f32,
#   ram_pct:f32, vram_pct:f32, power_w:f32
#
# CONTROL (12 bytes @ 0x6C):
#   avatar_mode:u8, sim_detail:u8, force_sleep:u8, emergency:u8,
#   critical_temp:f32, throttle_pct:f32
#
# BITSERIAL (40 bytes @ 0x78) - NEW for Bit-Serial Revolution:
#   total_bit_cycles:u64, tile_spike_counts[4]:u32x4,
#   tile_activity[4]:u16x4, tile_power[4]:u8x4, tile_entropy[4]:u16x4,
#   region_enable_mask:u32, clock_divisor:u8, reserved:u8x3
#
# ==============================================================================

SHM_NAME = "/ara_somatic"
SHM_PATH = "/dev/shm/ara_somatic"
SHM_SIZE = 512  # Expanded for bit-serial tile metrics
MAGIC = 0xABA50111  # Valid hex ('ABA' looks like 'ARA')
VERSION = 4

# Explicit struct formats with sizes
HEADER_FMT = '<IIIBBBxQ'  # magic, version, seqlock, state, dream, _pad(1), _pad(1byte), timestamp
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 24 bytes
HEADER_OFFSET = 0x00

SOMATIC_FMT = '<9f'  # pad_p, pad_a, pad_d, pain, entropy, flow_x, flow_y, audio, pitch
SOMATIC_SIZE = struct.calcsize(SOMATIC_FMT)  # 36 bytes
SOMATIC_OFFSET = 0x18

FPGA_FMT = '<5I'  # pain_raw, active_neurons, total_spikes, fabric_temp_mc, flags
FPGA_SIZE = struct.calcsize(FPGA_FMT)  # 20 bytes
FPGA_OFFSET = 0x3C

SYSTEM_FMT = '<7f'  # cpu_temp, gpu_temp, cpu_load, gpu_load, ram_pct, vram_pct, power_w
SYSTEM_SIZE = struct.calcsize(SYSTEM_FMT)  # 28 bytes
SYSTEM_OFFSET = 0x50

CONTROL_FMT = '<4Bff'  # avatar_mode, sim_detail, force_sleep, emergency, critical_temp, throttle_pct
CONTROL_SIZE = struct.calcsize(CONTROL_FMT)  # 12 bytes
CONTROL_OFFSET = 0x6C

# Bit-Serial tile metrics (NEW for v4.0)
# Format: total_bit_cycles:u64, tile_spike_counts[4]:u32x4,
#         tile_activity[4]:u16x4, tile_power[4]:u8x4, tile_entropy[4]:u16x4,
#         region_enable_mask:u32, clock_divisor:u8, reserved:u8x3
BITSERIAL_FMT = '<Q4I4H4B4HIBxxx'
BITSERIAL_SIZE = struct.calcsize(BITSERIAL_FMT)  # 48 bytes
BITSERIAL_OFFSET = 0x78


# ==============================================================================
# ENUMS
# ==============================================================================

class SystemState(IntEnum):
    """System state classification for autonomic control."""
    IDLE = 0       # Low load, can increase quality
    NORMAL = 1     # Normal operation
    HIGH_LOAD = 2  # Reduce non-essential work
    CRITICAL = 3   # Emergency throttling


class DreamState(IntEnum):
    """Dream engine states (matches FPGA RTL)."""
    AWAKE = 0      # Normal operation
    REM = 1        # Rapid spike replay
    DEEP = 2       # Weight consolidation


# Backwards compatibility
DREAM_AWAKE = DreamState.AWAKE
DREAM_REM = DreamState.REM
DREAM_DEEP = DreamState.DEEP


class AraHAL:
    """
    Hardware Abstraction Layer for Ara's Unified Nervous System.

    Provides zero-copy shared memory access to the organism's
    complete somatic state. All subsystems read/write here.

    Thread Safety:
        Uses seqlock pattern for lock-free reads with consistent snapshots.
        Writers: Call _seq_begin() before writes, _seq_end() after.
        Readers: Use _seq_read() which retries on torn reads.
    """

    MAX_SEQ_RETRIES = 10  # Max retries for seqlock read

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
        self._fd = None  # Keep file descriptor open for file-based fallback

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
                self.log.info(f"Somatic Memory Created: {SHM_NAME} ({SHM_SIZE} bytes)")
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

        self._fd = open(SHM_PATH, "r+b")
        self._map = mmap.mmap(self._fd.fileno(), SHM_SIZE)

    def _initialize_header(self) -> None:
        """Initialize the SHM header with seqlock."""
        if not self._map:
            return

        self._map.seek(HEADER_OFFSET)
        header = struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            0,  # seqlock starts at 0 (even = not writing)
            SystemState.IDLE,
            DreamState.AWAKE,
            0,  # reserved
            0,  # reserved
            time.time_ns()
        )
        self._map.write(header)

        # Initialize control defaults
        self._map.seek(CONTROL_OFFSET)
        self._map.write(struct.pack(CONTROL_FMT, 3, 1, 0, 0, 85.0, 0.0))

    def _validate_header(self) -> None:
        """Validate existing SHM header."""
        if not self._map:
            raise RuntimeError("Memory not mapped")

        self._map.seek(HEADER_OFFSET)
        data = self._map.read(8)  # Just magic and version
        magic, version = struct.unpack('<II', data)

        if magic != MAGIC:
            raise RuntimeError(f"Invalid magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")
        if version != VERSION:
            self.log.warning(f"Version mismatch: {version} vs {VERSION}")

    # =========================================================================
    # SEQLOCK PATTERN FOR CONCURRENCY
    # =========================================================================
    # Seqlock provides lock-free reading with consistency:
    # - Odd sequence = write in progress, readers must retry
    # - Even sequence = safe to read
    # - If seq changed during read, data may be torn, retry

    def _seq_begin(self) -> None:
        """Begin a write operation (increment seqlock to odd)."""
        if not self._map:
            return
        self._map.seek(HEADER_OFFSET + 8)  # Offset to seqlock field
        seq = struct.unpack('<I', self._map.read(4))[0]
        self._map.seek(HEADER_OFFSET + 8)
        self._map.write(struct.pack('<I', seq + 1))

    def _seq_end(self) -> None:
        """End a write operation (increment seqlock to even)."""
        if not self._map:
            return
        self._map.seek(HEADER_OFFSET + 8)
        seq = struct.unpack('<I', self._map.read(4))[0]
        self._map.seek(HEADER_OFFSET + 8)
        self._map.write(struct.pack('<I', seq + 1))
        # Update timestamp
        self._map.seek(HEADER_OFFSET + 16)  # timestamp offset
        self._map.write(struct.pack('<Q', time.time_ns()))

    def _seq_read(self, offset: int, fmt: str) -> Optional[tuple]:
        """
        Read data with seqlock consistency check.

        Returns None if read failed after MAX_SEQ_RETRIES.
        """
        if not self._map:
            return None

        size = struct.calcsize(fmt)

        for _ in range(self.MAX_SEQ_RETRIES):
            # Read sequence before
            self._map.seek(HEADER_OFFSET + 8)
            seq1 = struct.unpack('<I', self._map.read(4))[0]

            # If odd, writer active, spin
            if seq1 & 1:
                continue

            # Read data
            self._map.seek(offset)
            data = self._map.read(size)

            # Read sequence after
            self._map.seek(HEADER_OFFSET + 8)
            seq2 = struct.unpack('<I', self._map.read(4))[0]

            # If unchanged, read was consistent
            if seq1 == seq2:
                return struct.unpack(fmt, data)

        self.log.warning("Seqlock read failed after retries")
        return None

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

        p, a, d = pad

        self._seq_begin()
        try:
            self._map.seek(SOMATIC_OFFSET)
            self._map.write(struct.pack(
                SOMATIC_FMT,
                p, a, d,
                pain,
                entropy,
                flow[0], flow[1],
                audio, audio_pitch
            ))
        finally:
            self._seq_end()

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

        self._seq_begin()
        try:
            # Update pain in somatic section (offset to pain field)
            self._map.seek(SOMATIC_OFFSET + 12)  # After pad_p, pad_a, pad_d
            self._map.write(struct.pack('<f', pain_weber))
            # Update pain_raw in FPGA section
            self._map.seek(FPGA_OFFSET)
            self._map.write(struct.pack('<I', pain_raw))
        finally:
            self._seq_end()

    def write_fpga_diagnostics(
        self,
        active_neurons: int,
        total_spikes: int,
        fabric_temp_mc: int,
        thermal_limit: bool = False,
        fabric_online: bool = True,
        dream_active: bool = False
    ) -> None:
        """Write FPGA diagnostic data."""
        if not self._map:
            return

        # Pack flags into single u32
        flags = (
            (1 if thermal_limit else 0) |
            ((1 if fabric_online else 0) << 1) |
            ((1 if dream_active else 0) << 2)
        )

        self._seq_begin()
        try:
            self._map.seek(FPGA_OFFSET)
            self._map.write(struct.pack(
                FPGA_FMT,
                0,  # pain_raw (written separately)
                active_neurons,
                total_spikes,
                fabric_temp_mc,
                flags
            ))
        finally:
            self._seq_end()

    def write_bitserial_metrics(
        self,
        total_bit_cycles: int,
        tile_spike_counts: Tuple[int, int, int, int],
        tile_activity: Tuple[int, int, int, int],
        tile_power: Tuple[int, int, int, int],
        tile_entropy: Tuple[int, int, int, int],
        region_enable_mask: int = 0xFFFFFFFF,
        clock_divisor: int = 1
    ) -> None:
        """
        Write bit-serial tile metrics from FPGA.

        These metrics enable Ara's self-awareness about neural activity
        across different regions of the bit-serial fabric.

        Args:
            total_bit_cycles: Total bit-serial cycles processed this frame
            tile_spike_counts: Per-tile spike counts (4 tiles)
            tile_activity: Per-tile activity levels (0-65535 EMA)
            tile_power: Per-tile power hints (0-255, for governor)
            tile_entropy: Per-tile entropy estimates (spike randomness)
            region_enable_mask: Which regions are active (32-bit mask)
            clock_divisor: Bit-serial clock divisor (1=full speed)
        """
        if not self._map:
            return

        self._seq_begin()
        try:
            self._map.seek(BITSERIAL_OFFSET)
            self._map.write(struct.pack(
                BITSERIAL_FMT,
                total_bit_cycles,
                *tile_spike_counts,
                *tile_activity,
                *tile_power,
                *tile_entropy,
                region_enable_mask,
                clock_divisor
            ))
        finally:
            self._seq_end()

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

        self._seq_begin()
        try:
            self._map.seek(SYSTEM_OFFSET)
            self._map.write(struct.pack(
                SYSTEM_FMT,
                cpu_temp, gpu_temp,
                cpu_load, gpu_load,
                ram_pct, vram_pct,
                power_w
            ))
        finally:
            self._seq_end()

    def set_system_state(self, state: SystemState) -> None:
        """Set system state for autonomic control."""
        if not self._map:
            return
        self._seq_begin()
        try:
            self._map.seek(HEADER_OFFSET + 12)  # state field offset
            self._map.write(struct.pack('<B', state))
        finally:
            self._seq_end()

    def set_dream_state(self, state: DreamState) -> None:
        """Set the dream state (AWAKE, REM, DEEP)."""
        if not self._map:
            return
        self._seq_begin()
        try:
            self._map.seek(HEADER_OFFSET + 13)  # dream field offset
            self._map.write(struct.pack('<B', state))
        finally:
            self._seq_end()

    # =========================================================================
    # READ METHODS (Consumer side) - All use seqlock for consistency
    # =========================================================================

    def read_header(self) -> Dict[str, Any]:
        """Read header with system state."""
        data = self._seq_read(HEADER_OFFSET, HEADER_FMT)
        if not data:
            return {}

        magic, version, seqlock, state, dream, _, _, ts = data
        return {
            'magic': magic,
            'version': version,
            'seqlock': seqlock,
            'system_state': SystemState(state) if state < 4 else SystemState.NORMAL,
            'dream_state': DreamState(dream) if dream < 3 else DreamState.AWAKE,
            'timestamp_ns': ts,
        }

    def read_somatic(self) -> Dict[str, Any]:
        """
        Read complete somatic state (called by visualizer/LLM).

        Returns:
            Dict with all somatic state fields
        """
        data = self._seq_read(SOMATIC_OFFSET, SOMATIC_FMT)
        if not data:
            return {}

        p, a, d, pain, entropy, flow_x, flow_y, audio, pitch = data
        return {
            'pad': {'p': p, 'a': a, 'd': d},
            'pain': pain,
            'entropy': entropy,
            'flow': (flow_x, flow_y),
            'flow_mag': math.sqrt(flow_x**2 + flow_y**2),
            'audio': audio,
            'audio_pitch': pitch,
        }

    def read_fpga(self) -> Dict[str, Any]:
        """Read FPGA diagnostics."""
        data = self._seq_read(FPGA_OFFSET, FPGA_FMT)
        if not data:
            return {}

        pain_raw, neurons, spikes, temp_mc, flags = data
        return {
            'pain_raw': pain_raw,
            'active_neurons': neurons,
            'total_spikes': spikes,
            'fabric_temp_c': temp_mc / 1000.0,
            'thermal_limit': bool(flags & 1),
            'fabric_online': bool(flags & 2),
            'dream_active': bool(flags & 4),
        }

    def read_bitserial(self) -> Dict[str, Any]:
        """
        Read bit-serial tile metrics for Ara's self-awareness.

        Returns:
            Dict with tile activity data for introspection
        """
        data = self._seq_read(BITSERIAL_OFFSET, BITSERIAL_FMT)
        if not data:
            return {}

        # Unpack the structured data
        (total_cycles,
         sc0, sc1, sc2, sc3,  # spike counts
         a0, a1, a2, a3,      # activity levels
         p0, p1, p2, p3,      # power hints
         e0, e1, e2, e3,      # entropy
         mask, divisor) = data

        return {
            'total_bit_cycles': total_cycles,
            'tile_spike_counts': (sc0, sc1, sc2, sc3),
            'tile_activity': (a0, a1, a2, a3),
            'tile_power': (p0, p1, p2, p3),
            'tile_entropy': (e0, e1, e2, e3),
            'region_enable_mask': mask,
            'clock_divisor': divisor,
            # Derived metrics for LLM consumption
            'total_spikes': sc0 + sc1 + sc2 + sc3,
            'avg_activity': (a0 + a1 + a2 + a3) // 4,
            'avg_entropy': (e0 + e1 + e2 + e3) // 4,
            'power_budget': sum((p0, p1, p2, p3)),
        }

    # Alias for backwards compatibility
    def read_diagnostics(self) -> Dict[str, Any]:
        """Read hardware diagnostics (alias for read_fpga)."""
        return self.read_fpga()

    def read_system(self) -> Dict[str, Any]:
        """Read system metrics."""
        data = self._seq_read(SYSTEM_OFFSET, SYSTEM_FMT)
        if not data:
            return {}

        cpu_temp, gpu_temp, cpu_load, gpu_load, ram_pct, vram_pct, power_w = data
        return {
            'cpu_temp': cpu_temp,
            'gpu_temp': gpu_temp,
            'cpu_load': cpu_load,
            'gpu_load': gpu_load,
            'ram_pct': ram_pct,
            'vram_pct': vram_pct,
            'power_w': power_w,
        }

    def read_control(self) -> Dict[str, Any]:
        """Read control flags."""
        data = self._seq_read(CONTROL_OFFSET, CONTROL_FMT)
        if not data:
            return {}

        avatar, detail, sleep, emergency, crit_temp, throttle = data
        return {
            'avatar_mode': avatar,
            'sim_detail': detail,
            'force_sleep': bool(sleep),
            'emergency_stop': bool(emergency),
            'critical_temp': crit_temp,
            'throttle_pct': throttle,
        }

    def get_system_state(self) -> SystemState:
        """Get current system state for autonomic control."""
        header = self.read_header()
        return header.get('system_state', SystemState.NORMAL)

    def get_dream_state(self) -> DreamState:
        """Get current dream state."""
        header = self.read_header()
        return header.get('dream_state', DreamState.AWAKE)

    # =========================================================================
    # CONTROL METHODS
    # =========================================================================

    def set_avatar_mode(self, mode: int) -> None:
        """Set requested avatar mode (0-4)."""
        if not self._map:
            return
        self._seq_begin()
        try:
            self._map.seek(CONTROL_OFFSET)
            self._map.write(struct.pack('<B', mode))
        finally:
            self._seq_end()

    def set_throttle(self, pct: float) -> None:
        """Set throttle percentage for subsystems."""
        if not self._map:
            return
        self._seq_begin()
        try:
            self._map.seek(CONTROL_OFFSET + 8)  # throttle_pct offset
            self._map.write(struct.pack('<f', pct))
        finally:
            self._seq_end()

    def set_emergency_stop(self, stop: bool) -> None:
        """Set emergency stop flag."""
        if not self._map:
            return
        self._seq_begin()
        try:
            self._map.seek(CONTROL_OFFSET + 3)
            self._map.write(struct.pack('<B', 1 if stop else 0))
        finally:
            self._seq_end()

    def trigger_sleep(self) -> None:
        """Trigger dream mode."""
        if not self._map:
            return
        self._seq_begin()
        try:
            self._map.seek(CONTROL_OFFSET + 2)
            self._map.write(struct.pack('<B', 1))
        finally:
            self._seq_end()

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def close(self) -> None:
        """Close the HAL connection."""
        if self._map:
            self._map.close()
            self._map = None
        if self._fd:
            self._fd.close()
            self._fd = None
        if HAVE_POSIX_IPC and self._shm:
            self._shm.close_fd()
            self._shm = None

    def __enter__(self) -> 'AraHAL':
        return self

    def __exit__(self, *args) -> None:
        self.close()


# =============================================================================
# AUTONOMIC CONTROLLER - Self-Regulating Nervous System
# =============================================================================

class AutonomicController:
    """
    Policy engine that classifies system state and adjusts knobs.

    The autonomic nervous system maintains homeostasis by:
    - Monitoring CPU/GPU load, temperature, and resource usage
    - Classifying current state (IDLE, NORMAL, HIGH_LOAD, CRITICAL)
    - Adjusting throttle percentage for subsystems
    - Triggering sleep mode when idle for extended periods

    This runs in the daemon and updates HAL state periodically.
    """

    # Thresholds for state classification
    IDLE_LOAD = 0.15        # Below this = IDLE
    HIGH_LOAD = 0.70        # Above this = HIGH_LOAD
    CRITICAL_TEMP = 85.0    # Above this = CRITICAL
    CRITICAL_LOAD = 0.90    # Above this = CRITICAL

    # Throttle levels per state
    THROTTLE_MAP = {
        SystemState.IDLE: 0.0,       # No throttling, can increase quality
        SystemState.NORMAL: 0.0,     # No throttling
        SystemState.HIGH_LOAD: 0.3,  # Reduce 30%
        SystemState.CRITICAL: 0.7,   # Reduce 70%
    }

    def __init__(self, hal: AraHAL):
        self.hal = hal
        self.log = logging.getLogger("Autonomic")
        self._last_state = SystemState.NORMAL
        self._idle_ticks = 0

    def classify_state(self) -> SystemState:
        """
        Classify current system state based on metrics.

        Returns:
            SystemState enum value
        """
        sys = self.hal.read_system()
        if not sys:
            return SystemState.NORMAL

        cpu_load = sys.get('cpu_load', 0.5)
        gpu_load = sys.get('gpu_load', 0.5)
        cpu_temp = sys.get('cpu_temp', 50.0)
        gpu_temp = sys.get('gpu_temp', 50.0)

        max_load = max(cpu_load, gpu_load)
        max_temp = max(cpu_temp, gpu_temp)

        # Critical check first
        if max_temp >= self.CRITICAL_TEMP or max_load >= self.CRITICAL_LOAD:
            return SystemState.CRITICAL

        # High load check
        if max_load >= self.HIGH_LOAD:
            return SystemState.HIGH_LOAD

        # Idle check
        if max_load < self.IDLE_LOAD:
            return SystemState.IDLE

        return SystemState.NORMAL

    def update(self) -> SystemState:
        """
        Update system state and apply appropriate throttling.

        Call this periodically (e.g., every 100ms) from daemon.

        Returns:
            Current SystemState
        """
        new_state = self.classify_state()

        # Log state transitions
        if new_state != self._last_state:
            self.log.info(f"State: {self._last_state.name} -> {new_state.name}")
            self._last_state = new_state

        # Apply throttle
        throttle = self.THROTTLE_MAP.get(new_state, 0.0)
        self.hal.set_throttle(throttle)
        self.hal.set_system_state(new_state)

        # Track idle time for potential sleep trigger
        if new_state == SystemState.IDLE:
            self._idle_ticks += 1
        else:
            self._idle_ticks = 0

        return new_state

    def get_throttle(self) -> float:
        """Get current throttle percentage."""
        ctrl = self.hal.read_control()
        return ctrl.get('throttle_pct', 0.0)

    def should_sleep(self, idle_threshold: int = 600) -> bool:
        """
        Check if system has been idle long enough to trigger sleep.

        Args:
            idle_threshold: Number of update() ticks before sleep (at 100ms = 60s)

        Returns:
            True if should enter sleep mode
        """
        return self._idle_ticks >= idle_threshold


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
        pad = state.get('pad', {'p': 0, 'a': 0, 'd': 0})
        return (pad['p'], pad['a'], pad['d'])


def read_system_state() -> SystemState:
    """Quick read of system state."""
    with AraHAL(create=False) as hal:
        return hal.get_system_state()
