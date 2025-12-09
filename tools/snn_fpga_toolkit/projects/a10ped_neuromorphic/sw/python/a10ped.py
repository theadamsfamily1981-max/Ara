#!/usr/bin/env python3
"""
A10PED Neuromorphic AI Tile - Python API

High-level Python interface for BittWare A10PED AI Tile v0.

Features:
  - CSR register access
  - Memcopy operations
  - DDR4 memory management
  - Performance monitoring

Usage:
    from a10ped import AITile

    tile = AITile(tile_id=0)
    tile.memcopy(src=0x1000, dst=0x2000, length=4096)
    status = tile.get_status()

Author: A10PED Neuromorphic Project
License: MIT
"""

import os
import fcntl
import struct
import mmap
from typing import Optional, Dict
from dataclasses import dataclass
from enum import IntEnum


# ioctl commands (must match driver)
_IOC_WRITE = 1
_IOC_READ = 2
_IOC_NONE = 0

def _IOC(dir, type, nr, size):
    return (dir << 30) | (ord(type) << 8) | (nr << 0) | (size << 16)

def _IOW(type, nr, size):
    return _IOC(_IOC_WRITE, type, nr, size)

def _IOR(type, nr, size):
    return _IOC(_IOC_READ, type, nr, size)

def _IO(type, nr):
    return _IOC(_IOC_NONE, type, nr, 0)


# ioctl command codes
A10PED_IOCTL_MEMCOPY = _IOW('A', 1, 24)  # sizeof(struct a10ped_memcopy_cmd)
A10PED_IOCTL_RESET = _IO('A', 2)
A10PED_IOCTL_GET_STATUS = _IOR('A', 3, 4)  # sizeof(uint32_t)


# CSR register offsets (from ai_tile_registers.yaml)
class CSROffset(IntEnum):
    CTRL = 0x00
    STATUS = 0x04
    CMD_SRC_LO = 0x08
    CMD_SRC_HI = 0x0C
    CMD_DST_LO = 0x10
    CMD_DST_HI = 0x14
    CMD_LEN = 0x18
    CMD_CFG = 0x1C
    VERSION = 0x20
    CAPABILITIES = 0x24
    SNN_THRESHOLD = 0x28
    SNN_LEAK = 0x2C
    SNN_REFRACT = 0x30
    ERROR_CODE = 0x34
    PERF_CYCLES = 0x38
    DDR_BANDWIDTH = 0x3C
    TEMPERATURE = 0x40
    SCRATCH = 0x44


# Status bits
class StatusBits(IntEnum):
    BUSY = (1 << 0)
    DONE = (1 << 1)
    ERROR = (1 << 2)
    IRQ_PENDING = (1 << 3)
    DDR_READY = (1 << 4)
    THERMAL_WARNING = (1 << 5)


# Command modes
class CommandMode(IntEnum):
    MEMCOPY = 0x0
    SNN_INFER = 0x1
    TOPOLOGICAL = 0x2


# Precision modes
class Precision(IntEnum):
    INT8 = 0x0
    INT16 = 0x1
    FP16 = 0x2
    FP32 = 0x3


@dataclass
class TileStatus:
    """Parsed status register"""
    busy: bool
    done: bool
    error: bool
    irq_pending: bool
    ddr_ready: bool
    thermal_warning: bool
    raw: int


@dataclass
class TileInfo:
    """Tile hardware information"""
    version: tuple[int, int, int]  # (major, minor, patch)
    capabilities: int
    has_memcopy: bool
    has_snn: bool
    has_topological: bool
    has_irq: bool
    has_multi_precision: bool


class AITileError(Exception):
    """Base exception for AI tile errors"""
    pass


class AITileBusyError(AITileError):
    """Raised when tile is busy"""
    pass


class AITileCommandError(AITileError):
    """Raised when command fails"""
    pass


class AITile:
    """
    Python interface to A10PED AI Tile v0

    Example:
        tile = AITile(tile_id=0)
        tile.reset()
        tile.memcopy(src=0, dst=0x10000, length=4096)
        print(tile.get_status())
    """

    def __init__(self, tile_id: int = 0):
        """
        Initialize AI tile interface

        Args:
            tile_id: Tile ID (0 or 1 for dual-tile boards)
        """
        self.tile_id = tile_id
        self.dev_path = f"/dev/a10ped{tile_id}"

        if not os.path.exists(self.dev_path):
            raise FileNotFoundError(
                f"Device {self.dev_path} not found. "
                "Is the a10ped_driver kernel module loaded?"
            )

        # Open device
        self.fd = os.open(self.dev_path, os.O_RDWR)

        # mmap BAR0 for direct CSR access (optional, for performance)
        self.csr_mem: Optional[mmap.mmap] = None
        try:
            self.csr_mem = mmap.mmap(self.fd, 4096, offset=0)
        except OSError:
            # mmap not supported, fall back to ioctl
            pass

        # Read tile info
        self._info = self._read_tile_info()

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'csr_mem') and self.csr_mem:
            self.csr_mem.close()
        if hasattr(self, 'fd'):
            os.close(self.fd)

    def _read_csr32(self, offset: int) -> int:
        """Read 32-bit CSR register"""
        if self.csr_mem:
            # Fast path: direct memory access
            self.csr_mem.seek(offset)
            return struct.unpack('<I', self.csr_mem.read(4))[0]
        else:
            # Slow path: via ioctl (not implemented in driver yet)
            # Would need to add IOCTL_READ_CSR to driver
            raise NotImplementedError("CSR read via ioctl not yet implemented")

    def _write_csr32(self, offset: int, value: int):
        """Write 32-bit CSR register"""
        if self.csr_mem:
            # Fast path: direct memory access
            self.csr_mem.seek(offset)
            self.csr_mem.write(struct.pack('<I', value & 0xFFFFFFFF))
        else:
            # Slow path: via ioctl
            raise NotImplementedError("CSR write via ioctl not yet implemented")

    def _read_tile_info(self) -> TileInfo:
        """Read tile hardware information"""
        if not self.csr_mem:
            # If CSR access not available, return defaults
            return TileInfo(
                version=(1, 0, 0),
                capabilities=0x03,
                has_memcopy=True,
                has_snn=True,
                has_topological=False,
                has_irq=False,
                has_multi_precision=False
            )

        version_reg = self._read_csr32(CSROffset.VERSION)
        cap_reg = self._read_csr32(CSROffset.CAPABILITIES)

        major = (version_reg >> 16) & 0xFF
        minor = (version_reg >> 8) & 0xFF
        patch = version_reg & 0xFF

        return TileInfo(
            version=(major, minor, patch),
            capabilities=cap_reg,
            has_memcopy=(cap_reg & (1 << 0)) != 0,
            has_snn=(cap_reg & (1 << 1)) != 0,
            has_topological=(cap_reg & (1 << 2)) != 0,
            has_irq=(cap_reg & (1 << 3)) != 0,
            has_multi_precision=(cap_reg & (1 << 4)) != 0
        )

    def get_info(self) -> TileInfo:
        """Get tile hardware information"""
        return self._info

    def get_status(self) -> TileStatus:
        """Get current tile status"""
        # Use ioctl for status (always available)
        buf = bytearray(4)
        fcntl.ioctl(self.fd, A10PED_IOCTL_GET_STATUS, buf, True)
        status_raw = struct.unpack('<I', buf)[0]

        return TileStatus(
            busy=(status_raw & StatusBits.BUSY) != 0,
            done=(status_raw & StatusBits.DONE) != 0,
            error=(status_raw & StatusBits.ERROR) != 0,
            irq_pending=(status_raw & StatusBits.IRQ_PENDING) != 0,
            ddr_ready=(status_raw & StatusBits.DDR_READY) != 0,
            thermal_warning=(status_raw & StatusBits.THERMAL_WARNING) != 0,
            raw=status_raw
        )

    def reset(self):
        """Reset the AI tile"""
        fcntl.ioctl(self.fd, A10PED_IOCTL_RESET)

    def memcopy(self, src: int, dst: int, length: int, mode: int = CommandMode.MEMCOPY):
        """
        Perform memory copy operation

        Args:
            src: Source address in FPGA DDR4 (64-byte aligned)
            dst: Destination address in FPGA DDR4 (64-byte aligned)
            length: Transfer length in bytes (64-byte aligned)
            mode: Command mode (default: MEMCOPY)

        Raises:
            AITileBusyError: If tile is busy
            AITileCommandError: If command fails
        """
        # Validate alignment
        if src & 0x3F or dst & 0x3F or length & 0x3F:
            raise ValueError("src, dst, and length must be 64-byte aligned")

        if length > 16 * 1024 * 1024:
            raise ValueError("length must not exceed 16MB")

        # Pack command structure
        cmd_buf = struct.pack('<QQII', src, dst, length, mode)

        # Issue ioctl
        try:
            fcntl.ioctl(self.fd, A10PED_IOCTL_MEMCOPY, cmd_buf)
        except OSError as e:
            if e.errno == 16:  # EBUSY
                raise AITileBusyError("Tile is busy") from e
            elif e.errno == 5:  # EIO
                raise AITileCommandError("Command failed") from e
            else:
                raise

    def get_perf_cycles(self) -> int:
        """Get performance counter (cycles for last command)"""
        if not self.csr_mem:
            return 0
        return self._read_csr32(CSROffset.PERF_CYCLES)

    def get_error_code(self) -> int:
        """Get error code from last failed command"""
        if not self.csr_mem:
            return 0
        return self._read_csr32(CSROffset.ERROR_CODE) & 0xFF

    def get_temperature(self) -> float:
        """Get FPGA junction temperature in Celsius"""
        if not self.csr_mem:
            return 0.0
        temp_raw = self._read_csr32(CSROffset.TEMPERATURE) & 0xFFFF
        # Q8.8 fixed-point: divide by 256
        return temp_raw / 256.0

    def __repr__(self) -> str:
        info = self.get_info()
        status = self.get_status()
        return (
            f"AITile(tile_id={self.tile_id}, "
            f"version={info.version[0]}.{info.version[1]}.{info.version[2]}, "
            f"ddr_ready={status.ddr_ready}, busy={status.busy})"
        )


if __name__ == "__main__":
    # Simple test
    import sys

    try:
        tile = AITile(tile_id=0)
        print(tile)
        print(f"Status: {tile.get_status()}")
        print(f"Info: {tile.get_info()}")
        print(f"Temperature: {tile.get_temperature():.1f}°C")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Make sure the a10ped_driver kernel module is loaded:", file=sys.stderr)
        print("  cd ../driver && make && sudo insmod a10ped_driver.ko", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
