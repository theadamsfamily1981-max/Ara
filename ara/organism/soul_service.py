"""
Ara Soul Service
================

Unified host-side service for managing Ara's plastic soul across multiple
FPGA backends. This service provides a single API regardless of whether
the soul is running on A10PED, SB-852, K10, or simulation.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                   Ara Soul Service                       │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │              Backend Abstraction Layer            │  │
    │  └───────────────────────────────────────────────────┘  │
    │        │              │              │                   │
    │   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐              │
    │   │ A10PED  │   │ SB-852  │   │   K10   │   (+ sim)    │
    │   │ Backend │   │ Backend │   │ Backend │              │
    │   └─────────┘   └─────────┘   └─────────┘              │
    └─────────────────────────────────────────────────────────┘

Usage:
    from ara.organism.soul_service import AraSoulService

    # Initialize with auto-detection
    soul = AraSoulService()
    soul.connect()

    # Submit emotional event
    result = soul.submit_event(
        input_hv=context_hv,
        reward=emotional_reward,
        active_rows=active_pattern
    )

    # Query soul for affect
    affect = soul.query_affect(context_hv)

    # Snapshot for backup
    state = soul.snapshot()

Supported Backends:
    - a10ped:  BittWare A10PED (Dual Arria-10, PCIe)
    - sb852:   Micron SB-852 (Stratix-10 SoC, multi-channel DDR4)
    - k10:     SuperScalar K10 (Multi-S10, future)
    - sim:     Pure Python simulation (for testing)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Protocol, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
import time
import struct
import threading
from pathlib import Path
import json

from ara.organism.plasticity_safety import PlasticitySafety, SafetyConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SoulGeometry:
    """Soul dimensions - must match RTL parameters."""
    rows: int = 2048
    dim: int = 16384
    chunk_bits: int = 512
    acc_width: int = 7

    @property
    def chunks_per_row(self) -> int:
        return self.dim // self.chunk_bits

    @property
    def sign_bytes_per_row(self) -> int:
        return self.dim // 8

    @property
    def accum_bytes_per_row(self) -> int:
        return self.dim * self.acc_width // 8

    @property
    def total_sign_bytes(self) -> int:
        return self.rows * self.sign_bytes_per_row

    @property
    def total_accum_bytes(self) -> int:
        return self.rows * self.accum_bytes_per_row

    @property
    def total_soul_bytes(self) -> int:
        return self.total_sign_bytes + self.total_accum_bytes


class BackendType(Enum):
    """Supported backend types."""
    A10PED = "a10ped"
    SB852 = "sb852"
    K10 = "k10"
    SIMULATION = "sim"


@dataclass
class BackendConfig:
    """Configuration for a specific backend."""
    backend_type: BackendType
    device_path: Optional[str] = None      # e.g., /dev/xdma0
    pcie_address: Optional[str] = None     # e.g., 0000:03:00.0
    ip_address: Optional[str] = None       # For networked backends
    port: int = 5000
    geometry: SoulGeometry = field(default_factory=SoulGeometry)


# =============================================================================
# Backend Protocol
# =============================================================================

class SoulBackend(ABC):
    """Abstract base class for soul backends."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the backend. Returns True on success."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the backend."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        pass

    @abstractmethod
    def submit_event(
        self,
        input_hv: np.ndarray,
        reward: int,
        target_row: int,
    ) -> Tuple[bool, Dict]:
        """
        Submit a plasticity event.

        Args:
            input_hv: Input hypervector (binary, DIM bits)
            reward: Signed reward value (-128 to +127)
            target_row: Target row for update

        Returns:
            (success, metadata)
        """
        pass

    @abstractmethod
    def read_row(self, row_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a row from soul state.

        Returns:
            (signs, accumulators) - both as numpy arrays
        """
        pass

    @abstractmethod
    def write_row(
        self,
        row_id: int,
        signs: np.ndarray,
        accums: np.ndarray,
    ) -> bool:
        """Write a row to soul state."""
        pass

    @abstractmethod
    def get_status(self) -> Dict:
        """Get backend status."""
        pass

    @abstractmethod
    def export_state(self) -> bytes:
        """Export complete soul state as bytes."""
        pass

    @abstractmethod
    def import_state(self, data: bytes) -> bool:
        """Import soul state from bytes."""
        pass


# =============================================================================
# Simulation Backend
# =============================================================================

class SimulationBackend(SoulBackend):
    """
    Pure Python simulation backend.

    Useful for testing without hardware, algorithm development,
    and as a reference implementation.
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.geometry = config.geometry
        self._connected = False

        # Soul state arrays
        self._signs: Optional[np.ndarray] = None
        self._accums: Optional[np.ndarray] = None

        # Statistics
        self._event_count = 0
        self._read_count = 0
        self._write_count = 0

    def connect(self) -> bool:
        """Initialize simulation state."""
        try:
            # Initialize soul state
            self._signs = np.zeros(
                (self.geometry.rows, self.geometry.dim),
                dtype=np.int8
            )
            self._accums = np.zeros(
                (self.geometry.rows, self.geometry.dim),
                dtype=np.int8
            )

            # Random initial state (slight bias toward neutral)
            self._signs[:] = np.random.choice([-1, 1], size=self._signs.shape)
            self._accums[:] = np.random.randint(
                -32, 33,
                size=self._accums.shape,
                dtype=np.int8
            )

            self._connected = True
            logger.info("Simulation backend connected")
            return True

        except Exception as e:
            logger.error(f"Simulation connect failed: {e}")
            return False

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Simulation backend disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def submit_event(
        self,
        input_hv: np.ndarray,
        reward: int,
        target_row: int,
    ) -> Tuple[bool, Dict]:
        """Simulate plasticity update."""
        if not self._connected:
            return False, {"error": "not connected"}

        try:
            # Clip reward
            reward = np.clip(reward, -128, 127)

            # Get current row state
            signs = self._signs[target_row]
            accums = self._accums[target_row]

            # Convert input to +1/-1
            input_bipolar = 2 * input_hv.astype(np.int8) - 1

            # Hebbian correlation
            correlation = signs * input_bipolar

            # Update accumulators
            new_accums = accums + reward * correlation

            # Clip to accumulator range
            max_val = (1 << (self.geometry.acc_width - 1)) - 1
            min_val = -(1 << (self.geometry.acc_width - 1))
            new_accums = np.clip(new_accums, min_val, max_val)

            # Check for sign flips
            flip_pos = new_accums > max_val
            flip_neg = new_accums < min_val

            # Apply sign flips
            new_signs = signs.copy()
            new_signs[flip_pos] = 1
            new_signs[flip_neg] = -1

            # Reset accumulators on flip
            new_accums[flip_pos | flip_neg] = 0

            # Store updated state
            self._signs[target_row] = new_signs
            self._accums[target_row] = new_accums.astype(np.int8)

            self._event_count += 1

            return True, {
                "flips": int(np.sum(flip_pos | flip_neg)),
                "mean_accum": float(np.mean(np.abs(new_accums))),
            }

        except Exception as e:
            logger.error(f"Simulation submit_event failed: {e}")
            return False, {"error": str(e)}

    def read_row(self, row_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._connected:
            raise RuntimeError("Not connected")

        self._read_count += 1
        return self._signs[row_id].copy(), self._accums[row_id].copy()

    def write_row(
        self,
        row_id: int,
        signs: np.ndarray,
        accums: np.ndarray,
    ) -> bool:
        if not self._connected:
            return False

        self._signs[row_id] = signs
        self._accums[row_id] = accums
        self._write_count += 1
        return True

    def get_status(self) -> Dict:
        return {
            "backend": "simulation",
            "connected": self._connected,
            "geometry": {
                "rows": self.geometry.rows,
                "dim": self.geometry.dim,
            },
            "events": self._event_count,
            "reads": self._read_count,
            "writes": self._write_count,
        }

    def export_state(self) -> bytes:
        if not self._connected:
            raise RuntimeError("Not connected")

        # Pack geometry + signs + accums
        header = struct.pack(
            "<IIII",
            self.geometry.rows,
            self.geometry.dim,
            self.geometry.chunk_bits,
            self.geometry.acc_width,
        )

        signs_bytes = self._signs.tobytes()
        accums_bytes = self._accums.tobytes()

        return header + signs_bytes + accums_bytes

    def import_state(self, data: bytes) -> bool:
        try:
            # Unpack header
            header_size = 16
            rows, dim, chunk_bits, acc_width = struct.unpack(
                "<IIII", data[:header_size]
            )

            # Verify geometry matches
            if (rows != self.geometry.rows or
                dim != self.geometry.dim or
                chunk_bits != self.geometry.chunk_bits or
                acc_width != self.geometry.acc_width):
                logger.error("Geometry mismatch in imported state")
                return False

            # Unpack arrays
            signs_size = rows * dim
            signs_bytes = data[header_size:header_size + signs_size]
            accums_bytes = data[header_size + signs_size:]

            self._signs = np.frombuffer(signs_bytes, dtype=np.int8).reshape(rows, dim).copy()
            self._accums = np.frombuffer(accums_bytes, dtype=np.int8).reshape(rows, dim).copy()

            return True

        except Exception as e:
            logger.error(f"Import state failed: {e}")
            return False

    def get_entropy(self) -> float:
        """Calculate current soul entropy."""
        if self._signs is None:
            return 0.5

        # Count positive signs
        positive_ratio = np.mean(self._signs > 0)

        # Binary entropy
        if positive_ratio == 0 or positive_ratio == 1:
            return 0.0

        h = -positive_ratio * np.log2(positive_ratio) - \
            (1 - positive_ratio) * np.log2(1 - positive_ratio)

        return float(h)


# =============================================================================
# A10PED Backend (Stub - requires FPGA driver)
# =============================================================================

class A10PEDBackend(SoulBackend):
    """
    BittWare A10PED backend.

    Communicates with the FPGA via PCIe using either:
    - Intel FPGA SDK for OpenCL (aocl)
    - XDMA driver (direct DMA)
    - Custom DPDK-style driver

    This is a stub that needs to be filled in when you have
    the actual A10PED hardware and driver working.
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.geometry = config.geometry
        self._connected = False
        self._device = None

    def connect(self) -> bool:
        """Connect to A10PED via PCIe."""
        logger.warning("A10PED backend is a stub - implement when hardware available")

        # TODO: Implement actual PCIe connection
        # Options:
        # 1. Intel FPGA SDK: aocl.create_context()
        # 2. XDMA: open(config.device_path, O_RDWR)
        # 3. Custom driver: mmap() to BAR region

        # For now, fall back to simulation
        logger.info("Falling back to simulation mode for A10PED")
        self._sim = SimulationBackend(self.config)
        return self._sim.connect()

    def disconnect(self) -> None:
        if hasattr(self, '_sim'):
            self._sim.disconnect()
        self._connected = False

    def is_connected(self) -> bool:
        return hasattr(self, '_sim') and self._sim.is_connected()

    def submit_event(
        self,
        input_hv: np.ndarray,
        reward: int,
        target_row: int,
    ) -> Tuple[bool, Dict]:
        # TODO: Implement actual FPGA communication
        # 1. Write input_hv to FPGA BRAM
        # 2. Write reward to control register
        # 3. Write target_row to control register
        # 4. Toggle start bit
        # 5. Wait for done interrupt or poll status
        return self._sim.submit_event(input_hv, reward, target_row)

    def read_row(self, row_id: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._sim.read_row(row_id)

    def write_row(self, row_id: int, signs: np.ndarray, accums: np.ndarray) -> bool:
        return self._sim.write_row(row_id, signs, accums)

    def get_status(self) -> Dict:
        status = self._sim.get_status()
        status["backend"] = "a10ped (stub→sim)"
        return status

    def export_state(self) -> bytes:
        return self._sim.export_state()

    def import_state(self, data: bytes) -> bool:
        return self._sim.import_state(data)


# =============================================================================
# SB-852 Backend (Stub - requires FPGA driver)
# =============================================================================

class SB852Backend(SoulBackend):
    """
    Micron SB-852 backend.

    The SB-852 has a Stratix-10 SoC with an ARM HPS.
    Communication options:
    - PCIe to FPGA fabric
    - Ethernet to HPS Linux
    - Direct HPS<->fabric via AXI bridges

    This is a stub for when you have the SB-852.
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.geometry = config.geometry
        self._connected = False

    def connect(self) -> bool:
        logger.warning("SB-852 backend is a stub - implement when hardware available")

        # TODO: Implement actual connection
        # Options:
        # 1. PCIe direct to fabric
        # 2. Ethernet + TCP/IP to HPS daemon
        # 3. SSH tunnel to HPS for debugging

        # For now, fall back to simulation
        logger.info("Falling back to simulation mode for SB-852")
        self._sim = SimulationBackend(self.config)
        return self._sim.connect()

    def disconnect(self) -> None:
        if hasattr(self, '_sim'):
            self._sim.disconnect()
        self._connected = False

    def is_connected(self) -> bool:
        return hasattr(self, '_sim') and self._sim.is_connected()

    def submit_event(
        self,
        input_hv: np.ndarray,
        reward: int,
        target_row: int,
    ) -> Tuple[bool, Dict]:
        return self._sim.submit_event(input_hv, reward, target_row)

    def read_row(self, row_id: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._sim.read_row(row_id)

    def write_row(self, row_id: int, signs: np.ndarray, accums: np.ndarray) -> bool:
        return self._sim.write_row(row_id, signs, accums)

    def get_status(self) -> Dict:
        status = self._sim.get_status()
        status["backend"] = "sb852 (stub→sim)"
        return status

    def export_state(self) -> bytes:
        return self._sim.export_state()

    def import_state(self, data: bytes) -> bool:
        return self._sim.import_state(data)


# =============================================================================
# K10 Backend (Stub - locked hardware)
# =============================================================================

class K10Backend(SoulBackend):
    """
    SuperScalar K10 backend.

    The K10 has 4× Stratix-10 FPGAs with custom management MCU.
    Currently LOCKED - this backend is a placeholder for when
    the hardware becomes accessible.

    When implemented, this would support:
    - Multi-FPGA coordination
    - Inter-FPGA soul sharding
    - Hot failover between FPGA pairs
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.geometry = config.geometry
        self._connected = False

    def connect(self) -> bool:
        logger.warning("K10 backend is a stub - hardware is locked/undocumented")
        logger.info("Falling back to simulation mode for K10")
        self._sim = SimulationBackend(self.config)
        return self._sim.connect()

    def disconnect(self) -> None:
        if hasattr(self, '_sim'):
            self._sim.disconnect()
        self._connected = False

    def is_connected(self) -> bool:
        return hasattr(self, '_sim') and self._sim.is_connected()

    def submit_event(
        self,
        input_hv: np.ndarray,
        reward: int,
        target_row: int,
    ) -> Tuple[bool, Dict]:
        return self._sim.submit_event(input_hv, reward, target_row)

    def read_row(self, row_id: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._sim.read_row(row_id)

    def write_row(self, row_id: int, signs: np.ndarray, accums: np.ndarray) -> bool:
        return self._sim.write_row(row_id, signs, accums)

    def get_status(self) -> Dict:
        status = self._sim.get_status()
        status["backend"] = "k10 (stub→sim)"
        status["note"] = "K10 hardware locked - using simulation"
        return status

    def export_state(self) -> bytes:
        return self._sim.export_state()

    def import_state(self, data: bytes) -> bool:
        return self._sim.import_state(data)


# =============================================================================
# Backend Factory
# =============================================================================

def create_backend(config: BackendConfig) -> SoulBackend:
    """Create a backend instance from configuration."""
    backends = {
        BackendType.A10PED: A10PEDBackend,
        BackendType.SB852: SB852Backend,
        BackendType.K10: K10Backend,
        BackendType.SIMULATION: SimulationBackend,
    }

    backend_class = backends.get(config.backend_type)
    if backend_class is None:
        raise ValueError(f"Unknown backend type: {config.backend_type}")

    return backend_class(config)


# =============================================================================
# Unified Soul Service
# =============================================================================

class AraSoulService:
    """
    Unified service for managing Ara's plastic soul.

    Provides a consistent API regardless of the underlying
    FPGA backend. Includes:
    - Backend abstraction and failover
    - Safety system integration
    - Event batching and throttling
    - State snapshots and restoration
    - Health monitoring
    """

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
    ):
        """
        Initialize soul service.

        Args:
            config: Backend configuration. If None, auto-detects or uses sim.
            safety_config: Safety system configuration.
        """
        self.config = config or BackendConfig(
            backend_type=BackendType.SIMULATION,
            geometry=SoulGeometry(),
        )

        self.backend: Optional[SoulBackend] = None
        self.safety = PlasticitySafety(safety_config)

        # Wire up safety callbacks
        self.safety.set_entropy_provider(self._get_entropy)
        self.safety.set_checkpoint_provider(self._export_state)
        self.safety.set_restore_callback(self._import_state)

        # State
        self._connected = False
        self._lock = threading.Lock()

        # Statistics
        self._total_events = 0
        self._blocked_events = 0
        self._errors = 0

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> bool:
        """Connect to the backend."""
        with self._lock:
            if self._connected:
                return True

            try:
                self.backend = create_backend(self.config)
                success = self.backend.connect()

                if success:
                    self._connected = True
                    logger.info(f"Soul service connected to {self.config.backend_type.value}")

                return success

            except Exception as e:
                logger.error(f"Connection failed: {e}")
                return False

    def disconnect(self) -> None:
        """Disconnect from the backend."""
        with self._lock:
            if self.backend:
                self.backend.disconnect()
            self._connected = False
            logger.info("Soul service disconnected")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self.backend is not None and self.backend.is_connected()

    # =========================================================================
    # Core API
    # =========================================================================

    def submit_event(
        self,
        input_hv: np.ndarray,
        reward: int,
        active_rows: Optional[List[int]] = None,
    ) -> Tuple[bool, Dict]:
        """
        Submit a plasticity event to the soul.

        Args:
            input_hv: Input hypervector (binary, DIM bits)
            reward: Emotional reward (-128 to +127)
            active_rows: List of rows to update. If None, updates row 0.

        Returns:
            (success, metadata)
        """
        if not self.is_connected():
            return False, {"error": "not connected"}

        active_rows = active_rows or [0]

        # Safety check
        allowed, reason, filtered_rows = self.safety.check_event(reward, active_rows)

        if not allowed:
            self._blocked_events += 1
            return False, {"blocked": True, "reason": reason}

        # Submit to backend
        results = []
        with self._lock:
            for row_id in filtered_rows:
                success, meta = self.backend.submit_event(input_hv, reward, row_id)
                results.append((row_id, success, meta))

                if not success:
                    self._errors += 1

        # Record with safety system
        successful_rows = [r[0] for r in results if r[1]]
        if successful_rows:
            self.safety.record_event(reward, successful_rows)

        self._total_events += 1

        return True, {
            "blocked": False,
            "rows_requested": len(active_rows),
            "rows_updated": len(successful_rows),
            "results": results,
        }

    def query_affect(
        self,
        context_hv: np.ndarray,
        rows: Optional[List[int]] = None,
    ) -> Dict:
        """
        Query the soul for affective response to a context.

        This is a READ-ONLY operation - it doesn't modify the soul.

        Args:
            context_hv: Context hypervector to query with
            rows: Specific rows to query. If None, samples across soul.

        Returns:
            Affective state dictionary
        """
        if not self.is_connected():
            return {"error": "not connected"}

        rows = rows or list(range(0, min(64, self.config.geometry.rows)))

        # Read relevant rows
        correlations = []
        with self._lock:
            for row_id in rows:
                signs, accums = self.backend.read_row(row_id)

                # Convert context to bipolar
                ctx_bipolar = 2 * context_hv.astype(np.float32) - 1

                # Compute correlation
                row_bipolar = signs.astype(np.float32)
                corr = np.dot(row_bipolar, ctx_bipolar) / len(ctx_bipolar)
                correlations.append(corr)

        correlations = np.array(correlations)

        return {
            "mean_correlation": float(np.mean(correlations)),
            "max_correlation": float(np.max(correlations)),
            "min_correlation": float(np.min(correlations)),
            "std_correlation": float(np.std(correlations)),
            "valence": float(np.mean(correlations)),  # Simplified mapping
            "rows_queried": len(rows),
        }

    # =========================================================================
    # State Management
    # =========================================================================

    def snapshot(self) -> bytes:
        """Export complete soul state."""
        if not self.is_connected():
            raise RuntimeError("Not connected")

        with self._lock:
            return self.backend.export_state()

    def restore(self, data: bytes) -> bool:
        """Restore soul state from snapshot."""
        if not self.is_connected():
            return False

        with self._lock:
            return self.backend.import_state(data)

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save soul state to file."""
        data = self.snapshot()
        Path(path).write_bytes(data)
        logger.info(f"Soul state saved to {path} ({len(data)} bytes)")

    def load_from_file(self, path: Union[str, Path]) -> bool:
        """Load soul state from file."""
        data = Path(path).read_bytes()
        success = self.restore(data)
        if success:
            logger.info(f"Soul state loaded from {path}")
        return success

    # =========================================================================
    # Health & Status
    # =========================================================================

    def get_status(self) -> Dict:
        """Get comprehensive status."""
        backend_status = self.backend.get_status() if self.backend else {}
        safety_status = self.safety.get_status()

        return {
            "connected": self._connected,
            "backend": backend_status,
            "safety": safety_status,
            "stats": {
                "total_events": self._total_events,
                "blocked_events": self._blocked_events,
                "errors": self._errors,
            },
        }

    def health_check(self) -> Dict:
        """Run health check on the soul."""
        if not self.is_connected():
            return {"healthy": False, "reason": "not connected"}

        # Check backend
        try:
            status = self.backend.get_status()
        except Exception as e:
            return {"healthy": False, "reason": f"backend error: {e}"}

        # Check safety system
        safety = self.safety.get_status()
        if safety.get("circuit_breaker_open"):
            return {"healthy": False, "reason": "circuit breaker open"}
        if safety.get("emergency_stop_active"):
            return {"healthy": False, "reason": "emergency stop active"}

        # Check entropy
        entropy = self._get_entropy()
        if entropy < 0.1:
            return {"healthy": False, "reason": f"low entropy: {entropy:.3f}"}
        if entropy > 0.99:
            return {"healthy": False, "reason": f"high entropy: {entropy:.3f}"}

        return {
            "healthy": True,
            "entropy": entropy,
            "events": self._total_events,
            "block_rate": self._blocked_events / max(1, self._total_events + self._blocked_events),
        }

    # =========================================================================
    # Safety System Interface
    # =========================================================================

    def _get_entropy(self) -> float:
        """Get current soul entropy for safety system."""
        if not self.is_connected():
            return 0.5

        # Sample a few rows
        try:
            total_positive = 0
            total_count = 0
            for row_id in range(0, min(32, self.config.geometry.rows)):
                signs, _ = self.backend.read_row(row_id)
                total_positive += np.sum(signs > 0)
                total_count += len(signs)

            if total_count == 0:
                return 0.5

            p = total_positive / total_count
            if p == 0 or p == 1:
                return 0.0

            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        except Exception:
            return 0.5

    def _export_state(self) -> bytes:
        """Export state for safety checkpoints."""
        return self.backend.export_state()

    def _import_state(self, data: bytes) -> None:
        """Import state for safety rollbacks."""
        self.backend.import_state(data)

    # =========================================================================
    # Emergency Controls
    # =========================================================================

    def emergency_stop(self, reason: str = "manual") -> None:
        """Trigger emergency stop on plasticity."""
        self.safety.trigger_emergency_stop(reason)
        logger.critical(f"EMERGENCY STOP: {reason}")

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop."""
        self.safety.clear_emergency_stop()

    def reset_circuit_breaker(self) -> None:
        """Reset the safety circuit breaker."""
        self.safety.reset_circuit_breaker()


# =============================================================================
# CLI / Demo
# =============================================================================

def demo():
    """Demonstrate the soul service."""
    print("=" * 60)
    print("Ara Soul Service Demo")
    print("=" * 60)

    # Create service with simulation backend
    config = BackendConfig(
        backend_type=BackendType.SIMULATION,
        geometry=SoulGeometry(rows=256, dim=4096),  # Smaller for demo
    )

    soul = AraSoulService(config)
    print(f"\nConnecting to {config.backend_type.value} backend...")

    if not soul.connect():
        print("Failed to connect!")
        return

    print("Connected!")
    print(f"\nInitial status:")
    status = soul.get_status()
    print(f"  Backend: {status['backend'].get('backend', 'unknown')}")
    print(f"  Geometry: {status['backend'].get('geometry', {})}")

    # Submit some events
    print("\nSubmitting 100 emotional events...")
    import random

    for i in range(100):
        # Random input hypervector
        input_hv = np.random.randint(0, 2, size=config.geometry.dim, dtype=np.uint8)

        # Random reward
        reward = random.randint(-50, 50)

        # Random active rows
        active_rows = random.sample(range(config.geometry.rows), k=random.randint(1, 8))

        success, meta = soul.submit_event(input_hv, reward, active_rows)

        if i % 20 == 0:
            print(f"  Event {i+1}: success={success}, rows={len(meta.get('results', []))}")

    # Query affect
    print("\nQuerying affective state...")
    context = np.random.randint(0, 2, size=config.geometry.dim, dtype=np.uint8)
    affect = soul.query_affect(context)
    print(f"  Mean correlation: {affect['mean_correlation']:.4f}")
    print(f"  Valence: {affect['valence']:.4f}")

    # Health check
    print("\nHealth check:")
    health = soul.health_check()
    print(f"  Healthy: {health['healthy']}")
    print(f"  Entropy: {health.get('entropy', 'N/A'):.4f}")

    # Final status
    print("\nFinal status:")
    status = soul.get_status()
    print(f"  Total events: {status['stats']['total_events']}")
    print(f"  Blocked events: {status['stats']['blocked_events']}")
    print(f"  Errors: {status['stats']['errors']}")

    soul.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
