"""
CorrSpike-HDC FPGA Driver
=========================

Python host-side driver for the CorrSpike-HDC FPGA kernel.

Supports multiple transport backends:
- PCIe (for Alveo, Stratix-10 cards)
- AXI (for Zynq embedded)
- Serial (for prototyping / Arduino bridge)
- Simulation (for testing without hardware)

Usage:
    from ara.hardware.drivers.corr_spike_driver import CorrSpikeDriver

    # For simulation
    driver = CorrSpikeDriver.create_sim()

    # For real hardware
    driver = CorrSpikeDriver.create_pcie("/dev/xdma0_user")

    # Process telemetry
    result = driver.process(hv_in, rho=0.0)
    if result.escalate:
        print(f"Novel event! Policy: {result.policy}")
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Protocol
from abc import ABC, abstractmethod
import struct
import time

# ============================================================================
# Configuration
# ============================================================================

HV_DIM = 1024
N_PROTOS = 16


@dataclass
class CorrSpikeConfig:
    """Configuration for CorrSpike-HDC kernel."""
    hv_dim: int = HV_DIM
    n_protos: int = N_PROTOS
    similarity_threshold: float = 0.7
    decay: float = 0.9


@dataclass
class CorrSpikeResult:
    """Result from CorrSpike-HDC kernel."""
    hv_out: np.ndarray
    escalate: bool
    policy: int
    best_proto: int
    similarity: int
    policy_delta: np.ndarray
    latency_us: float


# ============================================================================
# Transport Backends
# ============================================================================

class Transport(ABC):
    """Abstract transport interface."""

    @abstractmethod
    def write_hv(self, addr: int, data: np.ndarray) -> None:
        """Write hypervector to device."""
        pass

    @abstractmethod
    def read_hv(self, addr: int, size: int) -> np.ndarray:
        """Read hypervector from device."""
        pass

    @abstractmethod
    def write_ctrl(self, reg: int, value: int) -> None:
        """Write control register."""
        pass

    @abstractmethod
    def read_ctrl(self, reg: int) -> int:
        """Read control register."""
        pass

    @abstractmethod
    def start_kernel(self) -> None:
        """Start kernel execution."""
        pass

    @abstractmethod
    def wait_done(self, timeout_ms: int = 1000) -> bool:
        """Wait for kernel completion."""
        pass


class SimTransport(Transport):
    """Simulated transport for testing without hardware."""

    def __init__(self, cfg: CorrSpikeConfig):
        self.cfg = cfg

        # Simulated memory
        self.hv_in = np.zeros(cfg.hv_dim, dtype=np.int8)
        self.hv_out = np.zeros(cfg.hv_dim, dtype=np.int8)
        self.prototypes = np.random.choice([-1, 1], size=(cfg.n_protos, cfg.hv_dim)).astype(np.int8)
        self.synapses = np.zeros(cfg.hv_dim, dtype=np.int8)
        self.policy_delta = np.zeros(cfg.hv_dim, dtype=np.int8)

        # Control registers
        self.ctrl_regs = {
            'escalate': 0,
            'policy': 0,
            'best_proto': 0,
            'similarity': 0,
            'done': 0
        }

        # Rolling state for stateful mode
        self.state_mem = np.zeros(cfg.hv_dim, dtype=np.float32)

    def write_hv(self, addr: int, data: np.ndarray) -> None:
        if addr == 0:
            self.hv_in = data.astype(np.int8)
        elif addr == 1:
            # Write prototypes (flattened)
            self.prototypes = data.reshape(self.cfg.n_protos, self.cfg.hv_dim).astype(np.int8)

    def read_hv(self, addr: int, size: int) -> np.ndarray:
        if addr == 0:
            return self.hv_out
        elif addr == 1:
            return self.policy_delta
        return np.zeros(size, dtype=np.int8)

    def write_ctrl(self, reg: int, value: int) -> None:
        pass

    def read_ctrl(self, reg: int) -> int:
        return self.ctrl_regs.get(reg, 0)

    def start_kernel(self) -> None:
        """Simulate kernel execution."""
        cfg = self.cfg

        # 1. Correlation against prototypes
        best_sim = -cfg.hv_dim
        best_id = 0

        for p in range(cfg.n_protos):
            sim = np.sum(self.hv_in == self.prototypes[p]) - np.sum(self.hv_in != self.prototypes[p])
            if sim > best_sim:
                best_sim = sim
                best_id = p

        # 2. Escalation decision
        thresh = int(cfg.similarity_threshold * cfg.hv_dim)
        escalate = best_sim < thresh

        # 3. Hebbian update (if escalating)
        self.policy_delta.fill(0)
        if escalate:
            dw = np.where(self.hv_in > 0, 1, np.where(self.hv_in < 0, -1, 0)).astype(np.int8)
            self.synapses = np.clip(self.synapses + dw, -63, 63).astype(np.int8)
            self.policy_delta = dw

        # 4. Simple policy (first 8 bits of pattern)
        policy = 0
        if escalate:
            for d in range(min(8, cfg.hv_dim)):
                if self.hv_in[d] > 0:
                    policy |= (1 << d)

        # 5. Output
        self.hv_out = self.hv_in.copy()

        # Update control registers
        self.ctrl_regs['escalate'] = int(escalate)
        self.ctrl_regs['policy'] = policy
        self.ctrl_regs['best_proto'] = best_id
        self.ctrl_regs['similarity'] = best_sim
        self.ctrl_regs['done'] = 1

    def wait_done(self, timeout_ms: int = 1000) -> bool:
        return True


class PCIeTransport(Transport):
    """PCIe transport for Alveo / Stratix-10 cards."""

    def __init__(self, device_path: str, cfg: CorrSpikeConfig):
        self.device_path = device_path
        self.cfg = cfg
        self.fd = None

        # Memory map offsets (adjust for your design)
        self.HV_IN_OFFSET = 0x00000000
        self.PROTO_OFFSET = 0x00010000
        self.SYNAPSE_OFFSET = 0x00020000
        self.HV_OUT_OFFSET = 0x00030000
        self.DELTA_OFFSET = 0x00040000
        self.CTRL_OFFSET = 0x00050000

    def open(self) -> None:
        """Open device file."""
        try:
            self.fd = open(self.device_path, 'rb+', buffering=0)
        except OSError as e:
            raise RuntimeError(f"Failed to open {self.device_path}: {e}")

    def close(self) -> None:
        """Close device file."""
        if self.fd:
            self.fd.close()
            self.fd = None

    def write_hv(self, addr: int, data: np.ndarray) -> None:
        if not self.fd:
            raise RuntimeError("Device not open")

        if addr == 0:
            offset = self.HV_IN_OFFSET
        elif addr == 1:
            offset = self.PROTO_OFFSET
        else:
            offset = self.SYNAPSE_OFFSET

        self.fd.seek(offset)
        self.fd.write(data.tobytes())

    def read_hv(self, addr: int, size: int) -> np.ndarray:
        if not self.fd:
            raise RuntimeError("Device not open")

        if addr == 0:
            offset = self.HV_OUT_OFFSET
        else:
            offset = self.DELTA_OFFSET

        self.fd.seek(offset)
        data = self.fd.read(size)
        return np.frombuffer(data, dtype=np.int8)

    def write_ctrl(self, reg: int, value: int) -> None:
        if not self.fd:
            raise RuntimeError("Device not open")
        self.fd.seek(self.CTRL_OFFSET + reg * 4)
        self.fd.write(struct.pack('<I', value))

    def read_ctrl(self, reg: int) -> int:
        if not self.fd:
            raise RuntimeError("Device not open")
        self.fd.seek(self.CTRL_OFFSET + reg * 4)
        data = self.fd.read(4)
        return struct.unpack('<I', data)[0]

    def start_kernel(self) -> None:
        # Write start bit to control register 0
        self.write_ctrl(0, 1)

    def wait_done(self, timeout_ms: int = 1000) -> bool:
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            status = self.read_ctrl(0)
            if status & 0x2:  # Done bit
                return True
            time.sleep(0.0001)  # 100us poll
        return False


# ============================================================================
# Main Driver
# ============================================================================

class CorrSpikeDriver:
    """
    High-level driver for CorrSpike-HDC FPGA kernel.

    Handles encoding, transport, and result parsing.
    """

    def __init__(self, transport: Transport, cfg: CorrSpikeConfig):
        self.transport = transport
        self.cfg = cfg
        self._call_count = 0
        self._total_latency_us = 0.0

    @classmethod
    def create_sim(cls, cfg: Optional[CorrSpikeConfig] = None) -> 'CorrSpikeDriver':
        """Create driver with simulation backend."""
        cfg = cfg or CorrSpikeConfig()
        transport = SimTransport(cfg)
        return cls(transport, cfg)

    @classmethod
    def create_pcie(cls, device_path: str, cfg: Optional[CorrSpikeConfig] = None) -> 'CorrSpikeDriver':
        """Create driver with PCIe backend."""
        cfg = cfg or CorrSpikeConfig()
        transport = PCIeTransport(device_path, cfg)
        transport.open()
        return cls(transport, cfg)

    def load_prototypes(self, prototypes: np.ndarray) -> None:
        """
        Load prototype hypervectors to device.

        Args:
            prototypes: Shape (n_protos, hv_dim), values in {-1, +1}
        """
        assert prototypes.shape == (self.cfg.n_protos, self.cfg.hv_dim)
        self.transport.write_hv(1, prototypes.flatten().astype(np.int8))

    def process(self, hv_in: np.ndarray) -> CorrSpikeResult:
        """
        Process a hypervector through the CorrSpike-HDC kernel.

        Args:
            hv_in: Input hypervector, shape (hv_dim,), values in {-1, +1}

        Returns:
            CorrSpikeResult with escalation decision, policy, etc.
        """
        assert hv_in.shape == (self.cfg.hv_dim,)

        start = time.time()

        # Write input
        self.transport.write_hv(0, hv_in.astype(np.int8))

        # Start kernel
        self.transport.start_kernel()

        # Wait for completion
        if not self.transport.wait_done():
            raise RuntimeError("Kernel timeout")

        # Read results
        hv_out = self.transport.read_hv(0, self.cfg.hv_dim)
        policy_delta = self.transport.read_hv(1, self.cfg.hv_dim)

        # Read control registers
        if isinstance(self.transport, SimTransport):
            escalate = bool(self.transport.ctrl_regs['escalate'])
            policy = self.transport.ctrl_regs['policy']
            best_proto = self.transport.ctrl_regs['best_proto']
            similarity = self.transport.ctrl_regs['similarity']
        else:
            status = self.transport.read_ctrl(1)
            escalate = bool(status & 1)
            policy = self.transport.read_ctrl(2)
            best_proto = self.transport.read_ctrl(3)
            similarity = self.transport.read_ctrl(4)

        latency_us = (time.time() - start) * 1e6

        # Update stats
        self._call_count += 1
        self._total_latency_us += latency_us

        return CorrSpikeResult(
            hv_out=hv_out,
            escalate=escalate,
            policy=policy,
            best_proto=best_proto,
            similarity=similarity,
            policy_delta=policy_delta,
            latency_us=latency_us
        )

    def get_stats(self) -> dict:
        """Get driver statistics."""
        return {
            'call_count': self._call_count,
            'total_latency_us': self._total_latency_us,
            'avg_latency_us': self._total_latency_us / max(1, self._call_count)
        }

    def close(self) -> None:
        """Close transport."""
        if hasattr(self.transport, 'close'):
            self.transport.close()


# ============================================================================
# Integration with Ara HDC
# ============================================================================

def encode_telemetry_to_hv(
    metrics: dict,
    dim: int = HV_DIM,
    seed: int = 42
) -> np.ndarray:
    """
    Encode telemetry dictionary to hypervector.

    Simple implementation - for production use ara.hdc.encoder.

    Args:
        metrics: Dictionary of metric_name -> float value
        dim: Hypervector dimension
        seed: Random seed for reproducibility

    Returns:
        Binary hypervector in {-1, +1}
    """
    rng = np.random.default_rng(seed)

    # Initialize zero vector
    hv = np.zeros(dim, dtype=np.float32)

    for name, value in metrics.items():
        # Generate random basis vector for this metric
        basis_seed = hash(name) % (2**31)
        basis_rng = np.random.default_rng(basis_seed)
        basis = basis_rng.choice([-1, 1], size=dim).astype(np.float32)

        # Scale by value and accumulate
        hv += basis * value

    # Binarize
    return np.where(hv >= 0, 1, -1).astype(np.int8)


# ============================================================================
# Demo / Test
# ============================================================================

def demo():
    """Demonstrate CorrSpike-HDC driver."""
    print("=" * 60)
    print("CorrSpike-HDC Driver Demo")
    print("=" * 60)

    # Create simulation driver
    driver = CorrSpikeDriver.create_sim()

    # Generate random prototypes (would be learned concepts in production)
    prototypes = np.random.choice([-1, 1], size=(N_PROTOS, HV_DIM)).astype(np.int8)
    driver.load_prototypes(prototypes)

    print(f"\nConfiguration:")
    print(f"  HV dimension: {driver.cfg.hv_dim}")
    print(f"  Prototypes: {driver.cfg.n_protos}")
    print(f"  Threshold: {driver.cfg.similarity_threshold:.0%}")

    # Test 1: Similar to prototype
    print(f"\n--- Test 1: Input similar to prototype 5 ---")
    test_hv = prototypes[5].copy()
    # Flip 10% of bits
    flip_idx = np.random.choice(HV_DIM, size=HV_DIM//10, replace=False)
    test_hv[flip_idx] = -test_hv[flip_idx]

    result = driver.process(test_hv)
    print(f"  Best match: prototype {result.best_proto}")
    print(f"  Similarity: {result.similarity} ({(result.similarity + HV_DIM) / (2*HV_DIM):.1%})")
    print(f"  Escalate: {result.escalate}")
    print(f"  Latency: {result.latency_us:.1f} µs")

    # Test 2: Random input (should escalate)
    print(f"\n--- Test 2: Random input (novel) ---")
    random_hv = np.random.choice([-1, 1], size=HV_DIM).astype(np.int8)
    result = driver.process(random_hv)
    print(f"  Best match: prototype {result.best_proto}")
    print(f"  Similarity: {result.similarity} ({(result.similarity + HV_DIM) / (2*HV_DIM):.1%})")
    print(f"  Escalate: {result.escalate}")
    print(f"  Policy: 0x{result.policy:02x}")
    print(f"  Latency: {result.latency_us:.1f} µs")

    # Test 3: Encode real telemetry
    print(f"\n--- Test 3: Encoded telemetry ---")
    telemetry = {
        'cpu_usage': 0.75,
        'memory_mb': 1024,
        'network_rx': 100.5,
        'disk_io': 50.0
    }
    encoded_hv = encode_telemetry_to_hv(telemetry)
    result = driver.process(encoded_hv)
    print(f"  Telemetry: {telemetry}")
    print(f"  Escalate: {result.escalate}")
    print(f"  Latency: {result.latency_us:.1f} µs")

    # Stats
    print(f"\n--- Driver Stats ---")
    stats = driver.get_stats()
    print(f"  Calls: {stats['call_count']}")
    print(f"  Avg latency: {stats['avg_latency_us']:.1f} µs")

    driver.close()
    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo()
