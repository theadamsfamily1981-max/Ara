#!/usr/bin/env python3
# ara/cathedral/a10ped_hdc.py
"""
BittWare A10PED HDC Sensor Fusion Interface

Hardware specs:
- Intel Arria 10 GX 1150 FPGA
- PCIe Gen3 x16 (16 GB/s bandwidth)
- 32GB DDR4 on-board
- 1.5 TFLOPS estimated

Specialization:
- 10,000-dimensional hypervector encoding
- Parallel XOR/Bundle operations in FPGA fabric
- Direct P2P output to SB-852 HBM2 (no CPU copy)
- <50us encoding latency
"""

import logging
import time
import struct
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import mmap
    MMAP_AVAILABLE = True
except ImportError:
    MMAP_AVAILABLE = False


@dataclass
class HDCEncodingResult:
    """Result from HDC encoding operation."""
    hd_vector: np.ndarray
    latency_us: float
    sparsity: float  # Fraction of 1s in binary vector
    input_dim: int
    output_dim: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'latency_us': self.latency_us,
            'sparsity': self.sparsity,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hd_vector_hash': hash(self.hd_vector.tobytes()) % 2**32
        }


class BittWareA10PEDInterface:
    """
    BittWare A10PED: Dedicated HDC encoding FPGA.

    Memory map:
    - 0x0000-0x0FFF: Control registers
    - 0x1000-0x1FFF: Input buffer (raw observations)
    - 0x10000-0x1FFFF: Output buffer (HD vectors)
    - 0x20000-0x2FFFF: Projection matrix cache
    - 0x30000+: DDR4 workspace

    Operations:
    - HD Encode: Dense vector -> 10K binary HD vector
    - HD Bind: XOR of two HD vectors
    - HD Bundle: Majority vote of multiple HD vectors
    - HD Permute: Cyclic rotation for sequence encoding
    """

    # Hardware constants
    DDR4_SIZE = 32 * 1024**3  # 32GB
    PCIE_BANDWIDTH_GBPS = 16  # Gen3 x16

    # Register offsets
    REG_CONTROL = 0x0000
    REG_STATUS = 0x0004
    REG_HD_DIM = 0x0008
    REG_INPUT_DIM = 0x000C
    REG_OP_TYPE = 0x0010

    # Operation codes
    OP_ENCODE = 0x01
    OP_BIND = 0x02
    OP_BUNDLE = 0x03
    OP_PERMUTE = 0x04

    def __init__(
        self,
        pcie_bar_addr: int = 0xE0000000,
        hd_dim: int = 10000,
        simulation_mode: bool = True
    ):
        self.pcie_bar_addr = pcie_bar_addr
        self.hd_dim = hd_dim
        self.simulation_mode = simulation_mode

        self.mem = None

        # Pre-computed projection matrix for encoding
        self._projection_matrix = None
        self._projection_seed = 42

        # Statistics
        self.total_encodings = 0
        self.total_latency_us = 0.0

        self._initialize()

    def _initialize(self):
        """Initialize A10PED interface."""
        if self.simulation_mode:
            logger.info("A10PED HDC encoder in simulation mode")
            self._init_simulation()
        else:
            self._init_hardware()

    def _init_hardware(self):
        """Initialize real hardware interface."""
        try:
            if MMAP_AVAILABLE:
                import os
                fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
                self.mem = mmap.mmap(
                    fd,
                    0x100000,  # 1MB BAR
                    mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=self.pcie_bar_addr
                )
                logger.info("A10PED BAR mapped at 0x%08X", self.pcie_bar_addr)

                # Configure HD dimension
                self._write_register(self.REG_HD_DIM, self.hd_dim)
            else:
                logger.warning("mmap not available, using simulation")
                self.simulation_mode = True
                self._init_simulation()

        except Exception as e:
            logger.warning("Hardware init failed: %s, using simulation", e)
            self.simulation_mode = True
            self._init_simulation()

    def _init_simulation(self):
        """Initialize simulation mode with pre-computed projection."""
        # Generate deterministic projection matrix
        np.random.seed(self._projection_seed)
        # Will be lazily initialized based on input dimension
        self._projection_matrix = None

        logger.info("A10PED simulation initialized")
        logger.info("  HD dimension: %d", self.hd_dim)

    def _ensure_projection_matrix(self, input_dim: int):
        """Ensure projection matrix exists for given input dimension."""
        if self._projection_matrix is None or self._projection_matrix.shape[0] != input_dim:
            np.random.seed(self._projection_seed)
            self._projection_matrix = np.random.randn(input_dim, self.hd_dim).astype(np.float32)
            logger.debug("Generated projection matrix: %s", self._projection_matrix.shape)

    def _write_register(self, offset: int, value: int):
        """Write to FPGA register."""
        if self.simulation_mode or self.mem is None:
            return
        self.mem[offset:offset+4] = struct.pack('<I', value)

    def _read_register(self, offset: int) -> int:
        """Read from FPGA register."""
        if self.simulation_mode or self.mem is None:
            return 0
        return struct.unpack('<I', self.mem[offset:offset+4])[0]

    def encode_observation(
        self,
        observation: np.ndarray
    ) -> HDCEncodingResult:
        """
        Encode observation to HD vector on FPGA.

        Process:
        1. Write observation to input buffer
        2. Trigger encode operation
        3. Read HD vector from output buffer

        Target latency: <50us

        Args:
            observation: Dense observation vector

        Returns:
            HDCEncodingResult with binary HD vector
        """
        start = time.perf_counter()

        input_dim = len(observation)
        observation = observation.astype(np.float32)

        if self.simulation_mode:
            hd_vector = self._simulate_encode(observation)
        else:
            hd_vector = self._hardware_encode(observation)

        latency_us = (time.perf_counter() - start) * 1e6

        # Update statistics
        self.total_encodings += 1
        self.total_latency_us += latency_us

        # Calculate sparsity
        sparsity = np.mean(hd_vector)

        return HDCEncodingResult(
            hd_vector=hd_vector,
            latency_us=latency_us,
            sparsity=sparsity,
            input_dim=input_dim,
            output_dim=self.hd_dim
        )

    def _simulate_encode(self, observation: np.ndarray) -> np.ndarray:
        """Simulate HD encoding in software."""
        self._ensure_projection_matrix(len(observation))

        # Project to HD space
        hd_continuous = np.dot(observation, self._projection_matrix)

        # Binarize
        hd_binary = (hd_continuous > 0).astype(np.float32)

        return hd_binary

    def _hardware_encode(self, observation: np.ndarray) -> np.ndarray:
        """Execute HD encoding on hardware."""
        # Write observation to input buffer
        obs_bytes = observation.tobytes()
        input_offset = 0x1000
        self.mem[input_offset:input_offset + len(obs_bytes)] = obs_bytes

        # Configure operation
        self._write_register(self.REG_INPUT_DIM, len(observation))
        self._write_register(self.REG_OP_TYPE, self.OP_ENCODE)

        # Trigger
        self._write_register(self.REG_CONTROL, 0x00000001)  # START

        # Poll for completion
        timeout_us = 1000  # 1ms timeout
        start = time.perf_counter()
        while True:
            status = self._read_register(self.REG_STATUS)
            if status & 0x00000002:  # DONE bit
                break
            if (time.perf_counter() - start) * 1e6 > timeout_us:
                logger.error("Encode timeout")
                break

        # Read HD vector from output buffer
        output_offset = 0x10000
        hd_bytes = bytes(self.mem[output_offset:output_offset + self.hd_dim // 8])
        hd_packed = np.frombuffer(hd_bytes, dtype=np.uint8)
        hd_vector = np.unpackbits(hd_packed)[:self.hd_dim].astype(np.float32)

        return hd_vector

    def bind(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """
        HD binding (XOR) operation.

        Creates a new HD vector representing the association of A and B.
        Binding is reversible: bind(bind(A, B), B) ≈ A

        Target latency: <10us (single cycle in FPGA)
        """
        if self.simulation_mode:
            return np.logical_xor(vec_a > 0.5, vec_b > 0.5).astype(np.float32)

        # Write vectors to FPGA
        bind_offset = 0x20000
        combined = np.concatenate([vec_a, vec_b]).astype(np.uint8)
        self.mem[bind_offset:bind_offset + len(combined)] = combined.tobytes()

        # Trigger bind
        self._write_register(self.REG_OP_TYPE, self.OP_BIND)
        self._write_register(self.REG_CONTROL, 0x00000001)

        # Poll and read result
        while not (self._read_register(self.REG_STATUS) & 0x02):
            pass

        result_bytes = bytes(self.mem[bind_offset:bind_offset + len(vec_a)])
        return np.frombuffer(result_bytes, dtype=np.uint8).astype(np.float32)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        HD bundling (majority vote) operation.

        Creates a new HD vector representing the set union.
        Used for combining multiple items into a set representation.

        Target latency: <50us for up to 100 vectors
        """
        if not vectors:
            return np.zeros(self.hd_dim, dtype=np.float32)

        if self.simulation_mode:
            stacked = np.stack(vectors, axis=0)
            summed = np.sum(stacked, axis=0)
            return (summed > len(vectors) / 2).astype(np.float32)

        # Hardware implementation would batch the operation
        # For now, iterative in software
        stacked = np.stack(vectors, axis=0)
        summed = np.sum(stacked, axis=0)
        return (summed > len(vectors) / 2).astype(np.float32)

    def permute(self, vector: np.ndarray, shifts: int = 1) -> np.ndarray:
        """
        HD permutation (cyclic rotation).

        Used for encoding position/sequence information.
        permute(A, 1) represents "A at position 1"

        Target latency: <5us (simple barrel shifter in FPGA)
        """
        return np.roll(vector, shifts)

    def encode_sequence(
        self,
        observations: List[np.ndarray]
    ) -> np.ndarray:
        """
        Encode a sequence of observations into a single HD vector.

        Uses permutation + binding for position encoding:
        sequence([A, B, C]) = bundle([
            permute(encode(A), 0),
            permute(encode(B), 1),
            permute(encode(C), 2)
        ])
        """
        encoded_items = []

        for i, obs in enumerate(observations):
            hd = self.encode_observation(obs).hd_vector
            permuted = self.permute(hd, shifts=i)
            encoded_items.append(permuted)

        return self.bundle(encoded_items)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between HD vectors.

        For binary vectors, this is equivalent to normalized Hamming similarity.
        """
        # For binary vectors: similarity = 1 - 2 * hamming_distance
        matches = np.sum(vec_a == vec_b)
        return 2 * (matches / len(vec_a)) - 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get A10PED statistics."""
        avg_latency = (
            self.total_latency_us / self.total_encodings
            if self.total_encodings > 0 else 0
        )

        return {
            'total_encodings': self.total_encodings,
            'total_latency_us': self.total_latency_us,
            'avg_latency_us': avg_latency,
            'hd_dim': self.hd_dim,
            'simulation_mode': self.simulation_mode,
            'ddr4_size_gb': self.DDR4_SIZE / 1e9,
            'pcie_bandwidth_gbps': self.PCIE_BANDWIDTH_GBPS
        }

    def close(self):
        """Release A10PED resources."""
        if self.mem is not None:
            self.mem.close()
        logger.info("A10PED closed")


# ============================================================================
# Example Usage
# ============================================================================

def example_a10ped():
    """Demonstrate A10PED HDC interface."""
    print("BittWare A10PED HDC Interface")
    print("=" * 70)

    hdc = BittWareA10PEDInterface(hd_dim=10000, simulation_mode=True)

    # Single encoding
    obs = np.random.randn(100).astype(np.float32)
    result = hdc.encode_observation(obs)
    print(f"\nSingle encoding:")
    print(f"  Input dim: {result.input_dim}")
    print(f"  Output dim: {result.output_dim}")
    print(f"  Latency: {result.latency_us:.1f} us")
    print(f"  Sparsity: {result.sparsity:.3f}")

    # Benchmark
    print("\nRunning encoding benchmark...")
    latencies = []
    for _ in range(1000):
        result = hdc.encode_observation(np.random.randn(100).astype(np.float32))
        latencies.append(result.latency_us)

    print(f"\nLatency Statistics (1000 encodings):")
    print(f"  Mean:   {np.mean(latencies):.1f} us")
    print(f"  Median: {np.median(latencies):.1f} us")
    print(f"  P95:    {np.percentile(latencies, 95):.1f} us")

    # Test HDC operations
    print("\nTesting HDC operations...")
    a = hdc.encode_observation(np.array([1.0, 0.0, 0.0])).hd_vector
    b = hdc.encode_observation(np.array([0.0, 1.0, 0.0])).hd_vector
    c = hdc.encode_observation(np.array([0.0, 0.0, 1.0])).hd_vector

    print(f"  sim(A, A): {hdc.similarity(a, a):.3f}")
    print(f"  sim(A, B): {hdc.similarity(a, b):.3f}")
    print(f"  sim(bind(A,B), bind(A,B)): {hdc.similarity(hdc.bind(a, b), hdc.bind(a, b)):.3f}")

    # Sequence encoding
    seq = [np.random.randn(100) for _ in range(5)]
    seq_hd = hdc.encode_sequence(seq)
    print(f"  Sequence HD shape: {seq_hd.shape}")

    print(f"\nStatistics:")
    stats = hdc.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    hdc.close()


if __name__ == "__main__":
    example_a10ped()
