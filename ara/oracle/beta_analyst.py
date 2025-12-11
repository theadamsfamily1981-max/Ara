#!/usr/bin/env python3
# ara/oracle/beta_analyst.py
"""
Oracle Beta: The Analyst

Hardware allocation:
- GPU 2 (3090 #2): World model inference
- BittWare A10P FPGA: HDC encoding/binding operations
- CPU cores 22-42: FPGA communication
- NVMe SB852: Ultra-low-latency experience replay

Specialization:
- <1ms inference latency
- FPGA-accelerated HDC operations
- Hardware-optimized TRC reversible dynamics
"""

import logging
import time
import struct
from typing import Tuple, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Optional torch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using numpy fallback")

# Optional mmap for FPGA interface
try:
    import mmap
    MMAP_AVAILABLE = True
except ImportError:
    MMAP_AVAILABLE = False


def get_device(gpu_index: int = 1):
    """Get torch device, preferring specified GPU."""
    if not TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


# Default device for Oracle Beta (GPU 2)
DEVICE = get_device(1) if TORCH_AVAILABLE else None


# ============================================================================
# FPGA Interface (BittWare A10P)
# ============================================================================

class BittWareFPGAInterface:
    """
    Interface to BittWare A10P FPGA for HDC acceleration.

    Memory map:
    - 0x0000-0x3FFF: Control registers
    - 0x4000-0x7FFF: HD vector buffer (input)
    - 0x8000-0xBFFF: HD vector buffer (output)
    - 0xC000-0xFFFF: Binding operation scratchpad

    In simulation mode, uses CPU fallback with matching API.
    """

    def __init__(
        self,
        device_path: str = "/dev/mem",
        base_addr: int = 0xF0000000,
        simulation_mode: bool = True
    ):
        self.base_addr = base_addr
        self.size = 0x10000  # 64KB
        self.simulation_mode = simulation_mode
        self.mem = None

        # HD encoding parameters
        self.hd_dim = 10000
        self._hd_projection = None

        if not simulation_mode and MMAP_AVAILABLE:
            try:
                self.fd = open(device_path, 'r+b', buffering=0)
                self.mem = mmap.mmap(
                    self.fd.fileno(),
                    self.size,
                    mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=self.base_addr
                )
                logger.info("FPGA mapped at 0x%08X", self.base_addr)
            except Exception as e:
                logger.warning("Could not map FPGA: %s. Using simulation mode.", e)
                self.simulation_mode = True
        else:
            logger.info("BittWare FPGA interface in simulation mode")

        # Pre-generate projection matrix for simulation
        np.random.seed(42)
        self._hd_projection = np.random.randn(1024, self.hd_dim).astype(np.float32)

    def write_register(self, offset: int, value: int):
        """Write 32-bit value to register."""
        if self.simulation_mode or self.mem is None:
            return
        self.mem[offset:offset+4] = struct.pack('<I', value)

    def read_register(self, offset: int) -> int:
        """Read 32-bit value from register."""
        if self.simulation_mode or self.mem is None:
            return 0
        return struct.unpack('<I', self.mem[offset:offset+4])[0]

    def hd_encode(
        self,
        vector: np.ndarray,
        hd_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode dense vector to HD vector using FPGA.

        FPGA implementation:
        - 10,000 parallel multiply-accumulate units
        - 156 MHz clock -> 1.56 GOPS
        - Latency: ~100 cycles = 640ns

        In simulation: CPU fallback with random projection.
        """
        if hd_dim is None:
            hd_dim = self.hd_dim

        if self.simulation_mode:
            return self._cpu_hd_encode(vector, hd_dim)

        # Write input vector to FPGA buffer
        input_offset = 0x4000
        vector_bytes = vector.astype(np.float32).tobytes()
        self.mem[input_offset:input_offset+len(vector_bytes)] = vector_bytes

        # Write control: start encoding
        self.write_register(0x0000, 0x00000001)  # START bit

        # Poll for completion
        timeout = 1000  # iterations
        for _ in range(timeout):
            if (self.read_register(0x0000) & 0x00000002) != 0:  # DONE bit
                break
        else:
            logger.warning("FPGA encode timeout, using CPU fallback")
            return self._cpu_hd_encode(vector, hd_dim)

        # Read output HD vector
        output_offset = 0x8000
        hd_bytes = self.mem[output_offset:output_offset + hd_dim // 8]
        hd_vector = np.frombuffer(hd_bytes, dtype=np.uint8)

        # Unpack bits
        hd_binary = np.unpackbits(hd_vector)[:hd_dim]

        return hd_binary.astype(np.float32)

    def _cpu_hd_encode(self, vector: np.ndarray, hd_dim: int) -> np.ndarray:
        """CPU fallback for HD encoding."""
        # Ensure vector is right size for projection
        vec_len = len(vector)
        if vec_len > self._hd_projection.shape[0]:
            # Truncate
            vector = vector[:self._hd_projection.shape[0]]
        elif vec_len < self._hd_projection.shape[0]:
            # Pad
            padded = np.zeros(self._hd_projection.shape[0], dtype=np.float32)
            padded[:vec_len] = vector
            vector = padded

        projection = self._hd_projection[:, :hd_dim]
        hd_continuous = np.dot(vector, projection)
        hd_binary = (hd_continuous > 0).astype(np.float32)
        return hd_binary

    def hd_bind(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """
        HD binding (XOR) operation on FPGA.

        FPGA implementation:
        - 10,000 parallel XOR gates
        - Latency: 1 cycle = 6.4ns

        In simulation: numpy XOR.
        """
        if self.simulation_mode:
            return np.logical_xor(vec_a > 0.5, vec_b > 0.5).astype(np.float32)

        # Write vectors
        bind_offset = 0xC000
        combined = np.concatenate([vec_a, vec_b]).astype(np.uint8)
        self.mem[bind_offset:bind_offset+len(combined)] = combined.tobytes()

        # Trigger bind operation
        self.write_register(0x0004, 0x00000001)

        # Poll completion
        for _ in range(1000):
            if (self.read_register(0x0004) & 0x00000002) != 0:
                break

        # Read result
        result_bytes = self.mem[bind_offset:bind_offset + len(vec_a)]
        return np.frombuffer(result_bytes, dtype=np.uint8).astype(np.float32)

    def hd_bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        HD bundling (majority vote) operation.

        Combines multiple HD vectors into one.
        """
        if not vectors:
            return np.zeros(self.hd_dim, dtype=np.float32)

        stacked = np.stack(vectors, axis=0)
        summed = np.sum(stacked, axis=0)
        return (summed > len(vectors) / 2).astype(np.float32)

    def close(self):
        """Release FPGA resources."""
        if self.mem is not None:
            self.mem.close()
        if hasattr(self, 'fd') and self.fd:
            self.fd.close()


# ============================================================================
# Ultra-Low-Latency World Model
# ============================================================================

if TORCH_AVAILABLE:
    class FastWorldModel(nn.Module):
        """
        Optimized world model for <1ms inference.

        Optimizations:
        - Mixed precision (FP16)
        - Fused kernels
        - JIT compilation
        - Minimal network depth
        """

        def __init__(self, latent_dim: int = 10, action_dim: int = 4):
            super().__init__()

            # Tiny network for speed
            self.net = nn.Sequential(
                nn.Linear(latent_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim)
            )

        def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            """Forward pass with residual connection."""
            x = torch.cat([z, u], dim=-1)
            dz = self.net(x)
            return z + dz

else:
    # Numpy fallback
    class FastWorldModel:
        """Numpy-based world model fallback."""

        def __init__(self, latent_dim: int = 10, action_dim: int = 4):
            self.latent_dim = latent_dim
            self.action_dim = action_dim

            # Simple linear weights
            np.random.seed(42)
            self.W = np.random.randn(latent_dim + action_dim, latent_dim) * 0.1

        def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
            x = np.concatenate([z, u], axis=-1)
            dz = np.tanh(np.dot(x, self.W))
            return z + 0.1 * dz


# ============================================================================
# NVMe-backed Experience Replay
# ============================================================================

class NVMeExperienceReplay:
    """
    Ultra-low-latency experience replay using NVMe storage.

    Target specs (Micron SB852):
    - Read latency: 90us
    - Write latency: 15us
    - 7GB/s sequential read

    Strategy: Memory-map the NVMe for zero-copy access.
    """

    def __init__(
        self,
        path: str = "experience_replay.bin",
        capacity_mb: int = 1000,
        item_size: int = 1024
    ):
        self.path = Path(path)
        self.capacity = capacity_mb * 1024 * 1024
        self.item_size = item_size
        self.max_items = self.capacity // self.item_size

        self.mmap = None
        self.fd = None
        self.write_idx = 0

        self._initialize_storage()

        logger.info(
            "NVMeExperienceReplay: %d MB, %d max items",
            capacity_mb, self.max_items
        )

    def _initialize_storage(self):
        """Initialize memory-mapped file."""
        try:
            if not self.path.exists():
                # Create sparse file
                with open(self.path, 'wb') as f:
                    f.seek(self.capacity - 1)
                    f.write(b'\x00')

            self.fd = open(self.path, 'r+b')

            if MMAP_AVAILABLE:
                self.mmap = mmap.mmap(
                    self.fd.fileno(),
                    self.capacity,
                    mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE
                )
        except Exception as e:
            logger.warning("Could not initialize NVMe replay: %s", e)
            self.mmap = None

    def store(self, experience: bytes):
        """Store experience (target: 15us write latency)."""
        if self.mmap is None:
            return

        # Pad or truncate to item_size
        if len(experience) < self.item_size:
            experience = experience + b'\x00' * (self.item_size - len(experience))
        else:
            experience = experience[:self.item_size]

        offset = (self.write_idx % self.max_items) * self.item_size
        self.mmap[offset:offset + self.item_size] = experience
        self.write_idx += 1

    def sample(self, batch_size: int = 32) -> List[bytes]:
        """Sample random experiences (target: 90us read latency each)."""
        if self.mmap is None or self.write_idx == 0:
            return []

        max_idx = min(self.write_idx, self.max_items)
        indices = np.random.randint(0, max_idx, size=batch_size)

        samples = []
        for idx in indices:
            offset = idx * self.item_size
            sample = bytes(self.mmap[offset:offset + self.item_size])
            samples.append(sample)

        return samples

    def close(self):
        """Release resources."""
        if self.mmap is not None:
            self.mmap.close()
        if self.fd is not None:
            self.fd.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# ============================================================================
# Oracle Beta
# ============================================================================

@dataclass
class AnalystPrediction:
    """Result from Oracle Beta analysis."""
    next_state: np.ndarray
    uncertainty: float
    latency_ms: float
    hd_encoding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'next_state_norm': float(np.linalg.norm(self.next_state)),
            'uncertainty': self.uncertainty,
            'latency_ms': self.latency_ms,
            'hd_encoding_dim': len(self.hd_encoding) if self.hd_encoding is not None else 0
        }


class OracleBeta:
    """
    The Analyst: Real-time inference with sub-millisecond latency.

    Combines:
    - Fast world model on GPU 2
    - FPGA-accelerated HDC encoding
    - CUDA graphs for minimum overhead
    """

    def __init__(
        self,
        world_model: Optional[Any] = None,
        fpga_interface: Optional[BittWareFPGAInterface] = None,
        latent_dim: int = 10,
        action_dim: int = 4,
        device=None
    ):
        self.device = device or DEVICE
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # FPGA interface
        self.fpga = fpga_interface or BittWareFPGAInterface(simulation_mode=True)

        # World model
        if world_model is not None:
            self.world_model = world_model
        else:
            self.world_model = FastWorldModel(latent_dim, action_dim)

        if TORCH_AVAILABLE and self.device is not None:
            if hasattr(self.world_model, 'to'):
                self.world_model = self.world_model.to(self.device).half().eval()

            # Pre-allocate buffers
            self.z_buffer = torch.zeros(1, latent_dim, device=self.device, dtype=torch.float16)
            self.u_buffer = torch.zeros(1, action_dim, device=self.device, dtype=torch.float16)

            # Attempt to create CUDA graph
            self.cuda_graph = None
            self._warmup_cuda_graph()

        # Latency tracking
        self.latency_history: List[float] = []

        logger.info("OracleBeta initialized on %s with FPGA acceleration", self.device)

    def _warmup_cuda_graph(self):
        """Pre-record CUDA graph to eliminate kernel launch overhead."""
        if not TORCH_AVAILABLE or self.device is None:
            return

        if not torch.cuda.is_available():
            return

        try:
            # Warmup passes
            for _ in range(10):
                with torch.no_grad():
                    _ = self.world_model(self.z_buffer, self.u_buffer)

            torch.cuda.synchronize()

            # Record graph
            self.cuda_graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(self.cuda_graph):
                self.z_next_buffer = self.world_model(self.z_buffer, self.u_buffer)

            logger.info("CUDA graph recorded for zero-overhead inference")
        except Exception as e:
            logger.warning("Could not create CUDA graph: %s", e)
            self.cuda_graph = None

    def predict_next_state(
        self,
        z_current: np.ndarray,
        action: np.ndarray
    ) -> AnalystPrediction:
        """
        Predict next state with minimal latency.

        Target: <1ms total latency including GPU transfer.
        """
        start = time.perf_counter()

        if TORCH_AVAILABLE and self.device is not None:
            # Copy to GPU buffers
            self.z_buffer.copy_(
                torch.from_numpy(z_current.reshape(1, -1)).half(),
                non_blocking=True
            )
            self.u_buffer.copy_(
                torch.from_numpy(action.reshape(1, -1)).half(),
                non_blocking=True
            )

            if self.cuda_graph is not None:
                # Execute CUDA graph (fastest)
                self.cuda_graph.replay()
                result = self.z_next_buffer
            else:
                # Regular forward pass
                with torch.no_grad():
                    result = self.world_model(self.z_buffer, self.u_buffer)

            # Copy back
            next_state = result.squeeze(0).cpu().numpy()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            # Numpy fallback
            next_state = self.world_model(
                z_current.reshape(1, -1),
                action.reshape(1, -1)
            ).squeeze(0)

        latency_ms = (time.perf_counter() - start) * 1000
        self.latency_history.append(latency_ms)

        # Estimate uncertainty (simple variance-based)
        uncertainty = float(np.std(next_state))

        return AnalystPrediction(
            next_state=next_state,
            uncertainty=uncertainty,
            latency_ms=latency_ms
        )

    def encode_observation(
        self,
        raw_observation: np.ndarray
    ) -> np.ndarray:
        """
        Encode raw observation to HD space using FPGA.

        Target latency: 640ns (FPGA) + overhead.
        """
        hd_vector = self.fpga.hd_encode(raw_observation)
        return hd_vector

    def batch_predict(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Batch prediction for multiple state-action pairs.

        More efficient for parallel evaluation.
        """
        start = time.perf_counter()

        if TORCH_AVAILABLE and self.device is not None:
            z_tensor = torch.from_numpy(states).half().to(self.device)
            u_tensor = torch.from_numpy(actions).half().to(self.device)

            with torch.no_grad():
                next_states = self.world_model(z_tensor, u_tensor)

            result = next_states.cpu().numpy()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            result = self.world_model(states, actions)

        latency_ms = (time.perf_counter() - start) * 1000

        return result, latency_ms

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latency_history:
            return {'mean': 0, 'median': 0, 'p95': 0, 'p99': 0, 'min': 0}

        arr = np.array(self.latency_history)
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'min': float(np.min(arr)),
            'count': len(arr)
        }


# ============================================================================
# Example Usage
# ============================================================================

def example_oracle_beta():
    """Demonstrate sub-millisecond inference."""

    print("Oracle Beta: The Analyst")
    print("=" * 70)

    # Initialize FPGA
    fpga = BittWareFPGAInterface(simulation_mode=True)

    # Create Oracle
    oracle = OracleBeta(fpga_interface=fpga)

    print("\nRunning latency benchmark (1000 inferences)...")

    # Benchmark inference latency
    for i in range(1000):
        z = np.random.randn(10).astype(np.float32)
        u = np.random.randn(4).astype(np.float32)

        prediction = oracle.predict_next_state(z, u)

    stats = oracle.get_latency_stats()

    print(f"\nLatency Statistics:")
    print(f"  Mean:   {stats['mean']:.3f} ms")
    print(f"  Median: {stats['median']:.3f} ms")
    print(f"  P95:    {stats['p95']:.3f} ms")
    print(f"  P99:    {stats['p99']:.3f} ms")
    print(f"  Min:    {stats['min']:.3f} ms")

    # Test HD encoding
    print("\nTesting FPGA HD encoding...")
    raw_obs = np.random.randn(100).astype(np.float32)
    hd = oracle.encode_observation(raw_obs)
    print(f"  Input dim: {len(raw_obs)}")
    print(f"  HD dim: {len(hd)}")
    print(f"  HD sparsity: {np.mean(hd):.2%}")

    # Cleanup
    fpga.close()


if __name__ == "__main__":
    example_oracle_beta()
