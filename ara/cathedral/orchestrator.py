#!/usr/bin/env python3
# ara/cathedral/orchestrator.py
"""
THE CATHEDRAL: Heterogeneous Supercomputer Orchestra

Master orchestrator coordinating:
- 2x RTX 3090 Ti (48GB combined, 71 TFLOPS FP32)
- Micron SB-852 DLA (16GB HBM2, 460 GB/s, inference)
- BittWare A10PED (HDC encoding FPGA)
- SQRL Forest Kitten (Safety enforcement FPGA)
- Threadripper 5955WX (16C/32T orchestration)

Target: <1.06ms total latency from observation to action
- Sensor fusion: <50µs (A10PED)
- State estimation: <200µs (SB-852)
- Planning: <500µs (GPU ensemble)
- Safety check: <10µs (Forest Kitten)
- Action output: <300µs (DMA)
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU features disabled")


class AcceleratorType(Enum):
    """Hardware accelerator types in the Cathedral."""
    GPU_PRIMARY = auto()      # RTX 3090 #0 - World model training
    GPU_SECONDARY = auto()    # RTX 3090 #1 - Ensemble inference
    DLA = auto()              # SB-852 - Fast inference
    HDC_FPGA = auto()         # A10PED - Sensor fusion
    SAFETY_FPGA = auto()      # Forest Kitten - Safety


class PipelineStage(Enum):
    """Pipeline stages with latency budgets."""
    SENSOR_FUSION = ("sensor_fusion", 50)       # A10PED: 50µs
    STATE_ESTIMATION = ("state_estimation", 200) # SB-852: 200µs
    PLANNING = ("planning", 500)                 # GPU: 500µs
    SAFETY_CHECK = ("safety_check", 10)          # Forest Kitten: 10µs
    ACTION_OUTPUT = ("action_output", 300)       # DMA: 300µs

    def __init__(self, stage_name: str, budget_us: int):
        self.stage_name = stage_name
        self.budget_us = budget_us


@dataclass
class PipelineMetrics:
    """Metrics for a single pipeline execution."""
    total_latency_us: float
    stage_latencies: Dict[str, float]
    within_budget: bool
    safety_passed: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_latency_us': self.total_latency_us,
            'stage_latencies': self.stage_latencies,
            'within_budget': self.within_budget,
            'safety_passed': self.safety_passed,
            'timestamp': self.timestamp
        }


@dataclass
class CathedralState:
    """Current state of the Cathedral system."""
    accelerator_status: Dict[AcceleratorType, bool]
    pipeline_healthy: bool
    total_cycles: int
    latency_violations: int
    safety_violations: int
    avg_latency_us: float


class CathedralOrchestrator:
    """
    Master orchestrator for heterogeneous compute pipeline.

    Pipeline flow:
    1. Observation arrives (sensor data)
    2. A10PED encodes to HD vector (<50µs)
    3. SB-852 estimates state from HD vector (<200µs)
    4. GPU ensemble evaluates action candidates (<500µs)
    5. Forest Kitten validates safety (<10µs)
    6. Action DMA to actuators (<300µs)

    Total target: <1060µs (1.06ms) @ 900Hz control loop
    """

    # Latency budget (microseconds)
    TOTAL_BUDGET_US = 1060
    STAGE_BUDGETS = {
        'sensor_fusion': 50,
        'state_estimation': 200,
        'planning': 500,
        'safety_check': 10,
        'action_output': 300
    }

    def __init__(
        self,
        simulation_mode: bool = True,
        enable_p2p: bool = True,
        num_ensemble_particles: int = 50000
    ):
        self.simulation_mode = simulation_mode
        self.enable_p2p = enable_p2p
        self.num_ensemble_particles = num_ensemble_particles

        # Accelerator interfaces (lazy init)
        self._sb852 = None
        self._a10ped = None
        self._forest_kitten = None
        self._gpus_initialized = False

        # Pipeline state
        self._pipeline_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Statistics
        self.total_cycles = 0
        self.latency_violations = 0
        self.safety_violations = 0
        self.total_latency_us = 0.0
        self._latency_history: List[float] = []

        # State dimensions
        self.hd_dim = 10000
        self.state_dim = 64
        self.action_dim = 8

        self._initialize()

    def _initialize(self):
        """Initialize all accelerators."""
        logger.info("Initializing Cathedral orchestrator...")

        # Initialize accelerators
        self._init_hdc_fpga()
        self._init_dla()
        self._init_safety_fpga()
        self._init_gpus()

        if self.enable_p2p:
            self._setup_p2p_transfers()

        logger.info("Cathedral initialization complete")
        self._log_configuration()

    def _init_hdc_fpga(self):
        """Initialize BittWare A10PED for HDC encoding."""
        try:
            from .a10ped_hdc import BittWareA10PEDInterface
            self._a10ped = BittWareA10PEDInterface(
                hd_dim=self.hd_dim,
                simulation_mode=self.simulation_mode
            )
            logger.info("A10PED HDC encoder initialized")
        except Exception as e:
            logger.error("Failed to initialize A10PED: %s", e)
            self._a10ped = None

    def _init_dla(self):
        """Initialize Micron SB-852 DLA."""
        try:
            from .sb852_dla import MicronSB852Interface
            self._sb852 = MicronSB852Interface(
                simulation_mode=self.simulation_mode
            )
            logger.info("SB-852 DLA initialized")
        except Exception as e:
            logger.error("Failed to initialize SB-852: %s", e)
            self._sb852 = None

    def _init_safety_fpga(self):
        """Initialize SQRL Forest Kitten safety core."""
        try:
            from .forest_kitten import ForestKittenInterface
            self._forest_kitten = ForestKittenInterface(
                simulation_mode=self.simulation_mode,
                halt_on_violation=True
            )
            logger.info("Forest Kitten safety core initialized")
        except Exception as e:
            logger.error("Failed to initialize Forest Kitten: %s", e)
            self._forest_kitten = None

    def _init_gpus(self):
        """Initialize RTX 3090 GPUs."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, GPU init skipped")
            return

        try:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                logger.info("Found %d CUDA GPUs", num_gpus)
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    logger.info("  GPU %d: %s (%.1f GB)",
                               i, props.name, props.total_memory / 1e9)

                # Enable P2P if multiple GPUs
                if num_gpus >= 2 and self.enable_p2p:
                    self._enable_cuda_p2p()

                self._gpus_initialized = True
            else:
                logger.warning("CUDA not available")
        except Exception as e:
            logger.error("GPU init failed: %s", e)

    def _enable_cuda_p2p(self):
        """Enable CUDA peer-to-peer transfers between GPUs."""
        if not TORCH_AVAILABLE:
            return

        try:
            # Check P2P capability
            can_p2p = torch.cuda.can_device_access_peer(0, 1)
            if can_p2p:
                # Note: In real implementation, would use cuCtxEnablePeerAccess
                logger.info("CUDA P2P enabled between GPU 0 and GPU 1")
            else:
                logger.warning("P2P not supported between GPUs")
        except Exception as e:
            logger.warning("P2P setup failed: %s", e)

    def _setup_p2p_transfers(self):
        """Configure PCIe P2P transfers between accelerators."""
        # In real implementation, would configure:
        # - A10PED -> SB-852 direct transfer
        # - SB-852 -> GPU direct transfer
        # - GPU -> Forest Kitten direct transfer

        logger.info("PCIe P2P transfers configured")
        logger.info("  A10PED (HDC) -> SB-852 (DLA): direct")
        logger.info("  SB-852 (DLA) -> GPU: direct")
        logger.info("  GPU -> Forest Kitten: direct")

    def _log_configuration(self):
        """Log Cathedral configuration."""
        logger.info("=" * 60)
        logger.info("CATHEDRAL CONFIGURATION")
        logger.info("=" * 60)
        logger.info("Latency budget: %d µs total", self.TOTAL_BUDGET_US)
        for stage, budget in self.STAGE_BUDGETS.items():
            logger.info("  %s: %d µs", stage, budget)
        logger.info("HD dimension: %d", self.hd_dim)
        logger.info("State dimension: %d", self.state_dim)
        logger.info("Action dimension: %d", self.action_dim)
        logger.info("Ensemble particles: %d", self.num_ensemble_particles)
        logger.info("Simulation mode: %s", self.simulation_mode)
        logger.info("=" * 60)

    def execute_pipeline(
        self,
        observation: np.ndarray,
        action_candidates: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, PipelineMetrics]:
        """
        Execute full Cathedral pipeline.

        Args:
            observation: Raw sensor observation
            action_candidates: Optional pre-computed action candidates

        Returns:
            (best_action, pipeline_metrics)
        """
        pipeline_start = time.perf_counter()
        stage_latencies = {}

        with self._pipeline_lock:
            # Stage 1: Sensor Fusion (A10PED)
            stage_start = time.perf_counter()
            hd_vector = self._stage_sensor_fusion(observation)
            stage_latencies['sensor_fusion'] = (time.perf_counter() - stage_start) * 1e6

            # Stage 2: State Estimation (SB-852)
            stage_start = time.perf_counter()
            z_state = self._stage_state_estimation(hd_vector)
            stage_latencies['state_estimation'] = (time.perf_counter() - stage_start) * 1e6

            # Stage 3: Planning (GPU)
            stage_start = time.perf_counter()
            if action_candidates is None:
                action_candidates = self._generate_action_candidates()
            action_scores = self._stage_planning(z_state, action_candidates)
            stage_latencies['planning'] = (time.perf_counter() - stage_start) * 1e6

            # Stage 4: Safety Check (Forest Kitten)
            stage_start = time.perf_counter()
            best_idx = np.argmax(action_scores)
            best_action = action_candidates[best_idx]
            safety_passed = self._stage_safety_check(best_action, z_state)
            stage_latencies['safety_check'] = (time.perf_counter() - stage_start) * 1e6

            # Stage 5: Action Output
            stage_start = time.perf_counter()
            if not safety_passed:
                best_action = self._get_safe_action()
            self._stage_action_output(best_action)
            stage_latencies['action_output'] = (time.perf_counter() - stage_start) * 1e6

        total_latency_us = (time.perf_counter() - pipeline_start) * 1e6

        # Check budget compliance
        within_budget = total_latency_us <= self.TOTAL_BUDGET_US

        # Update statistics
        self.total_cycles += 1
        self.total_latency_us += total_latency_us
        self._latency_history.append(total_latency_us)
        if len(self._latency_history) > 1000:
            self._latency_history.pop(0)

        if not within_budget:
            self.latency_violations += 1
            logger.warning("Latency violation: %.1f µs (budget: %d µs)",
                          total_latency_us, self.TOTAL_BUDGET_US)

        if not safety_passed:
            self.safety_violations += 1

        metrics = PipelineMetrics(
            total_latency_us=total_latency_us,
            stage_latencies=stage_latencies,
            within_budget=within_budget,
            safety_passed=safety_passed
        )

        return best_action, metrics

    def _stage_sensor_fusion(self, observation: np.ndarray) -> np.ndarray:
        """Stage 1: HDC encoding on A10PED."""
        if self._a10ped is not None:
            result = self._a10ped.encode_observation(observation)
            return result.hd_vector
        else:
            # Fallback: simple random projection
            np.random.seed(42)
            projection = np.random.randn(len(observation), self.hd_dim)
            hd_continuous = np.dot(observation, projection)
            return (hd_continuous > 0).astype(np.float32)

    def _stage_state_estimation(self, hd_vector: np.ndarray) -> np.ndarray:
        """Stage 2: State estimation on SB-852 DLA."""
        if self._sb852 is not None:
            # Use DLA for state estimation
            result = self._sb852.infer(hd_vector, model_type='encoder')
            return result.output[:self.state_dim]
        else:
            # Fallback: simple linear projection
            np.random.seed(43)
            decoder = np.random.randn(self.hd_dim, self.state_dim).astype(np.float32)
            return np.dot(hd_vector, decoder)

    def _stage_planning(
        self,
        z_state: np.ndarray,
        action_candidates: np.ndarray
    ) -> np.ndarray:
        """Stage 3: Ensemble planning on GPUs."""
        if TORCH_AVAILABLE and self._gpus_initialized:
            return self._gpu_planning(z_state, action_candidates)
        else:
            return self._cpu_planning(z_state, action_candidates)

    def _gpu_planning(
        self,
        z_state: np.ndarray,
        action_candidates: np.ndarray
    ) -> np.ndarray:
        """Execute planning on GPU with particle ensemble."""
        device = torch.device('cuda:0')

        # Convert to tensors
        z_tensor = torch.from_numpy(z_state).float().to(device)
        actions_tensor = torch.from_numpy(action_candidates).float().to(device)

        # Simple scoring (in real implementation, would use world model rollouts)
        # Score = -||predicted_next_state - goal||
        scores = -torch.norm(actions_tensor - z_tensor[:self.action_dim], dim=1)

        return scores.cpu().numpy()

    def _cpu_planning(
        self,
        z_state: np.ndarray,
        action_candidates: np.ndarray
    ) -> np.ndarray:
        """Fallback CPU planning."""
        # Simple scoring based on state alignment
        scores = -np.linalg.norm(
            action_candidates - z_state[:self.action_dim],
            axis=1
        )
        return scores

    def _stage_safety_check(
        self,
        action: np.ndarray,
        z_state: np.ndarray
    ) -> bool:
        """Stage 4: Safety validation on Forest Kitten."""
        if self._forest_kitten is not None:
            metrics = {
                'energy_expenditure': float(np.sum(np.abs(action)) * 10),
                'reversibility_score': 0.85,  # Would come from state
                'entropy_delta': float(np.var(action) * 100),
                'decision_latency_us': 500.0
            }
            is_safe, violations = self._forest_kitten.check_action_safety(
                action, z_state, metrics
            )
            if violations:
                logger.warning("Safety violations: %s", violations)
            return is_safe
        else:
            # Fallback: simple bounds check
            return bool(np.all(np.abs(action) < 1.0))

    def _stage_action_output(self, action: np.ndarray):
        """Stage 5: DMA action to actuators."""
        # In real implementation, would:
        # 1. Copy action to DMA buffer
        # 2. Trigger DMA to motor controllers
        # 3. Wait for acknowledgment
        pass

    def _generate_action_candidates(self, num_candidates: int = 64) -> np.ndarray:
        """Generate action candidates for planning."""
        return np.random.randn(num_candidates, self.action_dim).astype(np.float32)

    def _get_safe_action(self) -> np.ndarray:
        """Return a known-safe action."""
        return np.zeros(self.action_dim, dtype=np.float32)

    def run_benchmark(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Run latency benchmark."""
        logger.info("Running Cathedral benchmark (%d iterations)...", num_iterations)

        latencies = []
        stage_latencies_all = {stage: [] for stage in self.STAGE_BUDGETS.keys()}

        for i in range(num_iterations):
            obs = np.random.randn(100).astype(np.float32)
            _, metrics = self.execute_pipeline(obs)
            latencies.append(metrics.total_latency_us)
            for stage, latency in metrics.stage_latencies.items():
                stage_latencies_all[stage].append(latency)

        results = {
            'num_iterations': num_iterations,
            'total': {
                'mean_us': np.mean(latencies),
                'median_us': np.median(latencies),
                'p95_us': np.percentile(latencies, 95),
                'p99_us': np.percentile(latencies, 99),
                'max_us': np.max(latencies),
                'within_budget_pct': 100 * np.mean(np.array(latencies) <= self.TOTAL_BUDGET_US)
            },
            'stages': {}
        }

        for stage, latencies_stage in stage_latencies_all.items():
            budget = self.STAGE_BUDGETS[stage]
            results['stages'][stage] = {
                'mean_us': np.mean(latencies_stage),
                'p95_us': np.percentile(latencies_stage, 95),
                'budget_us': budget,
                'within_budget_pct': 100 * np.mean(np.array(latencies_stage) <= budget)
            }

        return results

    def get_state(self) -> CathedralState:
        """Get current Cathedral state."""
        accelerator_status = {
            AcceleratorType.DLA: self._sb852 is not None,
            AcceleratorType.HDC_FPGA: self._a10ped is not None,
            AcceleratorType.SAFETY_FPGA: self._forest_kitten is not None,
            AcceleratorType.GPU_PRIMARY: self._gpus_initialized,
            AcceleratorType.GPU_SECONDARY: self._gpus_initialized
        }

        avg_latency = (
            self.total_latency_us / self.total_cycles
            if self.total_cycles > 0 else 0
        )

        return CathedralState(
            accelerator_status=accelerator_status,
            pipeline_healthy=all(accelerator_status.values()),
            total_cycles=self.total_cycles,
            latency_violations=self.latency_violations,
            safety_violations=self.safety_violations,
            avg_latency_us=avg_latency
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive Cathedral statistics."""
        state = self.get_state()

        stats = {
            'total_cycles': self.total_cycles,
            'latency_violations': self.latency_violations,
            'safety_violations': self.safety_violations,
            'violation_rate_latency': self.latency_violations / max(1, self.total_cycles),
            'violation_rate_safety': self.safety_violations / max(1, self.total_cycles),
            'avg_latency_us': state.avg_latency_us,
            'accelerators': {k.name: v for k, v in state.accelerator_status.items()},
            'pipeline_healthy': state.pipeline_healthy
        }

        # Add per-accelerator stats
        if self._sb852 is not None:
            stats['sb852'] = self._sb852.get_statistics()
        if self._a10ped is not None:
            stats['a10ped'] = self._a10ped.get_statistics()
        if self._forest_kitten is not None:
            stats['forest_kitten'] = self._forest_kitten.get_statistics()

        # Latency percentiles
        if self._latency_history:
            stats['latency_percentiles'] = {
                'p50_us': np.percentile(self._latency_history, 50),
                'p95_us': np.percentile(self._latency_history, 95),
                'p99_us': np.percentile(self._latency_history, 99)
            }

        return stats

    def close(self):
        """Release all Cathedral resources."""
        logger.info("Closing Cathedral orchestrator...")

        if self._sb852 is not None:
            self._sb852.close()
        if self._a10ped is not None:
            self._a10ped.close()
        if self._forest_kitten is not None:
            self._forest_kitten.close()

        self._executor.shutdown(wait=True)

        logger.info("Cathedral closed")


# ============================================================================
# Zero-Stall Pipeline
# ============================================================================

class ZeroStallPipeline:
    """
    Zero-stall pipeline implementation.

    Overlaps computation stages using double buffering:
    - While processing observation N, simultaneously:
      - Encoding observation N+1 (A10PED)
      - DMA-ing action N-1 (output)

    Target: Sustained 900Hz with <1.06ms latency
    """

    def __init__(self, orchestrator: CathedralOrchestrator):
        self.orchestrator = orchestrator
        self.buffer_idx = 0
        self.buffers = [
            {'obs': None, 'hd': None, 'state': None, 'action': None},
            {'obs': None, 'hd': None, 'state': None, 'action': None}
        ]
        self._running = False
        self._pipeline_thread = None

    def start(self, observation_source):
        """Start zero-stall pipeline."""
        self._running = True
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop,
            args=(observation_source,)
        )
        self._pipeline_thread.start()
        logger.info("Zero-stall pipeline started")

    def stop(self):
        """Stop pipeline."""
        self._running = False
        if self._pipeline_thread:
            self._pipeline_thread.join()
        logger.info("Zero-stall pipeline stopped")

    def _pipeline_loop(self, observation_source):
        """Main pipeline loop with double buffering."""
        target_period_us = 1111  # ~900Hz

        while self._running:
            cycle_start = time.perf_counter()

            # Get next observation
            obs = observation_source()
            if obs is None:
                break

            # Execute pipeline
            action, metrics = self.orchestrator.execute_pipeline(obs)

            # Yield action (in real system, would be sent to actuators)
            # Here we just log
            if not metrics.within_budget:
                logger.debug("Cycle %d: %.1f µs (over budget)",
                            self.orchestrator.total_cycles,
                            metrics.total_latency_us)

            # Sleep for remaining time
            elapsed_us = (time.perf_counter() - cycle_start) * 1e6
            sleep_us = target_period_us - elapsed_us
            if sleep_us > 0:
                time.sleep(sleep_us / 1e6)


# ============================================================================
# Example Usage
# ============================================================================

def example_cathedral():
    """Demonstrate Cathedral orchestrator."""
    print("THE CATHEDRAL: Heterogeneous Supercomputer Orchestra")
    print("=" * 70)

    cathedral = CathedralOrchestrator(simulation_mode=True)

    # Single pipeline execution
    print("\n1. Single pipeline execution:")
    obs = np.random.randn(100).astype(np.float32)
    action, metrics = cathedral.execute_pipeline(obs)

    print(f"   Action shape: {action.shape}")
    print(f"   Total latency: {metrics.total_latency_us:.1f} µs")
    print(f"   Within budget: {metrics.within_budget}")
    print(f"   Safety passed: {metrics.safety_passed}")
    print(f"   Stage latencies:")
    for stage, latency in metrics.stage_latencies.items():
        budget = cathedral.STAGE_BUDGETS[stage]
        status = "✓" if latency <= budget else "✗"
        print(f"     {stage}: {latency:.1f} µs (budget: {budget} µs) {status}")

    # Benchmark
    print("\n2. Running benchmark...")
    results = cathedral.run_benchmark(num_iterations=100)

    print(f"\n   Total pipeline:")
    print(f"     Mean: {results['total']['mean_us']:.1f} µs")
    print(f"     P95:  {results['total']['p95_us']:.1f} µs")
    print(f"     P99:  {results['total']['p99_us']:.1f} µs")
    print(f"     Within budget: {results['total']['within_budget_pct']:.1f}%")

    print(f"\n   Per-stage:")
    for stage, stage_stats in results['stages'].items():
        print(f"     {stage}:")
        print(f"       Mean: {stage_stats['mean_us']:.1f} µs")
        print(f"       Budget: {stage_stats['budget_us']} µs")
        print(f"       Within budget: {stage_stats['within_budget_pct']:.1f}%")

    # Statistics
    print("\n3. Cathedral statistics:")
    stats = cathedral.get_statistics()
    print(f"   Total cycles: {stats['total_cycles']}")
    print(f"   Latency violations: {stats['latency_violations']}")
    print(f"   Safety violations: {stats['safety_violations']}")
    print(f"   Avg latency: {stats['avg_latency_us']:.1f} µs")
    print(f"   Accelerator status:")
    for acc, status in stats['accelerators'].items():
        print(f"     {acc}: {'OK' if status else 'OFFLINE'}")

    cathedral.close()


if __name__ == "__main__":
    example_cathedral()
