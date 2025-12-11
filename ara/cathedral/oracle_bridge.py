#!/usr/bin/env python3
# ara/cathedral/oracle_bridge.py
"""
Cathedral-Oracle Bridge: Hardware-Accelerated Prophecy

Connects the Cathedral's heterogeneous compute pipeline to the Oracle system,
enabling hardware-accelerated divination:

- A10PED FPGA: HDC encoding for Oracle Beta's fast inference
- SB-852 DLA: World model inference for both Alpha and Beta
- Forest Kitten: Hardware safety enforcement for Oracle Gamma
- GPU ensemble: Particle propagation for Oracle Alpha

Total target: <10ms for full Oracle consultation
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Import Cathedral components
from .orchestrator import CathedralOrchestrator
from .sb852_dla import MicronSB852Interface
from .a10ped_hdc import BittWareA10PEDInterface
from .forest_kitten import ForestKittenInterface

# Optional Oracle imports
try:
    from ..oracle.pythia_triad import PythiaTriad, OracleConsensus
    from ..oracle.alpha_visionary import OracleAlpha
    from ..oracle.beta_analyst import OracleBeta
    from ..oracle.gamma_arbiter import OracleGamma
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False
    logger.warning("Oracle module not available")


@dataclass
class AcceleratedConsultation:
    """Result of hardware-accelerated Oracle consultation."""
    consensus: Optional[Any]  # OracleConsensus when available
    total_latency_ms: float
    hardware_latency_breakdown: Dict[str, float]
    accelerators_used: List[str]
    fallback_used: bool


class CathedralOracleBridge:
    """
    Bridge connecting Cathedral hardware to Oracle divination.

    Provides hardware acceleration for:
    1. HDC encoding (Oracle Beta sensor fusion)
    2. World model inference (Oracle Alpha/Beta predictions)
    3. Safety enforcement (Oracle Gamma covenant checks)
    4. Particle ensemble (Oracle Alpha long-horizon planning)
    """

    def __init__(
        self,
        cathedral: Optional[CathedralOrchestrator] = None,
        pythia: Optional[Any] = None,
        simulation_mode: bool = True
    ):
        self.simulation_mode = simulation_mode

        # Initialize Cathedral if not provided
        if cathedral is None:
            self.cathedral = CathedralOrchestrator(
                simulation_mode=simulation_mode
            )
        else:
            self.cathedral = cathedral

        # Initialize Oracle if not provided and available
        if pythia is None and ORACLE_AVAILABLE:
            self.pythia = PythiaTriad(
                latent_dim=64,
                action_dim=8,
                num_particles=10000
            )
        else:
            self.pythia = pythia

        # Direct accelerator references for low-level access
        self._a10ped = self.cathedral._a10ped
        self._sb852 = self.cathedral._sb852
        self._forest_kitten = self.cathedral._forest_kitten

        # Statistics
        self.total_consultations = 0
        self.total_latency_ms = 0.0
        self.hardware_accelerated_count = 0

        logger.info("Cathedral-Oracle bridge initialized")

    def accelerated_encode(self, observation: np.ndarray) -> np.ndarray:
        """
        Hardware-accelerated HDC encoding using A10PED.

        Used by Oracle Beta for fast sensor fusion.
        Target: <50µs
        """
        if self._a10ped is not None:
            result = self._a10ped.encode_observation(observation)
            logger.debug("HDC encode: %.1f µs", result.latency_us)
            return result.hd_vector
        else:
            # Fallback CPU encoding
            np.random.seed(42)
            projection = np.random.randn(len(observation), 10000).astype(np.float32)
            hd = np.dot(observation, projection)
            return (hd > 0).astype(np.float32)

    def accelerated_inference(
        self,
        latent_state: np.ndarray,
        action: np.ndarray,
        model_type: str = 'world_model'
    ) -> np.ndarray:
        """
        Hardware-accelerated inference using SB-852 DLA.

        Used by Oracle Alpha/Beta for world model predictions.
        Target: <200µs
        """
        if self._sb852 is not None:
            # Pack state and action for DLA
            input_tensor = np.concatenate([latent_state, action])
            result = self._sb852.infer(input_tensor, model_type=model_type)
            logger.debug("DLA inference: %.1f µs", result.latency_us)
            return result.output
        else:
            # Fallback CPU inference (simple linear)
            np.random.seed(43)
            W = np.random.randn(len(latent_state) + len(action), len(latent_state)).astype(np.float32)
            return np.tanh(np.dot(np.concatenate([latent_state, action]), W))

    def accelerated_safety_check(
        self,
        action: np.ndarray,
        state: np.ndarray,
        metrics: Dict[str, float]
    ) -> tuple:
        """
        Hardware-accelerated safety check using Forest Kitten.

        Used by Oracle Gamma for covenant enforcement.
        Target: <10µs
        """
        if self._forest_kitten is not None:
            is_safe, violations = self._forest_kitten.check_action_safety(
                action, state, metrics
            )
            return is_safe, violations
        else:
            # Fallback: simple bounds check
            is_safe = bool(np.all(np.abs(action) < 1.0))
            violations = [] if is_safe else ["Action magnitude exceeds bounds"]
            return is_safe, violations

    def accelerated_mpc_rollout(
        self,
        z_current: np.ndarray,
        action_sequence: np.ndarray,
        horizon: int = 10
    ) -> Dict[str, Any]:
        """
        Hardware-accelerated MPC rollout using SB-852 DLA.

        Used by Oracle Alpha for trajectory evaluation.
        Target: <500µs for 10-step rollout
        """
        if self._sb852 is not None:
            result = self._sb852.mpc_rollout(z_current, action_sequence, horizon)
            return {
                'trajectory': result.output,
                'latency_us': result.latency_us,
                'hbm_bandwidth_used': result.hbm_bandwidth_gbps
            }
        else:
            # Fallback CPU rollout
            trajectory = [z_current.copy()]
            z = z_current.copy()
            for i in range(min(horizon, len(action_sequence))):
                action = action_sequence[i] if len(action_sequence.shape) > 1 else action_sequence
                z = self.accelerated_inference(z, action)
                trajectory.append(z)
            return {
                'trajectory': np.array(trajectory),
                'latency_us': horizon * 100,  # Estimated
                'hbm_bandwidth_used': 0
            }

    def consult_oracle(
        self,
        observation: np.ndarray,
        query: str = "What should I do?",
        use_hardware: bool = True
    ) -> AcceleratedConsultation:
        """
        Full Oracle consultation with hardware acceleration.

        Pipeline:
        1. A10PED encodes observation to HD vector
        2. SB-852 estimates latent state
        3. Oracle Alpha explores futures (GPU-accelerated particles)
        4. Oracle Beta provides fast tactical prediction (DLA)
        5. Oracle Gamma arbitrates with hardware safety check
        """
        start_time = time.perf_counter()
        latency_breakdown = {}
        accelerators_used = []
        fallback_used = False

        # Stage 1: HDC Encoding
        stage_start = time.perf_counter()
        hd_vector = self.accelerated_encode(observation)
        latency_breakdown['hdc_encode_ms'] = (time.perf_counter() - stage_start) * 1000
        if self._a10ped is not None:
            accelerators_used.append('A10PED')

        # Stage 2: State Estimation
        stage_start = time.perf_counter()
        # Use SB-852 for state estimation from HD vector
        if self._sb852 is not None:
            result = self._sb852.infer(hd_vector[:1024], model_type='encoder')  # Truncate for encoder
            z_state = result.output[:64]  # Take first 64 dims as state
            accelerators_used.append('SB-852')
        else:
            # Fallback
            np.random.seed(44)
            decoder = np.random.randn(len(hd_vector), 64).astype(np.float32)
            z_state = np.dot(hd_vector, decoder)
            fallback_used = True
        latency_breakdown['state_estimation_ms'] = (time.perf_counter() - stage_start) * 1000

        # Stage 3: Oracle Consultation
        consensus = None
        if self.pythia is not None and ORACLE_AVAILABLE:
            stage_start = time.perf_counter()

            # Inject hardware accelerators into Oracle
            # (In production, would set these as callbacks)
            consensus = self.pythia.divine(
                z_state,
                observation,
                query=query,
                horizon=100
            )
            latency_breakdown['oracle_consultation_ms'] = (time.perf_counter() - stage_start) * 1000
        else:
            # Minimal fallback if Oracle not available
            latency_breakdown['oracle_consultation_ms'] = 0
            fallback_used = True

        # Stage 4: Safety Verification
        stage_start = time.perf_counter()
        if consensus is not None:
            action = consensus.final_action
        else:
            action = np.zeros(8, dtype=np.float32)

        metrics = {
            'energy_expenditure': float(np.sum(np.abs(action)) * 10),
            'reversibility_score': 0.85,
            'entropy_delta': 25.0,
            'decision_latency_us': sum(latency_breakdown.values()) * 1000
        }
        is_safe, violations = self.accelerated_safety_check(action, z_state, metrics)
        latency_breakdown['safety_check_ms'] = (time.perf_counter() - stage_start) * 1000

        if self._forest_kitten is not None:
            accelerators_used.append('Forest_Kitten')

        if violations:
            logger.warning("Safety violations: %s", violations)

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Update statistics
        self.total_consultations += 1
        self.total_latency_ms += total_latency_ms
        if accelerators_used:
            self.hardware_accelerated_count += 1

        return AcceleratedConsultation(
            consensus=consensus,
            total_latency_ms=total_latency_ms,
            hardware_latency_breakdown=latency_breakdown,
            accelerators_used=accelerators_used,
            fallback_used=fallback_used
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        avg_latency = (
            self.total_latency_ms / self.total_consultations
            if self.total_consultations > 0 else 0
        )
        hw_rate = (
            self.hardware_accelerated_count / self.total_consultations
            if self.total_consultations > 0 else 0
        )

        stats = {
            'total_consultations': self.total_consultations,
            'avg_latency_ms': avg_latency,
            'hardware_acceleration_rate': hw_rate,
            'cathedral_stats': self.cathedral.get_statistics() if self.cathedral else {},
        }

        if self.pythia is not None:
            stats['oracle_stats'] = self.pythia.get_triad_status()

        return stats

    def close(self):
        """Release all resources."""
        if self.cathedral is not None:
            self.cathedral.close()
        logger.info("Cathedral-Oracle bridge closed")


# ============================================================================
# Hardware-Accelerated Oracle Factory
# ============================================================================

def create_accelerated_oracle(
    simulation_mode: bool = True,
    num_particles: int = 50000
) -> CathedralOracleBridge:
    """
    Factory function to create a fully hardware-accelerated Oracle system.

    This is the recommended way to instantiate the complete system.
    """
    logger.info("Creating hardware-accelerated Oracle system...")

    # Create Cathedral with full hardware support
    cathedral = CathedralOrchestrator(
        simulation_mode=simulation_mode,
        enable_p2p=True,
        num_ensemble_particles=num_particles
    )

    # Create Pythia Triad if available
    pythia = None
    if ORACLE_AVAILABLE:
        pythia = PythiaTriad(
            latent_dim=64,
            action_dim=8,
            num_particles=num_particles
        )

    # Create bridge
    bridge = CathedralOracleBridge(
        cathedral=cathedral,
        pythia=pythia,
        simulation_mode=simulation_mode
    )

    logger.info("Hardware-accelerated Oracle ready")
    return bridge


# ============================================================================
# Example Usage
# ============================================================================

def example_cathedral_oracle():
    """Demonstrate Cathedral-Oracle integration."""
    print("CATHEDRAL-ORACLE BRIDGE DEMONSTRATION")
    print("=" * 70)

    # Create accelerated Oracle
    bridge = create_accelerated_oracle(simulation_mode=True, num_particles=5000)

    # Generate test observation
    observation = np.random.randn(100).astype(np.float32)

    # Consult Oracle with hardware acceleration
    print("\n1. Consulting Oracle with hardware acceleration...")
    result = bridge.consult_oracle(
        observation,
        query="Should I increase power allocation to the cooling system?"
    )

    print(f"\n   Results:")
    print(f"   Total latency: {result.total_latency_ms:.2f} ms")
    print(f"   Accelerators used: {result.accelerators_used}")
    print(f"   Fallback used: {result.fallback_used}")
    print(f"\n   Latency breakdown:")
    for stage, latency in result.hardware_latency_breakdown.items():
        print(f"     {stage}: {latency:.2f} ms")

    if result.consensus is not None:
        print(f"\n   Consensus:")
        print(f"     Action: {result.consensus.final_action}")
        print(f"     Confidence: {result.consensus.confidence:.0%}")
        print(f"     Safe: {result.consensus.safe}")

    # Individual accelerator tests
    print("\n2. Individual accelerator tests:")

    # HDC encoding
    hd = bridge.accelerated_encode(observation)
    print(f"   HDC encode: shape={hd.shape}, sparsity={np.mean(hd):.3f}")

    # DLA inference
    z = np.random.randn(64).astype(np.float32)
    action = np.random.randn(8).astype(np.float32)
    next_z = bridge.accelerated_inference(z, action)
    print(f"   DLA inference: shape={next_z.shape}")

    # Safety check
    metrics = {'energy_expenditure': 50, 'reversibility_score': 0.8, 'entropy_delta': 20, 'decision_latency_us': 500}
    is_safe, violations = bridge.accelerated_safety_check(action, z, metrics)
    print(f"   Safety check: safe={is_safe}, violations={len(violations)}")

    # Statistics
    print("\n3. Bridge statistics:")
    stats = bridge.get_statistics()
    print(f"   Total consultations: {stats['total_consultations']}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f} ms")
    print(f"   HW acceleration rate: {stats['hardware_acceleration_rate']:.0%}")

    bridge.close()


if __name__ == "__main__":
    example_cathedral_oracle()
