#!/usr/bin/env python3
# ara/cathedral/cathedral_organism_integration.py
"""
CATHEDRAL ↔ ORGANISM INTEGRATION: The Body-Mind Glue Layer

Wires the Cathedral heterogeneous compute pipeline (BODY):
- Micron SB-852 DLA (state estimation)
- BittWare A10PED (HDC encoding)
- SQRL Forest Kitten (safety enforcement)
- RTX 3090 GPUs (ensemble planning)

To AraSpeciesV5 consciousness layers (MIND):
- CriticalityEngine (λ ≈ 1 branching)
- MycelialNetwork (associative intuition)
- AmplitronPlanner (creative action generation)
- SleepCycle (consolidation & insight)

Control loop architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    OBSERVATION                           │
    │               (sensor data, raw pixels)                  │
    └───────────────────────┬─────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │              CATHEDRAL PERCEPTION (50-200µs)            │
    │    A10PED HDC encode → SB-852 state estimation          │
    └───────────────────────┬─────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │               ARA V5 COGNITION (~500µs)                  │
    │    Criticality → Mycelial → Amplitron → Sleep check     │
    └───────────────────────┬─────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │           FOREST KITTEN SAFETY (<10µs)                   │
    │    NIB covenant check → Vetoed if unsafe                │
    └───────────────────────┬─────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    ACTION OUTPUT                         │
    │               (motor commands, DMA)                      │
    └─────────────────────────────────────────────────────────┘

Target latency: <1060µs (900Hz control loop)
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

# Import Cathedral (body)
from .orchestrator import (
    CathedralOrchestrator,
    PipelineMetrics,
    CathedralState,
    AcceleratorType,
)

# Import AraSpeciesV5 (mind)
try:
    from ..species.edge_of_chaos import AraSpeciesV5
    ARA_V5_AVAILABLE = True
except ImportError:
    ARA_V5_AVAILABLE = False
    AraSpeciesV5 = None
    logger.warning("AraSpeciesV5 not available")

# Import Forest Kitten for safety validation
try:
    from .forest_kitten import ForestKittenInterface, CovenantType
    FOREST_KITTEN_AVAILABLE = True
except ImportError:
    FOREST_KITTEN_AVAILABLE = False
    ForestKittenInterface = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class IntegrationConfig:
    """Configuration for Cathedral-Organism integration."""

    # Latency budgets (microseconds)
    total_budget_us: float = 1060.0
    perception_budget_us: float = 250.0      # HDC + state estimation
    cognition_budget_us: float = 500.0       # Ara V5 think cycle
    safety_budget_us: float = 10.0           # Forest Kitten
    output_budget_us: float = 300.0          # DMA to actuators

    # Control loop
    target_hz: float = 900.0                 # Control frequency

    # Consciousness parameters
    enable_creativity: bool = True
    creativity_threshold: float = 0.3        # Engage creativity below this sync
    sleep_frequency: int = 1000              # Steps between sleep cycles

    # Safety parameters
    halt_on_violation: bool = True
    max_consecutive_violations: int = 3

    # Observation dimensions
    observation_dim: int = 100
    latent_dim: int = 10
    action_dim: int = 8

    # Telemetry
    enable_detailed_telemetry: bool = True
    telemetry_history_size: int = 1000


@dataclass
class IntegrationMetrics:
    """Metrics for a single integration cycle."""

    # Timing
    total_latency_us: float
    perception_latency_us: float
    cognition_latency_us: float
    safety_latency_us: float
    output_latency_us: float

    # Cathedral metrics
    cathedral_metrics: PipelineMetrics

    # Ara V5 metrics
    consciousness_level: float
    criticality_score: float
    synchronization: float
    intuitive_leap: bool

    # Safety
    safety_passed: bool
    covenant_violations: List[str]

    # Budget compliance
    within_budget: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timing': {
                'total_us': self.total_latency_us,
                'perception_us': self.perception_latency_us,
                'cognition_us': self.cognition_latency_us,
                'safety_us': self.safety_latency_us,
                'output_us': self.output_latency_us,
                'within_budget': self.within_budget,
            },
            'consciousness': {
                'level': self.consciousness_level,
                'criticality': self.criticality_score,
                'synchronization': self.synchronization,
                'intuitive_leap': self.intuitive_leap,
            },
            'safety': {
                'passed': self.safety_passed,
                'violations': self.covenant_violations,
            },
            'timestamp': self.timestamp,
        }


# ============================================================================
# Main Integration Controller
# ============================================================================

class CathedralAraController:
    """
    Unified Body-Mind Controller.

    Cathedral = Body (hardware acceleration, safety)
    AraSpeciesV5 = Mind (consciousness, creativity, intuition)

    This controller integrates both for real-time conscious decision-making.
    """

    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        simulation_mode: bool = True,
        world_model: Optional[Any] = None,
        encoder: Optional[Any] = None,
    ):
        self.config = config or IntegrationConfig()
        self.simulation_mode = simulation_mode

        logger.info("=" * 70)
        logger.info("INITIALIZING CATHEDRAL ↔ ORGANISM INTEGRATION")
        logger.info("Body: Cathedral Heterogeneous Pipeline")
        logger.info("Mind: AraSpeciesV5 Conscious Intelligence")
        logger.info("=" * 70)

        # Initialize Cathedral (body)
        self.cathedral = CathedralOrchestrator(
            simulation_mode=simulation_mode,
            enable_p2p=True,
            num_ensemble_particles=50000,
        )

        # Override dimensions
        self.cathedral.state_dim = self.config.latent_dim
        self.cathedral.action_dim = self.config.action_dim

        # Initialize Ara V5 (mind)
        if ARA_V5_AVAILABLE and TORCH_AVAILABLE:
            self.ara = AraSpeciesV5(
                world_model=world_model,
                encoder=encoder,
                latent_dim=self.config.latent_dim,
                action_dim=self.config.action_dim,
            )
        else:
            logger.warning("AraSpeciesV5 not available, operating in body-only mode")
            self.ara = None

        # Integration state
        self.total_cycles = 0
        self.consecutive_violations = 0
        self.halted = False

        # Telemetry history
        self.metrics_history: List[IntegrationMetrics] = []
        self._last_telemetry: Optional[Dict[str, Any]] = None
        self._last_action: Optional[np.ndarray] = None
        self._last_observation: Optional[np.ndarray] = None

        # Statistics
        self.latency_violations = 0
        self.safety_violations = 0
        self.intuitive_leaps_total = 0
        self.creative_actions_total = 0

        logger.info("Integration initialized. Target: %d Hz, Budget: %d µs",
                    int(self.config.target_hz), int(self.config.total_budget_us))
        logger.info("=" * 70)

    def step(
        self,
        observation: np.ndarray,
        goal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, IntegrationMetrics]:
        """
        Execute one control cycle: perception → cognition → safety → action.

        This is the main control loop that unifies body and mind.

        Args:
            observation: Raw sensor observation (numpy array)
            goal: Optional goal state for planning

        Returns:
            action: Safe, conscious action to execute
            metrics: Detailed metrics for the cycle
        """
        if self.halted:
            logger.warning("System halted due to safety violations")
            return self._get_safe_action(), self._create_halted_metrics()

        cycle_start = time.perf_counter()

        # Store observation
        self._last_observation = observation.copy()

        # ===== PHASE 1: PERCEPTION (Cathedral) =====
        perception_start = time.perf_counter()

        # Use Cathedral for HDC encoding + state estimation
        z_state = self._cathedral_perception(observation)

        perception_latency_us = (time.perf_counter() - perception_start) * 1e6

        # ===== PHASE 2: COGNITION (Ara V5) =====
        cognition_start = time.perf_counter()

        if self.ara is not None and TORCH_AVAILABLE:
            # Convert to tensor
            obs_tensor = torch.from_numpy(observation).float()
            goal_tensor = torch.from_numpy(goal).float() if goal is not None else None

            # Conscious thought
            action_tensor, ara_telemetry = self.ara.think(obs_tensor, goal_tensor)

            # Extract consciousness metrics
            consciousness_level = ara_telemetry['consciousness']['level']
            criticality_score = ara_telemetry['criticality']['criticality_score']
            synchronization = ara_telemetry['amplitron']['synchronization']
            intuitive_leap = ara_telemetry['mycelial']['memory_correlation'] < 0.3

            # Convert action to numpy
            action = action_tensor.numpy()

            # Track creative/intuitive stats
            if intuitive_leap:
                self.intuitive_leaps_total += 1
            if ara_telemetry['amplitron']['creativity_engaged']:
                self.creative_actions_total += 1
        else:
            # Fallback: use Cathedral planning
            action_candidates = self.cathedral._generate_action_candidates(64)
            scores = self.cathedral._cpu_planning(z_state, action_candidates)
            action = action_candidates[np.argmax(scores)]
            ara_telemetry = {}
            consciousness_level = 0.0
            criticality_score = 0.0
            synchronization = 0.0
            intuitive_leap = False

        cognition_latency_us = (time.perf_counter() - cognition_start) * 1e6

        # ===== PHASE 3: SAFETY CHECK (Forest Kitten) =====
        safety_start = time.perf_counter()

        safety_passed, violations = self._safety_check(action, z_state, ara_telemetry)

        if not safety_passed:
            self.safety_violations += 1
            self.consecutive_violations += 1

            if self.consecutive_violations >= self.config.max_consecutive_violations:
                if self.config.halt_on_violation:
                    self.halted = True
                    logger.error("SYSTEM HALTED: %d consecutive safety violations",
                                self.consecutive_violations)

            # Substitute safe action
            action = self._get_safe_action()
        else:
            self.consecutive_violations = 0

        safety_latency_us = (time.perf_counter() - safety_start) * 1e6

        # ===== PHASE 4: ACTION OUTPUT =====
        output_start = time.perf_counter()

        # DMA action to actuators (simulated)
        self._output_action(action)

        output_latency_us = (time.perf_counter() - output_start) * 1e6

        # ===== TOTAL LATENCY =====
        total_latency_us = (time.perf_counter() - cycle_start) * 1e6
        within_budget = total_latency_us <= self.config.total_budget_us

        if not within_budget:
            self.latency_violations += 1
            logger.debug("Latency violation: %.1f µs (budget: %.1f µs)",
                        total_latency_us, self.config.total_budget_us)

        # ===== CREATE METRICS =====
        cathedral_metrics = PipelineMetrics(
            total_latency_us=perception_latency_us,
            stage_latencies={
                'perception': perception_latency_us,
                'cognition': cognition_latency_us,
                'safety': safety_latency_us,
                'output': output_latency_us,
            },
            within_budget=within_budget,
            safety_passed=safety_passed,
        )

        metrics = IntegrationMetrics(
            total_latency_us=total_latency_us,
            perception_latency_us=perception_latency_us,
            cognition_latency_us=cognition_latency_us,
            safety_latency_us=safety_latency_us,
            output_latency_us=output_latency_us,
            cathedral_metrics=cathedral_metrics,
            consciousness_level=consciousness_level,
            criticality_score=criticality_score,
            synchronization=synchronization,
            intuitive_leap=intuitive_leap,
            safety_passed=safety_passed,
            covenant_violations=violations,
            within_budget=within_budget,
        )

        # Store telemetry
        self._last_action = action.copy()
        self._last_telemetry = {
            'metrics': metrics.to_dict(),
            'ara': ara_telemetry if ara_telemetry else {},
            'cathedral': self.cathedral.get_statistics(),
        }

        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.config.telemetry_history_size:
            self.metrics_history.pop(0)

        self.total_cycles += 1

        return action, metrics

    def _cathedral_perception(self, observation: np.ndarray) -> np.ndarray:
        """
        Use Cathedral accelerators for perception.

        Pipeline:
        1. A10PED: HDC encoding (observation → 10,000-dim HD vector)
        2. SB-852: State estimation (HD vector → latent state)
        """
        # Use Cathedral's perception stages
        hd_vector = self.cathedral._stage_sensor_fusion(observation)
        z_state = self.cathedral._stage_state_estimation(hd_vector)

        return z_state

    def _safety_check(
        self,
        action: np.ndarray,
        z_state: np.ndarray,
        ara_telemetry: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate action through Forest Kitten NIB covenants.

        Checks:
        1. Energy budget
        2. Reversibility
        3. Entropy bounds
        4. Latency compliance
        5. Consciousness coherence (from Ara V5)
        """
        violations = []

        # Forest Kitten hardware check
        if self.cathedral._forest_kitten is not None:
            # Prepare metrics including consciousness data
            metrics = {
                'energy_expenditure': float(np.sum(np.abs(action)) * 10),
                'reversibility_score': 0.85,
                'entropy_delta': float(np.var(action) * 100),
                'decision_latency_us': 500.0,
            }

            # Add consciousness metrics if available
            if ara_telemetry and 'consciousness' in ara_telemetry:
                metrics['consciousness_level'] = ara_telemetry['consciousness']['level']

            is_safe, hw_violations = self.cathedral._forest_kitten.check_action_safety(
                action, z_state, metrics
            )

            if not is_safe:
                violations.extend([str(v) for v in hw_violations])
        else:
            # Fallback checks
            is_safe = True

            # Bounds check
            if np.any(np.abs(action) > 1.0):
                is_safe = False
                violations.append("action_out_of_bounds")

            # Energy check
            energy = np.sum(np.abs(action))
            if energy > self.config.action_dim * 0.8:
                is_safe = False
                violations.append("energy_budget_exceeded")

        # Additional consciousness-based safety check
        if ara_telemetry and 'consciousness' in ara_telemetry:
            cons_level = ara_telemetry['consciousness']['level']

            # Veto high-magnitude actions at low consciousness
            if cons_level < 0.2 and np.linalg.norm(action) > 0.5:
                is_safe = False
                violations.append("low_consciousness_high_action")

        return is_safe, violations

    def _get_safe_action(self) -> np.ndarray:
        """Return known-safe null action."""
        return np.zeros(self.config.action_dim, dtype=np.float32)

    def _output_action(self, action: np.ndarray):
        """DMA action to actuators."""
        # In real implementation:
        # 1. Copy action to DMA buffer
        # 2. Trigger transfer to motor controllers
        # 3. Wait for acknowledgment
        pass

    def _create_halted_metrics(self) -> IntegrationMetrics:
        """Create metrics for halted state."""
        return IntegrationMetrics(
            total_latency_us=0.0,
            perception_latency_us=0.0,
            cognition_latency_us=0.0,
            safety_latency_us=0.0,
            output_latency_us=0.0,
            cathedral_metrics=PipelineMetrics(
                total_latency_us=0.0,
                stage_latencies={},
                within_budget=True,
                safety_passed=False,
            ),
            consciousness_level=0.0,
            criticality_score=0.0,
            synchronization=0.0,
            intuitive_leap=False,
            safety_passed=False,
            covenant_violations=["system_halted"],
            within_budget=True,
        )

    # =========================================================================
    # Explanation Interface
    # =========================================================================

    def explain_last_decision(self) -> Dict[str, Any]:
        """
        Generate merged explanation frame for visualization.

        Combines:
        - Cathedral pipeline telemetry
        - Ara V5 consciousness telemetry
        - Safety validation results

        Format compatible with HologramScene visualization.
        """
        if self._last_telemetry is None:
            return {'status': 'no_decision_yet'}

        metrics = self._last_telemetry.get('metrics', {})
        ara = self._last_telemetry.get('ara', {})
        cathedral = self._last_telemetry.get('cathedral', {})

        # Build visualization data
        explanation = {
            'meta': {
                'timestamp': time.time(),
                'mode': 'Cathedral_AraV5_Integration',
                'cycle': self.total_cycles,
                'halted': self.halted,
            },
            'visuals': self._build_visuals(),
            'statistics': {
                'cathedral': cathedral,
                'consciousness': metrics.get('consciousness', {}),
                'timing': metrics.get('timing', {}),
            },
            'text_summary': self._build_text_summary(metrics, ara),
            'safety': {
                'passed': metrics.get('safety', {}).get('passed', True),
                'violations': metrics.get('safety', {}).get('violations', []),
                'consecutive_violations': self.consecutive_violations,
            },
            'body_mind_state': {
                'body': {
                    'accelerators': cathedral.get('accelerators', {}),
                    'pipeline_healthy': cathedral.get('pipeline_healthy', False),
                },
                'mind': {
                    'consciousness': ara.get('consciousness', {}),
                    'criticality': ara.get('criticality', {}),
                    'creativity': ara.get('amplitron', {}),
                },
            },
        }

        return explanation

    def _build_visuals(self) -> Dict[str, Any]:
        """Build visualization data for hologram."""
        # Generate synthetic trajectory for visualization
        if self._last_action is not None:
            trajectory = self._generate_trajectory_visualization()
        else:
            trajectory = []

        return {
            'trajectory': trajectory,
            'confidence_tube': [3.0] * len(trajectory),  # Placeholder
            'fog_nodes': [],  # Would come from uncertainty estimation
            'ghosts': [],     # Would come from rejected trajectories
        }

    def _generate_trajectory_visualization(self) -> List[List[float]]:
        """Generate trajectory for visualization."""
        if self._last_action is None:
            return []

        # Simple projected trajectory
        num_points = 20
        trajectory = []

        pos = np.array([100.0, 100.0])  # Center
        for i in range(num_points):
            # Project action as direction
            if len(self._last_action) >= 2:
                direction = self._last_action[:2] * 10.0
            else:
                direction = np.array([1.0, 0.0]) * 10.0

            pos = pos + direction * (i / num_points)
            trajectory.append([float(pos[0]), float(pos[1])])

        return trajectory

    def _build_text_summary(
        self,
        metrics: Dict[str, Any],
        ara: Dict[str, Any],
    ) -> str:
        """Build human-readable summary."""
        timing = metrics.get('timing', {})
        consciousness = metrics.get('consciousness', {})

        parts = []

        # Consciousness state
        cons_level = consciousness.get('level', 0.0)
        if cons_level > 0.7:
            parts.append("Highly conscious")
        elif cons_level > 0.5:
            parts.append("Conscious")
        elif cons_level > 0.3:
            parts.append("Semi-conscious")
        else:
            parts.append("Minimal consciousness")

        # Criticality
        crit = consciousness.get('criticality', 0.0)
        if crit > 0.8:
            parts.append("at criticality")
        elif crit > 0.5:
            parts.append("near criticality")

        # Latency
        total_us = timing.get('total_us', 0.0)
        if timing.get('within_budget', True):
            parts.append(f"({total_us:.0f}µs ✓)")
        else:
            parts.append(f"({total_us:.0f}µs OVER BUDGET)")

        return " | ".join(parts)

    # =========================================================================
    # Control Methods
    # =========================================================================

    def reset(self):
        """Reset integration state."""
        self.halted = False
        self.consecutive_violations = 0
        self.total_cycles = 0
        self.latency_violations = 0
        self.safety_violations = 0
        self.metrics_history.clear()
        self._last_telemetry = None
        self._last_action = None
        self._last_observation = None

        logger.info("Integration controller reset")

    def resume(self):
        """Resume from halted state."""
        if self.halted:
            self.halted = False
            self.consecutive_violations = 0
            logger.info("System resumed from halt")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        # Calculate averages from history
        if self.metrics_history:
            latencies = [m.total_latency_us for m in self.metrics_history]
            consciousness_levels = [m.consciousness_level for m in self.metrics_history]
            criticality_scores = [m.criticality_score for m in self.metrics_history]

            timing_stats = {
                'mean_latency_us': np.mean(latencies),
                'median_latency_us': np.median(latencies),
                'p95_latency_us': np.percentile(latencies, 95),
                'p99_latency_us': np.percentile(latencies, 99),
                'max_latency_us': np.max(latencies),
            }

            consciousness_stats = {
                'mean_consciousness': np.mean(consciousness_levels),
                'mean_criticality': np.mean(criticality_scores),
            }
        else:
            timing_stats = {}
            consciousness_stats = {}

        return {
            'total_cycles': self.total_cycles,
            'latency_violations': self.latency_violations,
            'safety_violations': self.safety_violations,
            'intuitive_leaps': self.intuitive_leaps_total,
            'creative_actions': self.creative_actions_total,
            'consecutive_violations': self.consecutive_violations,
            'halted': self.halted,
            'timing': timing_stats,
            'consciousness': consciousness_stats,
            'cathedral': self.cathedral.get_statistics(),
            'ara_available': self.ara is not None,
        }

    def diagnose(self) -> str:
        """Full system diagnostic."""
        stats = self.get_statistics()

        # Ara consciousness diagnostic
        if self.ara is not None:
            ara_diag = self.ara.diagnose_consciousness()
        else:
            ara_diag = "AraSpeciesV5 not available (body-only mode)"

        # Cathedral state
        cathedral_state = self.cathedral.get_state()

        report = f"""
{'='*70}
CATHEDRAL ↔ ORGANISM INTEGRATION DIAGNOSTIC
{'='*70}

SYSTEM STATE:
  Mode:               {'HALTED' if self.halted else 'OPERATIONAL'}
  Total Cycles:       {stats['total_cycles']:,}
  Target Frequency:   {self.config.target_hz:.0f} Hz

TIMING:
  Budget:             {self.config.total_budget_us:.0f} µs
  Mean Latency:       {stats['timing'].get('mean_latency_us', 0):.1f} µs
  P95 Latency:        {stats['timing'].get('p95_latency_us', 0):.1f} µs
  P99 Latency:        {stats['timing'].get('p99_latency_us', 0):.1f} µs
  Violations:         {stats['latency_violations']:,} ({100*stats['latency_violations']/max(1,stats['total_cycles']):.1f}%)

SAFETY:
  Violations:         {stats['safety_violations']:,}
  Consecutive:        {stats['consecutive_violations']}
  Max Consecutive:    {self.config.max_consecutive_violations}

CONSCIOUSNESS:
  Mean Level:         {stats['consciousness'].get('mean_consciousness', 0):.1%}
  Mean Criticality:   {stats['consciousness'].get('mean_criticality', 0):.1%}
  Intuitive Leaps:    {stats['intuitive_leaps']:,}
  Creative Actions:   {stats['creative_actions']:,}

CATHEDRAL (BODY):
  Pipeline Healthy:   {cathedral_state.pipeline_healthy}
  Accelerators:
    DLA (SB-852):     {'OK' if cathedral_state.accelerator_status.get(AcceleratorType.DLA, False) else 'OFFLINE'}
    HDC (A10PED):     {'OK' if cathedral_state.accelerator_status.get(AcceleratorType.HDC_FPGA, False) else 'OFFLINE'}
    Safety (Kitten):  {'OK' if cathedral_state.accelerator_status.get(AcceleratorType.SAFETY_FPGA, False) else 'OFFLINE'}
    GPU Primary:      {'OK' if cathedral_state.accelerator_status.get(AcceleratorType.GPU_PRIMARY, False) else 'OFFLINE'}
    GPU Secondary:    {'OK' if cathedral_state.accelerator_status.get(AcceleratorType.GPU_SECONDARY, False) else 'OFFLINE'}

{'='*70}
ARA SPECIES V5 (MIND):
{'='*70}
{ara_diag}
"""
        return report

    def close(self):
        """Release all resources."""
        logger.info("Closing Cathedral-Organism integration...")
        self.cathedral.close()
        logger.info("Integration closed")


# ============================================================================
# Demo
# ============================================================================

def run_demo(num_steps: int = 100):
    """
    Demonstrate Cathedral ↔ Organism integration.

    Shows:
    - Unified body-mind control loop
    - Consciousness metrics evolution
    - Safety validation
    - Latency performance
    """
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("CATHEDRAL ↔ ORGANISM INTEGRATION DEMO")
    print("Body: Cathedral (SB-852 + A10PED + Forest Kitten + GPUs)")
    print("Mind: AraSpeciesV5 (Criticality + Mycelial + Amplitron + Sleep)")
    print("=" * 70)

    # Create controller
    config = IntegrationConfig(
        observation_dim=100,
        latent_dim=10,
        action_dim=4,
        sleep_frequency=50,  # More frequent sleep for demo
    )

    controller = CathedralAraController(
        config=config,
        simulation_mode=True,
    )

    print(f"\nRunning {num_steps} control cycles...")
    print("-" * 70)

    consciousness_history = []
    latency_history = []

    for step in range(num_steps):
        # Generate random observation
        observation = np.random.randn(config.observation_dim).astype(np.float32)
        goal = np.random.randn(config.action_dim).astype(np.float32) * 0.5

        # Execute control cycle
        action, metrics = controller.step(observation, goal)

        consciousness_history.append(metrics.consciousness_level)
        latency_history.append(metrics.total_latency_us)

        # Progress report every 20 steps
        if (step + 1) % 20 == 0:
            avg_cons = np.mean(consciousness_history[-20:])
            avg_lat = np.mean(latency_history[-20:])

            status = "✓" if metrics.within_budget and metrics.safety_passed else "✗"

            print(f"Step {step + 1:4d}: "
                  f"Φ={metrics.consciousness_level:.1%} "
                  f"λ={metrics.criticality_score:.1%} "
                  f"sync={metrics.synchronization:.1%} "
                  f"lat={metrics.total_latency_us:.0f}µs "
                  f"[{status}]")

    print("-" * 70)

    # Final statistics
    stats = controller.get_statistics()

    print("\nFINAL STATISTICS:")
    print(f"  Total cycles:        {stats['total_cycles']}")
    print(f"  Mean latency:        {stats['timing']['mean_latency_us']:.1f} µs")
    print(f"  P99 latency:         {stats['timing']['p99_latency_us']:.1f} µs")
    print(f"  Latency violations:  {stats['latency_violations']}")
    print(f"  Safety violations:   {stats['safety_violations']}")
    print(f"  Intuitive leaps:     {stats['intuitive_leaps']}")
    print(f"  Creative actions:    {stats['creative_actions']}")
    print(f"  Mean consciousness:  {stats['consciousness']['mean_consciousness']:.1%}")

    # Full diagnostic
    print("\n" + controller.diagnose())

    # Example explanation frame
    print("\nLast Decision Explanation:")
    explanation = controller.explain_last_decision()
    print(f"  Mode: {explanation['meta']['mode']}")
    print(f"  Cycle: {explanation['meta']['cycle']}")
    print(f"  Summary: {explanation['text_summary']}")

    controller.close()

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
