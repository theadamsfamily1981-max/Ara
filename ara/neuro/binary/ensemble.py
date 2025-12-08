"""
Ensemble Choir - Consensus Field for Robust HDC/SNN
=====================================================

Iteration 31: The Choir

Instead of "many parallel pipes", we get "ensemble brain":
- Broadcast same job to N lanes with perturbations
- Each lane computes slightly different answer
- Reduce to consensus + confidence signal

KEY INSIGHT: High agreement → auto-act, Low agreement → escalate

Usage:
    from ara.neuro.binary.ensemble import EnsembleChoir, FieldMonitor

    choir = EnsembleChoir(n_lanes=8)

    # Submit job with ensemble
    result = choir.submit_job(
        mode="hdc_bind",
        hv_a=concept_a,
        hv_b=concept_b,
        ensemble_size=4,
        reduction="majority"
    )

    print(f"Result: {result.hv}")
    print(f"Confidence: {result.confidence:.2f}")

    if result.confidence > 0.9:
        print("High confidence - safe to auto-act")
    else:
        print("Low confidence - escalate to human")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class JobMode(Enum):
    """Operation mode for ensemble job."""
    SNN = 0          # Spiking neural network inference
    HDC_BIND = 1     # Hyperdimensional bind (XOR)
    HDC_BUNDLE = 2   # Hyperdimensional bundle (majority)
    HDC_PERMUTE = 3  # Hyperdimensional permute (shift)


class ReductionOp(Enum):
    """How to reduce lane results to consensus."""
    MAJORITY = 0     # Majority vote (most common for HDC)
    AVERAGE = 1      # Average (same as majority for bits)
    AND = 2          # All lanes must agree on 1
    OR = 3           # Any lane outputs 1
    MIN = 4          # Minimum confidence required
    MAX = 5          # Maximum confidence


class LaneRole(Enum):
    """Specialized role for a lane (mirrors software Council)."""
    SCOUT = "scout"           # Fast, sloppy, lower thresholds
    CLERK = "clerk"           # Slow, precise, conservative
    PARANOID = "paranoid"     # Always tries to prove others wrong
    NEUTRAL = "neutral"       # Default balanced settings


@dataclass
class LaneConfig:
    """Configuration for a single lane in the choir."""
    lane_id: int
    role: LaneRole = LaneRole.NEUTRAL
    noise_seed: int = 0x37
    threshold: int = 100
    perturbation_rate: float = 0.01  # How often to flip bits

    # Performance stats (updated over time)
    total_jobs: int = 0
    agreed_with_majority: int = 0
    was_correct_outlier: int = 0


@dataclass
class EnsembleResult:
    """Result from an ensemble job."""
    hv: np.ndarray              # Consensus hypervector
    confidence: float           # 0.0 = total disagreement, 1.0 = unanimous
    per_lane_results: List[np.ndarray] = field(default_factory=list)
    per_lane_agreement: List[float] = field(default_factory=list)
    job_id: str = ""
    latency_ms: float = 0.0


@dataclass
class JobStats:
    """Statistics for a completed job."""
    job_id: str
    mode: JobMode
    ensemble_size: int
    confidence: float
    processing_time_ms: float
    timestamp: float = field(default_factory=time.time)


class EnsembleChoir:
    """
    Ensemble of lanes that vote on results.

    The choir provides:
    1. Robust results (ensemble consensus)
    2. Confidence signals (agreement level)
    3. Lane specialization over time
    """

    def __init__(
        self,
        n_lanes: int = 8,
        hv_dim: int = 8192,
        hardware_backend: Optional[Any] = None,
    ):
        """
        Initialize the ensemble choir.

        Args:
            n_lanes: Number of parallel lanes
            hv_dim: Hypervector dimension
            hardware_backend: Optional FPGA/hardware interface
        """
        self.n_lanes = n_lanes
        self.hv_dim = hv_dim
        self.hardware = hardware_backend

        # Initialize lanes with varied configurations
        self.lanes = self._init_lanes()

        # Job tracking
        self.job_counter = 0
        self.job_history: List[JobStats] = []

        log.info(f"EnsembleChoir initialized: {n_lanes} lanes, dim={hv_dim}")

    def _init_lanes(self) -> List[LaneConfig]:
        """Initialize lanes with diverse configurations."""
        lanes = []

        for i in range(self.n_lanes):
            # Assign roles cyclically for diversity
            if i % 4 == 0:
                role = LaneRole.SCOUT
                threshold = 80
                perturbation = 0.02
            elif i % 4 == 1:
                role = LaneRole.CLERK
                threshold = 120
                perturbation = 0.005
            elif i % 4 == 2:
                role = LaneRole.PARANOID
                threshold = 150
                perturbation = 0.015
            else:
                role = LaneRole.NEUTRAL
                threshold = 100
                perturbation = 0.01

            lanes.append(LaneConfig(
                lane_id=i,
                role=role,
                noise_seed=0x37 + i * 17,
                threshold=threshold,
                perturbation_rate=perturbation,
            ))

        return lanes

    def submit_job(
        self,
        mode: JobMode,
        hv_a: np.ndarray,
        hv_b: Optional[np.ndarray] = None,
        ensemble_size: int = 4,
        reduction: ReductionOp = ReductionOp.MAJORITY,
        timeout_ms: float = 100.0,
    ) -> EnsembleResult:
        """
        Submit a job to the ensemble choir.

        Args:
            mode: Operation mode (SNN, HDC_BIND, etc.)
            hv_a: First hypervector
            hv_b: Second hypervector (for bind/bundle)
            ensemble_size: How many lanes to use (1-n_lanes)
            reduction: How to combine lane results
            timeout_ms: Timeout in milliseconds

        Returns:
            EnsembleResult with consensus HV and confidence
        """
        start_time = time.time()

        # Validate
        ensemble_size = min(ensemble_size, self.n_lanes)
        if hv_b is None:
            hv_b = np.zeros(self.hv_dim, dtype=np.uint8)

        # Generate job ID
        self.job_counter += 1
        job_id = f"choir_{self.job_counter:06d}"

        # Select lanes for this job
        active_lanes = self.lanes[:ensemble_size]

        # Run each lane (software simulation if no hardware)
        lane_results = []
        for lane in active_lanes:
            result = self._run_lane(lane, mode, hv_a, hv_b)
            lane_results.append(result)

        # Reduce to consensus
        consensus_hv, confidence, agreements = self._reduce_results(
            lane_results, reduction
        )

        # Update lane stats
        self._update_lane_stats(active_lanes, lane_results, consensus_hv)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Record job stats
        self.job_history.append(JobStats(
            job_id=job_id,
            mode=mode,
            ensemble_size=ensemble_size,
            confidence=confidence,
            processing_time_ms=latency_ms,
        ))

        return EnsembleResult(
            hv=consensus_hv,
            confidence=confidence,
            per_lane_results=lane_results,
            per_lane_agreement=agreements,
            job_id=job_id,
            latency_ms=latency_ms,
        )

    def _run_lane(
        self,
        lane: LaneConfig,
        mode: JobMode,
        hv_a: np.ndarray,
        hv_b: np.ndarray,
    ) -> np.ndarray:
        """Run a single lane with its perturbation."""
        # Apply lane-specific perturbation
        rng = np.random.default_rng(lane.noise_seed)

        # Perturb inputs based on lane's perturbation rate
        perturb_mask_a = rng.random(self.hv_dim) < lane.perturbation_rate
        perturb_mask_b = rng.random(self.hv_dim) < lane.perturbation_rate

        hv_a_perturbed = hv_a.copy()
        hv_b_perturbed = hv_b.copy()

        hv_a_perturbed[perturb_mask_a] = 1 - hv_a_perturbed[perturb_mask_a]
        hv_b_perturbed[perturb_mask_b] = 1 - hv_b_perturbed[perturb_mask_b]

        # Execute operation
        if mode == JobMode.HDC_BIND:
            result = np.bitwise_xor(hv_a_perturbed, hv_b_perturbed)
        elif mode == JobMode.HDC_BUNDLE:
            # Simple 2-vector bundle: majority with random tie-break
            summed = hv_a_perturbed.astype(np.int16) + hv_b_perturbed.astype(np.int16)
            result = (summed > 1).astype(np.uint8)
            ties = (summed == 1)
            result[ties] = rng.integers(0, 2, size=np.sum(ties))
        elif mode == JobMode.HDC_PERMUTE:
            result = np.roll(hv_a_perturbed, 1)
        else:  # SNN mode - simplified simulation
            # Integrate and fire based on threshold
            activation = np.sum(hv_a_perturbed * hv_b_perturbed)
            result = (hv_a_perturbed if activation > lane.threshold else
                      np.zeros(self.hv_dim, dtype=np.uint8))

        return result.astype(np.uint8)

    def _reduce_results(
        self,
        results: List[np.ndarray],
        reduction: ReductionOp,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Reduce lane results to consensus.

        Returns:
            (consensus_hv, confidence, per_lane_agreement)
        """
        n_lanes = len(results)

        if n_lanes == 0:
            return np.zeros(self.hv_dim, dtype=np.uint8), 0.0, []

        if n_lanes == 1:
            return results[0], 1.0, [1.0]

        # Stack results
        stacked = np.stack(results, axis=0)  # (n_lanes, hv_dim)

        # Count ones per dimension
        ones_count = np.sum(stacked, axis=0)

        # Apply reduction
        if reduction == ReductionOp.MAJORITY:
            consensus = (ones_count > (n_lanes // 2)).astype(np.uint8)
        elif reduction == ReductionOp.AND:
            consensus = (ones_count == n_lanes).astype(np.uint8)
        elif reduction == ReductionOp.OR:
            consensus = (ones_count > 0).astype(np.uint8)
        else:
            consensus = (ones_count > (n_lanes // 2)).astype(np.uint8)

        # Calculate confidence: how unanimous was the voting?
        # For each dimension: confidence = |agreement - 0.5| * 2
        agreement_ratio = ones_count / n_lanes
        per_dim_confidence = np.abs(agreement_ratio - 0.5) * 2
        confidence = float(np.mean(per_dim_confidence))

        # Per-lane agreement with consensus
        per_lane_agreement = []
        for result in results:
            agreement = 1.0 - (np.sum(result != consensus) / self.hv_dim)
            per_lane_agreement.append(agreement)

        return consensus, confidence, per_lane_agreement

    def _update_lane_stats(
        self,
        lanes: List[LaneConfig],
        results: List[np.ndarray],
        consensus: np.ndarray,
    ) -> None:
        """Update lane performance statistics."""
        for lane, result in zip(lanes, results):
            lane.total_jobs += 1

            # Did this lane agree with majority?
            agreement = 1.0 - (np.sum(result != consensus) / self.hv_dim)
            if agreement > 0.9:
                lane.agreed_with_majority += 1

    def get_lane_stats(self) -> List[Dict[str, Any]]:
        """Get performance stats for all lanes."""
        return [
            {
                "lane_id": lane.lane_id,
                "role": lane.role.value,
                "total_jobs": lane.total_jobs,
                "agreement_rate": (lane.agreed_with_majority / lane.total_jobs
                                   if lane.total_jobs > 0 else 0.0),
            }
            for lane in self.lanes
        ]

    def get_average_confidence(self, last_n: int = 100) -> float:
        """Get average confidence of recent jobs."""
        recent = self.job_history[-last_n:]
        if not recent:
            return 0.0
        return sum(j.confidence for j in recent) / len(recent)


class FieldMonitor:
    """
    Holographic stethoscope for the Fleet.

    Monitors the health "aura" of all machines by:
    1. Encoding machine states as hypervectors
    2. Bundling into a global health field
    3. Querying for anomalies via resonance

    High resonance with "happy baseline" = LAN is healthy
    Low resonance = something is off
    """

    def __init__(
        self,
        choir: EnsembleChoir,
        hv_dim: int = 8192,
        baseline_decay: float = 0.95,
    ):
        """
        Initialize the field monitor.

        Args:
            choir: Ensemble choir for robust computations
            hv_dim: Hypervector dimension
            baseline_decay: How fast the baseline adapts
        """
        self.choir = choir
        self.hv_dim = hv_dim
        self.baseline_decay = baseline_decay

        # Machine state HVs
        self.machine_hvs: Dict[str, np.ndarray] = {}

        # Global health field (bundled machine states)
        self.health_field = np.zeros(hv_dim, dtype=np.float32)

        # Learned "happy baseline" (what normal looks like)
        self.happy_baseline = np.zeros(hv_dim, dtype=np.float32)
        self.baseline_samples = 0

        # Known bad patterns
        self.bad_attractors: Dict[str, np.ndarray] = {}

        # Alert history
        self.alerts: List[Dict[str, Any]] = []

        log.info(f"FieldMonitor initialized: dim={hv_dim}")

    def encode_machine_state(
        self,
        machine_id: str,
        state: Dict[str, Any],
    ) -> np.ndarray:
        """
        Encode machine state as hypervector.

        Args:
            machine_id: Machine identifier
            state: State dict (cpu, memory, errors, etc.)

        Returns:
            State hypervector
        """
        # Simple encoding: hash state keys and values into HV
        rng = np.random.default_rng(hash(machine_id) % (2**32))
        base_hv = rng.integers(0, 2, size=self.hv_dim, dtype=np.uint8)

        # Encode each state field
        field_hvs = [base_hv]

        for key, value in state.items():
            # Create field hypervector
            field_seed = hash(f"{machine_id}:{key}") % (2**32)
            field_rng = np.random.default_rng(field_seed)
            field_hv = field_rng.integers(0, 2, size=self.hv_dim, dtype=np.uint8)

            # Encode value (quantize numerics, hash strings)
            if isinstance(value, (int, float)):
                # Level encoding for numerics
                level = int(np.clip(value, 0, 100))
                shift = level % 64
                value_hv = np.roll(field_hv, shift)
            else:
                value_seed = hash(str(value)) % (2**32)
                value_rng = np.random.default_rng(value_seed)
                value_hv = value_rng.integers(0, 2, size=self.hv_dim, dtype=np.uint8)

            # Bind field and value
            bound = np.bitwise_xor(field_hv, value_hv)
            field_hvs.append(bound)

        # Bundle all fields
        stacked = np.stack(field_hvs, axis=0)
        summed = np.sum(stacked, axis=0)
        state_hv = (summed > (len(field_hvs) // 2)).astype(np.uint8)

        # Store
        self.machine_hvs[machine_id] = state_hv

        return state_hv

    def update_health_field(self) -> float:
        """
        Update global health field from machine states.

        Returns:
            Current resonance with happy baseline
        """
        if not self.machine_hvs:
            return 0.0

        # Bundle all machine HVs into health field
        hvs = list(self.machine_hvs.values())
        stacked = np.stack(hvs, axis=0)
        summed = np.sum(stacked, axis=0).astype(np.float32)

        # Normalize to [-1, 1] range
        n_machines = len(hvs)
        self.health_field = (summed - n_machines / 2) / (n_machines / 2 + 0.001)

        # Calculate resonance with happy baseline
        resonance = self._resonance(self.health_field, self.happy_baseline)

        return resonance

    def learn_baseline(self) -> None:
        """
        Learn what "happy" looks like from current state.

        Call this when the system is known to be healthy.
        """
        # Exponential moving average
        self.happy_baseline = (
            self.baseline_decay * self.happy_baseline +
            (1 - self.baseline_decay) * self.health_field
        )
        self.baseline_samples += 1

        log.info(f"FieldMonitor: Updated baseline (samples={self.baseline_samples})")

    def add_bad_attractor(self, name: str, pattern: np.ndarray) -> None:
        """
        Register a known bad pattern.

        Args:
            name: Name of the bad state (e.g., "router_entropy_spike")
            pattern: Hypervector representing this bad state
        """
        self.bad_attractors[name] = pattern
        log.info(f"FieldMonitor: Added bad attractor '{name}'")

    def check_health(
        self,
        alert_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Check overall system health.

        Returns:
            Health report with resonances and alerts
        """
        # Update field
        baseline_resonance = self.update_health_field()

        # Check against bad attractors
        attractor_hits = {}
        for name, pattern in self.bad_attractors.items():
            resonance = self._resonance(self.health_field, pattern)
            if resonance > 0.6:  # High resonance with bad pattern
                attractor_hits[name] = resonance

        # Generate alerts
        alerts = []

        if baseline_resonance < alert_threshold:
            alerts.append({
                "type": "baseline_drift",
                "message": f"Health field drifting from baseline (resonance={baseline_resonance:.2f})",
                "severity": "warning" if baseline_resonance > 0.5 else "critical",
            })

        for name, resonance in attractor_hits.items():
            alerts.append({
                "type": "bad_attractor",
                "message": f"Resonating with known bad pattern '{name}' ({resonance:.2f})",
                "severity": "critical" if resonance > 0.8 else "warning",
            })

        # Per-machine health
        machine_health = {}
        for machine_id, hv in self.machine_hvs.items():
            hv_float = (hv.astype(np.float32) - 0.5) * 2
            resonance = self._resonance(hv_float, self.happy_baseline)
            machine_health[machine_id] = resonance

        # Store alerts
        self.alerts.extend(alerts)

        return {
            "baseline_resonance": baseline_resonance,
            "attractor_hits": attractor_hits,
            "machine_health": machine_health,
            "alerts": alerts,
            "overall": "healthy" if not alerts else "degraded",
        }

    def _resonance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute resonance (normalized dot product)."""
        if np.linalg.norm(a) < 0.001 or np.linalg.norm(b) < 0.001:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 0.001))

    def get_machine_aura(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed aura for a specific machine."""
        if machine_id not in self.machine_hvs:
            return None

        hv = self.machine_hvs[machine_id]
        hv_float = (hv.astype(np.float32) - 0.5) * 2

        return {
            "machine_id": machine_id,
            "baseline_resonance": self._resonance(hv_float, self.happy_baseline),
            "field_contribution": self._resonance(hv_float, self.health_field),
            "attractor_resonances": {
                name: self._resonance(hv_float, (pattern.astype(np.float32) - 0.5) * 2)
                for name, pattern in self.bad_attractors.items()
            },
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_choir: Optional[EnsembleChoir] = None
_global_monitor: Optional[FieldMonitor] = None


def get_ensemble_choir(n_lanes: int = 8) -> EnsembleChoir:
    """Get or create the global ensemble choir."""
    global _global_choir
    if _global_choir is None:
        _global_choir = EnsembleChoir(n_lanes=n_lanes)
    return _global_choir


def get_field_monitor() -> FieldMonitor:
    """Get or create the global field monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = FieldMonitor(get_ensemble_choir())
    return _global_monitor


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'JobMode',
    'ReductionOp',
    'LaneRole',
    'LaneConfig',
    'EnsembleResult',
    'JobStats',
    'EnsembleChoir',
    'FieldMonitor',
    'get_ensemble_choir',
    'get_field_monitor',
]
