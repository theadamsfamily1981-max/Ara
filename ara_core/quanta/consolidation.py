#!/usr/bin/env python3
"""
QUANTA v2.0 Consolidation Scheduler
====================================

Memory consolidation with the coupled optimization objective:

    max [T_s + A_g(σ*) - |D_NIB| - η_gap + C(r,L)/L]

subject to:
    - retention ≥ baseline
    - rank r ≤ r_opt(n) = c·n^0.25

Implements PRR-LoRA-style rank reduction with:
- Micro consolidation (σ_high, online RL)
- Replay consolidation (σ_medium)
- Structural consolidation (σ_low, SVD)

Timescale mapping: τ_hippocampus : τ_neocortex = 1:100
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

from .metrics import QUANTAMetrics, MetricStatus, compute_quanta_metrics


class ConsolidationPhase(str, Enum):
    """Memory consolidation phases."""
    MICRO = "micro"              # Online RL, high stress (σ_high)
    REPLAY = "replay"            # Experience replay, medium stress
    STRUCTURAL = "structural"    # SVD-based, low stress


@dataclass
class ConsolidationConfig:
    """Configuration for QUANTA consolidation."""
    # Stress levels per phase
    sigma_micro: float = 0.15
    sigma_replay: float = 0.10       # Optimal σ*
    sigma_structural: float = 0.05

    # Timescale ratios (τ_hippocampus : τ_neocortex = 1:100)
    micro_interval_s: float = 1.0    # Every second
    replay_interval_s: float = 60.0  # Every minute
    structural_interval_s: float = 3600.0  # Every hour

    # Rank constraints
    rank_coefficient: float = 1.0    # c in r_opt = c·n^0.25
    min_rank: int = 4
    max_rank_ratio: float = 0.5      # Max rank as fraction of dim

    # Optimization targets
    ts_target: float = 0.92
    ag_target: float = 0.01
    nib_target: float = 0.1
    gft_target_pct: float = 0.90

    # Retention constraint
    min_retention: float = 0.95


@dataclass
class ConsolidationEvent:
    """Record of a consolidation event."""
    event_id: str
    phase: ConsolidationPhase
    timestamp: float

    # Before/after metrics
    metrics_before: Optional[QUANTAMetrics] = None
    metrics_after: Optional[QUANTAMetrics] = None

    # Changes made
    rank_reduced: bool = False
    rank_before: int = 0
    rank_after: int = 0

    # Retention check
    retention: float = 1.0
    passed_constraints: bool = True


@dataclass
class ConsolidationSchedule:
    """Schedule for multi-phase consolidation."""
    schedule_id: str
    created_at: float = field(default_factory=time.time)

    # Phase timing
    last_micro: float = 0.0
    last_replay: float = 0.0
    last_structural: float = 0.0

    # Event history
    events: List[ConsolidationEvent] = field(default_factory=list)

    # Metrics trajectory
    metrics_history: List[QUANTAMetrics] = field(default_factory=list)

    def should_run_phase(self, phase: ConsolidationPhase,
                         config: ConsolidationConfig) -> bool:
        """Check if a phase should run based on timing."""
        now = time.time()

        if phase == ConsolidationPhase.MICRO:
            return (now - self.last_micro) >= config.micro_interval_s
        elif phase == ConsolidationPhase.REPLAY:
            return (now - self.last_replay) >= config.replay_interval_s
        elif phase == ConsolidationPhase.STRUCTURAL:
            return (now - self.last_structural) >= config.structural_interval_s

        return False


class QUANTAConsolidator:
    """
    Memory consolidation engine implementing QUANTA v2.0.

    Optimizes the coupled objective:
        max [T_s + A_g(σ*) - |D_NIB| - η_gap + C(r,L)/L]

    Usage:
        consolidator = QUANTAConsolidator()

        # Check if consolidation needed
        if consolidator.should_consolidate(current_metrics):
            new_weights = consolidator.consolidate(
                weights,
                phase=ConsolidationPhase.REPLAY
            )
    """

    def __init__(self, config: ConsolidationConfig = None):
        self.config = config or ConsolidationConfig()
        self.schedule = ConsolidationSchedule(schedule_id=f"sched_{int(time.time())}")
        self.current_metrics: Optional[QUANTAMetrics] = None

    def compute_objective(self, metrics: QUANTAMetrics) -> float:
        """
        Compute the coupled optimization objective.

        J = T_s + A_g - |ΔD| - η_gap + C_norm
        """
        ts = metrics.topology.value
        ag = metrics.antifragility.value
        delta_d = abs(metrics.nib.value)

        # η gap from critical (1.0)
        eta_gap = abs(metrics.gft.value - 1.0)

        # Normalized capacity
        c_norm = min(1.0, metrics.capacity.value / 20.0)  # 20 bits/layer = max

        # Weights for each term
        w_ts = 1.0
        w_ag = 2.0      # Bonus for antifragility
        w_nib = -1.5    # Penalty for identity drift
        w_eta = -0.5    # Penalty for non-critical damping
        w_cap = 0.5     # Bonus for capacity

        objective = (
            w_ts * ts +
            w_ag * ag +
            w_nib * delta_d +
            w_eta * eta_gap +
            w_cap * c_norm
        )

        return objective

    def optimal_rank(self, n_params: int) -> int:
        """
        Compute optimal rank: r_opt = c·n^0.25

        Based on capacity-retention tradeoff.
        """
        r_opt = int(self.config.rank_coefficient * (n_params ** 0.25))
        r_opt = max(self.config.min_rank, r_opt)

        max_rank = int(n_params * self.config.max_rank_ratio)
        r_opt = min(r_opt, max_rank)

        return r_opt

    def should_consolidate(self, metrics: QUANTAMetrics) -> bool:
        """
        Determine if consolidation is needed based on metrics.

        Triggers:
        - T_s < target
        - A_g < 0 (fragile, not antifragile)
        - NIB ΔD > target
        - GFT < target critical %
        """
        if metrics.topology.value < self.config.ts_target:
            return True
        if metrics.antifragility.value < 0:
            return True
        if metrics.nib.value > self.config.nib_target:
            return True
        if metrics.gft.critical_percentage < self.config.gft_target_pct:
            return True

        return False

    def get_recommended_action(self, metrics: QUANTAMetrics) -> Dict[str, Any]:
        """Get recommended consolidation action based on current state."""
        actions = []

        if metrics.topology.status == MetricStatus.RED:
            actions.append({
                "action": "increase_replay",
                "reason": f"T_s={metrics.topology.value:.3f} < {self.config.ts_target}",
                "param": "replay_frequency",
                "direction": "increase",
            })

        if metrics.antifragility.value < 0:
            actions.append({
                "action": "adjust_sigma",
                "reason": f"A_g={metrics.antifragility.value:.4f} < 0 (fragile)",
                "param": "sigma",
                "target": 0.10,
            })

        if metrics.nib.status == MetricStatus.RED:
            actions.append({
                "action": "pause_consolidation",
                "reason": f"NIB ΔD={metrics.nib.value:.3f} > {self.config.nib_target}",
                "param": "consolidation_rate",
                "direction": "decrease",
            })

        if metrics.gft.critical_percentage < 0.5:
            actions.append({
                "action": "boost_dissipation",
                "reason": f"GFT critical={metrics.gft.critical_percentage*100:.0f}% < 90%",
                "param": "layer_dissipation",
                "direction": "increase",
            })

        return {
            "should_act": len(actions) > 0,
            "actions": actions,
            "objective": self.compute_objective(metrics),
        }

    def consolidate(self,
                    weights: np.ndarray,
                    phase: ConsolidationPhase,
                    layer_weights: List[np.ndarray] = None) -> np.ndarray:
        """
        Perform consolidation based on phase.

        Micro: Small updates with high stress tolerance
        Replay: Experience replay with optimal stress
        Structural: SVD-based rank reduction
        """
        # Store original for metrics
        weights_original = weights.copy()

        # Get sigma for this phase
        if phase == ConsolidationPhase.MICRO:
            sigma = self.config.sigma_micro
        elif phase == ConsolidationPhase.REPLAY:
            sigma = self.config.sigma_replay
        else:
            sigma = self.config.sigma_structural

        # Phase-specific consolidation
        if phase == ConsolidationPhase.STRUCTURAL:
            weights_new = self._structural_consolidation(weights)
        elif phase == ConsolidationPhase.REPLAY:
            weights_new = self._replay_consolidation(weights, sigma)
        else:
            weights_new = self._micro_consolidation(weights, sigma)

        # Compute metrics
        if layer_weights is None:
            layer_weights = [weights_new]

        metrics_before = compute_quanta_metrics(
            weights_original, weights_original, [weights_original], sigma
        )
        metrics_after = compute_quanta_metrics(
            weights_original, weights_new, layer_weights, sigma
        )

        # Check retention constraint
        retention = self._compute_retention(weights_original, weights_new)

        if retention < self.config.min_retention:
            # Rollback - consolidation too aggressive
            weights_new = weights_original
            metrics_after = metrics_before

        # Record event
        event = ConsolidationEvent(
            event_id=f"evt_{int(time.time())}",
            phase=phase,
            timestamp=time.time(),
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            retention=retention,
            passed_constraints=retention >= self.config.min_retention,
        )

        self.schedule.events.append(event)
        self.schedule.metrics_history.append(metrics_after)

        # Update last run time
        if phase == ConsolidationPhase.MICRO:
            self.schedule.last_micro = time.time()
        elif phase == ConsolidationPhase.REPLAY:
            self.schedule.last_replay = time.time()
        else:
            self.schedule.last_structural = time.time()

        self.current_metrics = metrics_after
        return weights_new

    def _structural_consolidation(self, weights: np.ndarray) -> np.ndarray:
        """
        SVD-based rank reduction.

        Reduces rank to r_opt while preserving topology.
        """
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)

        try:
            U, S, Vt = np.linalg.svd(weights, full_matrices=False)

            # Optimal rank
            n_params = weights.size
            r_opt = self.optimal_rank(n_params)
            r_opt = min(r_opt, len(S))

            # Truncate
            U_r = U[:, :r_opt]
            S_r = S[:r_opt]
            Vt_r = Vt[:r_opt, :]

            # Reconstruct
            weights_reduced = U_r @ np.diag(S_r) @ Vt_r

            return weights_reduced.reshape(weights.shape)

        except Exception:
            return weights

    def _replay_consolidation(self, weights: np.ndarray, sigma: float) -> np.ndarray:
        """
        Experience replay consolidation.

        Adds controlled noise and smooths, simulating replay-based learning.
        """
        # Add controlled stress
        noise = np.random.normal(0, sigma * 0.5, weights.shape)

        # Smooth update (exponential moving average)
        alpha = 0.1
        weights_new = (1 - alpha) * weights + alpha * (weights + noise)

        return weights_new

    def _micro_consolidation(self, weights: np.ndarray, sigma: float) -> np.ndarray:
        """
        Micro consolidation - small online updates.
        """
        # Very small perturbation for stress tolerance building
        noise = np.random.normal(0, sigma * 0.1, weights.shape)
        weights_new = weights + noise * 0.01

        return weights_new

    def _compute_retention(self, weights_old: np.ndarray,
                           weights_new: np.ndarray) -> float:
        """Compute retention as correlation between old and new."""
        flat_old = weights_old.flatten()
        flat_new = weights_new.flatten()

        # Correlation coefficient
        corr = np.corrcoef(flat_old, flat_new)[0, 1]
        retention = max(0, corr)  # Clamp to [0, 1]

        return retention

    def get_health_summary(self) -> Dict[str, Any]:
        """Get current memory health summary."""
        if self.current_metrics is None:
            return {"status": "no_data"}

        return {
            "status": self.current_metrics.overall_status.value,
            "all_green": self.current_metrics.all_green,
            "metrics": self.current_metrics.to_dict(),
            "objective": self.compute_objective(self.current_metrics),
            "events_total": len(self.schedule.events),
            "recommendation": self.get_recommended_action(self.current_metrics),
        }


# Convenience functions
def create_consolidator(config: ConsolidationConfig = None) -> QUANTAConsolidator:
    """Create a QUANTA consolidator."""
    return QUANTAConsolidator(config)


def run_consolidation_cycle(weights: np.ndarray,
                            phases: List[ConsolidationPhase] = None) -> np.ndarray:
    """Run a full consolidation cycle through all phases."""
    if phases is None:
        phases = [
            ConsolidationPhase.MICRO,
            ConsolidationPhase.REPLAY,
            ConsolidationPhase.STRUCTURAL,
        ]

    consolidator = QUANTAConsolidator()

    for phase in phases:
        weights = consolidator.consolidate(weights, phase)

    return weights
