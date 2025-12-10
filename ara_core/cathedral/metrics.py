#!/usr/bin/env python3
"""
Cathedral OS - Core Metrics and Guarantees
==========================================

The unified theory of antifragile intelligence:

    T_s(n) = 1 - C/√n     # Complexity → Stability
    A_g(σ*) = +0.021      # Stress → Improvement (σ*=0.10)
    H_s = 97.7%           # Activity Bounded
    Yield/$ ↑ MoM         # Economic Scaling

Four Layers:
1. NEURAL:  T-FAN fields, T_s≥0.92, σ*=0.10
2. AGENTS:  Cities/Morons, H_influence>1.8, Bias Sentinel
3. HIVE:    Junkyard GPUs, Yield/$, Bee Scheduler
4. GOVERNANCE: Homeostatic controller w=10, α=0.12

13 Gates must pass for production deployment.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class GateStatus(str, Enum):
    """Status of a deployment gate."""
    GREEN = "green"       # Passed
    YELLOW = "yellow"     # Warning
    RED = "red"           # Failed
    UNKNOWN = "unknown"   # Not yet evaluated


@dataclass
class MetricValue:
    """A single metric measurement."""
    name: str
    value: float
    target: float
    operator: str = ">="  # ">=", ">", "<=", "<", "=="
    status: GateStatus = GateStatus.UNKNOWN
    unit: str = ""

    def evaluate(self) -> GateStatus:
        """Evaluate if metric meets target."""
        ops = {
            ">=": lambda v, t: v >= t,
            ">": lambda v, t: v > t,
            "<=": lambda v, t: v <= t,
            "<": lambda v, t: v < t,
            "==": lambda v, t: abs(v - t) < 0.001,
        }

        if ops.get(self.operator, lambda v, t: False)(self.value, self.target):
            self.status = GateStatus.GREEN
        else:
            # Check if close (within 10%)
            margin = abs(self.target * 0.1)
            if self.operator in [">=", ">"]:
                if self.value >= self.target - margin:
                    self.status = GateStatus.YELLOW
                else:
                    self.status = GateStatus.RED
            elif self.operator in ["<=", "<"]:
                if self.value <= self.target + margin:
                    self.status = GateStatus.YELLOW
                else:
                    self.status = GateStatus.RED
            else:
                self.status = GateStatus.RED

        return self.status


# =============================================================================
# CORE GUARANTEES (Theorem-derived)
# =============================================================================

@dataclass
class ComplexityStabilityGuarantee:
    """
    Theorem 5.1: T_s(n) = 1 - C/√n

    Complexity increases stability. Larger models are more stable.
    """
    name: str = "Complexity → Stability"
    formula: str = "T_s(n) = 1 - C/√n"

    # Empirical constant (from your validation)
    C: float = 0.3

    def compute_ts(self, n_params: int) -> float:
        """Compute expected T_s for given complexity."""
        return max(0, 1 - self.C / np.sqrt(n_params))

    def validate(self, measured_ts: float, n_params: int, tolerance: float = 0.1) -> bool:
        """Validate measurement against theorem."""
        expected = self.compute_ts(n_params)
        return abs(measured_ts - expected) <= tolerance


@dataclass
class HormesisGuarantee:
    """
    Theorem 1.2: A_g(σ*) = +0.021 at σ*=0.10

    Optimal stress σ* improves the system (antifragility).
    """
    name: str = "Stress → Improvement"
    formula: str = "A_g(σ*) = T_s(σ*) - T_s(0) > 0"

    # Optimal stress level
    sigma_star: float = 0.10

    # Expected gain at σ*
    expected_gain: float = 0.021

    def validate(self, ag_measured: float) -> bool:
        """Validate antifragility gain."""
        return ag_measured >= 0.01  # Must be positive


@dataclass
class HomeostasisGuarantee:
    """
    Theorem 1.3: H_s = 97.7% over 1200 steps

    Activity remains bounded with proper controller.
    """
    name: str = "Activity Bounded"
    formula: str = "H_s = P(|a_t - τ| < δτ) ≥ 0.95"

    # Golden controller parameters
    w: float = 10.0
    alpha: float = 0.12

    # Expected performance
    target_hs: float = 0.95
    empirical_hs: float = 0.977  # Your validation

    def validate(self, hs_measured: float) -> bool:
        """Validate homeostasis."""
        return hs_measured >= self.target_hs


@dataclass
class SafeMorphingGuarantee:
    """
    Phase 3 S2.2: ±10% morphing maintains T_s≥0.95

    Safe architecture changes within budget.
    """
    name: str = "Safe Morphing"
    formula: str = "Δparams ≤ ±10% → T_s ≥ 0.95"

    morph_budget: float = 0.10  # ±10%
    ts_threshold: float = 0.95


@dataclass
class DirectionalityGuarantee:
    """
    MAR Theorem: Undirected graphs 3.3x more efficient.

    Directionality limit for graph structures.
    """
    name: str = "Directionality Limit"
    formula: str = "E[undirected]/E[directed] = 3.3x"

    efficiency_ratio: float = 3.3


@dataclass
class EconomicScalingGuarantee:
    """
    Hive Economics: Yield/$ improves month-over-month.

    Economic viability through efficiency.
    """
    name: str = "Economic Scaling"
    formula: str = "Yield/$ ↑ MoM"

    target_multiplier: float = 5.0  # 5-15x target
    min_multiplier: float = 3.0


# =============================================================================
# DEPLOYMENT GATES
# =============================================================================

@dataclass
class NeuralGate:
    """
    6 Neural metrics required for deployment.

    All must be GREEN.
    """
    name: str = "Neural Gate"

    ts_sigma: MetricValue = field(default_factory=lambda: MetricValue(
        name="T_s(σ*)", value=0.0, target=0.95, operator=">="
    ))
    ag_sigma: MetricValue = field(default_factory=lambda: MetricValue(
        name="A_g(σ*)", value=0.0, target=0.01, operator=">"
    ))
    hs: MetricValue = field(default_factory=lambda: MetricValue(
        name="H_s", value=0.0, target=0.95, operator=">="
    ))
    tau_conv: MetricValue = field(default_factory=lambda: MetricValue(
        name="τ_conv", value=0.0, target=400, operator="<", unit="steps"
    ))
    controller_w: MetricValue = field(default_factory=lambda: MetricValue(
        name="Controller w", value=10.0, target=10.0, operator="=="
    ))
    controller_alpha: MetricValue = field(default_factory=lambda: MetricValue(
        name="Controller α", value=0.12, target=0.12, operator="=="
    ))

    def evaluate_all(self) -> Tuple[bool, Dict[str, GateStatus]]:
        """Evaluate all neural metrics."""
        metrics = [
            self.ts_sigma, self.ag_sigma, self.hs,
            self.tau_conv, self.controller_w, self.controller_alpha
        ]

        results = {}
        for m in metrics:
            m.evaluate()
            results[m.name] = m.status

        all_green = all(s == GateStatus.GREEN for s in results.values())
        return all_green, results


@dataclass
class HiveGate:
    """
    4 Hive metrics required for deployment.
    """
    name: str = "Hive Gate"

    e_media: MetricValue = field(default_factory=lambda: MetricValue(
        name="E_media", value=0.0, target=3.0, operator=">=", unit="x baseline"
    ))
    yield_dollar: MetricValue = field(default_factory=lambda: MetricValue(
        name="Yield/$", value=0.0, target=0.0, operator=">", unit="MoM improvement"
    ))
    cluster_ts: MetricValue = field(default_factory=lambda: MetricValue(
        name="Cluster T_s", value=0.0, target=0.92, operator=">="
    ))
    gpu_util: MetricValue = field(default_factory=lambda: MetricValue(
        name="GPU Util", value=0.0, target=80.0, operator=">=", unit="%"
    ))

    def evaluate_all(self) -> Tuple[bool, Dict[str, GateStatus]]:
        """Evaluate all hive metrics."""
        metrics = [self.e_media, self.yield_dollar, self.cluster_ts, self.gpu_util]

        results = {}
        for m in metrics:
            m.evaluate()
            results[m.name] = m.status

        all_green = all(s == GateStatus.GREEN for s in results.values())
        return all_green, results


@dataclass
class SwarmGate:
    """
    3 Swarm metrics required for deployment.
    """
    name: str = "Swarm Gate"

    h_influence: MetricValue = field(default_factory=lambda: MetricValue(
        name="H_influence", value=0.0, target=1.8, operator=">", unit="bits"
    ))
    ts_bias: MetricValue = field(default_factory=lambda: MetricValue(
        name="T_s_bias", value=0.0, target=0.92, operator=">="
    ))
    cost_reward: MetricValue = field(default_factory=lambda: MetricValue(
        name="Cost/Reward", value=0.0, target=2.0, operator=">", unit="x baseline"
    ))

    def evaluate_all(self) -> Tuple[bool, Dict[str, GateStatus]]:
        """Evaluate all swarm metrics."""
        metrics = [self.h_influence, self.ts_bias, self.cost_reward]

        results = {}
        for m in metrics:
            m.evaluate()
            results[m.name] = m.status

        all_green = all(s == GateStatus.GREEN for s in results.values())
        return all_green, results


# =============================================================================
# CATHEDRAL METRICS AGGREGATOR
# =============================================================================

@dataclass
class CathedralMetrics:
    """
    Complete Cathedral OS metrics.

    13 gates across 3 domains.
    """
    # Guarantees
    complexity_stability: ComplexityStabilityGuarantee = field(
        default_factory=ComplexityStabilityGuarantee
    )
    hormesis: HormesisGuarantee = field(default_factory=HormesisGuarantee)
    homeostasis: HomeostasisGuarantee = field(default_factory=HomeostasisGuarantee)
    safe_morphing: SafeMorphingGuarantee = field(default_factory=SafeMorphingGuarantee)
    directionality: DirectionalityGuarantee = field(default_factory=DirectionalityGuarantee)
    economic: EconomicScalingGuarantee = field(default_factory=EconomicScalingGuarantee)

    # Gates
    neural: NeuralGate = field(default_factory=NeuralGate)
    hive: HiveGate = field(default_factory=HiveGate)
    swarm: SwarmGate = field(default_factory=SwarmGate)

    # Timestamp
    evaluated_at: float = 0.0

    def update_neural(self, ts: float, ag: float, hs: float,
                      tau: float, w: float = 10.0, alpha: float = 0.12):
        """Update neural gate metrics."""
        self.neural.ts_sigma.value = ts
        self.neural.ag_sigma.value = ag
        self.neural.hs.value = hs
        self.neural.tau_conv.value = tau
        self.neural.controller_w.value = w
        self.neural.controller_alpha.value = alpha

    def update_hive(self, e_media: float, yield_delta: float,
                    cluster_ts: float, gpu_util: float):
        """Update hive gate metrics."""
        self.hive.e_media.value = e_media
        self.hive.yield_dollar.value = yield_delta
        self.hive.cluster_ts.value = cluster_ts
        self.hive.gpu_util.value = gpu_util

    def update_swarm(self, h_influence: float, ts_bias: float, cost_reward: float):
        """Update swarm gate metrics."""
        self.swarm.h_influence.value = h_influence
        self.swarm.ts_bias.value = ts_bias
        self.swarm.cost_reward.value = cost_reward

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate all gates and return status."""
        self.evaluated_at = time.time()

        neural_ok, neural_results = self.neural.evaluate_all()
        hive_ok, hive_results = self.hive.evaluate_all()
        swarm_ok, swarm_results = self.swarm.evaluate_all()

        total_gates = 13
        green_count = sum(
            1 for s in list(neural_results.values()) +
                       list(hive_results.values()) +
                       list(swarm_results.values())
            if s == GateStatus.GREEN
        )

        return {
            "neural": {
                "passed": neural_ok,
                "gates": neural_results,
                "score": f"{sum(1 for s in neural_results.values() if s == GateStatus.GREEN)}/6",
            },
            "hive": {
                "passed": hive_ok,
                "gates": hive_results,
                "score": f"{sum(1 for s in hive_results.values() if s == GateStatus.GREEN)}/4",
            },
            "swarm": {
                "passed": swarm_ok,
                "gates": swarm_results,
                "score": f"{sum(1 for s in swarm_results.values() if s == GateStatus.GREEN)}/3",
            },
            "overall": {
                "all_green": neural_ok and hive_ok and swarm_ok,
                "score": f"{green_count}/{total_gates}",
                "deploy_ready": neural_ok and hive_ok and swarm_ok,
            },
            "timestamp": self.evaluated_at,
        }

    def deploy_decision(self) -> str:
        """Get deployment decision."""
        result = self.evaluate()
        if result["overall"]["deploy_ready"]:
            return "DEPLOY_OK"
        elif result["neural"]["passed"]:
            return "PARTIAL_DEPLOY"  # Neural OK, hive/swarm issues
        else:
            return "REJECT_UNSTABLE"


# =============================================================================
# QUANTUM CONTROL METRICS (Hybrid Classical-Quantum)
# =============================================================================

@dataclass
class QuantumMetrics:
    """Metrics specific to quantum control layer."""

    # Core performance
    recall_accuracy: float = 0.0      # P(quantum_recall == classical_recall)
    speedup: float = 1.0              # t_classical / t_quantum
    control_fidelity: float = 0.0     # Average state fidelity

    # Robustness / antifragility
    noise_tolerance: float = 0.0      # Accuracy at noise level
    stress_gain: float = 0.0          # A_g_quant = post_tune - pre_stress

    # Efficiency
    hybrid_efficiency: float = 0.0    # Useful ops per joule
    routing_quality: float = 0.0      # Fraction where quantum was net win

    # Safety
    control_budget_used: float = 0.0  # Fraction of max pulse depth used
    failure_rate: float = 0.0         # Invalid/timeout per 1000 calls
    fallback_rate: float = 0.0        # Rate of falling back to classical


@dataclass
class QuantumGate:
    """
    Quantum control metrics for deployment.
    """
    name: str = "Quantum Gate"

    recall: MetricValue = field(default_factory=lambda: MetricValue(
        name="Recall Accuracy", value=0.0, target=0.95, operator=">="
    ))
    speedup: MetricValue = field(default_factory=lambda: MetricValue(
        name="Speedup", value=1.0, target=1.0, operator=">=", unit="x"
    ))
    fidelity: MetricValue = field(default_factory=lambda: MetricValue(
        name="Control Fidelity", value=0.0, target=0.90, operator=">="
    ))
    routing: MetricValue = field(default_factory=lambda: MetricValue(
        name="Routing Quality", value=0.0, target=0.80, operator=">="
    ))
    noise_tol: MetricValue = field(default_factory=lambda: MetricValue(
        name="Noise Tolerance", value=0.0, target=0.80, operator=">="
    ))

    def evaluate_all(self) -> Tuple[bool, Dict[str, GateStatus]]:
        """Evaluate all quantum metrics."""
        metrics = [self.recall, self.speedup, self.fidelity, self.routing, self.noise_tol]

        results = {}
        for m in metrics:
            m.evaluate()
            results[m.name] = m.status

        all_green = all(s == GateStatus.GREEN for s in results.values())
        return all_green, results


# =============================================================================
# HYBRID METRICS DASHBOARD
# =============================================================================

class HybridMetricsDashboard:
    """Render combined classical + quantum metrics."""

    def __init__(self, cathedral: CathedralMetrics, quantum: QuantumMetrics):
        self.cathedral = cathedral
        self.quantum = quantum

    def render(self, mode: str = "starfleet") -> str:
        """Render full hybrid dashboard."""
        lines = []

        mode_icons = {
            "starfleet": "🖖",
            "red_dwarf": "🐱",
            "time_lord": "👨‍⚕️",
            "colonial_fleet": "⚔️",
            "k10_toaster": "🧈",
        }
        icon = mode_icons.get(mode, "🏛️")

        # Compute J
        T_s = self.cathedral.neural.ts_sigma.value
        A_g = self.cathedral.neural.ag_sigma.value
        P_norm = 0.8  # placeholder
        J = 0.5 * T_s + 0.4 * A_g - 0.1 * P_norm

        lines.append("╔══════════════════════════════════════════════════════════════════════════╗")
        lines.append(f"║  {icon} CATHEDRAL OS METRICS - {mode.upper():<20}  J = {J:+.4f}           ║")
        lines.append("╠══════════════════════════════════════════════════════════════════════════╣")

        # Classical section
        lines.append("║  CLASSICAL STACK                                                         ║")
        lines.append(f"║    T_s = {T_s:.4f}  {self._bar(T_s, 0.9, 1.0)}  [{self._status(T_s, 0.95)}]       ║")
        lines.append(f"║    H_s = {self.cathedral.neural.hs.value:.4f}  {self._bar(self.cathedral.neural.hs.value, 0.9, 1.0)}  [{self._status(self.cathedral.neural.hs.value, 0.95)}]       ║")
        lines.append(f"║    A_g = {A_g:+.4f} {self._bar(A_g + 0.5, 0, 1)}  [{'GAIN' if A_g > 0 else 'LOSS'}]       ║")
        lines.append("╠══════════════════════════════════════════════════════════════════════════╣")

        # Quantum section
        lines.append("║  QUANTUM CONTROL LAYER                                                   ║")
        lines.append(f"║    Recall   = {self.quantum.recall_accuracy:.2%}  {self._bar(self.quantum.recall_accuracy, 0, 1)}  [{self._status(self.quantum.recall_accuracy, 0.95)}]       ║")
        lines.append(f"║    Speedup  = {self.quantum.speedup:.2f}x   {self._bar(min(self.quantum.speedup/5, 1), 0, 1)}  [{'FAST' if self.quantum.speedup > 1 else 'SLOW'}]       ║")
        lines.append(f"║    Fidelity = {self.quantum.control_fidelity:.2%}  {self._bar(self.quantum.control_fidelity, 0, 1)}  [{self._status(self.quantum.control_fidelity, 0.9)}]       ║")
        lines.append(f"║    Routing  = {self.quantum.routing_quality:.2%}  (fail={self.quantum.failure_rate:.1f}/1k, fb={self.quantum.fallback_rate:.2%})   ║")
        lines.append(f"║    Noise    = {self.quantum.noise_tolerance:.2%}  {self._bar(self.quantum.noise_tolerance, 0, 1)}  [{self._status(self.quantum.noise_tolerance, 0.8)}]       ║")
        lines.append("╚══════════════════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def _bar(self, value: float, min_v: float, max_v: float, width: int = 15) -> str:
        """Render a progress bar."""
        norm = (value - min_v) / (max_v - min_v + 1e-10)
        norm = max(0, min(1, norm))
        filled = int(norm * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"

    def _status(self, value: float, threshold: float) -> str:
        """Get status indicator."""
        if value >= threshold:
            return "OK"
        elif value >= threshold * 0.95:
            return "!!"
        else:
            return "XX"


def render_hybrid_dashboard(
    T_s: float = 0.99,
    A_g: float = 0.02,
    H_s: float = 0.977,
    recall_accuracy: float = 0.95,
    speedup: float = 1.5,
    fidelity: float = 0.92,
    routing: float = 0.85,
    noise_tol: float = 0.88,
    mode: str = "starfleet",
) -> str:
    """Quick render with provided metrics."""
    cathedral = CathedralMetrics()
    cathedral.update_neural(ts=T_s, ag=A_g, hs=H_s, tau=300)

    quantum = QuantumMetrics(
        recall_accuracy=recall_accuracy,
        speedup=speedup,
        control_fidelity=fidelity,
        routing_quality=routing,
        noise_tolerance=noise_tol,
    )

    dashboard = HybridMetricsDashboard(cathedral, quantum)
    return dashboard.render(mode)
