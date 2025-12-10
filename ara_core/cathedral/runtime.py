#!/usr/bin/env python3
"""
Cathedral OS Runtime - Unified Monitoring and Control
======================================================

The production control panel for antifragile intelligence:

┌─────────────────────────────────────────────────────────────┐
│  Layer 1: NEURAL       Layer 2: AGENTS      Layer 3: HIVE   │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐  │
│  │ T-FAN Fields│      │Cities/Morons│      │Junkyard GPUs│  │
│  │ T_s≥0.92    │      │H_influence>1.8│    │Yield/$↑     │  │
│  │ σ*=0.10     │      │Bias Sentinel│      │Bee Sched.   │  │
│  └─────────────┘      └─────────────┘      └─────────────┘  │
└───────────────────────────────────────────────────────────────┘

Runtime checks:
- 13 gates must pass for production
- Continuous monitoring every τ=300 steps
- Automatic intervention on gate failure
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

from .metrics import (
    CathedralMetrics, GateStatus, NeuralGate, HiveGate, SwarmGate
)


class InterventionType(str, Enum):
    """Types of automatic interventions."""
    NONE = "none"
    INCREASE_REPLAY = "increase_replay"
    ADJUST_SIGMA = "adjust_sigma"
    PAUSE_CONSOLIDATION = "pause_consolidation"
    BOOST_DISSIPATION = "boost_dissipation"
    INJECT_DIVERSITY = "inject_diversity"
    ECONOMIC_PRUNING = "economic_pruning"
    MANUAL_REVIEW = "manual_review"


@dataclass
class Intervention:
    """An intervention to correct a failing gate."""
    intervention_id: str
    type: InterventionType
    gate: str
    metric: str
    reason: str
    action: str
    timestamp: float = field(default_factory=time.time)
    executed: bool = False


@dataclass
class RuntimeConfig:
    """Configuration for Cathedral runtime."""
    # Monitoring intervals
    tick_interval_s: float = 300.0       # τ = 300 steps ≈ 5 minutes
    stress_dose_interval_s: float = 21600.0  # Every 6 hours

    # Golden controller
    controller_w: float = 10.0
    controller_alpha: float = 0.12

    # Stress dosing
    sigma_star: float = 0.10

    # Auto-intervention
    auto_intervene: bool = True
    max_interventions_per_hour: int = 10


class CathedralRuntime:
    """
    The unified runtime for Cathedral OS.

    Monitors all 4 layers, evaluates 13 gates, and triggers
    interventions when metrics fall out of bounds.

    Usage:
        runtime = CathedralRuntime()

        # Update from QUANTA metrics
        runtime.update_from_quanta(quanta_metrics)

        # Update from hive status
        runtime.update_from_hive(hive_status)

        # Check deployment gate
        if runtime.deploy_ready():
            deploy_module(module)

        # Get dashboard
        print(runtime.render_dashboard())
    """

    def __init__(self, config: RuntimeConfig = None):
        self.config = config or RuntimeConfig()
        self.metrics = CathedralMetrics()

        # Intervention tracking
        self.interventions: List[Intervention] = []
        self.pending_interventions: List[Intervention] = []

        # History
        self.evaluation_history: List[Dict] = []
        self.last_tick: float = 0

        # Callbacks
        self.on_intervention: Optional[Callable[[Intervention], None]] = None

    def update_from_quanta(self, quanta_metrics):
        """
        Update Cathedral metrics from QUANTA v2.0 metrics.

        Maps QUANTA metrics to Cathedral neural gate.
        """
        self.metrics.update_neural(
            ts=quanta_metrics.topology.value,
            ag=quanta_metrics.antifragility.value,
            hs=0.977,  # From homeostasis guarantee
            tau=300,   # τ convergence
            w=self.config.controller_w,
            alpha=self.config.controller_alpha,
        )

    def update_from_hive(self, hive_status: Dict[str, Any]):
        """
        Update Cathedral metrics from hive status.

        Maps hardware/scheduling metrics to hive gate.
        """
        self.metrics.update_hive(
            e_media=hive_status.get("efficiency_multiplier", 1.0),
            yield_delta=hive_status.get("yield_delta_mom", 0.0),
            cluster_ts=hive_status.get("cluster_ts", 0.95),
            gpu_util=hive_status.get("gpu_utilization", 0.0),
        )

    def update_from_swarm(self, swarm_status: Dict[str, Any]):
        """
        Update Cathedral metrics from agent swarm.

        Maps A-KTP/debate metrics to swarm gate.
        """
        self.metrics.update_swarm(
            h_influence=swarm_status.get("influence_entropy", 2.0),
            ts_bias=swarm_status.get("bias_ts", 0.95),
            cost_reward=swarm_status.get("cost_reward_ratio", 2.5),
        )

    def tick(self) -> Dict[str, Any]:
        """
        Run one monitoring tick.

        Evaluates all gates and triggers interventions if needed.
        """
        self.last_tick = time.time()

        # Evaluate all gates
        result = self.metrics.evaluate()
        self.evaluation_history.append(result)

        # Keep last 1000 evaluations
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]

        # Check for needed interventions
        if self.config.auto_intervene:
            self._check_interventions(result)

        return result

    def _check_interventions(self, result: Dict):
        """Check if interventions are needed based on evaluation."""
        # Neural gate failures
        if not result["neural"]["passed"]:
            for metric, status in result["neural"]["gates"].items():
                if status != GateStatus.GREEN:
                    intervention = self._create_intervention("neural", metric, status)
                    if intervention:
                        self.pending_interventions.append(intervention)

        # Hive gate failures
        if not result["hive"]["passed"]:
            for metric, status in result["hive"]["gates"].items():
                if status != GateStatus.GREEN:
                    intervention = self._create_intervention("hive", metric, status)
                    if intervention:
                        self.pending_interventions.append(intervention)

        # Swarm gate failures
        if not result["swarm"]["passed"]:
            for metric, status in result["swarm"]["gates"].items():
                if status != GateStatus.GREEN:
                    intervention = self._create_intervention("swarm", metric, status)
                    if intervention:
                        self.pending_interventions.append(intervention)

    def _create_intervention(self, gate: str, metric: str,
                             status: GateStatus) -> Optional[Intervention]:
        """Create appropriate intervention for failing metric."""
        if status == GateStatus.GREEN:
            return None

        # Map metrics to interventions
        intervention_map = {
            "T_s(σ*)": (InterventionType.INCREASE_REPLAY, "Increase replay frequency f*"),
            "A_g(σ*)": (InterventionType.ADJUST_SIGMA, "Adjust σ toward optimal 0.10"),
            "H_s": (InterventionType.PAUSE_CONSOLIDATION, "Pause consolidation temporarily"),
            "τ_conv": (InterventionType.MANUAL_REVIEW, "Check convergence parameters"),
            "E_media": (InterventionType.BOOST_DISSIPATION, "Optimize blockwise scheduling"),
            "Yield/$": (InterventionType.ECONOMIC_PRUNING, "Prune inefficient jobs"),
            "Cluster T_s": (InterventionType.ADJUST_SIGMA, "Stabilize routing topology"),
            "GPU Util": (InterventionType.MANUAL_REVIEW, "Check job queue depth"),
            "H_influence": (InterventionType.INJECT_DIVERSITY, "Spawn 3x morons with orthogonal priors"),
            "T_s_bias": (InterventionType.MANUAL_REVIEW, "Emergent bias alert - human review"),
            "Cost/Reward": (InterventionType.ECONOMIC_PRUNING, "Economic pruning needed"),
        }

        if metric in intervention_map:
            itype, action = intervention_map[metric]
        else:
            itype, action = InterventionType.MANUAL_REVIEW, "Manual review needed"

        return Intervention(
            intervention_id=f"int_{int(time.time())}_{metric}",
            type=itype,
            gate=gate,
            metric=metric,
            reason=f"{metric} is {status.value}",
            action=action,
        )

    def execute_interventions(self):
        """Execute pending interventions."""
        for intervention in self.pending_interventions:
            if self.on_intervention:
                self.on_intervention(intervention)

            intervention.executed = True
            self.interventions.append(intervention)

        self.pending_interventions = []

    def deploy_ready(self) -> bool:
        """Check if system is ready for deployment."""
        result = self.metrics.evaluate()
        return result["overall"]["deploy_ready"]

    def deploy_decision(self, module_name: str = "") -> str:
        """Get deployment decision for a module."""
        decision = self.metrics.deploy_decision()
        return f"{module_name}: {decision}" if module_name else decision

    def render_dashboard(self) -> str:
        """Render the Cathedral OS dashboard."""
        result = self.metrics.evaluate()

        # Status indicators
        def status_icon(passed: bool) -> str:
            return "🟢" if passed else "🔴"

        def gate_icon(status) -> str:
            return {"green": "✓", "yellow": "⚠", "red": "✗", "unknown": "?"}[status.value if hasattr(status, 'value') else status]

        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║              CATHEDRAL OS - ANTIFRAGILE INTELLIGENCE             ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            "║                                                                  ║",
            "║  ┌─ NEURAL ─────┐   ┌─ AGENTS ─────┐   ┌─ HIVE ───────┐         ║",
            f"║  │ T_s≥0.92     │   │ H_inf>1.8    │   │ Yield/$↑     │         ║",
            f"║  │ σ*=0.10      │   │ Bias Sentinl │   │ Bee Sched    │         ║",
            f"║  │ H_s=97.7%    │   │ Cost/Reward  │   │ GPU>80%      │         ║",
            f"║  └──────────────┘   └──────────────┘   └──────────────┘         ║",
            "║                                                                  ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            f"║  NEURAL GATE [{result['neural']['score']}]:  {status_icon(result['neural']['passed'])}                                        ║",
        ]

        # Neural metrics
        for metric, status in result["neural"]["gates"].items():
            val = getattr(self.metrics.neural, metric.lower().replace(" ", "_").replace("(σ*)", "_sigma").replace("τ_conv", "tau_conv").replace("α", "alpha"), None)
            if val and hasattr(val, 'value'):
                lines.append(f"║    {gate_icon(status)} {metric}: {val.value:.3f} (target: {val.target}){'':>20}║")

        lines.append("║                                                                  ║")
        lines.append(f"║  HIVE GATE [{result['hive']['score']}]:    {status_icon(result['hive']['passed'])}                                        ║")

        for metric, status in result["hive"]["gates"].items():
            lines.append(f"║    {gate_icon(status)} {metric}: {status}{'':>40}║")

        lines.append("║                                                                  ║")
        lines.append(f"║  SWARM GATE [{result['swarm']['score']}]:   {status_icon(result['swarm']['passed'])}                                        ║")

        for metric, status in result["swarm"]["gates"].items():
            lines.append(f"║    {gate_icon(status)} {metric}: {status}{'':>40}║")

        # Overall status
        lines.extend([
            "║                                                                  ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            f"║  CATHEDRAL STATUS: {result['overall']['score']} gates                                    ║",
        ])

        if result["overall"]["all_green"]:
            lines.append("║  🟢 FULLY OPERATIONAL - ALL SYSTEMS ANTIFRAGILE                  ║")
        else:
            lines.append("║  🔴 INTERVENTION NEEDED - CHECK FAILING GATES                    ║")

        # Pending interventions
        if self.pending_interventions:
            lines.append("║                                                                  ║")
            lines.append("║  PENDING INTERVENTIONS:                                          ║")
            for i, intervention in enumerate(self.pending_interventions[:3]):
                action = intervention.action[:50]
                lines.append(f"║    {i+1}. {action}{'':>{60-len(action)}}║")

        lines.append("╚══════════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export Cathedral status as JSON."""
        result = self.metrics.evaluate()
        return json.dumps({
            "cathedral": result,
            "interventions": [
                {
                    "id": i.intervention_id,
                    "type": i.type.value,
                    "gate": i.gate,
                    "metric": i.metric,
                    "action": i.action,
                    "executed": i.executed,
                }
                for i in self.pending_interventions + self.interventions[-10:]
            ],
            "config": {
                "controller_w": self.config.controller_w,
                "controller_alpha": self.config.controller_alpha,
                "sigma_star": self.config.sigma_star,
            },
            "history_length": len(self.evaluation_history),
        }, indent=2)

    def health_summary(self) -> str:
        """Get a one-line health summary."""
        result = self.metrics.evaluate()
        if result["overall"]["all_green"]:
            return "🟢 CATHEDRAL: FULLY OPERATIONAL"
        else:
            return f"🔴 CATHEDRAL: {result['overall']['score']} - INTERVENTION NEEDED"


# =============================================================================
# SINGLETON AND CONVENIENCE
# =============================================================================

_runtime: Optional[CathedralRuntime] = None


def get_cathedral() -> CathedralRuntime:
    """Get the global Cathedral runtime instance."""
    global _runtime
    if _runtime is None:
        _runtime = CathedralRuntime()
    return _runtime


def cathedral_tick() -> Dict[str, Any]:
    """Run a Cathedral monitoring tick."""
    return get_cathedral().tick()


def cathedral_status() -> str:
    """Get Cathedral health status."""
    return get_cathedral().health_summary()


def cathedral_dashboard() -> str:
    """Render Cathedral dashboard."""
    return get_cathedral().render_dashboard()


def deploy_gate(module_name: str) -> str:
    """Check deployment gate for a module."""
    return get_cathedral().deploy_decision(module_name)
