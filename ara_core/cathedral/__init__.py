"""
Cathedral OS - Unified Theory of Antifragile Intelligence
=========================================================

The operating system governing neural networks, agent swarms,
heterogeneous hardware, and economic optimization under stress.

Core Guarantees (Theorem-derived):
- T_s(n) = 1 - C/√n: Complexity → Stability
- A_g(σ*) = +0.021: Stress → Improvement (σ*=0.10)
- H_s = 97.7%: Activity Bounded
- Yield/$ ↑ MoM: Economic Scaling

Four Layers:
1. NEURAL: T-FAN fields, T_s≥0.92, σ*=0.10
2. AGENTS: Cities/Morons, H_influence>1.8, Bias Sentinel
3. HIVE: Junkyard GPUs, Yield/$, Bee Scheduler
4. GOVERNANCE: Homeostatic controller w=10, α=0.12

13 Production Gates:
- Neural (6/6): T_s, A_g, H_s, τ_conv, Controller w, Controller α
- Hive (4/4): E_media, Yield/$, Cluster T_s, GPU Util
- Swarm (3/3): H_influence, T_s_bias, Cost/Reward

Usage:
    from ara_core.cathedral import (
        get_cathedral, cathedral_tick, cathedral_status,
        cathedral_dashboard, deploy_gate
    )

    # Initialize and tick
    runtime = get_cathedral()
    runtime.update_from_quanta(quanta_metrics)
    result = cathedral_tick()

    # Check deployment
    if deploy_gate("ara_voice") == "ara_voice: DEPLOY_OK":
        deploy()

    # View dashboard
    print(cathedral_dashboard())
"""

from .metrics import (
    GateStatus,
    MetricValue,
    ComplexityStabilityGuarantee,
    HormesisGuarantee,
    HomeostasisGuarantee,
    SafeMorphingGuarantee,
    DirectionalityGuarantee,
    EconomicScalingGuarantee,
    NeuralGate,
    HiveGate,
    SwarmGate,
    CathedralMetrics,
)

from .runtime import (
    InterventionType,
    Intervention,
    RuntimeConfig,
    CathedralRuntime,
    get_cathedral,
    cathedral_tick,
    cathedral_status,
    cathedral_dashboard,
    deploy_gate,
)

from .integration import (
    StackConfig,
    CathedralStack,
    get_stack,
    stack_tick,
    stack_dashboard,
    stack_status,
)


__all__ = [
    # Metrics
    "GateStatus",
    "MetricValue",
    "ComplexityStabilityGuarantee",
    "HormesisGuarantee",
    "HomeostasisGuarantee",
    "SafeMorphingGuarantee",
    "DirectionalityGuarantee",
    "EconomicScalingGuarantee",
    "NeuralGate",
    "HiveGate",
    "SwarmGate",
    "CathedralMetrics",

    # Runtime
    "InterventionType",
    "Intervention",
    "RuntimeConfig",
    "CathedralRuntime",
    "get_cathedral",
    "cathedral_tick",
    "cathedral_status",
    "cathedral_dashboard",
    "deploy_gate",

    # Full Stack Integration
    "StackConfig",
    "CathedralStack",
    "get_stack",
    "stack_tick",
    "stack_dashboard",
    "stack_status",
]
