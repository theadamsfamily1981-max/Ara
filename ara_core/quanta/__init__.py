"""
QUANTA v2.0 - Antifragile Memory Consolidation
===============================================

Memory compression layer with the coupled optimization objective:

    max [T_s + A_g(σ*) - |D_NIB| - η_gap + C(r,L)/L]

Metrics:
- T_s: Topology stability (witness complex persistence)
- A_g: Antifragility gain at σ*=0.10
- NIB: Identity preservation (mutual information bound)
- GFT η: Geometric damping (critical = T/R)
- C: Stress-stable capacity (bits per layer)

Consolidation phases:
- Micro: Online RL, σ_high (every second)
- Replay: Experience replay, σ*=0.10 (every minute)
- Structural: SVD rank reduction (every hour)

Usage:
    from ara_core.quanta import (
        compute_quanta_metrics,
        QUANTAConsolidator,
        get_cockpit,
        update_cockpit,
    )

    # Compute metrics
    metrics = compute_quanta_metrics(weights_old, weights_new)
    print(metrics.summary())

    # Run consolidation
    consolidator = QUANTAConsolidator()
    weights_new = consolidator.consolidate(weights, ConsolidationPhase.REPLAY)

    # Update dashboard
    update_cockpit(metrics)
    print(render_cockpit())
"""

from .metrics import (
    MetricStatus,
    TopologyMetric,
    AntifragilityMetric,
    NIBMetric,
    GFTMetric,
    CapacityMetric,
    QUANTAMetrics,
    compute_quanta_metrics,
)

from .consolidation import (
    ConsolidationPhase,
    ConsolidationConfig,
    ConsolidationEvent,
    ConsolidationSchedule,
    QUANTAConsolidator,
    create_consolidator,
    run_consolidation_cycle,
)

from .cockpit import (
    AlertLevel,
    Alert,
    GaugeWidget,
    LineChartWidget,
    HistogramWidget,
    MemoryHealthCockpit,
    get_cockpit,
    update_cockpit,
    render_cockpit,
)


__all__ = [
    # Metrics
    "MetricStatus",
    "TopologyMetric",
    "AntifragilityMetric",
    "NIBMetric",
    "GFTMetric",
    "CapacityMetric",
    "QUANTAMetrics",
    "compute_quanta_metrics",

    # Consolidation
    "ConsolidationPhase",
    "ConsolidationConfig",
    "ConsolidationEvent",
    "ConsolidationSchedule",
    "QUANTAConsolidator",
    "create_consolidator",
    "run_consolidation_cycle",

    # Cockpit
    "AlertLevel",
    "Alert",
    "GaugeWidget",
    "LineChartWidget",
    "HistogramWidget",
    "MemoryHealthCockpit",
    "get_cockpit",
    "update_cockpit",
    "render_cockpit",
]
