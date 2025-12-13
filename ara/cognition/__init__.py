"""
Cognition: Ara's Thinking Infrastructure

This module connects Ara to her cognitive capabilities:
- LLM-based reasoning (Anthropic, OpenAI, local)
- Mood analysis
- Context understanding
- Criticality monitoring (edge-of-chaos dynamics)

Usage:
    from ara.cognition import get_brain_bridge

    bridge = get_brain_bridge()
    response = await bridge.reason("Help me with this", context_hv={"resonance": 0.8})

Criticality:
    from ara.cognition import CriticalityMonitor, get_criticality_monitor

    monitor = get_criticality_monitor()
    state = monitor.update(spectral_radius=0.95)
    print(f"Regime: {state.regime}, Curvature: {state.curvature}")
"""

from .brain_bridge import (
    BrainBridge,
    LLMConfig,
    LLMProvider,
    ReasoningContext,
    get_brain_bridge,
)

from .criticality import (
    CriticalityMonitor,
    CriticalityConfig,
    CriticalityState,
    CriticalityRegime,
    CriticalityBand,
    Phase,
    FisherProxy,
    CriticalityRegularizer,
    get_criticality_monitor,
    update_criticality,
    criticality_status,
    classify_band,
    band_to_meis_mode,
)

from .rnn_scaling_experiment import (
    run_experiment as run_rnn_scaling_experiment,
    ExperimentResults as RNNScalingResults,
    ScalingResult,
)

from .avalanche_analysis import (
    AvalancheAnalyzer,
    AnalysisResult as AvalancheResult,
    CriticalityMetrics,
    Avalanche,
    PowerLawFit,
    FisherEstimate,
    analyze_fim_vs_criticality,
    simulate_critical_dynamics,
)

__all__ = [
    # Brain Bridge
    'BrainBridge',
    'LLMConfig',
    'LLMProvider',
    'ReasoningContext',
    'get_brain_bridge',
    # Criticality
    'CriticalityMonitor',
    'CriticalityConfig',
    'CriticalityState',
    'CriticalityRegime',
    'CriticalityBand',
    'Phase',
    'FisherProxy',
    'CriticalityRegularizer',
    'get_criticality_monitor',
    'update_criticality',
    'criticality_status',
    'classify_band',
    'band_to_meis_mode',
    # RNN Scaling Experiment
    'run_rnn_scaling_experiment',
    'RNNScalingResults',
    'ScalingResult',
    # Avalanche Analysis
    'AvalancheAnalyzer',
    'AvalancheResult',
    'CriticalityMetrics',
    'Avalanche',
    'PowerLawFit',
    'FisherEstimate',
    'analyze_fim_vs_criticality',
    'simulate_critical_dynamics',
]
