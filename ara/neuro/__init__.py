"""
Neuromorphic Processing Module
==============================

On-chip spiking and Hebbian learning primitives.

This module implements the "protocol collapse" from NeuroSymbiosis v1 to v2:
- v1: Card sends STATE_HPV_QUERY → GPU LLM → NEW_POLICY_HDC back
- v2: On-chip Hebbian learning: pre × post × ρ → Δw

Key classes:
    HebbianPolicyLearner: Three-factor learning rule
    SpikeEncoder: HPV → spike stream conversion
    UnifiedHead: Reflex head with shared learnable weights
    ReflexUnit: Complete integrated unit
    HomeostaticLIFConfig: Config for homeostatic LIF network (PyTorch optional)
    HVProbeConfig: Config for hypervector probe (PyTorch optional)

The goal: eliminate GPU round-trip by doing policy adaptation on-chip.
"""

from ara.neuro.hebbian import HebbianPolicyLearner, HebbianConfig
from ara.neuro.spike_encoder import SpikeEncoder, SpikeEncoderConfig
from ara.neuro.unified_head import UnifiedHead, UnifiedHeadConfig, ReflexUnit
from ara.neuro.homeostatic_lif import HomeostaticLIFConfig, HAS_TORCH as HAS_TORCH_LIF
from ara.neuro.hv_probe import HVProbeConfig, HAS_TORCH as HAS_TORCH_PROBE

__all__ = [
    "HebbianPolicyLearner",
    "HebbianConfig",
    "SpikeEncoder",
    "SpikeEncoderConfig",
    "UnifiedHead",
    "UnifiedHeadConfig",
    "ReflexUnit",
    "HomeostaticLIFConfig",
    "HVProbeConfig",
]

# Conditionally export PyTorch-dependent classes
if HAS_TORCH_LIF:
    from ara.neuro.homeostatic_lif import (
        HomeostaticLIFLayer,
        HomeostaticLIFNet,
        HomeostaticLoss,
        HypervectorHead,
        export_for_fpga,
        export_c_header,
    )
    __all__.extend([
        "HomeostaticLIFLayer",
        "HomeostaticLIFNet",
        "HomeostaticLoss",
        "HypervectorHead",
        "export_for_fpga",
        "export_c_header",
    ])

if HAS_TORCH_PROBE:
    from ara.neuro.hv_probe import (
        HVProbe,
        ConceptCodebook,
        compute_cosine_stats,
        pca_plot,
        tsne_plot,
        analyze_status_hv,
    )
    __all__.extend([
        "HVProbe",
        "ConceptCodebook",
        "compute_cosine_stats",
        "pca_plot",
        "tsne_plot",
        "analyze_status_hv",
    ])
