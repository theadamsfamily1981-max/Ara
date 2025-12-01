"""
GPU-Emulated Spiking Neural Network (SNN) for TF-A-N.

Core components for building and training SNNs with TF-A-N integration.

Modules:
- neuron: LIF, PLIF, Izhikevich neurons with surrogate gradients
- layers: SpikingLinear, SpikingConv2d, SpikingResidualBlock
- encode: Rate, latency, delta encoders
- readout: Spike-count, membrane, CTC readouts
- lowrank_synapse: Low-rank masked weights for 97-99% parameter reduction
- lif_lowrank: LIF neurons with parameter-efficient connectivity
- temporal_kernels: Shared synaptic response dictionaries
- mask_tls: TLS-based sparse connectivity masks
- event_queue: Event-driven sparse processing
- auditors: Parameter counting and gate validation
"""

from .neuron import (
    LIF,
    PLIF,
    IzhikevichNeuron,
    NeuronState,
    get_surrogate_fn,
)

from .layers import (
    SpikingLinear,
    SpikingConv2d,
    SpikingResidualBlock,
    SpikingSelfAttention,
)

from .encode import (
    RateEncoder,
    LatencyEncoder,
    DeltaEncoder,
    create_encoder,
)

from .readout import (
    SpikeCountReadout,
    MembraneReadout,
    CTCReadout,
)

# Low-rank emulation for parameter reduction
from .lowrank_synapse import LowRankMaskedSynapse
from .lif_lowrank import LIFLayerLowRank, SurrogateSpikeFn, spike_surrogate
from .temporal_kernels import TemporalBasis, AlphaKernel, exp_kernel_state
from .mask_tls import (
    build_tls_mask_from_scores,
    build_uniform_random_mask,
    build_local_plus_random_mask,
    degree_from_csr,
    mask_density,
    verify_mask_properties,
)
from .event_queue import EventQueue, EventDrivenStepper
from .auditors import (
    dense_params,
    lowrank_params,
    param_reduction_pct,
    assert_param_gate,
    assert_degree_gate,
    assert_rank_gate,
    report,
    verify_all_gates,
    print_report,
)

__version__ = "0.2.0"  # Bumped for low-rank emulation

__all__ = [
    # Neurons
    "LIF",
    "PLIF",
    "IzhikevichNeuron",
    "NeuronState",
    "get_surrogate_fn",
    # Layers
    "SpikingLinear",
    "SpikingConv2d",
    "SpikingResidualBlock",
    "SpikingSelfAttention",
    # Encoders
    "RateEncoder",
    "LatencyEncoder",
    "DeltaEncoder",
    "create_encoder",
    # Readouts
    "SpikeCountReadout",
    "MembraneReadout",
    "CTCReadout",
    # Low-rank emulation
    "LowRankMaskedSynapse",
    "LIFLayerLowRank",
    "SurrogateSpikeFn",
    "spike_surrogate",
    "TemporalBasis",
    "AlphaKernel",
    "exp_kernel_state",
    "build_tls_mask_from_scores",
    "build_uniform_random_mask",
    "build_local_plus_random_mask",
    "degree_from_csr",
    "mask_density",
    "verify_mask_properties",
    "EventQueue",
    "EventDrivenStepper",
    "dense_params",
    "lowrank_params",
    "param_reduction_pct",
    "assert_param_gate",
    "assert_degree_gate",
    "assert_rank_gate",
    "report",
    "verify_all_gates",
    "print_report",
]
