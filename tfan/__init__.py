"""
TF-A-N: Transformer with Formal Alignment and Neuromodulation

Long-sequence, multi-modal inference at production latencies with
provable structural fidelity and homeostatic stability.
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring torch for submodule imports
# (e.g., tfan.memory.bloom should work with just numpy)

__all__ = [
    "TFANConfig",
    "TopologyRegularizer",
    "SparseAttention",
    "TLSLandmarkSelector",
    "ProofGatedUpdater",
    "TFANTrainer",
]


def __getattr__(name):
    """Lazy import of torch-dependent modules."""
    if name == "TFANConfig":
        from .config import TFANConfig
        return TFANConfig
    elif name == "TopologyRegularizer":
        from .topo import TopologyRegularizer
        return TopologyRegularizer
    elif name == "SparseAttention":
        from .attention import SparseAttention
        return SparseAttention
    elif name == "TLSLandmarkSelector":
        from .attention import TLSLandmarkSelector
        return TLSLandmarkSelector
    elif name == "ProofGatedUpdater":
        from .pgu import ProofGatedUpdater
        return ProofGatedUpdater
    elif name == "TFANTrainer":
        from .trainer import TFANTrainer
        return TFANTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
